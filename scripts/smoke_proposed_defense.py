#!/usr/bin/env python3
"""Measure a real stage-08 forward/backward memory floor on an official checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from poi.data import FEATURES, load_clean_splits
from poi.proposed_defense.data import feature_matrix, stratified_context
from poi.proposed_defense.losses import DefenseLossWeights, paired_defense_loss
from poi.proposed_defense.model import ProposedTabPFNDefense, load_official_tabpfn25
from poi.proposed_defense.predictor import (
    _validate_checkpoint_state_completeness,
    build_checkpoint_payload,
)
from poi.proposed_defense.schema import POIFeatureCodebook, region_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/poi_34k_seed42")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--token-file",
        default=os.environ.get("TABPFN_TOKEN_FILE", "/root/.config/poi/tabpfn_token"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-rows", type=int, default=256)
    parser.add_argument("--query-rows", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--output", default="artifacts/smoke/proposed_tabpfn25_gpu.json")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but torch.cuda.is_available() is false")
    token_path = Path(args.token_file)
    if not os.environ.get("TABPFN_TOKEN") and args.checkpoint is None:
        try:
            token_available = token_path.is_file()
        except PermissionError:
            token_available = False
        if token_available:
            os.environ["TABPFN_TOKEN"] = token_path.read_text().strip()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    clean, _ = load_clean_splits(args.data_dir, splits=("train",))
    for feature in FEATURES:
        clean[feature] = clean[feature].astype(float)
    regions = tuple(sorted(clean["region"].unique()))
    codebook = POIFeatureCodebook.fit(clean)
    context = stratified_context(clean, args.context_rows, seed=args.seed)
    remaining = clean.loc[~clean["POI_ID"].isin(set(context["POI_ID"]))]
    query = remaining.groupby("region", group_keys=False).sample(
        n=max(1, args.query_rows // len(regions)), random_state=args.seed
    )
    if len(query) < args.query_rows:
        extra = remaining.loc[~remaining["POI_ID"].isin(set(query["POI_ID"]))].sample(
            n=args.query_rows - len(query), random_state=args.seed
        )
        query = __import__("pandas").concat((query, extra), ignore_index=True)
    query = query.iloc[: args.query_rows]
    context_encoded = codebook.encode(feature_matrix(context), device=device)
    clean_query = codebook.encode(feature_matrix(query), device=device)
    attack_query = clean_query.clone()
    attack_query[:, :2] += torch.tensor([1e-4, -1e-4], device=device)
    valid_tuples = torch.as_tensor(
        codebook.valid_categorical_tuples, dtype=attack_query.dtype, device=device
    )
    attack_query[:, 2:] = valid_tuples[
        torch.arange(len(attack_query), device=device) % len(valid_tuples)
    ]
    context_labels = torch.as_tensor(
        region_indices(context["region"], regions), dtype=torch.long, device=device
    )
    query_labels = torch.as_tensor(
        region_indices(query["region"], regions), dtype=torch.long, device=device
    )
    changed = (attack_query != clean_query).float()

    base, provenance = load_official_tabpfn25(
        checkpoint_path=args.checkpoint,
        device=device,
        download_if_missing=args.checkpoint is None,
    )
    model = ProposedTabPFNDefense(
        base,
        codebook,
        regions,
        hidden_dim=args.hidden_dim,
        activation_checkpointing=True,
    ).to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=3e-4
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.mixed_precision
        else contextlib.nullcontext()
    )
    with autocast:
        clean_output = model(
            torch.cat((context_encoded, clean_query), dim=0), context_labels
        )
        attack_output = model(
            torch.cat((context_encoded, attack_query), dim=0), context_labels
        )
        loss, terms = paired_defense_loss(
            clean_output,
            attack_output,
            query_region_indices=query_labels,
            clean_encoded_query=clean_query,
            changed_feature_mask=changed,
            weights=DefenseLossWeights(),
        )
    if not torch.isfinite(loss) or any(not torch.isfinite(value) for value in terms.values()):
        raise RuntimeError("Official-checkpoint smoke produced a non-finite loss")
    region_only = terms["clean_region"] + terms["attack_region"]
    adapter_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("adapter.") and parameter.requires_grad
    ]
    region_gradients = torch.autograd.grad(
        region_only,
        [parameter for _, parameter in adapter_named],
        retain_graph=True,
        allow_unused=True,
    )
    finite_region_gradients = {
        name: float(gradient.detach().float().norm())
        for (name, _), gradient in zip(adapter_named, region_gradients)
        if gradient is not None and torch.isfinite(gradient).all()
    }
    nonzero_region_gradients = {
        name: norm for name, norm in finite_region_gradients.items() if norm > 0
    }
    if not nonzero_region_gradients:
        raise RuntimeError("Region-only loss has no finite nonzero adapter gradient")
    loss.backward()
    adapter_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("adapter.") and parameter.grad is not None
    ]
    if not adapter_gradients or any(not torch.isfinite(value).all() for value in adapter_gradients):
        raise RuntimeError("Combined loss produced missing or non-finite adapter gradients")
    optimizer.step()
    payload = build_checkpoint_payload(
        model,
        context_encoded=context_encoded,
        context_region_indices=context_labels,
        official_provenance=provenance,
        training_metadata={"scope": "official_checkpoint_current_source_smoke"},
    )
    serialized = io.BytesIO()
    torch.save(payload, serialized)
    serialized_sha256 = hashlib.sha256(serialized.getvalue()).hexdigest()
    serialized.seek(0)
    restored = torch.load(serialized, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(restored["trainable_state"], strict=False)
    _validate_checkpoint_state_completeness(
        model, restored["trainable_state"], list(missing), list(unexpected)
    )
    if not provenance.get("runtime_source_commit_verified"):
        raise RuntimeError("Imported official TabPFN source revision could not be attested")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "status": "pass",
        "context_rows": args.context_rows,
        "query_rows": args.query_rows,
        "ovr_prompts": 17,
        "two_forwards_before_backward": True,
        "activation_checkpointing": True,
        "checkpoint_compatibility_patch": model.checkpoint_compatibility_patch,
        "mixed_precision": args.mixed_precision,
        "loss": float(loss.detach()),
        "loss_terms": {name: float(value.detach()) for name, value in terms.items()},
        "all_loss_terms_finite": True,
        "region_only_adapter_gradient_parameter_count": len(nonzero_region_gradients),
        "region_only_adapter_gradient_norm_sum": sum(nonzero_region_gradients.values()),
        "all_combined_adapter_gradients_finite": True,
        "checkpoint_roundtrip": "pass",
        "serialized_trainable_checkpoint_sha256": serialized_sha256,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_bytes": (
            torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None
        ),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "official_tabpfn": provenance,
        "implementation_source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1] / "src/poi/proposed_defense/structural.py",
                Path(__file__).resolve().parents[1] / "src/poi/proposed_defense/model.py",
                Path(__file__).resolve().parents[1] / "src/poi/proposed_defense/losses.py",
                Path(__file__).resolve().parents[1] / "src/poi/proposed_defense/predictor.py",
                Path(__file__).resolve().parents[1] / "src/poi/proposed_defense/schema.py",
            )
        },
        "base_parameter_gradients": any(
            parameter.grad is not None for parameter in model.base_model.parameters()
        ),
    }
    if result["base_parameter_gradients"]:
        raise RuntimeError("Frozen official TabPFN parameters unexpectedly received gradients")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
