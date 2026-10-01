#!/usr/bin/env python3
"""Measure a real stage-08 forward/backward memory floor on an official checkpoint."""

from __future__ import annotations

import argparse
import contextlib
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
from poi.proposed_defense.schema import POIFeatureCodebook, region_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/poi_34k_seed42")
    parser.add_argument("--checkpoint")
    parser.add_argument("--token-file", default="/root/.config/poi/tabpfn_token")
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
    codebook = POIFeatureCodebook.fit(feature_matrix(clean))
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
    for feature, cardinality in enumerate(codebook.cardinalities):
        attack_query[:, feature + 2] = (attack_query[:, feature + 2] + 1) % cardinality
    context_labels = torch.as_tensor(
        region_indices(context["region"], regions), dtype=torch.long, device=device
    )
    query_labels = torch.as_tensor(
        region_indices(query["region"], regions), dtype=torch.long, device=device
    )
    changed = torch.ones(args.query_rows, 5, device=device)

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
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "status": "pass",
        "context_rows": args.context_rows,
        "query_rows": args.query_rows,
        "ovr_prompts": 17,
        "two_forwards_before_backward": True,
        "activation_checkpointing": True,
        "mixed_precision": args.mixed_precision,
        "loss": float(loss.detach()),
        "loss_terms": {name: float(value.detach()) for name, value in terms.items()},
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
