"""Checkpoint-backed inference API for proposed structural-defense explainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import ProposedTabPFNDefense, load_official_tabpfn25
from .schema import POIFeatureCodebook

CHECKPOINT_FORMAT = "poi-proposed-tabpfn25-v1"


class ProposedDefensePredictor:
    def __init__(
        self,
        model: ProposedTabPFNDefense,
        context_encoded: torch.Tensor,
        context_region_indices: torch.Tensor,
        *,
        device: str | torch.device,
        batch_size: int = 128,
    ) -> None:
        if len(context_encoded) != len(context_region_indices):
            raise ValueError("Stored context features and labels differ in length")
        self.model = model.to(device).eval()
        self.model.activation_checkpointing = False
        self.context_encoded = context_encoded.to(device)
        self.context_region_indices = context_region_indices.to(device)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.regions = model.regions

    def predict(self, matrix5: Any) -> dict[str, np.ndarray]:
        encoded = self.model.codebook.encode(matrix5, device=self.device)
        region_parts: list[torch.Tensor] = []
        attack_parts: list[torch.Tensor] = []
        feature_parts: list[torch.Tensor] = []
        reliability_parts: list[torch.Tensor] = []
        repaired_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(encoded), self.batch_size):
                query = encoded[start : start + self.batch_size]
                combined = torch.cat((self.context_encoded, query), dim=0)
                result = self.model(combined, self.context_region_indices)
                query_count = len(query)
                region_parts.append(torch.sigmoid(result.region_logits).cpu())
                attack_parts.append(
                    torch.sigmoid(result.structural.attack_logits[-query_count:]).cpu()
                )
                feature_parts.append(
                    torch.sigmoid(
                        result.structural.feature_corruption_logits[-query_count:]
                    ).cpu()
                )
                reliability_parts.append(result.structural.reliability[-query_count:].cpu())
                repaired_parts.append(
                    result.structural.repaired_features[-query_count:].cpu()
                )
        return {
            "region_probability": torch.cat(region_parts).numpy(),
            "attack_probability": torch.cat(attack_parts).numpy(),
            "feature_corruption_probability": torch.cat(feature_parts).numpy(),
            "reliability": torch.cat(reliability_parts).numpy(),
            "repaired_matrix": torch.cat(repaired_parts).numpy(),
        }

    def predict_region_proba(self, matrix5: Any) -> np.ndarray:
        """Factory-hook API used by the expanded explanation stage."""
        return self.predict(matrix5)["region_probability"]


def load_proposed_predictor(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    official_checkpoint: str | Path | None = None,
    batch_size: int = 128,
) -> ProposedDefensePredictor:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported proposed-defense checkpoint: {payload.get('format')}")
    codebook = POIFeatureCodebook.from_dict(payload["codebook"])
    official_path = official_checkpoint or payload["official_tabpfn"].get("checkpoint_path")
    base, provenance = load_official_tabpfn25(
        checkpoint_path=official_path,
        device=device,
        download_if_missing=official_path is None,
    )
    expected_sha = payload["official_tabpfn"].get("checkpoint_sha256")
    if expected_sha and provenance.get("checkpoint_sha256") != expected_sha:
        raise RuntimeError("Official TabPFN checkpoint hash differs from the training run")
    config = payload["model_config"]
    model = ProposedTabPFNDefense(
        base,
        codebook,
        tuple(payload["regions"]),
        hidden_dim=int(config["hidden_dim"]),
        numeric_scales=tuple(config["numeric_scales"]),
        activation_checkpointing=False,
    )
    unfrozen = int(config.get("unfrozen_blocks", 0))
    if unfrozen:
        model.unfreeze_last_blocks(unfrozen)
    missing, unexpected = model.load_state_dict(payload["trainable_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected proposed checkpoint keys: {unexpected}")
    required = {"region_weights", "region_bias"}
    if not required.issubset(payload["trainable_state"]):
        raise RuntimeError("Proposed checkpoint is missing its 17 OvR output heads")
    # Missing keys are expected for the frozen official checkpoint, which is loaded by
    # hash above.  Missing adapter/head keys are not allowed.
    bad_missing = [key for key in missing if not key.startswith("base_model.")]
    if bad_missing:
        raise RuntimeError(f"Proposed checkpoint is incomplete: {bad_missing}")
    return ProposedDefensePredictor(
        model,
        payload["context_encoded"],
        payload["context_region_indices"],
        device=device,
        batch_size=batch_size,
    )


def build_checkpoint_payload(
    model: ProposedTabPFNDefense,
    *,
    context_encoded: torch.Tensor,
    context_region_indices: torch.Tensor,
    official_provenance: dict[str, Any],
    training_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "regions": list(model.regions),
        "codebook": model.codebook.to_dict(),
        "model_config": {
            "hidden_dim": model.adapter.hidden_dim,
            "numeric_scales": list(model.adapter.numeric_scales),
            "unfrozen_blocks": model.unfrozen_block_count,
        },
        "official_tabpfn": official_provenance,
        "context_encoded": context_encoded.detach().cpu(),
        "context_region_indices": context_region_indices.detach().cpu(),
        "trainable_state": model.trainable_state_dict(),
        "training_metadata": training_metadata,
    }
