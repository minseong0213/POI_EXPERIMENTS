"""Checkpoint-backed inference API for proposed structural-defense explainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import ProposedTabPFNDefense, load_official_tabpfn25
from .schema import POIFeatureCodebook

CHECKPOINT_FORMAT = "poi-proposed-tabpfn25-v3"
REGIONAL_DECISION_RULE = "independent_ovr_probability_gte_threshold"


class ProposedDefensePredictor:
    def __init__(
        self,
        model: ProposedTabPFNDefense,
        context_encoded: torch.Tensor,
        context_region_indices: torch.Tensor,
        *,
        device: str | torch.device,
        batch_size: int = 128,
        classification_threshold: float = 0.5,
        regional_decision_rule: str = REGIONAL_DECISION_RULE,
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
        if regional_decision_rule != REGIONAL_DECISION_RULE:
            raise ValueError(f"Unsupported regional decision rule: {regional_decision_rule}")
        if not 0.0 < float(classification_threshold) < 1.0:
            raise ValueError("classification_threshold must be strictly between zero and one")
        self.classification_threshold = float(classification_threshold)
        self.regional_decision_rule = regional_decision_rule

    def assert_decision_contract(
        self, *, classification_threshold: float, regional_decision_rule: str
    ) -> None:
        """Fail if a caller would interpret the stored OvR heads differently."""
        if regional_decision_rule != self.regional_decision_rule:
            raise RuntimeError("Regional decision rule differs from proposed checkpoint")
        if float(classification_threshold) != self.classification_threshold:
            raise RuntimeError("Classification threshold differs from proposed checkpoint")

    def predict(self, matrix5: Any) -> dict[str, np.ndarray]:
        """Score rows and return a train-support-constrained deployment repair.

        ``region_probability`` is evaluated by the proposed model's own OvR heads after
        hard train-support projection. ``latent_region_probability`` exposes the soft
        differentiable training path under a separate name. Their discrepancy and the
        preprojection arrays make projection pressure directly measurable.
        """
        encoded = self.model.codebook.encode(matrix5, device=self.device)
        region_parts: list[torch.Tensor] = []
        latent_region_parts: list[torch.Tensor] = []
        discrepancy_parts: list[torch.Tensor] = []
        attack_parts: list[torch.Tensor] = []
        feature_parts: list[torch.Tensor] = []
        reliability_parts: list[torch.Tensor] = []
        repaired_parts: list[torch.Tensor] = []
        numeric_out_parts: list[torch.Tensor] = []
        tuple_invalid_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(encoded), self.batch_size):
                query = encoded[start : start + self.batch_size]
                combined = torch.cat((self.context_encoded, query), dim=0)
                result = self.model(combined, self.context_region_indices)
                deployment = self.model.adapter.project_for_deployment(
                    combined, result.structural
                )
                projected = self.model.forward_projected_features(
                    deployment.repaired_features, self.context_region_indices
                )
                query_count = len(query)
                deployed_probability = torch.sigmoid(projected.region_logits)
                latent_probability = torch.sigmoid(result.region_logits)
                region_parts.append(deployed_probability.cpu())
                latent_region_parts.append(latent_probability.cpu())
                discrepancy_parts.append(
                    (deployed_probability - latent_probability).abs().cpu()
                )
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
                    deployment.repaired_features[-query_count:].cpu()
                )
                numeric_out_parts.append(
                    deployment.preprojection_numeric_out_of_bounds[-query_count:].cpu()
                )
                tuple_invalid_parts.append(
                    deployment.preprojection_invalid_category_tuple[-query_count:].cpu()
                )
        region_probability = torch.cat(region_parts).numpy()
        return {
            "region_probability": region_probability,
            "region_prediction": (
                region_probability >= self.classification_threshold
            ).astype(np.int64),
            "latent_region_probability": torch.cat(latent_region_parts).numpy(),
            "projection_region_probability_absolute_delta": torch.cat(
                discrepancy_parts
            ).numpy(),
            "attack_probability": torch.cat(attack_parts).numpy(),
            "feature_corruption_probability": torch.cat(feature_parts).numpy(),
            "reliability": torch.cat(reliability_parts).numpy(),
            "repaired_matrix": torch.cat(repaired_parts).numpy(),
            "preprojection_numeric_out_of_bounds": torch.cat(
                numeric_out_parts
            ).numpy(),
            "preprojection_invalid_category_tuple": torch.cat(
                tuple_invalid_parts
            ).numpy(),
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
    stored_path = payload["official_tabpfn"].get("checkpoint_path")
    official_path = official_checkpoint or stored_path
    if official_checkpoint is None and official_path and not Path(official_path).is_file():
        official_path = None
    base, provenance = load_official_tabpfn25(
        checkpoint_path=official_path,
        device=device,
        download_if_missing=official_path is None,
    )
    expected_sha = payload["official_tabpfn"].get("checkpoint_sha256")
    if expected_sha and provenance.get("checkpoint_sha256") != expected_sha:
        raise RuntimeError("Official TabPFN checkpoint hash differs from the training run")
    config = payload["model_config"]
    threshold = config.get("classification_threshold")
    rule = config.get("regional_decision_rule")
    if threshold is None or rule is None:
        raise RuntimeError("Proposed checkpoint lacks its regional decision contract")
    training_metadata = payload.get("training_metadata", {})
    if (
        training_metadata.get("classification_threshold") != threshold
        or training_metadata.get("regional_decision_rule") != rule
    ):
        raise RuntimeError("Proposed checkpoint has inconsistent regional decision metadata")
    model = ProposedTabPFNDefense(
        base,
        codebook,
        tuple(payload["regions"]),
        hidden_dim=int(config["hidden_dim"]),
        numeric_scales=tuple(config["numeric_scales"]),
        activation_checkpointing=False,
        use_reliability_gate=bool(config.get("use_reliability_gate", True)),
        use_reconstruction_path=bool(config.get("use_reconstruction_path", True)),
        use_multiscale_encoder=bool(config.get("use_multiscale_encoder", True)),
        use_feature_encoders=bool(config.get("use_feature_encoders", True)),
        use_category_relations=bool(config.get("use_category_relations", True)),
        use_corruption_heads=bool(config.get("use_corruption_heads", True)),
    )
    unfrozen = int(config.get("unfrozen_blocks", 0))
    if unfrozen:
        model.unfreeze_last_blocks(unfrozen)
    state = payload["trainable_state"]
    _validate_finite_state(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    _validate_checkpoint_state_completeness(model, state, missing, unexpected)
    return ProposedDefensePredictor(
        model,
        payload["context_encoded"],
        payload["context_region_indices"],
        device=device,
        batch_size=batch_size,
        classification_threshold=float(threshold),
        regional_decision_rule=str(rule),
    )


def build_checkpoint_payload(
    model: ProposedTabPFNDefense,
    *,
    context_encoded: torch.Tensor,
    context_region_indices: torch.Tensor,
    official_provenance: dict[str, Any],
    training_metadata: dict[str, Any],
) -> dict[str, Any]:
    trainable_state = model.trainable_state_dict()
    _validate_finite_state(trainable_state)
    expected_base = _expected_unfrozen_base_keys(model)
    missing_base = expected_base.difference(trainable_state)
    if missing_base:
        raise RuntimeError(f"Cannot serialize missing trained block keys: {sorted(missing_base)}")
    threshold = float(training_metadata.get("classification_threshold", 0.5))
    rule = str(training_metadata.get("regional_decision_rule", REGIONAL_DECISION_RULE))
    if rule != REGIONAL_DECISION_RULE:
        raise ValueError(f"Unsupported regional decision rule: {rule}")
    if not 0.0 < threshold < 1.0:
        raise ValueError("classification_threshold must be strictly between zero and one")
    stored_training_metadata = {
        **training_metadata,
        "classification_threshold": threshold,
        "regional_decision_rule": rule,
    }
    return {
        "format": CHECKPOINT_FORMAT,
        "regions": list(model.regions),
        "codebook": model.codebook.to_dict(),
        "model_config": {
            "hidden_dim": model.adapter.hidden_dim,
            "numeric_scales": list(model.adapter.numeric_scales),
            "unfrozen_blocks": model.unfrozen_block_count,
            "checkpoint_compatibility_patch": model.checkpoint_compatibility_patch,
            **model.architecture_flags,
            "differentiable_forward_repair": (
                "soft expected category codes and unclipped numeric reconstruction"
            ),
            "deployment_projection": (
                "clip numeric values to clean-train min/max; select maximum summed "
                "log-probability categorical tuple observed in clean train"
            ),
            "deployment_projection_uses_region_or_paired_clean_oracle": False,
            "classification_threshold": threshold,
            "regional_decision_rule": rule,
        },
        "official_tabpfn": official_provenance,
        "context_encoded": context_encoded.detach().cpu(),
        "context_region_indices": context_region_indices.detach().cpu(),
        "trainable_state": trainable_state,
        "training_metadata": stored_training_metadata,
    }


def _expected_unfrozen_base_keys(model: ProposedTabPFNDefense) -> set[str]:
    blocks = list(getattr(model.base_model, "blocks", []))
    if not model.unfrozen_block_count:
        return set()
    first = len(blocks) - model.unfrozen_block_count
    return {
        key
        for key in model.state_dict()
        if key.startswith("base_model.blocks.") and int(key.split(".")[2]) >= first
    }


def _validate_finite_state(state: dict[str, torch.Tensor]) -> None:
    bad = [key for key, value in state.items() if not torch.isfinite(value).all()]
    if bad:
        raise RuntimeError(f"Proposed checkpoint has non-finite tensors: {bad}")


def _validate_checkpoint_state_completeness(
    model: ProposedTabPFNDefense,
    state: dict[str, torch.Tensor],
    missing: list[str],
    unexpected: list[str],
) -> None:
    if unexpected:
        raise RuntimeError(f"Unexpected proposed checkpoint keys: {unexpected}")
    required = {"region_weights", "region_bias"}
    if not required.issubset(state):
        raise RuntimeError("Proposed checkpoint is missing its 17 OvR output heads")
    expected_base = _expected_unfrozen_base_keys(model)
    missing_base = sorted(expected_base.difference(state))
    if missing_base:
        raise RuntimeError(
            f"Proposed checkpoint is missing trained official block keys: {missing_base}"
        )
    bad_missing = [
        key for key in missing
        if not key.startswith("base_model.") or key in expected_base
    ]
    if bad_missing:
        raise RuntimeError(f"Proposed checkpoint is incomplete: {bad_missing}")
