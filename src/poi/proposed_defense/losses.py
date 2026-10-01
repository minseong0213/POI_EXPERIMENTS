"""Loss terms from stage 08 of the dated robustness plan."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import ProposedDefenseOutput


@dataclass(frozen=True)
class DefenseLossWeights:
    adversarial_region: float = 1.0
    consistency: float = 0.25
    feature_detection: float = 1.0
    reconstruction: float = 1.0
    attack_detection: float = 1.0


def paired_defense_loss(
    clean: ProposedDefenseOutput,
    attacked: ProposedDefenseOutput,
    *,
    query_region_indices: torch.Tensor,
    clean_encoded_query: torch.Tensor,
    changed_feature_mask: torch.Tensor,
    weights: DefenseLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute every planned loss without hiding terms inside a combined metric."""
    query_rows, region_count = clean.region_logits.shape
    if attacked.region_logits.shape != clean.region_logits.shape:
        raise ValueError("Clean and attacked query logits must have the same shape")
    if query_region_indices.shape != (query_rows,):
        raise ValueError("query_region_indices has an incompatible shape")
    if changed_feature_mask.shape != (query_rows, 5):
        raise ValueError("changed_feature_mask must have shape [query rows, 5]")
    targets = F.one_hot(query_region_indices, num_classes=region_count).float()
    clean_region = F.binary_cross_entropy_with_logits(clean.region_logits, targets)
    attack_region = F.binary_cross_entropy_with_logits(attacked.region_logits, targets)
    consistency = F.mse_loss(
        torch.sigmoid(attacked.region_logits), torch.sigmoid(clean.region_logits).detach()
    )

    clean_feature_logits = clean.structural.feature_corruption_logits[-query_rows:]
    attacked_feature_logits = attacked.structural.feature_corruption_logits[-query_rows:]
    feature_detection = 0.5 * (
        F.binary_cross_entropy_with_logits(clean_feature_logits, torch.zeros_like(changed_feature_mask))
        + F.binary_cross_entropy_with_logits(attacked_feature_logits, changed_feature_mask.float())
    )
    clean_attack_logits = clean.structural.attack_logits[-query_rows:]
    attacked_attack_logits = attacked.structural.attack_logits[-query_rows:]
    attack_detection = 0.5 * (
        F.binary_cross_entropy_with_logits(clean_attack_logits, torch.zeros_like(clean_attack_logits))
        + F.binary_cross_entropy_with_logits(
            attacked_attack_logits, torch.ones_like(attacked_attack_logits)
        )
    )

    reconstruction = reconstruction_loss(
        attacked,
        clean_encoded_query=clean_encoded_query,
        changed_feature_mask=changed_feature_mask,
    )

    terms = {
        "clean_region": clean_region,
        "attack_region": attack_region,
        "consistency": consistency,
        "feature_detection": feature_detection,
        "reconstruction": reconstruction,
        "attack_detection": attack_detection,
    }
    total = (
        terms["clean_region"]
        + weights.adversarial_region * terms["attack_region"]
        + weights.consistency * terms["consistency"]
        + weights.feature_detection * terms["feature_detection"]
        + weights.reconstruction * terms["reconstruction"]
        + weights.attack_detection * terms["attack_detection"]
    )
    return total, terms


def reconstruction_loss(
    attacked: ProposedDefenseOutput,
    *,
    clean_encoded_query: torch.Tensor,
    changed_feature_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked clean-target reconstruction for numeric and categorical features."""
    # ``reconstruction_numeric`` is standardized.  Production model.forward attaches
    # the exact train-only scaling buffers to the structural output.
    statistics = getattr(attacked.structural, "numeric_statistics", None)
    if statistics is None:
        # Unit tests and callers can supply already-standardized numeric targets via a
        # private marker.  Production model.forward always attaches exact statistics.
        mean = clean_encoded_query.new_zeros(2)
        scale = clean_encoded_query.new_ones(2)
    else:
        mean, scale = statistics
    numeric_target = (clean_encoded_query[:, :2] - mean) / scale
    numeric_mask = changed_feature_mask[:, :2].float()
    numeric_error = F.smooth_l1_loss(
        attacked.structural.reconstruction_numeric[-len(clean_encoded_query):],
        numeric_target,
        reduction="none",
    )
    numeric_denom = numeric_mask.sum().clamp_min(1.0)
    numeric_loss = (numeric_error * numeric_mask).sum() / numeric_denom

    categorical_losses = []
    for feature, logits in enumerate(attacked.structural.reconstruction_categorical_logits):
        mask = changed_feature_mask[:, feature + 2].bool()
        if mask.any():
            categorical_losses.append(
                F.cross_entropy(
                    logits[-len(clean_encoded_query):][mask],
                    clean_encoded_query[:, feature + 2].long()[mask],
                )
            )
    if categorical_losses:
        categorical_loss = torch.stack(categorical_losses).mean()
    else:
        categorical_loss = numeric_loss.new_zeros(())
    return numeric_loss + categorical_loss
