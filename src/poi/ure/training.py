"""Bilevel alternating URE training with a cost-constrained relaxed inner attack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .model import URERegionModel


@dataclass(frozen=True)
class URETrainingConfig:
    epochs: int = 30
    batch_size: int = 512
    classifier_learning_rate: float = 1e-3
    embedding_learning_rate: float = 1e-2
    weight_decay: float = 5e-4
    attack_budget: float = 1.0
    attack_steps: int = 5
    attack_step_size: float = 0.5
    embedding_adversarial_steps_per_clean_step: int = 1


def train_universal_robust_embeddings(
    model: URERegionModel,
    indices: np.ndarray | torch.Tensor,
    region_labels: np.ndarray | torch.Tensor,
    transition_costs: tuple[np.ndarray | torch.Tensor, ...],
    *,
    config: URETrainingConfig,
    device: str | torch.device,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Train URE with separate classifier and embedding optimizers.

    The official repository alternates clean classifier updates and adversarial
    embedding-only updates and normalizes embedding columns.  This implementation keeps
    those mechanics while using a device-neutral relaxed cost attack, because the
    official inner loop is dataset-specific and hard-codes CUDA.
    """
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("URE epochs and batch_size must be positive")
    if (
        config.classifier_learning_rate <= 0
        or config.embedding_learning_rate <= 0
        or config.attack_budget <= 0
        or config.attack_steps <= 0
        or config.attack_step_size <= 0
        or config.embedding_adversarial_steps_per_clean_step <= 0
    ):
        raise ValueError("URE optimization and adversarial settings must be positive")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(indices, dtype=torch.long)
    y = torch.as_tensor(region_labels, dtype=torch.long)
    if x.ndim != 2 or x.shape[1] != len(model.embedding.cardinalities) or len(x) != len(y):
        raise ValueError("URE training arrays have incompatible shapes")
    costs = tuple(torch.as_tensor(matrix, dtype=torch.float32, device=device) for matrix in transition_costs)
    _validate_costs(costs, model.embedding.cardinalities)
    model.to(device)
    classifier_optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=config.classifier_learning_rate,
        weight_decay=config.weight_decay,
    )
    embedding_optimizer = torch.optim.Adam(
        model.embedding.parameters(), lr=config.embedding_learning_rate, weight_decay=0.0
    )
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        permutation = rng.permutation(len(x))
        clean_total = adversarial_total = 0.0
        batches = 0
        for start in range(0, len(x), config.batch_size):
            batch_index = permutation[start : start + config.batch_size]
            xb = x[batch_index].to(device)
            yb = y[batch_index].to(device)

            _set_requires_grad(model.embedding, False)
            _set_requires_grad(model.classifier, True)
            classifier_optimizer.zero_grad(set_to_none=True)
            clean_loss = F.cross_entropy(model(xb), yb)
            # The official URE `emb_only --emb_fine` path updates downstream
            # classifier parameters only on clean batches.
            clean_loss.backward()
            classifier_optimizer.step()

            adversarial_batch_total = 0.0
            for _ in range(config.embedding_adversarial_steps_per_clean_step):
                _set_requires_grad(model.embedding, True)
                _set_requires_grad(model.classifier, False)
                embedding_optimizer.zero_grad(set_to_none=True)
                relaxed = relaxed_cost_attack(
                    model,
                    xb,
                    yb,
                    costs,
                    budget=config.attack_budget,
                    steps=config.attack_steps,
                    step_size=config.attack_step_size,
                )
                embedding_adversarial = F.cross_entropy(model.forward_relaxed(relaxed), yb)
                # Conversely, adversarial batches update only embedding columns.
                embedding_adversarial.backward()
                torch.nn.utils.clip_grad_norm_(model.embedding.parameters(), 1.0)
                embedding_optimizer.step()
                model.embedding.normalize_columns()
                adversarial_batch_total += float(embedding_adversarial.detach())

            clean_total += float(clean_loss.detach())
            adversarial_total += adversarial_batch_total / max(
                config.embedding_adversarial_steps_per_clean_step, 1
            )
            batches += 1
        _set_requires_grad(model.embedding, True)
        _set_requires_grad(model.classifier, True)
        history.append(
            {
                "epoch": float(epoch + 1),
                "clean_region_loss": clean_total / max(batches, 1),
                "adversarial_region_loss": adversarial_total / max(batches, 1),
            }
        )
    return history


def relaxed_cost_attack(
    model: URERegionModel,
    indices: torch.Tensor,
    labels: torch.Tensor,
    transition_costs: tuple[torch.Tensor, ...],
    *,
    budget: float,
    steps: int,
    step_size: float,
) -> tuple[torch.Tensor, ...]:
    if budget < 0 or steps < 0 or step_size <= 0:
        raise ValueError("Invalid URE attack budget/steps/step_size")
    originals = tuple(
        F.one_hot(indices[:, feature], num_classes=cardinality).float()
        for feature, cardinality in enumerate(model.embedding.cardinalities)
    )
    probabilities = tuple(value.detach().clone() for value in originals)
    for _ in range(steps):
        variables = tuple(value.detach().requires_grad_(True) for value in probabilities)
        loss = F.cross_entropy(model.forward_relaxed(variables), labels)
        gradients = torch.autograd.grad(loss, variables)
        proposals = tuple(
            _project_simplex(value + step_size * gradient / gradient.abs().sum(dim=1, keepdim=True).clamp_min(1e-12))
            for value, gradient in zip(variables, gradients)
        )
        probabilities = _project_global_cost_ray(
            proposals, originals, indices, transition_costs, budget
        )
    return tuple(value.detach() for value in probabilities)


def expected_transition_cost(
    probabilities: tuple[torch.Tensor, ...],
    source_indices: torch.Tensor,
    transition_costs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    result = torch.zeros(len(source_indices), device=source_indices.device)
    for feature, (probability, cost_matrix) in enumerate(zip(probabilities, transition_costs)):
        source_costs = cost_matrix[source_indices[:, feature]]
        result = result + (probability * source_costs).sum(dim=1)
    return result


def _project_simplex(values: torch.Tensor) -> torch.Tensor:
    sorted_values, _ = torch.sort(values, dim=1, descending=True)
    cumulative = sorted_values.cumsum(dim=1) - 1.0
    rank = torch.arange(1, values.shape[1] + 1, device=values.device, dtype=values.dtype)
    support = sorted_values - cumulative / rank > 0
    rho = support.sum(dim=1).clamp_min(1)
    theta = cumulative.gather(1, (rho - 1).unsqueeze(1)).squeeze(1) / rho
    return (values - theta.unsqueeze(1)).clamp_min(0.0)


def _project_global_cost_ray(
    proposals: tuple[torch.Tensor, ...],
    originals: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    costs: tuple[torch.Tensor, ...],
    budget: float,
) -> tuple[torch.Tensor, ...]:
    total = expected_transition_cost(proposals, indices, costs)
    factor = torch.clamp(torch.as_tensor(budget, device=total.device) / total.clamp_min(1e-12), max=1.0)
    return tuple(
        original + factor.unsqueeze(1) * (proposal - original)
        for proposal, original in zip(proposals, originals)
    )


def _validate_costs(costs: tuple[torch.Tensor, ...], cardinalities: Iterable[int]) -> None:
    cardinalities = tuple(cardinalities)
    if len(costs) != len(cardinalities):
        raise ValueError("URE needs one transition-cost matrix per field")
    for feature, (matrix, cardinality) in enumerate(zip(costs, cardinalities)):
        if matrix.shape != (cardinality, cardinality):
            raise ValueError(f"URE cost matrix {feature} has an invalid shape")
        if torch.any(matrix < 0) or not torch.isfinite(matrix).all():
            raise ValueError(f"URE cost matrix {feature} is invalid")
        if not torch.allclose(torch.diag(matrix), torch.zeros(cardinality, device=matrix.device)):
            raise ValueError(f"URE cost matrix {feature} diagonal must be zero")


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
