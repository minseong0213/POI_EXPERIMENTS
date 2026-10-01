"""Numeric/categorical corruption, reconstruction, and reliability-gating layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .schema import POIFeatureCodebook


@dataclass
class StructuralDefenseOutput:
    repaired_features: torch.Tensor
    feature_corruption_logits: torch.Tensor
    attack_logits: torch.Tensor
    reconstruction_numeric: torch.Tensor
    reconstruction_categorical_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    reliability: torch.Tensor
    observed_tokens: torch.Tensor
    reconstructed_tokens: torch.Tensor


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


class StructuralDefenseAdapter(nn.Module):
    """Repair five encoded POI features before the official TabPFN input encoder.

    Numeric coordinates use several distance scales.  Each categorical field has its
    own embedding and also receives a learned three-field relation representation.  A
    feature is reconstructed only from the other four feature tokens.  Its corruption
    probability then controls the reliability gate between the observation and the
    reconstruction.
    """

    def __init__(
        self,
        codebook: POIFeatureCodebook,
        *,
        hidden_dim: int = 64,
        numeric_scales: tuple[float, ...] = (0.25, 1.0, 4.0, 16.0),
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if not numeric_scales or any(scale <= 0 for scale in numeric_scales):
            raise ValueError("numeric_scales must contain positive values")
        self.codebook = codebook
        self.hidden_dim = hidden_dim
        self.numeric_scales = tuple(float(scale) for scale in numeric_scales)
        self.register_buffer("numeric_mean", torch.tensor(codebook.numeric_mean))
        self.register_buffer("numeric_scale", torch.tensor(codebook.numeric_scale))
        for index, values in enumerate(codebook.categorical_values):
            self.register_buffer(f"category_values_{index}", torch.tensor(values))

        numeric_input_dim = 3 * len(self.numeric_scales)
        self.numeric_encoders = nn.ModuleList(
            [_mlp(numeric_input_dim, hidden_dim, hidden_dim) for _ in range(2)]
        )
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, hidden_dim) for cardinality in codebook.cardinalities]
        )
        self.category_relation = _mlp(3 * hidden_dim, 2 * hidden_dim, hidden_dim)
        self.category_relation_gates = nn.Parameter(torch.zeros(3, hidden_dim))

        self.cross_feature_reconstructors = nn.ModuleList(
            [_mlp(hidden_dim, 2 * hidden_dim, hidden_dim) for _ in range(5)]
        )
        self.numeric_decoders = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(2)])
        self.category_decoders = nn.ModuleList(
            [nn.Linear(hidden_dim, cardinality) for cardinality in codebook.cardinalities]
        )
        self.corruption_heads = nn.ModuleList(
            [_mlp(3 * hidden_dim, hidden_dim, 1) for _ in range(5)]
        )
        self.attack_head = _mlp(hidden_dim + 5, hidden_dim, 1)

    def _numeric_basis(self, value: torch.Tensor) -> torch.Tensor:
        scales = torch.as_tensor(
            self.numeric_scales, dtype=value.dtype, device=value.device
        )
        scaled = value.unsqueeze(-1) / scales
        return torch.cat((scaled, torch.sin(scaled), torch.cos(scaled)), dim=-1)

    def _encode_numeric(self, standardized: torch.Tensor) -> list[torch.Tensor]:
        return [
            encoder(self._numeric_basis(standardized[..., index]))
            for index, encoder in enumerate(self.numeric_encoders)
        ]

    def _validate_indices(self, x: torch.Tensor) -> list[torch.Tensor]:
        indices = [x[..., index + 2].long() for index in range(3)]
        for feature, (values, cardinality) in enumerate(zip(indices, self.codebook.cardinalities)):
            if torch.any(values < 0) or torch.any(values >= cardinality):
                raise ValueError(
                    f"Encoded categorical feature {feature} is outside [0, {cardinality})"
                )
        return indices

    def forward(self, encoded_features: torch.Tensor) -> StructuralDefenseOutput:
        if encoded_features.shape[-1] != 5:
            raise ValueError(
                f"StructuralDefenseAdapter expects five features, got {encoded_features.shape}"
            )
        dtype = next(self.parameters()).dtype
        x = encoded_features.to(dtype=dtype)
        numeric = (x[..., :2] - self.numeric_mean) / self.numeric_scale
        numeric_tokens = self._encode_numeric(numeric)
        categorical_indices = self._validate_indices(x)
        categorical_tokens = [
            embedding(indices)
            for embedding, indices in zip(self.category_embeddings, categorical_indices)
        ]
        relation = self.category_relation(torch.cat(categorical_tokens, dim=-1))
        categorical_tokens = [
            token + torch.sigmoid(gate) * relation
            for token, gate in zip(categorical_tokens, self.category_relation_gates)
        ]
        observed = torch.stack((*numeric_tokens, *categorical_tokens), dim=-2)

        token_sum = observed.sum(dim=-2)
        other_means = [
            (token_sum - observed[..., feature, :]) / 4.0 for feature in range(5)
        ]
        reconstructed = torch.stack(
            [module(other) for module, other in zip(self.cross_feature_reconstructors, other_means)],
            dim=-2,
        )
        reconstructed_numeric = torch.cat(
            [
                decoder(reconstructed[..., feature, :])
                for feature, decoder in enumerate(self.numeric_decoders)
            ],
            dim=-1,
        )
        reconstructed_categorical_logits = tuple(
            decoder(reconstructed[..., feature + 2, :])
            for feature, decoder in enumerate(self.category_decoders)
        )

        corruption_logits = torch.cat(
            [
                head(
                    torch.cat(
                        (
                            observed[..., feature, :],
                            other_means[feature],
                            torch.abs(observed[..., feature, :] - reconstructed[..., feature, :]),
                        ),
                        dim=-1,
                    )
                )
                for feature, head in enumerate(self.corruption_heads)
            ],
            dim=-1,
        )
        reliability = 1.0 - torch.sigmoid(corruption_logits)

        repaired_numeric_standard = (
            reliability[..., :2] * numeric
            + (1.0 - reliability[..., :2]) * reconstructed_numeric
        )
        repaired_numeric = (
            repaired_numeric_standard * self.numeric_scale + self.numeric_mean
        )
        repaired_categories: list[torch.Tensor] = []
        for feature, logits in enumerate(reconstructed_categorical_logits):
            values = getattr(self, f"category_values_{feature}").to(dtype=dtype)
            observed_value = values[categorical_indices[feature]]
            reconstructed_value = F.softmax(logits, dim=-1) @ values
            gate = reliability[..., feature + 2]
            repaired_categories.append(
                (gate * observed_value + (1.0 - gate) * reconstructed_value).unsqueeze(-1)
            )
        repaired = torch.cat((repaired_numeric, *repaired_categories), dim=-1)
        attack_logits = self.attack_head(
            torch.cat((observed.mean(dim=-2), corruption_logits), dim=-1)
        ).squeeze(-1)
        return StructuralDefenseOutput(
            repaired_features=repaired,
            feature_corruption_logits=corruption_logits,
            attack_logits=attack_logits,
            reconstruction_numeric=reconstructed_numeric,
            reconstruction_categorical_logits=reconstructed_categorical_logits,  # type: ignore[arg-type]
            reliability=reliability,
            observed_tokens=observed,
            reconstructed_tokens=reconstructed,
        )
