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
    numeric_mean: torch.Tensor
    numeric_scale: torch.Tensor


@dataclass
class DeploymentProjectionOutput:
    """Hard, train-support-constrained repair for external downstream models."""

    repaired_features: torch.Tensor
    preprojection_numeric_out_of_bounds: torch.Tensor
    preprojection_invalid_category_tuple: torch.Tensor
    selected_category_tuple_indices: torch.Tensor


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
    own embedding.  A target-masked relation representation is built separately for
    each categorical reconstruction from the other two categorical fields only.  A
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
        use_reliability_gate: bool = True,
        use_reconstruction_path: bool = True,
        use_multiscale_encoder: bool = True,
        use_feature_encoders: bool = True,
        use_category_relations: bool = True,
        use_corruption_heads: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if not numeric_scales or any(scale <= 0 for scale in numeric_scales):
            raise ValueError("numeric_scales must contain positive values")
        self.codebook = codebook
        self.hidden_dim = hidden_dim
        self.numeric_scales = tuple(float(scale) for scale in numeric_scales)
        self.use_reliability_gate = bool(use_reliability_gate)
        self.use_reconstruction_path = bool(use_reconstruction_path)
        self.use_multiscale_encoder = bool(use_multiscale_encoder)
        self.use_feature_encoders = bool(use_feature_encoders)
        self.use_category_relations = bool(use_category_relations)
        self.use_corruption_heads = bool(use_corruption_heads)
        self.register_buffer("numeric_mean", torch.tensor(codebook.numeric_mean))
        self.register_buffer("numeric_scale", torch.tensor(codebook.numeric_scale))
        self.register_buffer("numeric_min", torch.tensor(codebook.numeric_model_min))
        self.register_buffer("numeric_max", torch.tensor(codebook.numeric_model_max))
        self.register_buffer(
            "valid_category_tuple_indices",
            torch.tensor(codebook.valid_categorical_tuples, dtype=torch.long),
        )
        for index, values in enumerate(codebook.categorical_values):
            self.register_buffer(f"category_values_{index}", torch.tensor(values))

        numeric_input_dim = (
            3 * len(self.numeric_scales) if self.use_multiscale_encoder else 1
        )
        self.numeric_encoders = nn.ModuleList(
            [_mlp(numeric_input_dim, hidden_dim, hidden_dim) for _ in range(2)]
            if self.use_feature_encoders else []
        )
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, hidden_dim) for cardinality in codebook.cardinalities]
            if self.use_feature_encoders else []
        )
        # There is one categorical relation network per reconstruction target.  Its
        # input contains only the *other* two category embeddings.  Sharing one
        # three-category relation here would leak the target category into its own
        # reconstruction even after removing its observed token from the mean.
        self.category_relation_reconstructors = nn.ModuleList(
            [_mlp(2 * hidden_dim, 2 * hidden_dim, hidden_dim) for _ in range(3)]
        )
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
            if self.use_corruption_heads
            else []
        )
        self.attack_head = _mlp(hidden_dim + 5, hidden_dim, 1)

    def _numeric_basis(self, value: torch.Tensor) -> torch.Tensor:
        if not self.use_multiscale_encoder:
            return value.unsqueeze(-1)
        scales = torch.as_tensor(
            self.numeric_scales, dtype=value.dtype, device=value.device
        )
        scaled = value.unsqueeze(-1) / scales
        return torch.cat((scaled, torch.sin(scaled), torch.cos(scaled)), dim=-1)

    def _encode_numeric(self, standardized: torch.Tensor) -> list[torch.Tensor]:
        if not self.use_feature_encoders:
            return [
                standardized[..., index].unsqueeze(-1).expand(*standardized.shape[:-1], self.hidden_dim)
                for index in range(2)
            ]
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
        if self.use_feature_encoders:
            categorical_tokens = [
                embedding(indices)
                for embedding, indices in zip(self.category_embeddings, categorical_indices)
            ]
        else:
            categorical_tokens = [
                (indices.to(dtype=dtype) / max(cardinality - 1, 1)).unsqueeze(-1).expand(
                    *indices.shape, self.hidden_dim
                )
                for indices, cardinality in zip(categorical_indices, self.codebook.cardinalities)
            ]
        observed = torch.stack((*numeric_tokens, *categorical_tokens), dim=-2)

        token_sum = observed.sum(dim=-2)
        other_means = [
            (token_sum - observed[..., feature, :]) / 4.0 for feature in range(5)
        ]
        reconstruction_contexts = list(other_means)
        if self.use_category_relations:
            for target in range(3):
                other_categories = torch.cat(
                    [
                        token
                        for index, token in enumerate(categorical_tokens)
                        if index != target
                    ],
                    dim=-1,
                )
                relation = self.category_relation_reconstructors[target](other_categories)
                reconstruction_contexts[target + 2] = (
                    reconstruction_contexts[target + 2]
                    + torch.sigmoid(self.category_relation_gates[target]) * relation
                )
        if self.use_reconstruction_path:
            reconstructed = torch.stack(
                [
                    module(other)
                    for module, other in zip(
                        self.cross_feature_reconstructors, reconstruction_contexts
                    )
                ],
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
        else:
            # Preserve the output contract for common logging/loss code while making
            # the reconstruction modules absent from this forward graph.
            reconstructed = torch.zeros_like(observed)
            reconstructed_numeric = numeric.new_zeros((*numeric.shape[:-1], 2))
            reconstructed_categorical_logits = tuple(
                numeric.new_zeros((*numeric.shape[:-1], cardinality))
                for cardinality in self.codebook.cardinalities
            )

        if self.use_corruption_heads:
            corruption_logits = torch.cat(
                [
                    head(
                        torch.cat(
                            (
                                observed[..., feature, :],
                                reconstruction_contexts[feature],
                                torch.abs(
                                    observed[..., feature, :]
                                    - reconstructed[..., feature, :]
                                ),
                            ),
                            dim=-1,
                        )
                    )
                    for feature, head in enumerate(self.corruption_heads)
                ],
                dim=-1,
            )
        else:
            corruption_logits = observed.new_zeros((*observed.shape[:-2], 5))
        if not self.use_reconstruction_path:
            # With no reconstruction to select, the observation is the only valid
            # structural input to the official model.
            reliability = torch.ones_like(corruption_logits)
        elif not self.use_reliability_gate or not self.use_corruption_heads:
            # A fixed blend retains the reconstruction branch while removing the
            # learned, corruption-dependent gate.  This cleanly isolates the gate in
            # the Stage-14 structural ablation.
            reliability = torch.full_like(corruption_logits, 0.5)
        else:
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
            numeric_mean=self.numeric_mean,
            numeric_scale=self.numeric_scale,
        )

    def project_for_deployment(
        self,
        encoded_features: torch.Tensor,
        output: StructuralDefenseOutput,
    ) -> DeploymentProjectionOutput:
        """Project a soft repair onto numeric bounds and observed category tuples.

        The differentiable forward above deliberately keeps expected category codes so
        gradients from the official TabPFN region heads can reach every decoder.  An
        external downstream model instead receives this hard projection.  It clips
        numeric values to clean-train bounds and selects the allowed train tuple with
        maximum summed log probability under the three gated categorical marginals.
        No region label, paired clean row, or geographic boundary oracle is used.
        """
        if encoded_features.ndim != 2 or encoded_features.shape[1] != 5:
            raise ValueError("Deployment projection expects encoded_features [rows, 5]")
        if output.repaired_features.shape != encoded_features.shape:
            raise ValueError("Structural output and encoded features differ in shape")
        tensors = (
            encoded_features,
            output.repaired_features,
            output.reliability,
            output.reconstruction_numeric,
            *output.reconstruction_categorical_logits,
        )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("Deployment projection received non-finite structural values")
        dtype = output.repaired_features.dtype
        x = encoded_features.to(dtype=dtype)
        numeric = output.repaired_features[:, :2]
        numeric_out = (numeric < self.numeric_min) | (numeric > self.numeric_max)
        numeric_projected = torch.maximum(
            torch.minimum(numeric, self.numeric_max), self.numeric_min
        )

        indices = self._validate_indices(x)
        marginals = []
        for feature, (observed_index, logits) in enumerate(
            zip(indices, output.reconstruction_categorical_logits)
        ):
            reconstructed = F.softmax(logits, dim=-1)
            observed = F.one_hot(
                observed_index, num_classes=self.codebook.cardinalities[feature]
            ).to(dtype=dtype)
            gate = output.reliability[:, feature + 2].unsqueeze(1)
            marginals.append(gate * observed + (1.0 - gate) * reconstructed)

        valid = self.valid_category_tuple_indices
        scores = output.repaired_features.new_zeros((len(x), len(valid)))
        independent_indices = []
        for feature, probabilities in enumerate(marginals):
            independent_indices.append(probabilities.argmax(dim=1))
            scores = scores + probabilities.clamp_min(1e-12).log()[:, valid[:, feature]]
        independent = torch.stack(independent_indices, dim=1)
        preprojection_invalid = ~(
            independent[:, None, :] == valid[None, :, :]
        ).all(dim=2).any(dim=1)
        selected = valid[scores.argmax(dim=1)]
        projected_categories = []
        for feature in range(3):
            values = getattr(self, f"category_values_{feature}").to(dtype=dtype)
            projected_categories.append(values[selected[:, feature]].unsqueeze(1))
        repaired = torch.cat((numeric_projected, *projected_categories), dim=1)
        if not torch.isfinite(repaired).all():
            raise RuntimeError("Deployment projection produced non-finite features")
        return DeploymentProjectionOutput(
            repaired_features=repaired,
            preprojection_numeric_out_of_bounds=numeric_out,
            preprojection_invalid_category_tuple=preprojection_invalid,
            selected_category_tuple_indices=selected,
        )
