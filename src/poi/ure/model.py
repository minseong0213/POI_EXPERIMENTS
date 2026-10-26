"""Universal robust embedding and region-supervision network."""

from __future__ import annotations

import torch
from torch import nn


class UREEmbedding(nn.Module):
    """One bias-free linear layer per one-hot field, matching official URE."""

    def __init__(self, cardinalities: tuple[int, ...], embedding_dim: int = 8) -> None:
        super().__init__()
        if embedding_dim <= 0 or any(cardinality <= 0 for cardinality in cardinalities):
            raise ValueError("URE embedding dimensions and cardinalities must be positive")
        self.cardinalities = tuple(int(value) for value in cardinalities)
        self.embedding_dim = int(embedding_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(cardinality, embedding_dim, bias=False) for cardinality in cardinalities]
        )
        self.normalize_columns()

    @property
    def output_dim(self) -> int:
        return len(self.cardinalities) * self.embedding_dim

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 2 or indices.shape[1] != len(self.layers):
            raise ValueError("URE indices must have shape [rows, fields]")
        return torch.cat(
            [layer.weight.t()[indices[:, field].long()] for field, layer in enumerate(self.layers)],
            dim=-1,
        )

    def forward_relaxed(self, probabilities: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(probabilities) != len(self.layers):
            raise ValueError("URE relaxed probabilities differ from field count")
        return torch.cat(
            [layer(probability) for layer, probability in zip(self.layers, probabilities)],
            dim=-1,
        )

    @torch.no_grad()
    def normalize_columns(self) -> None:
        # Official code normalizes each category's embedding vector, i.e. every column
        # of Linear.weight (output_dim x cardinality).
        for layer in self.layers:
            layer.weight.div_(layer.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))


class URERegionModel(nn.Module):
    """Region-supervised neural model used only to learn transferable embeddings."""

    def __init__(
        self,
        cardinalities: tuple[int, ...],
        *,
        region_count: int = 17,
        embedding_dim: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = UREEmbedding(cardinalities, embedding_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding.output_dim),
            nn.Linear(self.embedding.output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, region_count),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(indices))

    def forward_relaxed(self, probabilities: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.classifier(self.embedding.forward_relaxed(probabilities))
