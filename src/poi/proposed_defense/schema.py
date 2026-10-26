"""Train-only feature schema for the five POI model inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch

FEATURE_NAMES = (
    "X_COORD",
    "Y_COORD",
    "ASORT_LCLASDC",
    "ASORT_MLSFCDC",
    "ASORT_SDASDC",
)


@dataclass(frozen=True)
class POIFeatureCodebook:
    """Numeric statistics and categorical vocabularies fitted on clean train only.

    Encoded tensors retain raw numeric coordinates in columns 0 and 1.  Columns 2--4
    contain integer vocabulary indices.  This lets a trainable categorical encoder sit
    before TabPFN while the structural adapter can decode repaired categories back to
    their original numeric codes for the official TabPFN forward.
    """

    numeric_mean: tuple[float, float]
    numeric_scale: tuple[float, float]
    numeric_min: tuple[float, float]
    numeric_max: tuple[float, float]
    categorical_values: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    valid_categorical_tuples: tuple[tuple[int, int, int], ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def fit(cls, matrix: Any) -> "POIFeatureCodebook":
        values = _as_matrix(matrix)
        mean = values[:, :2].mean(axis=0)
        scale = values[:, :2].std(axis=0)
        scale[scale < 1e-12] = 1.0
        vocabularies = tuple(
            tuple(float(v) for v in np.unique(values[:, column]))
            for column in range(2, 5)
        )
        if any(len(vocabulary) == 0 for vocabulary in vocabularies):
            raise ValueError("Categorical vocabularies must be non-empty")
        categorical_indices = []
        for offset, vocabulary in enumerate(vocabularies, start=2):
            lookup = {value: index for index, value in enumerate(vocabulary)}
            categorical_indices.append(
                np.asarray([lookup[value] for value in values[:, offset]], dtype=np.int64)
            )
        valid_tuples = tuple(
            tuple(int(value) for value in row)
            for row in np.unique(np.column_stack(categorical_indices), axis=0)
        )
        return cls(
            numeric_mean=(float(mean[0]), float(mean[1])),
            numeric_scale=(float(scale[0]), float(scale[1])),
            numeric_min=(float(values[:, 0].min()), float(values[:, 1].min())),
            numeric_max=(float(values[:, 0].max()), float(values[:, 1].max())),
            categorical_values=vocabularies,  # type: ignore[arg-type]
            valid_categorical_tuples=valid_tuples,  # type: ignore[arg-type]
        )

    @property
    def cardinalities(self) -> tuple[int, int, int]:
        return tuple(len(values) for values in self.categorical_values)  # type: ignore[return-value]

    @property
    def numeric_model_min(self) -> tuple[float, float]:
        """Smallest float32 coordinates that remain inside the raw float64 bounds."""
        return tuple(
            _inward_float32_bound(value, lower=True) for value in self.numeric_min
        )  # type: ignore[return-value]

    @property
    def numeric_model_max(self) -> tuple[float, float]:
        """Largest float32 coordinates that remain inside the raw float64 bounds."""
        return tuple(
            _inward_float32_bound(value, lower=False) for value in self.numeric_max
        )  # type: ignore[return-value]

    def encode(self, matrix: Any, *, device: str | torch.device | None = None) -> torch.Tensor:
        values = _as_matrix(matrix)
        encoded = np.empty(values.shape, dtype=np.float32)
        # Official TabPFN consumes float32.  Rounding a raw minimum downward (or a
        # maximum upward) would put an otherwise valid boundary row outside the true
        # clean-train bounds.  Clip to inward-representable float32 endpoints.
        encoded[:, :2] = np.clip(
            values[:, :2],
            np.asarray(self.numeric_model_min, dtype=np.float32),
            np.asarray(self.numeric_model_max, dtype=np.float32),
        )
        for offset, vocabulary in enumerate(self.categorical_values, start=2):
            lookup = {value: index for index, value in enumerate(vocabulary)}
            unknown = sorted(set(values[:, offset]).difference(lookup))
            if unknown:
                preview = ", ".join(str(value) for value in unknown[:5])
                raise ValueError(
                    f"Unknown category in {self.feature_names[offset]}: {preview}. "
                    "Fit the codebook on clean train and freeze it before validation."
                )
            encoded[:, offset] = np.fromiter(
                (lookup[value] for value in values[:, offset]),
                dtype=np.float32,
                count=len(values),
            )
        return torch.as_tensor(encoded, device=device)

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode an encoded tensor without detaching numeric gradients."""
        if encoded.shape[-1] != 5:
            raise ValueError(f"Expected five encoded features, got {encoded.shape}")
        columns = [encoded[..., :2]]
        for index, values in enumerate(self.categorical_values):
            table = torch.as_tensor(values, dtype=encoded.dtype, device=encoded.device)
            indices = encoded[..., index + 2].long()
            if torch.any(indices < 0) or torch.any(indices >= len(values)):
                raise ValueError(f"Categorical index {index} is outside its codebook")
            columns.append(table[indices].unsqueeze(-1))
        return torch.cat(columns, dim=-1)

    def to_dict(self) -> dict[str, Any]:
        state = asdict(self)
        state["numeric_mean"] = list(self.numeric_mean)
        state["numeric_scale"] = list(self.numeric_scale)
        state["numeric_min"] = list(self.numeric_min)
        state["numeric_max"] = list(self.numeric_max)
        state["categorical_values"] = [list(values) for values in self.categorical_values]
        state["valid_categorical_tuples"] = [
            list(values) for values in self.valid_categorical_tuples
        ]
        state["feature_names"] = list(self.feature_names)
        return state

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "POIFeatureCodebook":
        missing_projection = {
            "numeric_min",
            "numeric_max",
            "valid_categorical_tuples",
        }.difference(state)
        if missing_projection:
            raise ValueError(
                "Proposed-defense checkpoint predates the train-only deployment "
                f"projection contract: missing {sorted(missing_projection)}"
            )
        return cls(
            numeric_mean=tuple(float(v) for v in state["numeric_mean"]),  # type: ignore[arg-type]
            numeric_scale=tuple(float(v) for v in state["numeric_scale"]),  # type: ignore[arg-type]
            numeric_min=tuple(float(v) for v in state["numeric_min"]),  # type: ignore[arg-type]
            numeric_max=tuple(float(v) for v in state["numeric_max"]),  # type: ignore[arg-type]
            categorical_values=tuple(
                tuple(float(v) for v in values)
                for values in state["categorical_values"]
            ),  # type: ignore[arg-type]
            valid_categorical_tuples=tuple(
                tuple(int(v) for v in values)
                for values in state["valid_categorical_tuples"]
            ),  # type: ignore[arg-type]
            feature_names=tuple(state.get("feature_names", FEATURE_NAMES)),
        )


def _as_matrix(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "loc"):
        matrix = matrix.loc[:, list(FEATURE_NAMES)].to_numpy()
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError(f"Expected a [rows, 5] feature matrix, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("POI features must all be finite")
    return values


def _inward_float32_bound(value: float, *, lower: bool) -> float:
    raw = float(value)
    candidate = np.float32(raw)
    if lower and float(candidate) < raw:
        candidate = np.nextafter(candidate, np.float32(np.inf), dtype=np.float32)
    elif not lower and float(candidate) > raw:
        candidate = np.nextafter(candidate, np.float32(-np.inf), dtype=np.float32)
    return float(candidate)


def region_indices(labels: Iterable[str], regions: tuple[str, ...]) -> np.ndarray:
    lookup = {region: index for index, region in enumerate(regions)}
    try:
        return np.asarray([lookup[label] for label in labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Unknown region label: {error.args[0]}") from error
