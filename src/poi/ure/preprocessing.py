"""Train-only binning and categorical indexing for URE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from poi.proposed_defense.schema import FEATURE_NAMES


@dataclass(frozen=True)
class UREPreprocessor:
    """Represent both numeric coordinates and categorical codes as discrete fields.

    Kireev et al. bin numerical features into ten bins before robust-embedding training.
    Edges and categorical vocabularies here are fitted on clean train only.
    """

    numeric_edges: tuple[tuple[float, ...], tuple[float, ...]]
    categorical_values: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def fit(cls, matrix: Any, *, numeric_bins: int = 10) -> "UREPreprocessor":
        if numeric_bins < 2:
            raise ValueError("numeric_bins must be at least 2")
        values = _matrix(matrix)
        quantiles = np.linspace(0.0, 1.0, numeric_bins + 1)[1:-1]
        edges = []
        for column in range(2):
            # Repeated quantiles are removed, so a constant coordinate remains valid.
            edges.append(tuple(float(v) for v in np.unique(np.quantile(values[:, column], quantiles))))
        categorical = tuple(
            tuple(float(v) for v in np.unique(values[:, column]))
            for column in range(2, 5)
        )
        return cls(
            numeric_edges=tuple(edges),  # type: ignore[arg-type]
            categorical_values=categorical,  # type: ignore[arg-type]
        )

    @property
    def cardinalities(self) -> tuple[int, int, int, int, int]:
        return (
            len(self.numeric_edges[0]) + 1,
            len(self.numeric_edges[1]) + 1,
            *(len(values) for values in self.categorical_values),
        )  # type: ignore[return-value]

    def transform(self, matrix: Any) -> np.ndarray:
        values = _matrix(matrix)
        result = np.empty(values.shape, dtype=np.int64)
        for column, edges in enumerate(self.numeric_edges):
            result[:, column] = np.searchsorted(np.asarray(edges), values[:, column], side="right")
        for feature, vocabulary in enumerate(self.categorical_values):
            lookup = {value: index for index, value in enumerate(vocabulary)}
            column = feature + 2
            unknown = sorted(set(values[:, column]).difference(lookup))
            if unknown:
                raise ValueError(
                    f"URE encountered a category outside clean train in {self.feature_names[column]}: "
                    f"{unknown[:5]}"
                )
            result[:, column] = [lookup[value] for value in values[:, column]]
        return result

    def to_dict(self) -> dict[str, Any]:
        state = asdict(self)
        state["numeric_edges"] = [list(edges) for edges in self.numeric_edges]
        state["categorical_values"] = [list(values) for values in self.categorical_values]
        state["feature_names"] = list(self.feature_names)
        return state

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "UREPreprocessor":
        return cls(
            numeric_edges=tuple(tuple(float(v) for v in edge) for edge in state["numeric_edges"]),  # type: ignore[arg-type]
            categorical_values=tuple(
                tuple(float(v) for v in values) for values in state["categorical_values"]
            ),  # type: ignore[arg-type]
            feature_names=tuple(state.get("feature_names", FEATURE_NAMES)),
        )


def load_transition_costs(
    path: str,
    cardinalities: tuple[int, int, int, int, int],
) -> tuple[np.ndarray, ...]:
    """Load explicit per-value transition costs; never invent a URE threat model."""
    if not path:
        raise ValueError(
            "URE requires transition_costs_npz from the frozen attack threat model; "
            "uniform costs would be an unapproved replacement for Kireev et al.'s method"
        )
    archive = np.load(path, allow_pickle=False)
    matrices = []
    for feature, cardinality in enumerate(cardinalities):
        key = f"feature_{feature}"
        if key not in archive:
            raise ValueError(f"URE transition-cost archive is missing {key}")
        matrix = np.asarray(archive[key], dtype=np.float32)
        if matrix.shape != (cardinality, cardinality):
            raise ValueError(
                f"URE {key} costs have shape {matrix.shape}, expected {(cardinality, cardinality)}"
            )
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError(f"URE {key} costs must be finite and nonnegative")
        if not np.allclose(np.diag(matrix), 0.0):
            raise ValueError(f"URE {key} costs must be zero on the diagonal")
        matrices.append(matrix)
    return tuple(matrices)


def build_poi_additive_transition_costs(
    preprocessor: UREPreprocessor,
    clean_train: Any,
    *,
    numeric_cost_unit_m: float,
    categorical_change_cost: float = 1.0,
) -> tuple[np.ndarray, ...]:
    """Map the POI threat model to URE's additive per-field cost matrices.

    Coordinate-bin costs are approximate metres divided by an explicit unit.  Category
    changes use the plan's L0 cost.  The independent matrices cannot encode the valid
    three-code tuple relation; that limitation must remain in reports.
    """
    if numeric_cost_unit_m <= 0 or categorical_change_cost <= 0:
        raise ValueError("URE POI cost units must be positive")
    values = _matrix(clean_train)
    indices = preprocessor.transform(values)
    representatives = []
    for feature in range(2):
        reps = []
        for index in range(preprocessor.cardinalities[feature]):
            field_values = values[indices[:, feature] == index, feature]
            if len(field_values) == 0:
                raise ValueError(f"Empty numeric URE bin {feature}/{index}")
            reps.append(float(np.median(field_values)))
        representatives.append(np.asarray(reps))
    mean_latitude_radians = np.deg2rad(values[:, 1].mean())
    metres_per_degree = (111_320.0 * np.cos(mean_latitude_radians), 110_574.0)
    matrices: list[np.ndarray] = []
    for feature in range(2):
        distances = np.abs(
            representatives[feature][:, None] - representatives[feature][None, :]
        )
        matrices.append(
            (distances * metres_per_degree[feature] / numeric_cost_unit_m).astype(np.float32)
        )
    for cardinality in preprocessor.cardinalities[2:]:
        matrix = np.full((cardinality, cardinality), categorical_change_cost, dtype=np.float32)
        np.fill_diagonal(matrix, 0.0)
        matrices.append(matrix)
    return tuple(matrices)


def _matrix(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "loc"):
        matrix = matrix.loc[:, list(FEATURE_NAMES)].to_numpy()
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError("URE input must be a finite [rows, 5] matrix")
    return values
