"""Random Forest attack detector transferred onto frozen URE embeddings."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier

from .model import UREEmbedding
from .preprocessing import UREPreprocessor
from .source import IMPLEMENTATION_NOTES, OFFICIAL_COMMIT, OFFICIAL_REPOSITORY, PAPER_URL

URE_CHECKPOINT_FORMAT = "poi-ure-rf-v2"


@dataclass(frozen=True)
class UREEmbeddingMerger:
    """Frozen train-only implementation of paper Algorithm 3.

    One global Euclidean-distance percentile defines the complete-linkage maximum
    within-cluster distance.  Every category in a cluster is then represented by the
    cluster's mean first-layer embedding before fitting or scoring the tree.
    """

    percentile: float
    threshold: float
    merged_weights: tuple[np.ndarray, ...]  # each [embedding_dim, cardinality]
    cluster_labels: tuple[np.ndarray, ...]

    @classmethod
    def fit(cls, embedding: UREEmbedding, *, percentile: float) -> "UREEmbeddingMerger":
        if not 0.0 <= percentile <= 1.0:
            raise ValueError("URE merge percentile must be in [0, 1]")
        raw = tuple(layer.weight.detach().cpu().numpy().copy() for layer in embedding.layers)
        distances = []
        for weight in raw:
            points = weight.T
            for left in range(len(points)):
                for right in range(left + 1, len(points)):
                    distances.append(float(np.linalg.norm(points[left] - points[right], ord=2)))
        threshold = float(np.quantile(distances, percentile)) if distances else 0.0
        merged: list[np.ndarray] = []
        labels_out: list[np.ndarray] = []
        for weight in raw:
            points = weight.T
            if len(points) <= 1:
                labels = np.arange(len(points), dtype=np.int64)
            else:
                arguments = dict(
                    n_clusters=None,
                    linkage="complete",
                    distance_threshold=threshold + 1e-12,
                )
                try:
                    labels = AgglomerativeClustering(
                        metric="euclidean", **arguments
                    ).fit_predict(points)
                except TypeError:  # scikit-learn < 1.2
                    labels = AgglomerativeClustering(
                        affinity="euclidean", **arguments
                    ).fit_predict(points)
            merged_weight = weight.copy()
            for label in np.unique(labels):
                members = labels == label
                merged_weight[:, members] = points[members].mean(axis=0)[:, None]
            merged.append(merged_weight.astype(np.float32, copy=False))
            labels_out.append(np.asarray(labels, dtype=np.int64))
        return cls(percentile, threshold, tuple(merged), tuple(labels_out))

    def transform_indices(self, indices: np.ndarray) -> np.ndarray:
        if indices.ndim != 2 or indices.shape[1] != len(self.merged_weights):
            raise ValueError("URE merge input shape does not match frozen fields")
        return np.concatenate(
            [weight[:, indices[:, feature]].T for feature, weight in enumerate(self.merged_weights)],
            axis=1,
        ).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "paper_algorithm_3_euclidean_complete_linkage_mean",
            "percentile": self.percentile,
            "threshold": self.threshold,
            "merged_weights": list(self.merged_weights),
            "cluster_labels": list(self.cluster_labels),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UREEmbeddingMerger":
        return cls(
            percentile=float(value["percentile"]),
            threshold=float(value["threshold"]),
            merged_weights=tuple(np.asarray(x, dtype=np.float32) for x in value["merged_weights"]),
            cluster_labels=tuple(np.asarray(x, dtype=np.int64) for x in value["cluster_labels"]),
        )


class UREAttackDetector:
    """External ``URE embedding -> RF`` detector from plan section 4.1.

    This is an adaptation built for the POI attack-detection comparison.  It is not an
    original-paper attack detector and is not a modified TabPFN model.
    """

    result_label = "URE-RF attack detector"

    def __init__(
        self,
        preprocessor: UREPreprocessor,
        embedding: UREEmbedding,
        *,
        random_state: int = 42,
        n_estimators: int = 500,
        min_samples_leaf: int = 2,
        merge_percentile: float = 0.1,
        device: str | torch.device = "cpu",
        merger: UREEmbeddingMerger | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.embedding = embedding.to(device).eval()
        self.device = torch.device(device)
        # Fitted once from embeddings learned only on clean train.  It is frozen before
        # any attack labels are supplied to the Random Forest.
        self.merger = merger or UREEmbeddingMerger.fit(
            self.embedding, percentile=merge_percentile
        )
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    def transform(self, matrix5: Any, *, batch_size: int = 4096) -> np.ndarray:
        indices = self.preprocessor.transform(matrix5)
        parts = [
            self.merger.transform_indices(indices[start : start + batch_size])
            for start in range(0, len(indices), batch_size)
        ]
        return np.concatenate(parts, axis=0)

    def fit(self, matrix5: Any, attack_labels: np.ndarray) -> "UREAttackDetector":
        labels = np.asarray(attack_labels, dtype=np.int64)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("URE-RF requires normal=0 and attack=1 training labels")
        self.classifier.fit(self.transform(matrix5), labels)
        return self

    def predict_proba(self, matrix5: Any) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(matrix5))


def save_ure_attack_detector(
    checkpoint_path: str | Path,
    detector: UREAttackDetector,
    *,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    checkpoint_path = Path(checkpoint_path)
    forest_path = checkpoint_path.with_suffix(".joblib")
    payload = {
        "format": URE_CHECKPOINT_FORMAT,
        "preprocessor": detector.preprocessor.to_dict(),
        "cardinalities": list(detector.embedding.cardinalities),
        "embedding_dim": detector.embedding.embedding_dim,
        "embedding_state": {
            key: value.detach().cpu() for key, value in detector.embedding.state_dict().items()
        },
        "embedding_merger": detector.merger.to_dict(),
        "forest_path": forest_path.name,
        "official_source": {
            "paper": PAPER_URL,
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "implementation_notes": IMPLEMENTATION_NOTES,
        },
        "result_label": detector.result_label,
        "metadata": metadata,
    }
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    joblib.dump(detector.classifier, forest_path)
    return checkpoint_path, forest_path


def load_ure_attack_detector(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> UREAttackDetector:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != URE_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported URE-RF checkpoint: {payload.get('format')}")
    preprocessor = UREPreprocessor.from_dict(payload["preprocessor"])
    embedding = UREEmbedding(
        tuple(payload["cardinalities"]), embedding_dim=int(payload["embedding_dim"])
    )
    embedding.load_state_dict(payload["embedding_state"], strict=True)
    detector = UREAttackDetector(
        preprocessor,
        embedding,
        device=device,
        merger=UREEmbeddingMerger.from_dict(payload["embedding_merger"]),
    )
    detector.classifier = joblib.load(checkpoint_path.with_name(payload["forest_path"]))
    return detector
