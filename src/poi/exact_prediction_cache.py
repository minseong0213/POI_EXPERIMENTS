"""Bounded exact in-memory memoization for frozen region-model predictions.

This module deliberately provides no disk persistence.  A cache entry is reusable
only when both the frozen fitted-model identity and every byte of the ordered
float32 input matrix match.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class ExactPredictionCacheError(RuntimeError):
    """Raised when an exact-cache identity or prediction violates its contract."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ExactPredictionCacheError("Cache identity must be canonical JSON data") from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_frozen_model_identity(
    *,
    model_id: str,
    model_seed: int,
    feature_order: Sequence[str],
    region_order: Sequence[str],
    fit_attestations: Sequence[Mapping[str, Any]],
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Bind a cache namespace to actual persisted fits, sources, and runtimes."""
    regions = tuple(map(str, region_order))
    features = tuple(map(str, feature_order))
    if not model_id or not regions or len(set(regions)) != len(regions):
        raise ExactPredictionCacheError("Model and unique ordered regions are required")
    if not features or len(set(features)) != len(features):
        raise ExactPredictionCacheError("Unique ordered features are required")
    sources = {str(path): str(digest).lower() for path, digest in source_sha256.items()}
    if not sources or any(not _valid_sha256(digest) for digest in sources.values()):
        raise ExactPredictionCacheError("Exact source SHA-256 bindings are required")
    if len(fit_attestations) != len(regions):
        raise ExactPredictionCacheError("One fitted-state attestation per ordered region is required")

    runtime_fields = (
        "tabpfn_version", "torch_version", "cuda_version", "cuda_available",
        "gpu_device_name", "requested_device", "restored_on_device",
    )
    required = (
        "format", "model_id", "model_seed", "region", "feature_order", "train_matrix_sha256",
        "truth_sha256", "official_checkpoint_sha256", "fitted_state_path",
        "fitted_state_sha256", "fitted_state_size_bytes", "fitted_state_format",
        "fitted_state_api", "probe_prediction_sha256",
        "probe_repeat_prediction_sha256", "restored_probe_prediction_sha256",
    )
    fit_bindings: list[dict[str, Any]] = []
    runtime_bindings: list[dict[str, Any]] = []
    checkpoint_hashes: set[str] = set()
    for expected_region, raw in zip(regions, fit_attestations):
        fit = dict(raw)
        missing = [name for name in required if name not in fit]
        if missing:
            raise ExactPredictionCacheError(
                f"Frozen fit attestation is incomplete for {expected_region}: {missing}"
            )
        if (
            fit["format"] != "poi-tabpfn25-fit-replay-v1"
            or fit["fitted_state_format"] != "tabpfn_official_fit_state_v1"
            or fit["fitted_state_api"]
            != "TabPFNClassifier.save_fit_state/load_from_fit_state"
            or str(fit["model_id"]) != model_id
            or int(fit["model_seed"]) != int(model_seed)
            or str(fit["region"]) != expected_region
            or tuple(map(str, fit["feature_order"])) != features
        ):
            raise ExactPredictionCacheError("Frozen fit identity/order differs from cache namespace")
        for name in (
            "train_matrix_sha256", "truth_sha256", "official_checkpoint_sha256",
            "fitted_state_sha256", "probe_prediction_sha256",
            "probe_repeat_prediction_sha256", "restored_probe_prediction_sha256",
        ):
            if not _valid_sha256(str(fit[name]).lower()):
                raise ExactPredictionCacheError(f"Invalid {name} for {expected_region}")
        probe_hashes = {
            str(fit["probe_prediction_sha256"]).lower(),
            str(fit["probe_repeat_prediction_sha256"]).lower(),
            str(fit["restored_probe_prediction_sha256"]).lower(),
        }
        if len(probe_hashes) != 1:
            raise ExactPredictionCacheError(
                f"Frozen fit lacks exact original/repeat/restored probe agreement: {expected_region}"
            )
        state_path = Path(str(fit["fitted_state_path"]))
        if not state_path.is_file():
            raise ExactPredictionCacheError(f"Frozen fitted state is absent: {state_path}")
        state_size = int(fit["fitted_state_size_bytes"])
        if state_size <= 0 or state_path.stat().st_size != state_size:
            raise ExactPredictionCacheError(f"Frozen fitted-state size differs: {state_path}")
        state_digest = _file_sha256(state_path)
        if state_digest != str(fit["fitted_state_sha256"]).lower():
            raise ExactPredictionCacheError(f"Frozen fitted-state SHA-256 differs: {state_path}")
        if any(name not in fit for name in runtime_fields):
            raise ExactPredictionCacheError(
                f"Frozen runtime identity is incomplete for {expected_region}"
            )
        runtime = {name: fit[name] for name in runtime_fields}
        if runtime["requested_device"] != runtime["restored_on_device"]:
            raise ExactPredictionCacheError("Requested/restored frozen devices differ")
        if str(runtime["requested_device"]).startswith("cuda") and not runtime[
            "cuda_available"
        ]:
            raise ExactPredictionCacheError("Frozen CUDA fit identity reports CUDA unavailable")
        checkpoint_hashes.add(str(fit["official_checkpoint_sha256"]).lower())
        runtime_bindings.append(runtime)
        fit_bindings.append({
            "region": expected_region,
            "attestation_sha256": _canonical_sha256(fit),
            "fitted_state_path": str(state_path.resolve()),
            "fitted_state_sha256": state_digest,
            "train_matrix_sha256": str(fit["train_matrix_sha256"]).lower(),
            "truth_sha256": str(fit["truth_sha256"]).lower(),
            "probe_prediction_sha256": next(iter(probe_hashes)),
        })
    if len(checkpoint_hashes) != 1:
        raise ExactPredictionCacheError("Frozen region fits do not share one checkpoint")

    identity = {
        "format": "poi-exact-frozen-prediction-cache-identity-v1",
        "model_id": model_id,
        "model_seed": int(model_seed),
        "feature_order": list(features),
        "region_order": list(regions),
        "official_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "source_sha256": dict(sorted(sources.items())),
        "source_identity_sha256": _canonical_sha256(dict(sorted(sources.items()))),
        "runtime_identity_sha256": _canonical_sha256(runtime_bindings),
        "fit_bindings": fit_bindings,
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


@dataclass
class _Entry:
    prediction: np.ndarray
    prediction_seconds: float
    input_shape: tuple[int, int]
    input_bytes: bytes

    @property
    def resident_bytes(self) -> int:
        return self.prediction.nbytes + len(self.input_bytes)


class ExactFloat32PredictionCache:
    """LRU cache for exact full-matrix predictions from one frozen fitted model grid."""

    def __init__(
        self,
        predictor: Callable[[np.ndarray], np.ndarray],
        model_identity: Mapping[str, Any],
        *,
        feature_order: Sequence[str],
        output_order: Sequence[str],
        max_entries: int = 128,
        max_bytes: int = 64 * 1024 * 1024,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None] = lambda: None,
    ) -> None:
        self._predictor = predictor
        self.feature_order = tuple(map(str, feature_order))
        self.output_order = tuple(map(str, output_order))
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._clock = clock
        self._synchronize = synchronize
        if self.max_entries <= 0 or self.max_bytes <= 0:
            raise ValueError("Exact cache bounds must be positive")
        identity = dict(model_identity)
        recorded_digest = identity.pop("identity_sha256", None)
        identity_digest = _canonical_sha256(identity)
        if recorded_digest != identity_digest:
            raise ExactPredictionCacheError("Frozen model identity SHA-256 is absent or invalid")
        if (
            tuple(map(str, identity.get("feature_order", ()))) != self.feature_order
            or tuple(map(str, identity.get("region_order", ()))) != self.output_order
            or not identity.get("fit_bindings")
            or not _valid_sha256(identity.get("source_identity_sha256"))
            or not _valid_sha256(identity.get("runtime_identity_sha256"))
        ):
            raise ExactPredictionCacheError("Frozen model identity dimensions are incomplete")
        self.model_identity_sha256 = str(recorded_digest)
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._resident_bytes = 0
        self._requests = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._oversized_not_cached = 0
        self._hash_seconds = 0.0
        self._lookup_seconds = 0.0
        self._prediction_seconds = 0.0
        self._avoided_prediction_seconds_estimate = 0.0

    def key_for(self, matrix5: np.ndarray) -> str:
        array = np.asarray(matrix5)
        if array.dtype != np.dtype(np.float32):
            raise ValueError("Exact prediction cache accepts float32 input only")
        if array.ndim != 2 or array.shape[1] != len(self.feature_order):
            raise ValueError("Exact prediction cache input has the wrong feature shape")
        if not np.isfinite(array).all():
            raise ValueError("Exact prediction cache input must be finite")
        ordered = np.ascontiguousarray(array)
        header = {
            "format": "poi-exact-float32-matrix-v1",
            "shape": list(ordered.shape),
            "dtype": ordered.dtype.str,
            "feature_order": list(self.feature_order),
        }
        digest = hashlib.sha256()
        digest.update(_canonical_bytes(header))
        digest.update(ordered.tobytes(order="C"))
        input_digest = digest.hexdigest()
        return hashlib.sha256(
            f"{self.model_identity_sha256}:{input_digest}".encode("ascii")
        ).hexdigest()

    def predict(self, matrix5: np.ndarray) -> np.ndarray:
        input_array = np.asarray(matrix5)
        # Detach the predictor input from the caller.  This also canonicalizes memory
        # layout without changing logical row or feature order.
        detached = np.array(input_array, dtype=input_array.dtype, order="C", copy=True)
        hash_started = self._clock()
        key = self.key_for(detached)
        exact_input_bytes = detached.tobytes(order="C")
        exact_input_shape = tuple(detached.shape)
        self._hash_seconds += self._clock() - hash_started
        self._requests += 1
        lookup_started = self._clock()
        cached = self._entries.pop(key, None)
        self._lookup_seconds += self._clock() - lookup_started
        if cached is not None:
            if (
                cached.input_shape != exact_input_shape
                or cached.input_bytes != exact_input_bytes
            ):
                raise ExactPredictionCacheError("SHA-256 key collision across unequal exact inputs")
            self._entries[key] = cached
            self._hits += 1
            self._avoided_prediction_seconds_estimate += cached.prediction_seconds
            return cached.prediction.copy()

        self._misses += 1
        self._synchronize()
        predict_started = self._clock()
        prediction = np.asarray(self._predictor(detached))
        self._synchronize()
        prediction_seconds = self._clock() - predict_started
        if prediction_seconds < 0 or not math.isfinite(prediction_seconds):
            raise ExactPredictionCacheError("Prediction duration must be finite and nonnegative")
        self._prediction_seconds += prediction_seconds
        if (
            prediction.shape != (len(detached), len(self.output_order))
            or not np.issubdtype(prediction.dtype, np.floating)
            or not np.isfinite(prediction).all()
            or ((prediction < 0) | (prediction > 1)).any()
        ):
            raise ExactPredictionCacheError("Frozen predictor returned invalid probabilities")
        stored = np.array(prediction, order="C", copy=True)
        resident_bytes = stored.nbytes + len(exact_input_bytes)
        if resident_bytes > self.max_bytes:
            self._oversized_not_cached += 1
            return stored.copy()
        while self._entries and (
            len(self._entries) >= self.max_entries
            or self._resident_bytes + resident_bytes > self.max_bytes
        ):
            _, evicted = self._entries.popitem(last=False)
            self._resident_bytes -= evicted.resident_bytes
            self._evictions += 1
        self._entries[key] = _Entry(
            stored, prediction_seconds, exact_input_shape, exact_input_bytes
        )
        self._resident_bytes += resident_bytes
        return stored.copy()

    def statistics(self) -> dict[str, Any]:
        return {
            "format": "poi-exact-float32-prediction-cache-statistics-v1",
            "scope": "process_memory_only_no_disk_resume",
            "model_identity_sha256": self.model_identity_sha256,
            "feature_order": list(self.feature_order),
            "output_order": list(self.output_order),
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "requests": self._requests,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / self._requests if self._requests else 0.0,
            "entries": len(self._entries),
            "resident_bytes": self._resident_bytes,
            "evictions": self._evictions,
            "oversized_not_cached": self._oversized_not_cached,
            "input_hash_seconds": self._hash_seconds,
            "lookup_seconds": self._lookup_seconds,
            "underlying_prediction_seconds": self._prediction_seconds,
            "avoided_prediction_seconds_estimate": self._avoided_prediction_seconds_estimate,
            "probability_reuse_contract": (
                "exact first-computed probability bytes for identical model identity and "
                "ordered float32 input matrix"
            ),
        }
