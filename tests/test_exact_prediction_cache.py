import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from poi.data import FEATURES
from poi.defense_pipeline import (
    _cache_runtime_statistics,
    _observations,
    _timed_probability_call,
)
from poi.exact_prediction_cache import (
    ExactFloat32PredictionCache,
    ExactPredictionCacheError,
    build_frozen_model_identity,
)


REGIONS = ("A", "B")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_stage10_config_requires_bounded_process_memory_cache():
    cfg = yaml.safe_load(Path("configs/stages/10_defense_pipeline.yaml").read_text())
    cache = cfg["exact_prediction_cache"]
    assert cache == {
        "enabled": True,
        "max_entries_per_model_seed": 128,
        "max_bytes_per_model_seed": 67_108_864,
    }
    expected_requests = 3 * (
        1 + len(cfg["generation_seeds"]) * cfg["expected_attack_conditions"]
    )
    assert expected_requests == 291
    assert expected_requests * len(cfg["seeds"]) * 17 == 14_841


def _identity(tmp_path: Path, *, seed: int = 42, probe_match: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    attestations = []
    for index, region in enumerate(REGIONS):
        content = f"state::{seed}::{region}".encode()
        state = tmp_path / f"seed{seed}-{index}.tabpfn_fit"
        state.write_bytes(content)
        probe = _digest(f"probe::{seed}::{region}".encode())
        repeat = probe if probe_match else _digest(b"changed")
        attestations.append({
            "format": "poi-tabpfn25-fit-replay-v1",
            "model_id": "tabpfn_v2_5",
            "model_seed": seed,
            "region": region,
            "feature_order": FEATURES,
            "train_matrix_sha256": _digest(b"train"),
            "truth_sha256": _digest(region.encode()),
            "official_checkpoint_sha256": _digest(b"checkpoint"),
            "fitted_state_path": str(state),
            "fitted_state_sha256": _digest(content),
            "fitted_state_size_bytes": len(content),
            "fitted_state_format": "tabpfn_official_fit_state_v1",
            "fitted_state_api": "TabPFNClassifier.save_fit_state/load_from_fit_state",
            "probe_prediction_sha256": probe,
            "probe_repeat_prediction_sha256": repeat,
            "restored_probe_prediction_sha256": probe,
            "tabpfn_version": "fixture",
            "torch_version": "fixture",
            "cuda_version": None,
            "cuda_available": False,
            "gpu_device_name": None,
            "requested_device": "cpu",
            "restored_on_device": "cpu",
        })
    return build_frozen_model_identity(
        model_id="tabpfn_v2_5",
        model_seed=seed,
        feature_order=FEATURES,
        region_order=REGIONS,
        fit_attestations=attestations,
        source_sha256={"pipeline.py": _digest(b"source")},
    )


def _cache(tmp_path: Path, identity=None, predictor=None, **kwargs):
    if identity is None:
        identity = _identity(tmp_path)
    if predictor is None:
        predictor = lambda matrix: np.full((len(matrix), len(REGIONS)), 0.25)
    return ExactFloat32PredictionCache(
        predictor,
        identity,
        feature_order=FEATURES,
        output_order=REGIONS,
        **kwargs,
    )


def test_key_binds_exact_values_row_order_dtype_and_frozen_model(tmp_path):
    matrix = np.arange(15, dtype=np.float32).reshape(3, 5)
    first = _cache(tmp_path / "a", identity=_identity(tmp_path / "a", seed=42))
    second = _cache(tmp_path / "b", identity=_identity(tmp_path / "b", seed=202))

    base_key = first.key_for(matrix)
    changed = matrix.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(np.inf))
    assert first.key_for(changed) != base_key
    assert first.key_for(matrix[::-1]) != base_key
    assert second.key_for(matrix) != base_key
    with pytest.raises(ValueError, match="float32"):
        first.key_for(matrix.astype(np.float64))


def test_cache_reuses_exact_probabilities_without_mutating_inputs_or_outputs(tmp_path):
    calls = 0

    def mutating_predictor(matrix):
        nonlocal calls
        calls += 1
        result = np.column_stack((matrix[:, 0] / 10, matrix[:, 1] / 10))
        matrix[:] = -999  # The predictor receives a detached copy.
        return result

    cache = _cache(tmp_path, predictor=mutating_predictor)
    matrix = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=np.float32)
    original = matrix.copy()
    first = cache.predict(matrix)
    expected = first.copy()
    first[:] = 0  # A caller cannot mutate the resident cache entry.
    second = cache.predict(matrix)

    np.testing.assert_array_equal(matrix, original)
    np.testing.assert_array_equal(second, expected)
    assert calls == 1
    statistics = cache.statistics()
    assert statistics["requests"] == 2
    assert statistics["hits"] == 1
    assert statistics["misses"] == 1


def test_cache_is_bounded_and_evicts_least_recently_used_entry(tmp_path):
    calls = 0

    def predictor(matrix):
        nonlocal calls
        calls += 1
        return np.full((len(matrix), len(REGIONS)), calls / 10)

    cache = _cache(tmp_path, predictor=predictor, max_entries=1, max_bytes=1024)
    first = np.ones((2, 5), dtype=np.float32)
    second = np.full((2, 5), 2, dtype=np.float32)
    cache.predict(first)
    cache.predict(second)
    cache.predict(first)

    assert calls == 3
    statistics = cache.statistics()
    assert statistics["entries"] == 1
    assert statistics["evictions"] == 2
    assert statistics["resident_bytes"] <= statistics["max_bytes"]


def test_cache_fails_closed_if_unequal_inputs_resolve_to_same_key(tmp_path, monkeypatch):
    cache = _cache(tmp_path)
    first = np.ones((2, 5), dtype=np.float32)
    second = np.full((2, 5), 2, dtype=np.float32)
    forced_key = cache.key_for(first)
    monkeypatch.setattr(cache, "key_for", lambda matrix: forced_key)

    cache.predict(first)
    with pytest.raises(ExactPredictionCacheError, match="collision"):
        cache.predict(second)


def test_model_identity_fails_closed_on_probe_or_fitted_state_changes(tmp_path):
    with pytest.raises(ExactPredictionCacheError, match="probe agreement"):
        _identity(tmp_path / "probe", probe_match=False)

    identity_dir = tmp_path / "state"
    identity = _identity(identity_dir)
    state = next(identity_dir.glob("*.tabpfn_fit"))
    state.write_bytes(b"tampered")
    # Building a new identity re-hashes the actual state and rejects stale metadata.
    with pytest.raises(ExactPredictionCacheError, match="size differs|SHA-256 differs"):
        _identity_from_existing(identity_dir)
    assert identity["identity_sha256"]


def _identity_from_existing(directory: Path):
    """Recreate fixture metadata while retaining any modified state bytes."""
    attestations = []
    for index, region in enumerate(REGIONS):
        original = f"state::42::{region}".encode()
        state = directory / f"seed42-{index}.tabpfn_fit"
        probe = _digest(f"probe::42::{region}".encode())
        attestations.append({
            "format": "poi-tabpfn25-fit-replay-v1",
            "model_id": "tabpfn_v2_5", "model_seed": 42, "region": region,
            "feature_order": FEATURES, "train_matrix_sha256": _digest(b"train"),
            "truth_sha256": _digest(region.encode()),
            "official_checkpoint_sha256": _digest(b"checkpoint"),
            "fitted_state_path": str(state), "fitted_state_sha256": _digest(original),
            "fitted_state_size_bytes": len(original), "probe_prediction_sha256": probe,
            "fitted_state_format": "tabpfn_official_fit_state_v1",
            "fitted_state_api": "TabPFNClassifier.save_fit_state/load_from_fit_state",
            "probe_repeat_prediction_sha256": probe,
            "restored_probe_prediction_sha256": probe, "tabpfn_version": "fixture",
            "torch_version": "fixture", "cuda_version": None, "cuda_available": False,
            "gpu_device_name": None, "requested_device": "cpu", "restored_on_device": "cpu",
        })
    return build_frozen_model_identity(
        model_id="tabpfn_v2_5", model_seed=42, feature_order=FEATURES,
        region_order=REGIONS, fit_attestations=attestations,
        source_sha256={"pipeline.py": _digest(b"source")},
    )


def test_observations_cache_is_numerically_identical_and_counts_three_exact_inputs(tmp_path):
    frame = pd.DataFrame({
        "POI_ID": ["a", "b"], "region": ["A", "B"],
        "X_COORD": [1.0, 2.0], "Y_COORD": [2.0, 3.0],
        "ASORT_LCLASDC": [1, 2], "ASORT_MLSFCDC": [10, 11],
        "ASORT_SDASDC": [100, 101],
    })

    class Proposed:
        regions = REGIONS

        def predict(self, matrix):
            return {
                "attack_probability": np.full(len(matrix), .1),
                "region_probability": np.tile([.8, .2], (len(matrix), 1)),
                "repaired_matrix": matrix.copy(),
                "preprojection_numeric_out_of_bounds": np.zeros((len(matrix), 2), dtype=bool),
                "preprojection_invalid_category_tuple": np.zeros(len(matrix), dtype=bool),
            }

    class URE:
        def predict_proba(self, matrix):
            return np.tile([.9, .1], (len(matrix), 1))

    class Fixed:
        def __init__(self):
            self.calls = 0

        def predict(self, matrix):
            self.calls += 1
            return np.tile([.75, .25], (len(matrix), 1))

    class Repair:
        def repair(self, values):
            return values.copy()

    plain_fixed = Fixed()
    plain = _observations(frame, Proposed(), URE(), plain_fixed, Repair(), REGIONS)
    cached_fixed = Fixed()
    cache = _cache(tmp_path, predictor=cached_fixed.predict)
    cached = _observations(
        frame, Proposed(), URE(), cached_fixed, Repair(), REGIONS, fixed_cache=cache
    )

    for name in ("truth", "fixed", "shared", "native", "projected"):
        np.testing.assert_array_equal(cached[name], plain[name])
    assert plain_fixed.calls == 3
    assert cached_fixed.calls == 1
    assert cache.statistics()["hits"] == 2
    progress = _cache_runtime_statistics(
        cache, model_seed=42, region_count=2, expected_requests_per_seed=6,
        model_seed_count=3, status="fixture_running",
    )
    assert progress["observation_groups_completed"] == 1
    assert progress["requested_region_predictor_calls"] == 6
    assert progress["executed_region_predictor_calls"] == 2
    assert progress["avoided_region_predictor_calls"] == 4
    assert "not a runtime guarantee" in progress["forecast_limitation"]


def test_timing_probe_call_synchronizes_and_rejects_predictor_mutation():
    sync_calls = 0

    def synchronize():
        nonlocal sync_calls
        sync_calls += 1

    matrix = np.ones((2, 5), dtype=np.float32)
    clock = iter([10.0, 12.5]).__next__
    probability, elapsed = _timed_probability_call(
        lambda values: np.full(len(values), .4), matrix,
        synchronize=synchronize, clock=clock,
    )
    np.testing.assert_array_equal(probability, [.4, .4])
    np.testing.assert_array_equal(matrix, np.ones((2, 5), dtype=np.float32))
    assert elapsed == 2.5
    assert sync_calls == 2

    def mutating(values):
        values[0, 0] = 0
        return np.full(len(values), .4)

    with pytest.raises(RuntimeError, match="mutated"):
        _timed_probability_call(mutating, matrix, synchronize=lambda: None)
    np.testing.assert_array_equal(matrix, np.ones((2, 5), dtype=np.float32))
