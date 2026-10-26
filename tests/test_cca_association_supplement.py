import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from poi.cca_association_supplement import (
    FORMAT,
    SupplementError,
    add_fdr,
    cramers_v,
    fixed_direction_bootstrap,
    heldout_cca_supplement,
    ordered_id_sha256,
    paired_cramers_v_bootstrap,
    preflight,
    region_category_association,
    stable_seed,
    validate_config,
)
from poi.expanded_analysis import heldout_cca


def paired_features(rows=120, seed=7):
    rng = np.random.default_rng(seed)
    clean = pd.DataFrame({
        "POI_ID": [f"p{index:03d}" for index in range(rows)],
        "X_COORD": rng.normal(127, .4, rows),
        "Y_COORD": rng.normal(36, .5, rows),
        "ASORT_LCLASDC": rng.integers(0, 4, rows).astype(float).astype(str),
        "ASORT_MLSFCDC": rng.integers(0, 3, rows).astype(float).astype(str),
        "ASORT_SDASDC": rng.integers(0, 6, rows).astype(float).astype(str),
    })
    attack = clean.copy()
    attack["X_COORD"] += rng.normal(0, .02, rows)
    attack["Y_COORD"] += rng.normal(0, .02, rows)
    changed = np.arange(rows) % 7 == 0
    attack.loc[changed, "ASORT_SDASDC"] = "5.0"
    return clean, attack


def test_cca_point_estimates_match_existing_heldout_semantics():
    clean, attack = paired_features()
    existing = pd.DataFrame(heldout_cca(clean, attack, seed=42, permutations=19))
    correlations, loadings, split = heldout_cca_supplement(
        clean, attack, seed=42, components=3, pca_components_max=8,
        permutations=19, bootstrap_repeats=40, bootstrap_seed=99,
        bootstrap_chunk_size=7,
    )
    np.testing.assert_allclose(correlations.correlation, existing.correlation, rtol=0, atol=1e-12)
    np.testing.assert_allclose(correlations.p_value, existing.p_value, rtol=0, atol=0)
    assert correlations.bootstrap_repeats.eq(40).all()
    assert correlations.inference.str.contains("fixed").all()
    assert len(loadings) == split["encoded_features"] * 3 * 2
    assert set(loadings.side) == {"clean", "adversarial"}
    assert split["fit_inference_overlap"] == 0
    assert split["fit_rows"] + split["inference_rows"] == len(clean)
    assert split["fit_ordered_id_sha256"] != split["inference_ordered_id_sha256"]


def test_fixed_direction_bootstrap_is_chunked_deterministic_and_paired():
    rng = np.random.default_rng(3)
    left = rng.normal(size=(40, 3))
    right = left + rng.normal(scale=.2, size=(40, 3))
    first = fixed_direction_bootstrap(left, right, repeats=37, seed=123, chunk_size=5)
    second = fixed_direction_bootstrap(left, right, repeats=37, seed=123, chunk_size=11)
    np.testing.assert_allclose(first, second)
    assert first.shape == (37, 3)
    assert np.nanmean(first) > .8
    with pytest.raises(SupplementError):
        fixed_direction_bootstrap(left, right[:-1], repeats=10, seed=1, chunk_size=2)


def test_id_hash_and_stable_seed_are_order_sensitive():
    assert ordered_id_sha256(["a", "b"]) != ordered_id_sha256(["b", "a"])
    with pytest.raises(SupplementError, match="unique"):
        ordered_id_sha256(["a", "a"])
    assert stable_seed(42, "a", 1) == stable_seed(42, "a", 1)
    assert stable_seed(42, "a", 1) != stable_seed(42, "b", 1)


def test_region_category_association_reports_effect_and_sparse_diagnostics():
    regions = np.repeat(["N", "S"], 50)
    perfect = np.repeat(["a", "b"], 50)
    result = region_category_association(regions, perfect)
    assert result["p_value"] < 1e-10
    assert result["cramers_v"] > .95
    assert result["degrees_of_freedom"] == 1
    assert result["expected_cells_lt5"] == 0

    sparse = region_category_association(
        ["N"] * 5 + ["S"] * 5,
        ["a", "a", "a", "b", "rare", "a", "a", "b", "b", "b"],
    )
    assert sparse["expected_cells_lt5"] > 0
    assert sparse["asymptotic_reliability"] == "SPARSE_EXPECTED_COUNTS"


def test_bias_corrected_cramers_v_handles_degenerate_tables():
    assert cramers_v(np.array([[5], [6]])) == 0
    assert cramers_v(np.array([[25, 0], [0, 25]])) > .95
    with pytest.raises(SupplementError):
        cramers_v(np.array([[1, -1], [2, 3]]))


def test_paired_association_bootstrap_preserves_no_change_null():
    regions = np.repeat(["N", "S", "E"], 20)
    categories = np.tile(["a", "b", "c", "a", "b"], 12)
    result = paired_cramers_v_bootstrap(
        regions, categories, categories, repeats=100, seed=8, chunk_size=13,
    )
    assert result["delta_ci_low"] == pytest.approx(0)
    assert result["delta_ci_high"] == pytest.approx(0)
    assert result["delta_bootstrap_mean"] == pytest.approx(0)
    assert result["bootstrap_unit"].startswith("paired_POI")


def test_association_and_paired_contrast_are_explicitly_separate():
    regions = np.repeat(["N", "S"], 40)
    clean = np.tile(["a", "b"], 40)
    attack = clean.copy()
    attack[:20] = "a"
    cross_sectional = region_category_association(regions, attack)
    contrast = paired_cramers_v_bootstrap(
        regions, clean, attack, repeats=80, seed=4, chunk_size=16,
    )
    assert "p_value" in cross_sectional
    assert "delta_p_value" not in contrast
    assert "delta_q_value" not in contrast
    assert 0 <= contrast["bootstrap_sign_tail_fraction"] <= 1
    assert "not_null_calibrated_p_value" in contrast["sign_tail_interpretation"]


def test_fdr_is_global_over_supplied_family():
    frame = pd.DataFrame({"p_value": [.001, .02, np.nan, .8]})
    result = add_fdr(frame)
    assert result.q_value.notna().sum() == 3
    assert result.loc[0, "q_value"] >= result.loc[0, "p_value"]


def config(tmp_path: Path) -> dict:
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "poi_data_region.csv").write_text("placeholder")
    (data / "sample_manifest.csv").write_text("placeholder")
    attack = tmp_path / "attacks.parquet"
    attack.write_bytes(b"placeholder")
    return {
        "format": FORMAT, "data_dir": str(data), "attacks_file": str(attack),
        "output": str(tmp_path / "output"), "analysis_split": "train",
        "attack_seeds": [42, 202, 340], "expected_train_pois": 23800,
        "expected_conditions": 32, "expected_regions": 17,
        "cca_split_seed": 42, "cca_components": 3, "pca_components_max": 8,
        "cca_permutations": 499, "bootstrap_repeats": 2000,
        "bootstrap_seed": 20261001, "bootstrap_chunk_size": 16,
        "association_bootstrap_chunk_size": 64, "thread_limit": 1,
        "memory_limit_mb": 1500,
    }


def test_preflight_is_read_only_and_config_is_fail_closed(tmp_path):
    cfg = config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = preflight(path)
    assert result["status"] == "READY_FOR_TRAIN_ONLY_EXECUTION"
    assert result["test_features_read"] == 0
    assert not Path(cfg["output"]).exists()

    cfg["bootstrap_repeats"] = 1999
    with pytest.raises(SupplementError, match="2000"):
        validate_config(cfg)
    cfg = config(tmp_path / "second")
    cfg["analysis_split"] = "validation"
    with pytest.raises(SupplementError, match="train-only"):
        validate_config(cfg)
