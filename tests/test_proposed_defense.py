import json
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
from torch import nn

from poi.proposed_defense.losses import DefenseLossWeights, paired_defense_loss
from poi.proposed_defense.model import ProposedTabPFNDefense
from poi.proposed_defense.predictor import (
    ProposedDefensePredictor,
    REGIONAL_DECISION_RULE,
    _expected_unfrozen_base_keys,
    _validate_checkpoint_state_completeness,
    build_checkpoint_payload,
)
from poi.proposed_defense.preflight import (
    FullAttackInventoryValidator,
    _canonical_sha256,
    _sha256 as preflight_sha256,
    _validate_producer_lineage,
    run as run_preflight,
)
from poi.proposed_defense.schema import POIFeatureCodebook
from poi.proposed_defense.structural import StructuralDefenseAdapter
from poi.proposed_defense.data import (
    _validate_attack_integrity,
    attack_intent_diagnostics,
    balanced_attack_sample,
    changed_mask,
    eligible_training_attacks,
)
from poi.proposed_defense.ablation import resolve_ablation_configs
from poi.proposed_defense.train import _ovr_binary_predictions, _require_current_data_preflight
from poi.adversarial import (
    ATTACK_SCHEMA_VERSION,
    QUERY_COUNT_SCOPE,
    SURROGATE_ID,
    attack_condition_specs,
)
import pandas as pd
import poi.proposed_defense.preflight as preflight_module


RAW = np.asarray(
    [
        [126.0, 37.0, 1, 10, 100],
        [126.5, 37.5, 2, 20, 200],
        [127.0, 36.0, 1, 20, 100],
        [128.0, 35.0, 2, 10, 200],
        [129.0, 34.0, 1, 10, 200],
    ],
    dtype=np.float32,
)


class FakeOfficialTabPFN25(nn.Module):
    embedding_dim = 12

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(5, self.embedding_dim)
        self.decoder = nn.Linear(self.embedding_dim, 10)
        self.blocks = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim)])
        self.last_call = None

    def forward(self, x, y, **kwargs):
        self.last_call = (x, y, kwargs)
        query = self.projection(x[len(y) :])
        return {"test_embeddings": query, "standard": self.decoder(query)}


def test_structural_adapter_has_all_required_outputs_and_gradients():
    codebook = POIFeatureCodebook.fit(RAW)
    encoded = codebook.encode(RAW)
    adapter = StructuralDefenseAdapter(codebook, hidden_dim=16)
    result = adapter(encoded)
    assert result.repaired_features.shape == (5, 5)
    assert result.feature_corruption_logits.shape == (5, 5)
    assert result.attack_logits.shape == (5,)
    assert result.reliability.min() >= 0 and result.reliability.max() <= 1
    assert [value.shape[-1] for value in result.reconstruction_categorical_logits] == [2, 2, 2]
    result.repaired_features.sum().backward()
    assert adapter.corruption_heads[0][-1].weight.grad is not None


def test_deployment_projection_clips_numeric_and_selects_train_observed_tuple():
    codebook = POIFeatureCodebook.fit(RAW)
    encoded = codebook.encode(RAW)
    adapter = StructuralDefenseAdapter(codebook, hidden_dim=16)
    output = adapter(encoded)
    output.repaired_features[:, 0] = codebook.numeric_max[0] + 100.0
    output.reliability[:, 2:] = 0.0
    desired = (1, 1, 0)  # Individually valid indices; absent as a tuple in RAW.
    logits = []
    for feature, cardinality in enumerate(codebook.cardinalities):
        value = torch.full((len(RAW), cardinality), -20.0)
        value[:, desired[feature]] = 20.0
        logits.append(value)
    output.reconstruction_categorical_logits = tuple(logits)
    projected = adapter.project_for_deployment(encoded, output)
    assert projected.preprojection_numeric_out_of_bounds[:, 0].all()
    assert projected.preprojection_invalid_category_tuple.all()
    assert torch.all(projected.repaired_features[:, 0] == codebook.numeric_max[0])
    allowed_raw = {
        tuple(
            codebook.categorical_values[feature][index]
            for feature, index in enumerate(indices)
        )
        for indices in codebook.valid_categorical_tuples
    }
    assert all(
        tuple(row) in allowed_raw
        for row in projected.repaired_features[:, 2:].detach().numpy()
    )


def test_codebook_checkpoint_state_includes_bounds_and_joint_tuple_support():
    codebook = POIFeatureCodebook.fit(RAW)
    restored = POIFeatureCodebook.from_dict(codebook.to_dict())
    assert restored.numeric_min == codebook.numeric_min
    assert restored.numeric_max == codebook.numeric_max
    assert restored.valid_categorical_tuples == codebook.valid_categorical_tuples
    legacy = codebook.to_dict()
    legacy.pop("valid_categorical_tuples")
    with pytest.raises(ValueError, match="predates"):
        POIFeatureCodebook.from_dict(legacy)


def test_raw_float64_bounds_use_inward_float32_model_endpoints():
    raw = RAW.astype(np.float64)
    raw[0, 0] = 126.0000001
    raw[-1, 0] = 128.9999999
    raw[0, 1] = 34.0000001
    raw[-1, 1] = 37.9999999
    codebook = POIFeatureCodebook.fit(raw)
    assert codebook.numeric_min == (126.0000001, 34.0000001)
    assert codebook.numeric_max == (128.9999999, 37.9999999)
    encoded = codebook.encode(raw).numpy()
    assert np.all(encoded[:, :2] >= np.asarray(codebook.numeric_min))
    assert np.all(encoded[:, :2] <= np.asarray(codebook.numeric_max))
    assert np.all(np.asarray(codebook.numeric_model_min) >= np.asarray(codebook.numeric_min))
    assert np.all(np.asarray(codebook.numeric_model_max) <= np.asarray(codebook.numeric_max))


def test_all_five_reconstructors_are_invariant_to_their_own_input():
    codebook = POIFeatureCodebook.fit(RAW)
    encoded = codebook.encode(RAW)
    adapter = StructuralDefenseAdapter(codebook, hidden_dim=16).eval()
    original = adapter(encoded)
    for feature in range(5):
        changed = encoded.clone()
        if feature < 2:
            changed[0, feature] += 10.0
        else:
            changed[0, feature] = 1 - changed[0, feature]
        updated = adapter(changed)
        before = (
            original.reconstruction_numeric[:, feature]
            if feature < 2
            else original.reconstruction_categorical_logits[feature - 2]
        )
        after = (
            updated.reconstruction_numeric[:, feature]
            if feature < 2
            else updated.reconstruction_categorical_logits[feature - 2]
        )
        assert torch.allclose(before, after, atol=1e-7)


def test_predictor_primary_region_probability_uses_hard_valid_projection():
    codebook = POIFeatureCodebook.fit(RAW)
    model = ProposedTabPFNDefense(
        FakeOfficialTabPFN25(), codebook,
        tuple(f"region-{index}" for index in range(17)),
        hidden_dim=16, activation_checkpointing=False,
    )
    predictor = ProposedDefensePredictor(
        model, codebook.encode(RAW[:3]), torch.tensor([0, 1, 2]),
        device="cpu", batch_size=2,
    )
    result = predictor.predict(RAW[3:])
    assert result["region_probability"].shape == (2, 17)
    assert result["latent_region_probability"].shape == (2, 17)
    assert np.array_equal(
        result["region_prediction"], result["region_probability"] >= 0.5
    )
    predictor.assert_decision_contract(
        classification_threshold=0.5,
        regional_decision_rule=REGIONAL_DECISION_RULE,
    )
    with pytest.raises(RuntimeError, match="threshold"):
        predictor.assert_decision_contract(
            classification_threshold=0.4,
            regional_decision_rule=REGIONAL_DECISION_RULE,
        )
    assert np.allclose(
        result["projection_region_probability_absolute_delta"],
        np.abs(result["region_probability"] - result["latent_region_probability"]),
    )
    allowed = {tuple(row) for row in RAW[:, 2:]}
    assert all(tuple(row) in allowed for row in result["repaired_matrix"][:, 2:])


def test_checkpoint_roundtrip_preserves_and_applies_ovr_decision_contract(tmp_path):
    codebook = POIFeatureCodebook.fit(RAW)
    model = ProposedTabPFNDefense(
        FakeOfficialTabPFN25(), codebook,
        tuple(f"region-{index}" for index in range(17)),
        hidden_dim=16, activation_checkpointing=False,
    )
    context = codebook.encode(RAW[:3])
    labels = torch.tensor([0, 1, 2])
    payload = build_checkpoint_payload(
        model,
        context_encoded=context,
        context_region_indices=labels,
        official_provenance={"checkpoint_sha256": "a" * 64},
        training_metadata={
            "classification_threshold": 0.4,
            "regional_decision_rule": REGIONAL_DECISION_RULE,
        },
    )
    path = tmp_path / "proposed.pt"
    torch.save(payload, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert restored["model_config"]["classification_threshold"] == 0.4
    assert restored["training_metadata"]["regional_decision_rule"] == REGIONAL_DECISION_RULE
    predictor = ProposedDefensePredictor(
        model, context, labels, device="cpu", batch_size=2,
        classification_threshold=restored["model_config"]["classification_threshold"],
        regional_decision_rule=restored["model_config"]["regional_decision_rule"],
    )
    result = predictor.predict(RAW[3:])
    assert np.array_equal(
        result["region_prediction"], result["region_probability"] >= 0.4
    )


def test_proposed_model_calls_official_forward_as_17_ovr_prompts():
    codebook = POIFeatureCodebook.fit(RAW)
    base = FakeOfficialTabPFN25()
    model = ProposedTabPFNDefense(
        base,
        codebook,
        tuple(f"region-{index}" for index in range(17)),
        hidden_dim=16,
        activation_checkpointing=False,
    )
    result = model(codebook.encode(RAW), torch.tensor([0, 1, 2]))
    assert result.region_logits.shape == (2, 17)
    assert result.base_logits.shape == (2, 17, 10)
    called_x, called_y, kwargs = base.last_call
    assert called_x.shape == (5, 17, 5)
    assert called_y.shape == (3, 17)
    assert kwargs["only_return_standard_out"] is False
    result.region_logits.sum().backward()
    assert model.region_weights.grad is not None
    assert not any(parameter.grad is not None for parameter in base.parameters())


def test_paired_loss_trains_detection_reconstruction_and_region_terms():
    codebook = POIFeatureCodebook.fit(RAW)
    model = ProposedTabPFNDefense(
        FakeOfficialTabPFN25(),
        codebook,
        tuple(f"region-{index}" for index in range(17)),
        hidden_dim=16,
        activation_checkpointing=False,
    )
    encoded = codebook.encode(RAW)
    attacked = encoded.clone()
    attacked[-2:, 0] += 0.1
    attacked[-2:, 2] = 1 - attacked[-2:, 2]
    clean_output = model(encoded, torch.tensor([0, 1, 2]))
    attack_output = model(torch.cat((encoded[:3], attacked[-2:])), torch.tensor([0, 1, 2]))
    changed = torch.tensor([[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]], dtype=torch.float32)
    loss, terms = paired_defense_loss(
        clean_output,
        attack_output,
        query_region_indices=torch.tensor([3, 4]),
        clean_encoded_query=encoded[-2:],
        changed_feature_mask=changed,
        weights=DefenseLossWeights(),
    )
    assert set(terms) == {
        "clean_region",
        "attack_region",
        "consistency",
        "feature_detection",
        "reconstruction",
        "attack_detection",
    }
    assert torch.isfinite(loss)
    loss.backward()
    assert model.adapter.category_decoders[0].weight.grad is not None


def test_zero_consistency_weight_removes_only_consistency_from_objective():
    codebook = POIFeatureCodebook.fit(RAW)
    model = ProposedTabPFNDefense(
        FakeOfficialTabPFN25(),
        codebook,
        tuple(f"region-{index}" for index in range(17)),
        hidden_dim=16,
        activation_checkpointing=False,
    )
    encoded = codebook.encode(RAW)
    attacked = encoded.clone()
    attacked[-2:, 0] += 0.1
    clean_output = model(encoded, torch.tensor([0, 1, 2]))
    attack_output = model(
        torch.cat((encoded[:3], attacked[-2:])), torch.tensor([0, 1, 2])
    )
    # Make the consistency component large enough to remain visible next to the
    # deliberately class-balanced 17-way region losses in float32.
    attack_output.region_logits = attack_output.region_logits + 5.0
    arguments = {
        "query_region_indices": torch.tensor([3, 4]),
        "clean_encoded_query": encoded[-2:],
        "changed_feature_mask": torch.tensor(
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float32
        ),
    }
    full, terms = paired_defense_loss(
        clean_output,
        attack_output,
        weights=DefenseLossWeights(consistency=0.25),
        **arguments,
    )
    ablated, ablated_terms = paired_defense_loss(
        clean_output,
        attack_output,
        weights=DefenseLossWeights(consistency=0.0),
        **arguments,
    )
    assert torch.equal(terms["consistency"], ablated_terms["consistency"])
    assert torch.allclose(
        full - ablated, 0.25 * terms["consistency"], atol=1e-5, rtol=1e-4
    )


def test_fixed_ovr_positive_weight_changes_only_region_terms():
    codebook = POIFeatureCodebook.fit(RAW)
    model = ProposedTabPFNDefense(
        FakeOfficialTabPFN25(),
        codebook,
        tuple(f"region-{index}" for index in range(17)),
        hidden_dim=16,
        activation_checkpointing=False,
    )
    encoded = codebook.encode(RAW)
    clean = model(encoded, torch.tensor([0, 1, 2]))
    attack = model(encoded, torch.tensor([0, 1, 2]))
    clean.region_logits = torch.zeros_like(clean.region_logits)
    attack.region_logits = torch.zeros_like(attack.region_logits)
    arguments = {
        "query_region_indices": torch.tensor([3, 4]),
        "clean_encoded_query": encoded[-2:],
        "changed_feature_mask": torch.zeros((2, 5)),
    }
    _, unweighted = paired_defense_loss(
        clean,
        attack,
        weights=DefenseLossWeights(region_positive_weight=1.0),
        **arguments,
    )
    _, weighted = paired_defense_loss(
        clean,
        attack,
        weights=DefenseLossWeights(region_positive_weight=16.0),
        **arguments,
    )
    assert weighted["clean_region"] > unweighted["clean_region"]
    assert weighted["attack_region"] > unweighted["attack_region"]
    for name in (
        "consistency",
        "feature_detection",
        "reconstruction",
        "attack_detection",
    ):
        assert torch.equal(weighted[name], unweighted[name])


def test_attack_epoch_sampling_preserves_pairs_and_balances_available_strata():
    rows = []
    for poi in range(12):
        for generation_seed in (42, 202):
            for condition in ("a", "b"):
                if poi == 0 and condition == "a":
                    continue  # validity filtering can remove a stratum for a POI.
                rows.append(
                    {
                        "POI_ID": f"p{poi}",
                        "generation_seed": generation_seed,
                        "attack_condition": condition,
                        "region": "R",
                    }
                )
    selected = balanced_attack_sample(pd.DataFrame(rows), seed=42, max_pois=8)
    assert len(selected) == 8
    assert selected["POI_ID"].is_unique
    counts = selected.groupby(["generation_seed", "attack_condition"]).size()
    assert counts.max() - counts.min() <= 1


def test_attack_diagnostics_retain_unchanged_intent_and_feature_collisions():
    clean = pd.DataFrame(
        RAW[:2],
        columns=[
            "X_COORD",
            "Y_COORD",
            "ASORT_LCLASDC",
            "ASORT_MLSFCDC",
            "ASORT_SDASDC",
        ],
    )
    attacks = pd.DataFrame(
        {
            **{
                f"{feature}_clean": clean[feature]
                for feature in clean.columns
            },
            **{
                f"{feature}_adv": clean[feature]
                for feature in clean.columns
            },
            **{
                f"changed_{feature.lower()}": [False, False]
                for feature in clean.columns
            },
            "attack_attempted": [True, False],
            "attack_changed_any": [False, False],
            "attack_success_source": [False, False],
            "constraints_valid": [True, False],
            "label_preserved": [True, False],
            "primary_evaluation_eligible": [True, False],
        }
    )
    diagnostics = attack_intent_diagnostics(attacks, clean)
    assert diagnostics["attack_intent_rows"] == 2
    assert diagnostics["unchanged_attack_rows"] == 2
    assert diagnostics["attempted_unchanged_rows"] == 1
    assert diagnostics["unattempted_attack_intent_rows"] == 1
    assert diagnostics["clean_attack_feature_collision_rows"] == 2
    assert diagnostics["primary_evaluation_ineligible_rows"] == 1


def test_training_pool_filters_validity_but_not_source_attack_success():
    frame = pd.DataFrame(
        {
            "row": ["valid_failure", "valid_success", "invalid_success"],
            "primary_evaluation_eligible": [True, True, False],
            "constraints_valid": [True, True, False],
            "label_preserved": [True, True, True],
            "attack_success_source": [False, True, True],
        }
    )
    selected = eligible_training_attacks(frame)
    assert selected["row"].tolist() == ["valid_failure", "valid_success"]


def test_attack_integrity_fails_closed_on_source_and_mask_mismatch():
    columns = ["X_COORD", "Y_COORD", "ASORT_LCLASDC", "ASORT_MLSFCDC", "ASORT_SDASDC"]
    clean = pd.DataFrame(RAW[:2], columns=columns)
    clean["POI_ID"] = ["p0", "p1"]
    clean["split"] = "train"
    clean["region"] = ["r0", "r1"]
    adv = RAW[0].copy()
    adv[0] += 0.00005
    mean_latitude = np.deg2rad((RAW[0, 1] + adv[1]) / 2)
    distance = abs(adv[0] - RAW[0, 0]) * 111_320.0 * np.cos(mean_latitude)
    attacks = pd.DataFrame({
        "POI_ID": ["p0"], "split": ["train"], "region": ["r0"],
        **{f"{name}_clean": [RAW[0, i]] for i, name in enumerate(columns)},
        **{f"{name}_adv": [adv[i]] for i, name in enumerate(columns)},
        **{f"changed_{name.lower()}": [i == 0] for i, name in enumerate(columns)},
        "attack_changed_any": [True], "distance_m": [distance], "categorical_l0": [0],
        "numeric_budget_m": [10.0], "numeric_budget_was_implicit_zero": [False],
        "categorical_budget_l0": [0],
        "distance_metric": ["local_equirectangular_wgs84_approximation"],
        "distance_tolerance_m": [0.05], "valid_numeric_train_range": [True],
        "valid_category_codebook": [True], "valid_clean_category_combo": [True],
        "valid_attack_category_combo": [True], "valid_category_combo": [True],
        "within_budget": [True], "label_preserved": [True],
        "constraints_valid": [True], "primary_evaluation_eligible": [True],
    })
    _validate_attack_integrity(clean, attacks)
    bad_source = attacks.copy()
    bad_source.loc[0, "X_COORD_clean"] += 0.1
    with pytest.raises(ValueError, match="clean features"):
        _validate_attack_integrity(clean, bad_source)
    bad_mask = attacks.copy()
    bad_mask.loc[0, "changed_x_coord"] = False
    with pytest.raises(ValueError, match="masks"):
        _validate_attack_integrity(clean, bad_mask)
    precision_mismatch = attacks.copy()
    precision_mismatch["X_COORD_clean"] = precision_mismatch["X_COORD_clean"].astype(np.float64)
    precision_mismatch.loc[0, "X_COORD_clean"] += 1e-7
    with pytest.raises(ValueError, match="clean features"):
        _validate_attack_integrity(clean, precision_mismatch)
    for invalid in (2.0, np.nan, 0.5, "1"):
        bad_value = attacks.copy()
        bad_value["changed_x_coord"] = pd.Series(
            [invalid], dtype=object if isinstance(invalid, str) else float
        )
        with pytest.raises(ValueError, match="mask"):
            _validate_attack_integrity(clean, bad_value)


def test_changed_masks_accept_only_boolean_or_finite_exact_binary_values():
    names = [
        "changed_x_coord", "changed_y_coord", "changed_asort_lclasdc",
        "changed_asort_mlsfcdc", "changed_asort_sdasdc",
    ]
    boolean = pd.DataFrame({name: [True, False] for name in names})
    numeric = pd.DataFrame({name: [1, 0] for name in names})
    assert np.array_equal(changed_mask(boolean), changed_mask(numeric))
    for invalid in (2.0, np.nan, 0.5, "1"):
        frame = numeric.copy()
        frame[names[0]] = pd.Series(
            [invalid, 0], dtype=object if isinstance(invalid, str) else float
        )
        with pytest.raises(ValueError, match="mask"):
            changed_mask(frame)


def test_primary_ovr_threshold_rule_is_not_replaced_by_argmax():
    probabilities = np.asarray([[0.60, 0.55, 0.10], [0.49, 0.48, 0.47]])
    independent = _ovr_binary_predictions(probabilities, 0.5)
    argmax = np.eye(3, dtype=np.int64)[probabilities.argmax(axis=1)]
    assert independent.tolist() == [[1, 1, 0], [0, 0, 0]]
    assert not np.array_equal(independent, argmax)


def _small_inventory_fixture():
    clean = pd.DataFrame(
        {
            "POI_ID": ["train-1", "validation-1"],
            "split": ["train", "validation"],
            "region": ["Seoul", "Busan"],
        }
    )
    cfg = yaml.safe_load(open("configs/stages/01_attack_generation.yaml"))
    specs = attack_condition_specs(cfg)
    rows = []
    for seed in (42, 202, 340):
        for spec in specs:
            for item in clean.to_dict(orient="records"):
                rows.append(
                    {
                        **item,
                        "generation_seed": seed,
                        "attack_condition": spec["condition"],
                        "attack_method": spec["method"],
                        "numeric_budget_m": (
                            0.0 if spec["numeric_budget_m"] is None
                            else float(spec["numeric_budget_m"])
                        ),
                        "numeric_budget_was_implicit_zero": (
                            spec["numeric_budget_m"] is None
                        ),
                        "categorical_budget_l0": spec["categorical_budget_l0"],
                        "generation_identity_sha256": "a" * 64,
                        "surrogate_checkpoint_sha256": "b" * 64,
                        "attack_schema_version": ATTACK_SCHEMA_VERSION,
                        "query_count_scope": QUERY_COUNT_SCOPE,
                        "source_model": SURROGATE_ID,
                        "victim_model": SURROGATE_ID,
                        "victim_evaluation_scope": "direct_white_box_source_model",
                        "transfer_evaluated": False,
                        "attack_attempted": True,
                    }
                )
    return clean, specs, pd.DataFrame(rows)


def _small_inventory(clean, specs):
    return FullAttackInventoryValidator(
        clean,
        specs=specs,
        generation_seeds=[42, 202, 340],
        generation_identity_sha256="a" * 64,
        surrogate_checkpoint_sha256="b" * 64,
        expected_split_counts={"train": 1, "validation": 1},
    )


def test_full_inventory_requires_complete_unique_three_by_thirty_two_grid():
    clean, specs, frame = _small_inventory_fixture()
    inventory = _small_inventory(clean, specs)
    inventory.consume(frame.iloc[:97])
    inventory.consume(frame.iloc[97:])
    summary = inventory.finalize()
    assert summary["status"] == "PASS"
    assert summary["rows"] == 2 * 3 * 32
    assert summary["groups"] == 192
    assert summary["clean_development_pois"] == 2

    duplicate = _small_inventory(clean, specs)
    duplicate.consume(frame.iloc[:1])
    with pytest.raises(ValueError, match="Duplicate POI membership"):
        duplicate.consume(frame.iloc[:1])

    missing = _small_inventory(clean, specs)
    missing.consume(frame.iloc[:-1])
    with pytest.raises(ValueError, match="Incomplete attack group"):
        missing.finalize()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("generation_seed", 999, "undeclared generation seed"),
        ("attack_condition", "extra", "undeclared attack condition"),
        ("split", "validation", "split differs"),
        ("region", "wrong", "region differs"),
    ],
)
def test_full_inventory_rejects_seed_condition_split_and_region_mutations(
    column, value, message
):
    clean, specs, frame = _small_inventory_fixture()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        _small_inventory(clean, specs).consume(frame.iloc[:1])


@pytest.mark.parametrize("mode", ["missing", "empty"])
def test_real_configured_preflight_fails_closed_without_complete_attack_artifact(
    tmp_path, mode
):
    attack = tmp_path / "01_attack_generation" / "tables" / "attacks.parquet"
    if mode == "empty":
        attack.parent.mkdir(parents=True)
        attack.write_bytes(b"")
    output = tmp_path / f"preflight-{mode}"
    with pytest.raises((FileNotFoundError, ValueError)):
        run_preflight(
            data_dir="data/poi_34k_seed42",
            attack_table=attack,
            attack_generation_config="configs/stages/01_attack_generation.yaml",
            output=output,
        )
    report = yaml.safe_load((output / "preflight.json").read_text())
    assert report["status"] == "FAIL"
    assert report["training_ready"] is False
    assert report["test_rows_read"] == 0


def test_training_gate_rejects_self_declared_pass_without_full_inventory(tmp_path):
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    (preflight / "preflight.json").write_text(
        """{
          "status": "PASS",
          "training_ready": true,
          "test_rows_read": 0,
          "attack": {
            "complete_validated_inventory": {
              "status": "PASS", "rows": 2, "expected_rows": 2,
              "groups": 2, "expected_groups": 2,
              "unique_complete_poi_membership": true,
              "split_region_manifest_truth": true
            }
          }
        }"""
    )
    with pytest.raises(RuntimeError, match="exact full attack inventory"):
        _require_current_data_preflight({"preflight_output": str(preflight)})


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _complete_lineage_fixture(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name, content in {
        "poi_data_region.csv": b"clean-features",
        "sample_manifest.csv": b"clean-manifest",
        "report.json": b"clean-report",
    }.items():
        (data_dir / name).write_bytes(content)
    generation_config = tmp_path / "01_attack_generation.yaml"
    generation_config.write_bytes(
        Path("configs/stages/01_attack_generation.yaml").read_bytes()
    )
    cfg = yaml.safe_load(generation_config.read_text())
    specs = attack_condition_specs(cfg)
    seeds = [42, 202, 340]
    stage = tmp_path / "01_attack_generation"
    tables = stage / "tables"
    work = stage / "work"
    surrogate = work / "surrogate"
    tables.mkdir(parents=True)
    surrogate.mkdir(parents=True)
    attack = tables / "attacks.parquet"
    attack.write_bytes(b"synthetic-complete-table-bytes")
    checkpoint = surrogate / "surrogate.pt"
    encoder = surrogate / "surrogate_encoder.json"
    history = surrogate / "surrogate_training.csv"
    checkpoint.write_bytes(b"checkpoint")
    encoder.write_bytes(b"encoder")
    history.write_bytes(b"history")
    clean_hashes = {
        name: preflight_sha256(data_dir / name)
        for name in ("poi_data_region.csv", "sample_manifest.csv", "report.json")
    }
    request = {
        "format": "poi-generation-request-v2",
        "attack_schema_version": ATTACK_SCHEMA_VERSION,
        "config_sha256": preflight_sha256(generation_config),
        "dataset_sha256": clean_hashes,
        "features": [
            "X_COORD", "Y_COORD", "ASORT_LCLASDC", "ASORT_MLSFCDC", "ASORT_SDASDC"
        ],
        "splits": ["train", "validation"],
        "generation_seeds": seeds,
        "conditions": specs,
        "expected_rows": 2_611_200,
        "expected_shards": 192,
        "query_count_scope": QUERY_COUNT_SCOPE,
        "code_sha256": {
            f"/producer/src/poi/{name}": preflight_sha256(Path("src/poi") / name)
            for name in ("adversarial.py", "data.py", "geospatial.py")
        },
    }
    identity = {
        "format": "poi-generation-identity-v2",
        "generation_request_sha256": _canonical_sha256(request),
        "surrogate_checkpoint_sha256": preflight_sha256(checkpoint),
        "surrogate_encoder_sha256": preflight_sha256(encoder),
        "surrogate_training_history_sha256": preflight_sha256(history),
    }
    identity["generation_identity_sha256"] = _canonical_sha256(identity)
    shards = {
        f"{split}-g{seed}-{spec['condition']}.parquet": {
            "sha256": "a" * 64,
            "rows": 23_800 if split == "train" else 3_400,
            "ordered_poi_ids_sha256": "b" * 64,
            "key_sha256": "c" * 64,
        }
        for seed in seeds
        for split in ("train", "validation")
        for spec in specs
    }
    manifest = {
        "format": "poi-shard-manifest-v2",
        "generation_identity_sha256": identity["generation_identity_sha256"],
        "shards": shards,
    }
    request_path = work / "generation_request.json"
    identity_path = work / "generation_identity.json"
    manifest_path = work / "shard_manifest.json"
    _write_json(request_path, request)
    _write_json(identity_path, identity)
    _write_json(manifest_path, manifest)
    metadata = {
        "status": "generated_pending_fresh_independent_review",
        "source_model": SURROGATE_ID,
        "generation_seeds": seeds,
        "attack_conditions": 32,
        "attack_rows": 2_611_200,
        "loaded_feature_splits": ["train", "validation"],
        "test_used": False,
        "dataset_sha256": {
            name: clean_hashes[name]
            for name in ("poi_data_region.csv", "sample_manifest.csv")
        },
        "generation_request": {
            "path": str(request_path),
            "sha256": _canonical_sha256(request),
        },
        "generation_identity": {"path": str(identity_path), **identity},
        "shard_manifest": {
            "path": str(manifest_path),
            "sha256": preflight_sha256(manifest_path),
            "registered_shards": 192,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": preflight_sha256(checkpoint),
        },
        "tables": {
            "attacks": {
                "path": str(attack),
                "sha256": preflight_sha256(attack),
                "rows": 2_611_200,
            }
        },
    }
    metadata_path = stage / "metadata.json"
    marker_path = stage / "_PENDING_REVIEW.json"
    _write_json(metadata_path, metadata)
    _write_json(marker_path, metadata)
    paths = {
        "checkpoint": checkpoint,
        "encoder": encoder,
        "history": history,
        "manifest": manifest_path,
        "metadata": metadata_path,
        "marker": marker_path,
        "request": request_path,
        "identity": identity_path,
    }
    return attack, generation_config, data_dir, paths


def _full_launch_fixture(tmp_path):
    attack, generation_config, data_dir, paths = _complete_lineage_fixture(tmp_path)
    lineage = _validate_producer_lineage(attack, generation_config, data_dir)
    module_dir = Path("src/poi/proposed_defense")
    preflight_output = tmp_path / "preflight"
    preflight_output.mkdir()
    report = {
        "status": "PASS",
        "training_ready": True,
        "test_rows_read": 0,
        "data_dir": str(data_dir),
        "attack_table": str(attack),
        "attack_generation_config": str(generation_config),
        "surrogate_artifact_dir": str(paths["checkpoint"].parent),
        "implementation_sha256": {
            name: preflight_sha256(module_dir / name)
            for name in ("preflight.py", "data.py", "schema.py")
        },
        "dependency_sha256": {
            f"src/poi/{name}": preflight_sha256(Path("src/poi") / name)
            for name in ("adversarial.py", "data.py", "geospatial.py")
        },
        "ure_preprocessing_sha256": preflight_sha256(
            Path("src/poi/ure/preprocessing.py")
        ),
        "clean": {
            "source_sha256": {
                name: preflight_sha256(data_dir / name)
                for name in ("poi_data_region.csv", "sample_manifest.csv", "report.json")
            }
        },
        "attack": {
            "sha256": preflight_sha256(attack),
            "stable_snapshot_before_after": True,
            "producer_lineage": lineage["report"],
            "complete_validated_inventory": {
                "status": "PASS",
                "rows": 2_611_200,
                "expected_rows": 2_611_200,
                "clean_development_pois": 27_200,
                "expected_clean_development_pois": 27_200,
                "clean_pois_by_split": {"train": 23_800, "validation": 3_400},
                "expected_clean_pois_by_split": {
                    "train": 23_800, "validation": 3_400
                },
                "groups": 192,
                "expected_groups": 192,
                "generation_seed_count": 3,
                "attack_condition_count": 32,
                "unique_complete_poi_membership": True,
                "split_region_manifest_truth": True,
            },
        },
    }
    report_path = preflight_output / "preflight.json"
    _write_json(report_path, report)
    cfg = {
        "preflight_output": str(preflight_output),
        "attack_table": str(attack),
        "attack_generation_config": str(generation_config),
        "data_dir": str(data_dir),
        "surrogate_artifact_dir": str(paths["checkpoint"].parent),
    }
    return cfg, report_path, paths


@pytest.mark.parametrize("artifact", ["checkpoint", "encoder", "history"])
def test_lineage_requires_all_three_surrogate_artifacts(tmp_path, artifact):
    attack, generation_config, data_dir, paths = _complete_lineage_fixture(tmp_path)
    paths[artifact].unlink()
    with pytest.raises(FileNotFoundError, match="surrogate artifact"):
        _validate_producer_lineage(attack, generation_config, data_dir)


def test_lineage_verifies_all_available_shard_bytes(tmp_path):
    attack, generation_config, data_dir, paths = _complete_lineage_fixture(tmp_path)
    manifest = json.loads(paths["manifest"].read_text())
    shard_directory = paths["manifest"].parent / "shards"
    shard_directory.mkdir()
    for name, entry in manifest["shards"].items():
        path = shard_directory / name
        path.write_bytes(name.encode())
        entry["sha256"] = preflight_sha256(path)
    _write_json(paths["manifest"], manifest)
    metadata = json.loads(paths["metadata"].read_text())
    metadata["shard_manifest"]["sha256"] = preflight_sha256(paths["manifest"])
    _write_json(paths["metadata"], metadata)
    _write_json(paths["marker"], metadata)
    result = _validate_producer_lineage(attack, generation_config, data_dir)
    assert result["report"]["shard_artifact_bytes"] == (
        "verified_all_192_manifest_hashes"
    )
    first = next(iter(shard_directory.glob("*.parquet")))
    first.write_bytes(first.read_bytes() + b"mutated")
    with pytest.raises(ValueError, match="shard hash differs"):
        _validate_producer_lineage(attack, generation_config, data_dir)


@pytest.mark.parametrize(
    "artifact",
    ["checkpoint", "manifest", "metadata", "marker", "request", "identity"],
)
def test_training_gate_reopens_and_rejects_six_lineage_mutations(tmp_path, artifact):
    cfg, _, paths = _full_launch_fixture(tmp_path)
    _require_current_data_preflight(cfg)
    with paths[artifact].open("ab") as stream:
        stream.write(b" \n")
    with pytest.raises(RuntimeError, match="producer lineage"):
        _require_current_data_preflight(cfg)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("implementation_sha256", "preflight.py"),
        ("implementation_sha256", "data.py"),
        ("implementation_sha256", "schema.py"),
        ("dependency_sha256", "src/poi/adversarial.py"),
        ("dependency_sha256", "src/poi/data.py"),
        ("dependency_sha256", "src/poi/geospatial.py"),
        ("ure_preprocessing_sha256", None),
    ],
)
def test_training_gate_rejects_each_preflight_dependency_hash_edit(
    tmp_path, section, key
):
    cfg, report_path, _ = _full_launch_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    if key is None:
        report[section] = "0" * 64
    else:
        report[section][key] = "0" * 64
    _write_json(report_path, report)
    with pytest.raises(RuntimeError, match="source|dependency|URE"):
        _require_current_data_preflight(cfg)


def test_training_gate_rejects_geospatial_source_mutation(tmp_path, monkeypatch):
    cfg, _, _ = _full_launch_fixture(tmp_path)
    _require_current_data_preflight(cfg)
    original = preflight_module._sha256

    def changed_geospatial(path):
        path = Path(path)
        if path.resolve() == Path("src/poi/geospatial.py").resolve():
            return "0" * 64
        return original(path)

    monkeypatch.setattr(preflight_module, "_sha256", changed_geospatial)
    with pytest.raises(RuntimeError, match="producer lineage"):
        _require_current_data_preflight(cfg)


def test_reliability_gate_ablation_uses_fixed_blend_and_changes_forward_path():
    codebook = POIFeatureCodebook.fit(RAW)
    encoded = codebook.encode(RAW)
    gated = StructuralDefenseAdapter(codebook, hidden_dim=16)
    ungated = StructuralDefenseAdapter(
        codebook, hidden_dim=16, use_reliability_gate=False
    )
    ungated.load_state_dict(gated.state_dict())
    gated_output = gated(encoded)
    ungated_output = ungated(encoded)
    assert torch.equal(
        ungated_output.reliability, torch.full_like(ungated_output.reliability, 0.5)
    )
    assert not torch.allclose(
        gated_output.repaired_features, ungated_output.repaired_features
    )


def test_reconstruction_ablation_bypasses_modules_and_returns_observations():
    codebook = POIFeatureCodebook.fit(RAW)
    encoded = codebook.encode(RAW)
    adapter = StructuralDefenseAdapter(
        codebook, hidden_dim=16, use_reconstruction_path=False
    )
    output = adapter(encoded)
    assert torch.equal(output.repaired_features, codebook.decode(encoded))
    assert torch.count_nonzero(output.reconstructed_tokens) == 0
    assert torch.count_nonzero(output.reconstruction_numeric) == 0
    assert all(
        torch.count_nonzero(logits) == 0
        for logits in output.reconstruction_categorical_logits
    )
    output.attack_logits.sum().backward()
    assert all(
        parameter.grad is None
        for module in adapter.cross_feature_reconstructors
        for parameter in module.parameters()
    )


def test_multiscale_ablation_replaces_basis_with_one_scalar():
    codebook = POIFeatureCodebook.fit(RAW)
    full = StructuralDefenseAdapter(codebook, hidden_dim=16)
    ablated = StructuralDefenseAdapter(
        codebook, hidden_dim=16, use_multiscale_encoder=False
    )
    assert full.numeric_encoders[0][0].in_features == 12
    assert ablated.numeric_encoders[0][0].in_features == 1
    ablated(codebook.encode(RAW)).attack_logits.sum().backward()
    assert ablated.numeric_encoders[0][0].weight.grad is not None


def test_corruption_head_ablation_removes_heads_and_uses_fixed_blend():
    codebook = POIFeatureCodebook.fit(RAW)
    adapter = StructuralDefenseAdapter(codebook, hidden_dim=16, use_corruption_heads=False)
    output = adapter(codebook.encode(RAW))
    assert len(adapter.corruption_heads) == 0
    assert torch.count_nonzero(output.feature_corruption_logits) == 0
    assert torch.equal(output.reliability, torch.full_like(output.reliability, 0.5))


def test_encoder_ablation_removes_learned_feature_encoders_from_forward():
    codebook = POIFeatureCodebook.fit(RAW)
    adapter = StructuralDefenseAdapter(
        codebook, hidden_dim=16, use_multiscale_encoder=False,
        use_feature_encoders=False, use_category_relations=False,
    )
    assert len(adapter.numeric_encoders) == 0
    assert len(adapter.category_embeddings) == 0
    output = adapter(codebook.encode(RAW))
    output.attack_logits.sum().backward()
    assert adapter.cross_feature_reconstructors[0][0].weight.grad is not None


def test_frozen_reference_and_unfrozen_last_block_change_trainability():
    codebook = POIFeatureCodebook.fit(RAW)
    base = FakeOfficialTabPFN25()
    model = ProposedTabPFNDefense(
        base,
        codebook,
        tuple(f"region-{index}" for index in range(17)),
        hidden_dim=16,
        activation_checkpointing=False,
    )
    assert not any(parameter.requires_grad for parameter in base.parameters())
    model.unfreeze_last_blocks(1)
    assert all(parameter.requires_grad for parameter in base.blocks[-1].parameters())
    assert not any(parameter.requires_grad for parameter in base.projection.parameters())
    model.unfreeze_last_blocks(0)
    assert not any(parameter.requires_grad for parameter in base.parameters())
    with pytest.raises(ValueError, match="Cannot unfreeze"):
        model.unfreeze_last_blocks(-1)
    with pytest.raises(ValueError, match="Cannot unfreeze"):
        model.unfreeze_last_blocks(2)
    model.unfreeze_last_blocks(1)
    expected = _expected_unfrozen_base_keys(model)
    assert expected and expected.issubset(model.trainable_state_dict())
    incomplete = model.trainable_state_dict()
    removed = next(iter(expected))
    incomplete.pop(removed)
    with pytest.raises(RuntimeError, match="trained official block"):
        _validate_checkpoint_state_completeness(
            model, incomplete, [removed], []
        )


def test_stage14_matrix_distinguishes_structure_objective_and_optimization():
    variants = {
        name: (train, config)
        for name, train, config in resolve_ablation_configs(
            "configs/stages/14_proposed_structure_ablation.yaml"
        )
    }
    assert set(variants) == {
        "full_model_staged",
        "no_reliability_gate",
        "no_reconstruction_path",
        "no_encoder_enhancements",
        "no_corruption_heads",
        "no_consistency_loss",
        "no_fine_tuning_block",
    }
    assert variants["full_model_staged"][0] is False
    assert variants["no_reliability_gate"][1]["ablation_kind"] == "structural"
    assert variants["no_reconstruction_path"][1]["loss_weights"]["reconstruction"] == 0
    assert variants["no_encoder_enhancements"][1]["model"]["use_multiscale_encoder"] is False
    assert variants["no_encoder_enhancements"][1]["model"]["use_feature_encoders"] is False
    assert variants["no_encoder_enhancements"][1]["model"]["use_category_relations"] is False
    assert variants["no_corruption_heads"][1]["model"]["use_corruption_heads"] is False
    assert variants["no_corruption_heads"][1]["loss_weights"]["feature_detection"] == 0
    assert variants["no_consistency_loss"][1]["ablation_kind"] == "objective"
    assert variants["no_consistency_loss"][1]["loss_weights"]["consistency"] == 0
    assert variants["no_fine_tuning_block"][1]["ablation_kind"] == "optimization"
    assert variants["no_fine_tuning_block"][1]["fine_tuning"]["mode"] == "disabled"
