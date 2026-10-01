import os

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip('torch')

from poi.adversarial import (
    PENDING_STATUS,
    QUERY_COUNT_SCOPE,
    _ensure_immutable_json,
    _load_shard_manifest,
    categorical_exact,
    categorical_pcaa,
    condition_rng_seed,
    load_resumed_shard,
    mixed_evolution,
    numeric_attack_batch,
    persist_shard,
    records,
    load_surrogate,
    train_surrogate,
    validate_full_key_coverage,
    validate_shard_frame,
    valid_tuples,
)
from poi.data import CATEGORICAL, FEATURES, NUMERIC, load_clean_splits


class _BoundaryValidator:
    def validate(self, coordinates, expected_regions):
        matched = coordinates[:, 0] < 126.00005
        return {
            'matches': matched,
            'status': np.where(matched, 'consensus_match', 'source_disagreement'),
            'consensus_region': np.where(matched, expected_regions, ''),
            'source_labels': {
                'source_a': np.where(matched, expected_regions, ''),
                'source_b': np.where(matched, expected_regions, 'B'),
            },
        }


class _AllMatchBoundary:
    def validate(self, coordinates, expected_regions):
        count = len(coordinates)
        expected_regions = np.asarray(expected_regions, dtype=object)
        return {
            'matches': np.ones(count, dtype=bool),
            'status': np.full(count, 'consensus_match', dtype=object),
            'consensus_region': expected_regions.copy(),
            'source_labels': {
                'source_a': expected_regions.copy(),
                'source_b': expected_regions.copy(),
            },
        }


def _clean_boundary(frame):
    return _AllMatchBoundary().validate(
        frame[NUMERIC].to_numpy(float), frame.region.to_numpy())


def _attack_cfg():
    return {
        'attack_batch_size': 16,
        'attacks': {
            'fgsm': {'steps': 1},
            'pgd': {'steps': 2},
            'cw_l2': {'steps': 2, 'kappa': 0.0, 'distance_weight': 0.05},
            'capgd': {'steps': 2},
            'pcaa': {
                'steps': 2, 'learning_rate': 0.1,
                'entropy_weight': 0.01, 'candidate_draws': 2,
            },
            'moeva': {
                'generations': 2, 'population': 4, 'batch_size': 8,
                'mutation_scale': 0.1, 'category_mutation': 0.2,
            },
        },
        'distance_validation': {
            'metric': 'local_equirectangular_wgs84_approximation',
            'numeric_budget_tolerance_m': 0.05,
            'categorical_only_tolerance_m': 1e-6,
        },
    }


def _encoder():
    return {
        'regions': ['A', 'B'],
        'category_values': {
            'ASORT_LCLASDC': [1.0, 2.0],
            'ASORT_MLSFCDC': [2.0, 3.0],
            'ASORT_SDASDC': [3.0, 4.0],
        },
        'numeric_mean': np.array([126.0, 37.0], dtype=np.float32),
        'numeric_std': np.array([1.0, 1.0], dtype=np.float32),
    }


class _CountingModel(__import__('torch').nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.evaluated_rows = 0

    def forward(self, values):
        self.calls += 1
        self.evaluated_rows += len(values)
        score = values[:, 0] + 0.25 * values[:, 1]
        return __import__('torch').stack((-score, score), dim=1)


def test_records_require_train_ranges_real_polygon_consensus_and_actual_victim():
    frame = pd.DataFrame({
        'POI_ID': ['p1', 'p2'], 'region': ['A', 'A'], 'split': ['validation'] * 2,
        'X_COORD': [126.0, 126.0], 'Y_COORD': [37.0, 37.0],
        'ASORT_LCLASDC': [1.0, 1.0], 'ASORT_MLSFCDC': [2.0, 2.0],
        'ASORT_SDASDC': [3.0, 3.0],
    })
    attacked_coords = np.array([[126.0, 37.0], [126.0001, 37.0]])
    categories = frame[CATEGORICAL].to_numpy(np.float32)
    encoder = {
        'regions': ['A', 'B'],
        'category_values': {name: [float(index + 1)]
                            for index, name in enumerate(CATEGORICAL)},
    }
    clean_boundary = {
        'matches': np.array([True, True]),
        'status': np.array(['consensus_match', 'consensus_match']),
        'consensus_region': np.array(['A', 'A']),
        'source_labels': {'source_a': np.array(['A', 'A']),
                          'source_b': np.array(['A', 'A'])},
    }
    result = records(
        frame, attacked_coords, categories, np.array([0, 0]),
        np.array([0, 0]), np.array([1, 1]), 'pgd', 'pgd_m20', 20, 0,
        np.array([2, 2]), categories[:1], 202, 42,
        {'min': np.array([125.0, 36.0]), 'max': np.array([126.00005, 38.0])},
        encoder, _BoundaryValidator(), clean_boundary)

    assert result.constraints_valid.tolist() == [True, False]
    assert result.label_preserved.tolist() == [True, False]
    assert result.valid_numeric_train_range.tolist() == [True, False]
    assert result.generation_seed.tolist() == [202, 202]
    assert result.attack_rng_seed.nunique() == 1
    assert result.attack_rng_seed.iloc[0] == condition_rng_seed(
        202, 'validation', 'pgd_m20')
    assert result.victim_model.unique().tolist() == ['poi_mlp_surrogate_v1']
    assert result.victim_evaluation_scope.unique().tolist() == [
        'direct_white_box_source_model']
    assert not result.transfer_evaluated.any()
    assert 'attack_label_source_disagreement' in result.quarantine_reason.iloc[1]
    assert 'numeric_outside_train_range' in result.quarantine_reason.iloc[1]


def test_valid_category_tuples_are_derived_from_given_train_frame_only():
    train = pd.DataFrame({
        'ASORT_LCLASDC': [1.0, 1.0], 'ASORT_MLSFCDC': [2.0, 2.0],
        'ASORT_SDASDC': [3.0, 3.0],
    })
    encoder = {'category_values': {
        'ASORT_LCLASDC': [1.0], 'ASORT_MLSFCDC': [2.0], 'ASORT_SDASDC': [3.0]}}
    tuples, _ = valid_tuples(train, encoder)

    assert tuples.tolist() == [[1.0, 2.0, 3.0]]


def test_attack_config_has_three_generation_seeds_and_locked_splits():
    config = yaml.safe_load(open('configs/stages/01_attack_generation.yaml'))

    assert config['generation_seeds'] == [42, 202, 340]
    assert config['splits'] == ['train', 'validation']
    assert 'test' not in config['splits']


def test_condition_rng_seed_is_resume_stable_and_generation_specific():
    first = condition_rng_seed(42, 'validation', 'capgd_m100')

    assert first == condition_rng_seed(42, 'validation', 'capgd_m100')
    assert len({condition_rng_seed(seed, 'validation', 'capgd_m100')
                for seed in [42, 202, 340]}) == 3
    assert first != condition_rng_seed(42, 'train', 'capgd_m100')


def test_unseen_unchanged_tuple_is_quarantined_under_train_only_rule():
    frame = pd.DataFrame({
        'POI_ID': ['p1'], 'region': ['A'], 'split': ['validation'],
        'X_COORD': [126.0], 'Y_COORD': [37.0],
        'ASORT_LCLASDC': [1.0], 'ASORT_MLSFCDC': [20.0],
        'ASORT_SDASDC': [300.0],
    })
    categories = frame[CATEGORICAL].to_numpy(np.float32)
    encoder = {
        'regions': ['A', 'B'],
        'category_values': {
            'ASORT_LCLASDC': [1.0, 2.0],
            'ASORT_MLSFCDC': [10.0, 20.0],
            'ASORT_SDASDC': [100.0, 300.0],
        },
    }
    train_tuples = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 300.0]], np.float32)
    result = records(
        frame, frame[NUMERIC].to_numpy(float), categories,
        np.array([0]), np.array([0]), np.array([0]), 'pgd', 'pgd_m10',
        10, 0, np.array([3]), train_tuples, 42, 42,
        {'min': np.array([125.0, 36.0]), 'max': np.array([127.0, 38.0])},
        encoder, _AllMatchBoundary(), _clean_boundary(frame))

    assert not result.valid_clean_category_combo.iloc[0]
    assert not result.valid_attack_category_combo.iloc[0]
    assert not result.constraints_valid.iloc[0]
    assert 'clean_category_tuple_not_observed_in_train' in result.quarantine_reason.iloc[0]


def test_real_validation_has_six_unseen_source_tuples_and_all_are_quarantined():
    clean, _ = load_clean_splits(
        os.environ.get('DATA_DIR', 'data/poi_34k_seed42'),
        splits=('train', 'validation'), verify_hashes=False)
    for feature in FEATURES:
        clean[feature] = pd.to_numeric(clean[feature], errors='raise')
    train = clean.loc[clean.split.eq('train')].reset_index(drop=True)
    validation = clean.loc[clean.split.eq('validation')].reset_index(drop=True)
    tuples, _ = valid_tuples(train, {
        **_encoder(),
        'category_values': {
            name: sorted(train[name].astype(float).unique()) for name in CATEGORICAL
        },
    })
    valid_set = {tuple(row) for row in tuples.tolist()}
    unseen = ~validation[CATEGORICAL].apply(
        lambda row: tuple(row.astype(float)) in valid_set, axis=1)
    assert unseen.sum() == 6

    encoder = {
        'regions': sorted(train.region.unique()),
        'category_values': {
            name: sorted(train[name].astype(float).unique()) for name in CATEGORICAL
        },
    }
    result = records(
        validation, validation[NUMERIC].to_numpy(float),
        validation[CATEGORICAL].to_numpy(np.float32),
        np.zeros(len(validation), dtype=int), np.zeros(len(validation), dtype=int),
        np.zeros(len(validation), dtype=int), 'category_exact', 'category_exact_l1',
        None, 1, np.full(len(validation), len(tuples)), tuples, 42, 42,
        {'min': train[NUMERIC].min().to_numpy(float),
         'max': train[NUMERIC].max().to_numpy(float)},
        encoder, _AllMatchBoundary(), _clean_boundary(validation))
    assert (~result.valid_clean_category_combo).sum() == 6
    assert not result.loc[unseen.to_numpy(), 'constraints_valid'].any()


@pytest.mark.parametrize('numeric_budget,categorical_budget', [
    (float('nan'), 0), (-1, 0), (None, -1), (None, float('inf')),
])
def test_nonfinite_or_negative_budgets_fail_closed(numeric_budget, categorical_budget):
    frame = pd.DataFrame({
        'POI_ID': ['p'], 'region': ['A'], 'split': ['validation'],
        'X_COORD': [126.0], 'Y_COORD': [37.0],
        'ASORT_LCLASDC': [1.0], 'ASORT_MLSFCDC': [2.0],
        'ASORT_SDASDC': [3.0],
    })
    with pytest.raises(ValueError, match='finite|nonnegative'):
        records(
            frame, frame[NUMERIC].to_numpy(float), frame[CATEGORICAL].to_numpy(np.float32),
            np.array([0]), np.array([0]), np.array([0]), 'x', 'x',
            numeric_budget, categorical_budget, np.array([0]),
            frame[CATEGORICAL].to_numpy(np.float32), 42, 42,
            {'min': np.array([125.0, 36.0]), 'max': np.array([127.0, 38.0])},
            _encoder(), _AllMatchBoundary(), _clean_boundary(frame))


def test_categorical_only_none_budget_requires_coordinate_immutability():
    frame = pd.DataFrame({
        'POI_ID': ['p'], 'region': ['A'], 'split': ['validation'],
        'X_COORD': [126.0], 'Y_COORD': [37.0],
        'ASORT_LCLASDC': [1.0], 'ASORT_MLSFCDC': [2.0],
        'ASORT_SDASDC': [3.0],
    })
    attacked = frame[NUMERIC].to_numpy(float).copy()
    attacked[0, 0] += 0.001
    result = records(
        frame, attacked, frame[CATEGORICAL].to_numpy(np.float32),
        np.array([0]), np.array([0]), np.array([0]), 'category_exact',
        'category_exact_l1', None, 1, np.array([2]),
        frame[CATEGORICAL].to_numpy(np.float32), 42, 42,
        {'min': np.array([125.0, 36.0]), 'max': np.array([127.0, 38.0])},
        _encoder(), _AllMatchBoundary(), _clean_boundary(frame))

    assert result.numeric_budget_m.iloc[0] == 0
    assert result.numeric_budget_was_implicit_zero.iloc[0]
    assert result.distance_m.iloc[0] > 80
    assert not result.within_budget.iloc[0]
    assert not result.constraints_valid.iloc[0]


def test_numeric_query_counts_and_cw_selected_state_match_scored_state_cpu():
    import torch

    cfg = _attack_cfg()
    encoder = _encoder()
    base = torch.tensor([[126.0, 37.0]], dtype=torch.float32)
    categories = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)

    pgd_model = _CountingModel()
    _, pgd_queries = numeric_attack_batch(
        'pgd', base, categories, labels, pgd_model, encoder, 10, cfg)
    assert pgd_queries.tolist() == [3]
    assert pgd_model.calls == 3

    cw_cfg = _attack_cfg()
    cw_cfg['attacks']['cw_l2']['steps'] = 1
    cw_model = _CountingModel()
    cw_coords, cw_queries = numeric_attack_batch(
        'cw_l2', base, categories, labels, cw_model, encoder, 10, cw_cfg)
    assert cw_queries.tolist() == [1]
    assert cw_model.calls == 1
    # With one step, only delta=0 was scored. The returned state must therefore
    # remain delta=0 rather than the post-optimizer state from the historical bug.
    assert np.allclose(cw_coords.detach().cpu().numpy(), base.numpy())


def test_dense_exact_query_count_records_all_evaluated_candidates_cpu():
    import torch

    coords = np.array([[126.0, 37.0]], np.float32)
    categories = np.array([[1.0, 2.0, 3.0]], np.float32)
    tuples = np.array([
        [1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [2.0, 3.0, 4.0],
    ], np.float32)
    tuple_vectors = np.concatenate([
        np.eye(2)[[0, 1, 1]], np.eye(2)[[0, 0, 1]], np.eye(2)[[0, 0, 1]],
    ], axis=1).astype(np.float32)
    model = _CountingModel()
    _, queries = categorical_exact(
        coords, categories, np.array([0]), model, _encoder(), tuples,
        tuple_vectors, 1, _attack_cfg(), torch.device('cpu'))
    assert queries.tolist() == [3]
    assert model.evaluated_rows == 3


def _run_small_attacks(device):
    import torch

    cfg = _attack_cfg()
    encoder = _encoder()
    coords = np.array([[126.0, 37.0], [126.001, 37.001]], np.float32)
    categories = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    tuples = categories.copy()
    tuple_vectors = np.concatenate([
        np.eye(2), np.eye(2), np.eye(2)
    ], axis=1).astype(np.float32)
    for method in ('fgsm', 'pgd', 'cw_l2', 'capgd'):
        base = torch.from_numpy(coords).to(device)
        cats = torch.from_numpy(categories).to(device)
        truth = torch.from_numpy(labels).to(device)
        attacked, queries = numeric_attack_batch(
            method, base, cats, truth, _CountingModel().to(device), encoder, 10, cfg)
        assert attacked.shape == base.shape
        assert np.isfinite(attacked.detach().cpu().numpy()).all()
        assert (queries > 0).all()
    exact, exact_queries = categorical_exact(
        coords, categories, labels, _CountingModel().to(device), encoder,
        tuples, tuple_vectors, 1, cfg, device)
    pcaa, pcaa_queries = categorical_pcaa(
        coords, categories, labels, _CountingModel().to(device), encoder,
        tuples, tuple_vectors, 1, cfg, device)
    evo_coords, evo_categories, evo_queries = mixed_evolution(
        coords, categories, labels, _CountingModel().to(device), encoder,
        tuples, tuple_vectors, 10, 1, cfg, device, 42)
    assert exact.shape == pcaa.shape == evo_categories.shape == categories.shape
    assert evo_coords.shape == coords.shape
    assert (exact_queries > 0).all() and (pcaa_queries > 0).all()
    assert (evo_queries == 8).all()


def test_small_actual_attacks_cpu():
    import torch
    _run_small_attacks(torch.device('cpu'))


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='CUDA unavailable')
def test_small_actual_attacks_gpu_when_available():
    import torch
    _run_small_attacks(torch.device('cuda'))


def test_shard_resume_is_bound_to_hash_identity_ids_and_constraints(tmp_path):
    frame = pd.DataFrame({
        'POI_ID': ['p1', 'p2'], 'region': ['A', 'A'], 'split': ['validation'] * 2,
        'X_COORD': [126.0, 126.00001], 'Y_COORD': [37.0, 37.0],
        'ASORT_LCLASDC': [1.0, 1.0], 'ASORT_MLSFCDC': [2.0, 2.0],
        'ASORT_SDASDC': [3.0, 3.0],
    })
    tuples = frame[CATEGORICAL].drop_duplicates().to_numpy(np.float32)
    identity = {
        'generation_identity_sha256': 'identity-hash',
        'surrogate_checkpoint_sha256': 'checkpoint-hash',
    }
    cfg = _attack_cfg()
    spec = {'method': 'fgsm', 'condition': 'fgsm_m20',
            'numeric_budget_m': 20.0, 'categorical_budget_l0': 0}
    boundary = _AllMatchBoundary()
    clean_boundary = _clean_boundary(frame)
    bounds = {'min': np.array([125.0, 36.0]), 'max': np.array([127.0, 38.0])}
    part = records(
        frame, frame[NUMERIC].to_numpy(float), frame[CATEGORICAL].to_numpy(np.float32),
        np.array([0, 0]), np.array([0, 0]), np.array([0, 0]),
        'fgsm', 'fgsm_m20', 20, 0, np.ones(2, dtype=int), tuples,
        42, 42, bounds, _encoder(), boundary, clean_boundary,
        cfg['distance_validation'], identity['generation_identity_sha256'],
        identity['surrogate_checkpoint_sha256'])
    shards = tmp_path / 'shards'
    shards.mkdir()
    manifest_path = tmp_path / 'manifest.json'
    manifest = _load_shard_manifest(
        manifest_path, identity['generation_identity_sha256'], shards)
    path = shards / 'validation-g42-fgsm_m20.parquet'
    persist_shard(
        path, part, manifest, manifest_path, frame, spec, 42, cfg, tuples,
        bounds, _encoder(), boundary, clean_boundary, identity)
    load_resumed_shard(
        path, manifest['shards'][path.name], frame, spec, 42, cfg, tuples,
        bounds, _encoder(), boundary, clean_boundary, identity)

    wrong_ids = part.copy()
    wrong_ids['POI_ID'] = ['x1', 'x2']
    with pytest.raises(ValueError, match='ordered POI IDs'):
        validate_shard_frame(
            wrong_ids, frame, spec, 42, cfg, tuples, bounds, _encoder(),
            boundary, clean_boundary, identity)

    with path.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_resumed_shard(
            path, manifest['shards'][path.name], frame, spec, 42, cfg, tuples,
            bounds, _encoder(), boundary, clean_boundary, identity)


def test_immutable_request_rejects_config_change_and_pending_status_is_never_complete(tmp_path):
    path = tmp_path / 'generation_request.json'
    _ensure_immutable_json(path, {'config_sha256': 'a'}, 'generation request')
    with pytest.raises(ValueError, match='differs'):
        _ensure_immutable_json(path, {'config_sha256': 'b'}, 'generation request')
    assert PENDING_STATUS == 'generated_pending_fresh_independent_review'
    assert 'complete' not in PENDING_STATUS
    assert QUERY_COUNT_SCOPE.startswith('attack_search_candidate')


def test_unregistered_cached_shard_is_rejected(tmp_path):
    shards = tmp_path / 'shards'
    shards.mkdir()
    pd.DataFrame({'POI_ID': ['wrong']}).to_parquet(
        shards / 'validation-g42-fgsm_m10.parquet', index=False)
    with pytest.raises(ValueError, match='no immutable shard manifest'):
        _load_shard_manifest(tmp_path / 'manifest.json', 'identity', shards)


def test_persisted_surrogate_loads_exactly_and_is_never_retrained_over(tmp_path):
    import torch

    rows = []
    for region_index, region in enumerate(('A', 'B')):
        for index in range(5):
            rows.append({
                'POI_ID': f'{region}-{index}', 'region': region, 'split': 'train',
                'X_COORD': 126.0 + region_index * 0.1 + index * 0.001,
                'Y_COORD': 37.0 + region_index * 0.1 + index * 0.001,
                'ASORT_LCLASDC': float(region_index + 1),
                'ASORT_MLSFCDC': float(region_index + 2),
                'ASORT_SDASDC': float(region_index + 3),
            })
    train = pd.DataFrame(rows)
    cfg = {
        'surrogate_seed': 42,
        'surrogate': {
            'holdout_fraction': 0.2, 'batch_size': 8, 'epochs': 1,
            'patience': 1, 'learning_rate': 0.001, 'weight_decay': 0.0,
        },
    }
    model, encoder, _, checkpoint = train_surrogate(
        train, cfg, tmp_path / 'surrogate', torch.device('cpu'), 'request-a')
    digest = checkpoint.read_bytes()
    loaded, loaded_encoder, _, loaded_checkpoint = load_surrogate(
        train, cfg, tmp_path / 'surrogate', torch.device('cpu'), 'request-a')
    assert loaded_checkpoint == checkpoint
    assert checkpoint.read_bytes() == digest
    assert loaded_encoder['regions'] == encoder['regions']
    matrix = torch.from_numpy(
        np.asarray([[0.0] * model.network[0].in_features], dtype=np.float32))
    assert torch.allclose(model(matrix), loaded(matrix))
    with pytest.raises(FileExistsError, match='overwrite'):
        train_surrogate(
            train, cfg, tmp_path / 'surrogate', torch.device('cpu'), 'request-a')
    with pytest.raises(ValueError, match='different generation request'):
        load_surrogate(
            train, cfg, tmp_path / 'surrogate', torch.device('cpu'), 'request-b')


def test_full_key_coverage_rejects_same_size_wrong_pois():
    clean = pd.DataFrame({
        'POI_ID': ['t1', 't2', 'v1', 'v2'],
        'split': ['train', 'train', 'validation', 'validation'],
    })
    specs = [{'condition': 'fgsm_m10'}]
    attacks = pd.DataFrame({
        'POI_ID': ['t1', 't2', 'v1', 'v2'],
        'split': ['train', 'train', 'validation', 'validation'],
        'generation_seed': [42] * 4,
        'attack_condition': ['fgsm_m10'] * 4,
    })
    validate_full_key_coverage(attacks, clean, specs, [42])
    attacks.loc[3, 'POI_ID'] = 'wrong-but-same-size'
    with pytest.raises(ValueError, match='ordered POI coverage'):
        validate_full_key_coverage(attacks, clean, specs, [42])
