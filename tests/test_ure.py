import numpy as np
import pytest

torch = pytest.importorskip("torch")

from poi.ure.detector import (
    UREAttackDetector,
    UREEmbeddingMerger,
    load_ure_attack_detector,
    save_ure_attack_detector,
)
from poi.ure.model import UREEmbedding, URERegionModel
from poi.ure.preprocessing import (
    UREPreprocessor,
    build_poi_additive_transition_costs,
    load_transition_costs,
)
from poi.ure.training import (
    URETrainingConfig,
    expected_transition_cost,
    relaxed_cost_attack,
    train_universal_robust_embeddings,
)


RAW = np.asarray(
    [
        [126.0, 37.0, 1, 10, 100],
        [126.5, 37.5, 2, 20, 200],
        [127.0, 36.0, 1, 20, 100],
        [128.0, 35.0, 2, 10, 200],
    ],
    dtype=np.float32,
)


def test_ure_preprocessor_bins_numeric_and_freezes_categories(tmp_path):
    preprocessor = UREPreprocessor.fit(RAW, numeric_bins=3)
    indices = preprocessor.transform(RAW)
    assert indices.shape == RAW.shape
    assert preprocessor.cardinalities[:2] == (3, 3)
    unknown = RAW.copy()
    unknown[0, 2] = 999
    with pytest.raises(ValueError, match="outside clean train"):
        preprocessor.transform(unknown)
    path = tmp_path / "costs.npz"
    np.savez(
        path,
        **{
            f"feature_{feature}": np.ones((size, size), dtype=np.float32) - np.eye(size)
            for feature, size in enumerate(preprocessor.cardinalities)
        },
    )
    matrices = load_transition_costs(str(path), preprocessor.cardinalities)
    assert len(matrices) == 5


def test_ure_columns_are_unit_norm_and_relaxed_attack_respects_cost_budget():
    cardinalities = (3, 3, 2, 2, 2)
    model = URERegionModel(cardinalities, region_count=3, embedding_dim=4, hidden_dim=8)
    for layer in model.embedding.layers:
        assert torch.allclose(layer.weight.norm(dim=0), torch.ones(layer.weight.shape[1]))
    indices = torch.tensor([[0, 1, 0, 1, 0], [2, 0, 1, 0, 1]])
    labels = torch.tensor([0, 1])
    costs = tuple(torch.ones(size, size) - torch.eye(size) for size in cardinalities)
    relaxed = relaxed_cost_attack(
        model, indices, labels, costs, budget=1.25, steps=3, step_size=0.5
    )
    for probability in relaxed:
        assert torch.allclose(probability.sum(dim=1), torch.ones(2), atol=1e-5)
        assert probability.min() >= 0
    assert torch.all(expected_transition_cost(relaxed, indices, costs) <= 1.25001)


def test_ure_embedding_is_exact_bias_free_one_hot_linear_projection():
    embedding = UREEmbedding((3,), embedding_dim=2)
    indices = torch.tensor([[0], [2]])
    expected = embedding.layers[0].weight[:, [0, 2]].t()
    assert torch.allclose(embedding(indices), expected)


def test_ure_detector_round_trip_and_explicit_external_label(tmp_path):
    preprocessor = UREPreprocessor.fit(RAW, numeric_bins=3)
    embedding = UREEmbedding(preprocessor.cardinalities, embedding_dim=2)
    detector = UREAttackDetector(
        preprocessor, embedding, n_estimators=10, min_samples_leaf=1
    )
    matrix = np.concatenate((RAW, RAW + np.asarray([0.01, -0.01, 0, 0, 0])))
    labels = np.asarray([0] * len(RAW) + [1] * len(RAW))
    detector.fit(matrix, labels)
    checkpoint, _ = save_ure_attack_detector(
        tmp_path / "ure_rf.pt", detector, metadata={"split": "train"}
    )
    restored = load_ure_attack_detector(checkpoint)
    assert restored.result_label == "URE-RF attack detector"
    assert np.allclose(restored.predict_proba(matrix), detector.predict_proba(matrix))


def test_algorithm3_merging_is_frozen_complete_linkage_cluster_mean():
    embedding = UREEmbedding((4,), embedding_dim=2)
    with torch.no_grad():
        embedding.layers[0].weight.copy_(
            torch.tensor([[0.0, 0.01, 1.0, 2.0], [1.0, 1.0, 0.0, 0.0]])
        )
    merger = UREEmbeddingMerger.fit(embedding, percentile=0.2)
    transformed = merger.transform_indices(np.asarray([[0], [1], [2], [3]]))
    assert merger.cluster_labels[0][0] == merger.cluster_labels[0][1]
    assert np.allclose(transformed[0], transformed[1])
    assert np.allclose(transformed[0], np.asarray([0.005, 1.0]))
    assert merger.to_dict()["algorithm"] == "paper_algorithm_3_euclidean_complete_linkage_mean"


def test_poi_additive_costs_are_explicit_and_have_zero_diagonal():
    preprocessor = UREPreprocessor.fit(RAW, numeric_bins=3)
    costs = build_poi_additive_transition_costs(
        preprocessor, RAW, numeric_cost_unit_m=100.0, categorical_change_cost=1.0
    )
    assert len(costs) == 5
    for matrix in costs:
        assert np.allclose(np.diag(matrix), 0.0)
        assert np.all(matrix >= 0)


def test_bilevel_ure_training_runs_separate_embedding_and_classifier_updates():
    cardinalities = (3, 3, 2, 2, 2)
    model = URERegionModel(cardinalities, region_count=3, embedding_dim=2, hidden_dim=8)
    indices = np.asarray(
        [[a % 3, (a + 1) % 3, a % 2, (a + 1) % 2, a % 2] for a in range(12)]
    )
    labels = np.asarray([a % 3 for a in range(12)])
    costs = tuple(np.ones((size, size), dtype=np.float32) - np.eye(size) for size in cardinalities)
    before_embedding = model.embedding.layers[0].weight.detach().clone()
    before_classifier = model.classifier[-1].weight.detach().clone()
    history = train_universal_robust_embeddings(
        model,
        indices,
        labels,
        costs,
        config=URETrainingConfig(
            epochs=1,
            batch_size=6,
            attack_steps=1,
            embedding_adversarial_steps_per_clean_step=1,
        ),
        device="cpu",
    )
    assert len(history) == 1
    assert not torch.allclose(before_embedding, model.embedding.layers[0].weight)
    assert not torch.allclose(before_classifier, model.classifier[-1].weight)
    for layer in model.embedding.layers:
        assert torch.allclose(layer.weight.norm(dim=0), torch.ones(layer.weight.shape[1]), atol=1e-5)
