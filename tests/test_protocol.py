from pathlib import Path

import pandas as pd
import pytest
import yaml


def test_experiment_matrix_is_complete_and_unambiguous():
    matrix = yaml.safe_load(Path('configs/experiment_matrix.yaml').read_text())
    assert matrix['task']['type'] == 'binary_one_vs_rest'
    assert len(matrix['regions']) == len(set(matrix['regions'])) == 17
    assert len(matrix['models']) == 8
    assert len({model['id'] for model in matrix['models']}) == 8
    assert matrix['seeds'] == [42, 202, 340]
    assert set(['precision', 'recall', 'f1', 'accuracy', 'roc_auc']).issubset(matrix['metrics'])
    assert set(matrix['conditions']) == {'clean', 'adversarial'}


def test_all_protocol_stages_are_linked_and_present():
    index = Path('docs/experiments.md').read_text()
    for stage in range(11):
        prefix = f'{stage:02d}_'
        matches = list(Path('docs/experiments').glob(prefix + '*.md'))
        assert len(matches) == 1
        assert str(matches[0].relative_to('docs')) in index
    for template in ['eda_report.md', 'model_report.md']:
        assert (Path('docs/templates') / template).is_file()


def test_tabpfn_config_uses_supported_versions_and_no_secret():
    config = yaml.safe_load(Path('configs/tabpfn_benchmark.yaml').read_text())
    assert config['versions'] == ['v2.5', 'v2.6', 'v3']
    assert config['device'] == 'cuda'
    assert config['batch_retries'] == 3
    assert config['resume_partial'] is True
    assert 'token' not in str(config).lower()


def test_selected_model_ablation_covers_every_feature_group():
    config = yaml.safe_load(Path('configs/tabpfn_ablation.yaml').read_text())
    assert config['seeds'] == [42, 202, 340]
    assert config['resume_partial'] is True
    assert set(config['feature_sets']) == {
        'full', 'no_coordinates', 'coordinates_only', 'drop_x_coord', 'drop_y_coord',
        'drop_lclass', 'drop_mclass', 'drop_sclass',
    }
    assert 'token' not in str(config).lower()


def test_selected_model_explanation_is_bounded_and_secret_free():
    config = yaml.safe_load(Path('configs/tabpfn_explain.yaml').read_text())
    assert config['representative_region'] == 'Seoul'
    assert 1 <= config['explain_per_class'] <= 10
    assert config['background_rows'] <= 100
    assert config['resume_partial'] is True
    assert 'token' not in str(config).lower()


def test_final_ensemble_uses_validation_top_three():
    ranking = pd.read_csv('configs/final_model_selection.csv').sort_values(
        'mean', ascending=False)
    config = yaml.safe_load(Path('configs/defense_final.yaml').read_text())
    assert config['ensemble_models'] == ranking.model.head(3).tolist()


def test_final_test_refuses_an_incomplete_model_ranking(tmp_path):
    from poi.final_test import select_model

    ranking = tmp_path / 'ranking.csv'
    pd.DataFrame({'model': ['lightgbm', 'xgboost'], 'mean': [.98, .97]}).to_csv(
        ranking, index=False)
    with pytest.raises(ValueError, match='complete eight-model ranking'):
        select_model(ranking)


def test_final_model_selection_is_complete_and_selects_tabpfn_v2_5():
    from poi.final_test import select_model

    assert select_model('configs/final_model_selection.csv') == 'tabpfn_v2_5'
