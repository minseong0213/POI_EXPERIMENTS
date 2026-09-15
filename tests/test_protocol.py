from pathlib import Path
import hashlib
import json
import re

import pandas as pd
import pytest
import yaml


def test_experiment_matrix_is_complete_and_unambiguous():
    matrix = yaml.safe_load(Path('configs/protocol.yaml').read_text())
    assert matrix['task']['type'] == 'binary_one_vs_rest'
    assert len(matrix['regions']) == len(set(matrix['regions'])) == 17
    assert len(matrix['models']) == 8
    assert len({model['id'] for model in matrix['models']}) == 8
    assert matrix['seeds'] == [42, 202, 340]
    assert set(['precision', 'recall', 'f1', 'accuracy', 'roc_auc']).issubset(matrix['metrics'])
    assert set(matrix['conditions']) == {'clean', 'adversarial'}


def test_all_protocol_stages_are_linked_and_present():
    index = Path('docs/experiments.md').read_text()
    for stage in range(16):
        prefix = f'{stage:02d}_'
        matches = list(Path('docs/experiments').glob(prefix + '*.md'))
        assert len(matches) == 1
        assert str(matches[0].relative_to('docs')) in index
    for template in ['eda_report.md', 'model_report.md']:
        assert (Path('docs/templates') / template).is_file()


def test_tabpfn_config_uses_supported_versions_and_no_secret():
    config = yaml.safe_load(Path('configs/stages/06_region_models_tabpfn.yaml').read_text())
    assert config['versions'] == ['v2.5', 'v2.6', 'v3']
    assert config['device'] == 'cuda'
    assert config['batch_retries'] == 3
    assert config['resume_partial'] is True
    assert 'token' not in str(config).lower()


def test_selected_model_ablation_covers_every_feature_group():
    config = yaml.safe_load(Path('configs/stages/07_backbone_ablation_tabpfn.yaml').read_text())
    assert config['seeds'] == [42, 202, 340]
    assert config['resume_partial'] is True
    assert config['bootstrap_repeats'] >= 1000
    assert set(config['feature_sets']) == {
        'full', 'no_coordinates', 'coordinates_only', 'drop_x_coord', 'drop_y_coord',
        'drop_lclass', 'drop_mclass', 'drop_sclass',
    }
    assert 'token' not in str(config).lower()


def test_selected_model_explanation_is_bounded_and_secret_free():
    config = yaml.safe_load(Path('configs/legacy/13_explainability_tabpfn25.yaml').read_text())
    assert config['representative_region'] == 'Seoul'
    assert 1 <= config['explain_per_class'] <= 10
    assert config['background_rows'] <= 100
    assert config['resume_partial'] is True
    assert 'token' not in str(config).lower()


def test_config_layout_has_no_ambiguous_completion_names():
    ambiguous = {'defense_final.yaml', 'robustness_complete.yaml',
                 'robustness_enhanced.yaml', 'robustness_final.yaml'}
    assert not ambiguous.intersection(path.name for path in Path('configs').rglob('*.yaml'))
    for path in Path('configs').rglob('*.yaml'):
        assert yaml.safe_load(path.read_text()) is not None


def test_results_hub_exposes_review_verdict_and_every_stage():
    hub = Path('docs/results.md').read_text()
    assert '**FAIL**' in hub
    for stage in range(16):
        assert f'| {stage:02d} |' in hub
    for verdict in ['PASS', 'PARTIAL', 'FAIL', 'NOT_RUN']:
        assert verdict in hub


def test_local_markdown_links_resolve():
    missing = []
    markdown = [Path('README.md'), *Path('docs').rglob('*.md')]
    for source in markdown:
        for target in re.findall(r'\[[^]]+\]\(([^)]+)\)', source.read_text()):
            if target.startswith(('http://', 'https://', '#')):
                continue
            if 'artifacts/' in target:
                missing.append((source, f'ignored artifact link: {target}'))
                continue
            path = target.split('#', 1)[0]
            if path and not (source.parent / path).resolve().exists():
                missing.append((source, target))
    assert not missing


def test_documentation_result_assets_match_manifest():
    root = Path('docs/assets/poi-adversarial-20260915-001')
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['overall_verdict'] == 'FAIL'
    assert len(manifest['files']) >= 1
    for record in manifest['files']:
        path = root / record['path']
        assert path.stat().st_size == record['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']


def test_code_and_run_descriptors_reference_existing_configs():
    sources = [*Path('src').rglob('*.py'), *Path('scripts').rglob('*.py'),
               *Path('scripts').rglob('*.sh'),
               *Path('experiments').glob('*.env')]
    missing = []
    for source in (p for p in sources if p.is_file()):
        for target in re.findall(r'configs/[A-Za-z0-9_./-]+\.yaml', source.read_text()):
            if not Path(target).is_file():
                missing.append((source, target))
    assert not missing


def test_final_test_refuses_an_incomplete_model_ranking(tmp_path):
    from poi.final_test import select_model

    ranking = tmp_path / 'ranking.csv'
    pd.DataFrame({'model': ['lightgbm', 'xgboost'], 'mean': [.98, .97]}).to_csv(
        ranking, index=False)
    with pytest.raises(ValueError, match='complete eight-model ranking'):
        select_model(ranking)


def test_final_model_selection_uses_the_new_ranking_top_model(tmp_path):
    from poi.final_test import select_model

    ranking = tmp_path / 'ranking.csv'
    models = ['xgboost', 'lightgbm', 'random_forest', 'catboost', 'decision_tree',
              'tabpfn_v2_5', 'tabpfn_v2_6', 'tabpfn_v3']
    pd.DataFrame({'model': models, 'mean': range(len(models))}).to_csv(ranking, index=False)
    assert select_model(ranking) == 'tabpfn_v3'
