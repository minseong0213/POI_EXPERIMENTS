"""Audit the study bundle against every required experiment and final evaluation gate."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_MODELS = {
    'decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'catboost',
    'tabpfn_v2_5', 'tabpfn_v2_6', 'tabpfn_v3',
}
REQUIRED = {
    '01_descriptive': {
        'figures': {'class_balance_train', 'paired_change_rate'},
        'tables': {'table_one_categorical_by_region.csv', 'table_one_numeric_by_region.csv'},
    },
    '02_distribution': {
        'figures': {'boxplot_region_condition', 'histograms_and_deltas',
                    'paired_coordinate_scatter'},
        'tables': {'distribution_summary.csv', 'paired_numeric_deltas.csv'},
    },
    '03_embedding': {
        'figures': {'tsne_region_condition', 'umap_region_condition',
                    'dendrogram_region_condition'},
        'tables': {'embedding_coordinates.csv', 'dendrogram_centroids.csv'},
    },
    '04_statistics': {
        'figures': {'cca_correlations', 'correlation_heatmap', 'paired_t_effects'},
        'tables': {'cca_correlations.csv', 'correlations.csv', 'statistical_tests.csv'},
    },
    '05_region_models': {
        'figures': {'confusion_matrices_normalized', 'pr_summary_seed_42',
                    'roc_summary_seed_42'},
        'tables': {'metrics_by_region.csv', 'all_predictions.parquet',
                   'model_ranking_validation.csv', 'tabpfn_validation_performance.csv'},
    },
    '06_model_selection_ablation': {
        'figures': {'ablation_delta', 'ablation_heatmap'},
        'tables': {'ablation_metrics.csv', 'ablation_summary.csv'},
    },
    '07_explainability': {
        'figures': {'lime_representative', 'odds_ratio_forest_representative',
                    'shap_global_summary', 'shap_dependence_x_coord',
                    'shap_dependence_y_coord', 'shap_waterfall_representative'},
        'tables': {'lime_representative.csv', 'odds_ratios.csv',
                   'shap_global_summary.csv'},
    },
    '08_attack_detection': {
        'figures': {'attack_detection_confusion', 'attack_detection_metrics',
                    'attack_detection_pr_roc'},
        'tables': {'attack_detection_metrics.csv', 'attack_detection_odds_ratios.csv',
                   'attack_detection_predictions.parquet'},
    },
    '09_ensemble_defense': {
        'figures': {'defense_coverage', 'defense_f1_summary', 'risk_coverage'},
        'tables': {'defense_metrics_by_region.csv', 'defense_predictions.parquet',
                   'risk_coverage.csv'},
    },
    '10_robustness': {
        'figures': {'final_ablation', 'model_rank_heatmap',
                    'sensitivity_attack_ratio', 'sensitivity_thresholds',
                    'stability_robust_f1', 'defense_ablation'},
        'tables': {'final_ablation_summary.csv', 'model_rank_summary.csv',
                   'rank_stability_kendall.csv', 'sensitivity_summary.csv',
                   'threshold_sensitivity.csv', 'defense_ablation_summary.csv'},
    },
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(bundle):
    bundle = Path(bundle)
    reports = bundle / 'reports'
    checks, failures = {}, []
    for stage, requirements in REQUIRED.items():
        stage_dir = reports / stage
        missing_core = [name for name in ['report.md', 'metrics.json', 'metadata.json']
                        if not (stage_dir / name).is_file()]
        missing_tables = sorted(name for name in requirements['tables']
                                if not (stage_dir / 'tables' / name).is_file())
        missing_figures = []
        for stem in requirements['figures']:
            for suffix in ['.png', '.svg']:
                if not (stage_dir / 'figures' / f'{stem}{suffix}').is_file():
                    missing_figures.append(f'{stem}{suffix}')
        ok = not (missing_core or missing_tables or missing_figures)
        checks[stage] = {'ok': ok, 'missing_core': missing_core,
                         'missing_tables': missing_tables, 'missing_figures': missing_figures}
        if not ok:
            failures.append(f'{stage}: required artifacts missing')

    manifest_path = bundle / 'manifest.json'
    manifest_ok = False
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest_ok = True
        for relative, expected in manifest.get('files', {}).items():
            path = bundle / relative
            if (not path.is_file() or path.stat().st_size != expected['bytes'] or
                    digest(path) != expected['sha256']):
                manifest_ok = False
                break
    checks['manifest_integrity'] = {'ok': manifest_ok}
    if not manifest_ok:
        failures.append('manifest: hash or size verification failed')

    stage5 = reports / '05_region_models' / 'tables'
    observed_models, metric_rows, prediction_rows = set(), 0, 0
    if (stage5 / 'metrics_by_region.csv').is_file():
        metrics = pd.read_csv(stage5 / 'metrics_by_region.csv')
        observed_models = set(metrics.model.unique())
        metric_rows = len(metrics)
    if (stage5 / 'all_predictions.parquet').is_file():
        predictions = pd.read_parquet(stage5 / 'all_predictions.parquet', columns=['model'])
        prediction_rows = len(predictions)
    model_matrix_ok = (observed_models == EXPECTED_MODELS and metric_rows == 816 and
                       prediction_rows == 2_774_400)
    checks['eight_model_matrix'] = {
        'ok': model_matrix_ok, 'observed_models': sorted(observed_models),
        'missing_models': sorted(EXPECTED_MODELS - observed_models),
        'metric_rows': metric_rows, 'prediction_rows': prediction_rows,
    }
    if not model_matrix_ok:
        failures.append('model matrix: eight-model validation output is incomplete')

    selected_model = None
    ranking_path = stage5 / 'model_ranking_validation.csv'
    if ranking_path.is_file():
        ranking = pd.read_csv(ranking_path)
        if len(ranking):
            score_column = 'mean' if 'mean' in ranking.columns else 'mean_robust_f1'
            selected_model = ranking.sort_values(score_column, ascending=False).iloc[0].model
    downstream_models = {}
    downstream_ok = selected_model is not None and model_matrix_ok
    for stage in ['06_model_selection_ablation', '07_explainability']:
        metadata_path = reports / stage / 'metadata.json'
        model = None
        if metadata_path.is_file():
            model = json.loads(metadata_path.read_text()).get('model')
        downstream_models[stage] = model
        downstream_ok = downstream_ok and model == selected_model
    checks['selected_model_consistency'] = {
        'ok': downstream_ok, 'selected_model': selected_model,
        'downstream_models': downstream_models,
    }
    if not downstream_ok:
        failures.append('model selection: stages 06 and 07 are not final for the eight-model winner')

    rank_path = reports / '10_robustness' / 'tables' / 'model_rank_summary.csv'
    ranked_models = set()
    if rank_path.is_file():
        ranked_models = set(pd.read_csv(rank_path).model.unique())
    robustness_models_ok = ranked_models == EXPECTED_MODELS
    checks['eight_model_robustness'] = {
        'ok': robustness_models_ok, 'observed_models': sorted(ranked_models),
        'missing_models': sorted(EXPECTED_MODELS - ranked_models),
    }
    if not robustness_models_ok:
        failures.append('robustness: sensitivity/stability/ranking does not include all eight models')

    test_path = reports / '10_robustness' / 'tables' / 'final_test_metrics.csv'
    test_prediction_path = reports / '10_robustness' / 'tables' / 'final_test_predictions.parquet'
    final_test_ok = False
    final_test_rows, final_prediction_rows = 0, 0
    if test_path.is_file():
        test = pd.read_csv(test_path)
        final_test_rows = len(test)
        test_predictions = pd.DataFrame(columns=['model', 'split', 'condition'])
        if test_prediction_path.is_file():
            test_predictions = pd.read_parquet(
                test_prediction_path, columns=['model', 'split', 'condition'])
            final_prediction_rows = len(test_predictions)
        final_test_ok = (
            final_test_rows == 102 and final_prediction_rows == 693_600 and
            set(test.model.unique()) == {selected_model} and
            set(test.split.unique()) == {'test'} and
            set(test.condition.unique()) == {'clean', 'adversarial'} and
            set(test_predictions.model.unique()) == {selected_model} and
            set(test_predictions.split.unique()) == {'test'} and
            set(test_predictions.condition.unique()) == {'clean', 'adversarial'})
    checks['locked_final_test'] = {
        'ok': final_test_ok, 'path': str(test_path), 'metric_rows': final_test_rows,
        'prediction_rows': final_prediction_rows, 'selected_model': selected_model,
    }
    if not final_test_ok:
        failures.append('final test: locked clean/adversarial evaluation is missing')

    tabpfn_validation = reports / '05_region_models' / 'tables' / 'tabpfn_validation_performance.csv'
    tabpfn_test = reports / '11_locked_test' / 'tables' / 'tabpfn_test_performance.csv'
    tabpfn_report = bundle / 'TABPFN_PERFORMANCE.md'
    performance_tables_ok = False
    if tabpfn_validation.is_file() and tabpfn_test.is_file() and tabpfn_report.is_file():
        val = pd.read_csv(tabpfn_validation)
        test_summary = pd.read_csv(tabpfn_test)
        performance_tables_ok = (
            len(val) == 6 and set(val.model) == {'tabpfn_v2_5', 'tabpfn_v2_6', 'tabpfn_v3'} and
            set(val.condition) == {'clean', 'adversarial'} and len(test_summary) == 2 and
            set(test_summary.model) == {'tabpfn_v2_5'} and
            set(test_summary.condition) == {'clean', 'adversarial'})
    checks['tabpfn_performance_tables'] = {'ok': performance_tables_ok}
    if not performance_tables_ok:
        failures.append('TabPFN performance tables are missing or incomplete')

    return {
        'status': 'complete' if not failures else 'incomplete',
        'bundle': str(bundle), 'checks': checks, 'failures': failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', default='artifacts/poi-validation-study-001')
    parser.add_argument('--output')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = audit(args.bundle)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end='')
    if args.require_complete and result['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
