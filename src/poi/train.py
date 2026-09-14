"""Single experiment entry point shared by local and orchestrated runs."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import joblib
import yaml

from .data import FEATURES, FILES, features, load_dataset, select_split, sha256
from .evaluate import evaluate_ovr
from .model import build_model


def run(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    seed = config['seed']
    data_dir = Path(os.environ.get('DATA_DIR', config['data']['directory'])).resolve()
    frames, manifest = load_dataset(data_dir, verify_hashes=config['data']['verify_hashes'])
    exp_id = os.environ.get('EXP_ID', config['name'])
    if not exp_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in exp_id) or exp_id in {'.', '..'}:
        raise ValueError('Invalid EXP_ID')
    directories = {name: Path(os.environ.get(env, f'artifacts/{exp_id}/{name}')).resolve()
                   for name, env in [('checkpoints', 'CHECKPOINT_DIR'), ('results', 'RESULT_DIR'), ('logs', 'LOG_DIR')]}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    # Orchestrator creates directories first; atomically reserve this run within them.
    for key in ['results', 'checkpoints']:
        with (directories[key] / '.poi-run').open('x') as stream:
            stream.write(exp_id + '\n')
    started = time.time()
    train = select_split(frames['clean'], 'train', config['sampling']['train_per_region'], seed)
    regions = sorted(manifest.region.unique())
    models = {}
    train_matrix = features(train)
    for target_region in regions:
        y_train = train.region.eq(target_region).astype(int)
        if set(y_train) != {0, 1}:
            raise ValueError(f'Training requires both classes for {target_region}')
        model = build_model(config['model'], seed)
        model.fit(train_matrix, y_train)
        models[target_region] = model
    joblib.dump(models, directories['checkpoints'] / 'model.joblib')
    # Verify model serialization preserves predictions before claiming success.
    restored = joblib.load(directories['checkpoints'] / 'model.joblib')
    probe = features(train.head(20))
    for target_region in regions:
        if models[target_region].predict_proba(probe).tolist() != restored[target_region].predict_proba(probe).tolist():
            raise RuntimeError(f'Checkpoint round-trip changed predictions for {target_region}')
    metrics = {}
    used_ids = [train[['POI_ID', 'region', 'split']]]
    splits = config['evaluation']['splits']
    if not splits or len(set(splits)) != len(splits) or not set(splits).issubset({'validation', 'test'}):
        raise ValueError('Evaluation splits must be unique validation/test entries')
    for split in splits:
        clean = select_split(frames['clean'], split, config['sampling']['eval_per_region'], seed)
        attack = frames['attack'].set_index('POI_ID').loc[clean.POI_ID].reset_index()
        used_ids.append(clean[['POI_ID', 'region', 'split']])
        for condition, subset in [('clean', clean), ('attack', attack)]:
            key = f'{split}_{condition}'
            metrics[key], predictions = evaluate_ovr(models, subset, config['evaluation']['threshold'])
            predictions['condition'] = condition
            predictions['split'] = split
            predictions['model'] = config['model']['name']
            predictions['seed'] = seed
            predictions.to_csv(directories['results'] / f'{key}_predictions.csv', index=False)
    import pandas as pd
    pd.concat(used_ids).to_csv(directories['results'] / 'used_manifest.csv', index=False)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    metadata = {'experiment_id': exp_id, 'config': config, 'data_directory': str(data_dir),
                'task': 'binary_one_vs_rest', 'regions': regions,
                'feature_columns': FEATURES, 'train_rows': len(train),
                'dataset_sha256': {name: sha256(data_dir / name) for name in FILES},
                'git_commit': commit, 'git_dirty': dirty, 'python': platform.python_version(),
                'versions': {name: importlib.metadata.version(name) for name in ['poi', 'numpy', 'pandas', 'scikit-learn', 'joblib', 'PyYAML']},
                'seconds': time.time() - started}
    (directories['results'] / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (directories['results'] / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (directories['results'] / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (directories['logs'] / 'summary.json').write_text(json.dumps({'experiment_id': exp_id, 'seconds': metadata['seconds'], 'status': 'complete'}, indent=2) + '\n')
    print(json.dumps({key: value['summary'] for key, value in metrics.items()}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baseline.yaml')
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
