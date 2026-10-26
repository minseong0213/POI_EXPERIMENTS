"""Validate development coordinates against independent real ADM1 polygons."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .data import NUMERIC, load_clean_splits, sha256
from .geospatial import AdministrativeBoundaryValidator


def run(data_dir, boundary_config, output, prior_exploratory_test_feature_access=False):
    started = time.time()
    data_dir, output = Path(data_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clean, _ = load_clean_splits(
        data_dir, splits=('train', 'validation'), verify_hashes=True)
    coords = clean[NUMERIC].apply(pd.to_numeric, errors='raise').to_numpy(np.float64)
    validator = AdministrativeBoundaryValidator.from_config(boundary_config)
    result = validator.validate(coords, clean.region.to_numpy())
    train = clean.split.eq('train').to_numpy()
    train_coords = coords[train]
    lower, upper = train_coords.min(axis=0), train_coords.max(axis=0)
    numeric_valid = ((coords >= lower) & (coords <= upper)).all(axis=1)

    rows = clean[['POI_ID', 'region', 'split']].copy()
    rows['X_COORD'] = coords[:, 0]
    rows['Y_COORD'] = coords[:, 1]
    rows['boundary_status'] = result['status']
    rows['boundary_consensus_region'] = result['consensus_region']
    rows['boundary_label_valid'] = result['matches']
    rows['numeric_train_range_valid'] = numeric_valid
    for source_id, labels in result['source_labels'].items():
        rows[f'polygon_region_{source_id}'] = labels
    rows.to_parquet(output / 'development_boundary_rows.parquet', index=False)

    summary = (rows.groupby(['split', 'boundary_status'], as_index=False)
               .agg(rows=('POI_ID', 'size'),
                    boundary_label_valid=('boundary_label_valid', 'sum'),
                    numeric_train_range_valid=('numeric_train_range_valid', 'sum')))
    summary.to_csv(output / 'development_boundary_summary.csv', index=False)
    metadata = {
        'status': 'diagnostic',
        'scope': ['train', 'validation'],
        'feature_splits_loaded_by_this_run': sorted(clean.split.unique().tolist()),
        'test_features_loaded_by_this_run': False,
        'prior_exploratory_test_feature_access': bool(prior_exploratory_test_feature_access),
        'prior_access_note': (
            'Before the isolated loader was implemented, exploratory CRS and boundary checks '
            'loaded the full 34,000-row clean coordinate file, including test features.'
            if prior_exploratory_test_feature_access else None),
        'rows': len(rows),
        'boundary_valid_rows': int(rows.boundary_label_valid.sum()),
        'boundary_quarantined_rows': int((~rows.boundary_label_valid).sum()),
        'numeric_train_range_valid_rows': int(rows.numeric_train_range_valid.sum()),
        'numeric_train_range_quarantined_rows': int((~rows.numeric_train_range_valid).sum()),
        'numeric_train_bounds': {
            feature: {'min': float(lower[index]), 'max': float(upper[index])}
            for index, feature in enumerate(NUMERIC)},
        'boundary_validation': validator.provenance(),
        'dataset_sha256': {
            name: sha256(data_dir / name)
            for name in ['poi_data_region.csv', 'sample_manifest.csv']},
        'python': platform.python_version(),
        'seconds': time.time() - started,
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/poi_34k_seed42')
    parser.add_argument(
        '--boundary-config', default='configs/geospatial/korea_adm1_consensus.yaml')
    parser.add_argument(
        '--output',
        default='artifacts/poi-remediation-20261001/reports/01_attack_generation/work/'
                'boundary_validation_development')
    parser.add_argument('--prior-exploratory-test-feature-access', action='store_true')
    arguments = parser.parse_args()
    run(arguments.data_dir, arguments.boundary_config, arguments.output,
        arguments.prior_exploratory_test_feature_access)


if __name__ == '__main__':
    main()
