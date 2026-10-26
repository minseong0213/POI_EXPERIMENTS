"""Correct reporting of existing CCA evidence without pretending to refit statistics."""
import argparse
import json
from pathlib import Path
import shutil
import pandas as pd
from .data import sha256
from .cca_association_supplement import _forest_figures


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    cca_path = source/'tables/cca_correlations_bootstrap.csv'
    contrast_path = source/'tables/region_category_association_contrasts.csv'
    files = [cca_path, contrast_path, source/'metadata.json', Path(__file__),
             Path(__file__).with_name('cca_association_supplement.py')]
    before = {str(path): sha256(path) for path in files}
    cca = pd.read_csv(cca_path)
    contrast = pd.read_csv(contrast_path)
    if len(cca) != 5184 or len(contrast) != 288:
        raise ValueError('Unexpected original supplement grids')
    contrast = contrast.rename(columns={'delta_p_value':'bootstrap_sign_tail_fraction'})
    contrast = contrast.drop(columns=['delta_q_value'])
    contrast['sign_tail_interpretation'] = (
        'descriptive_uncentered_bootstrap_fraction_not_null_calibrated_p_value')
    for name in ['tables','figures','provenance']:
        (output/name).mkdir(parents=True)
    # Original CCA permutation p/q and intervals are unchanged byte-for-byte.
    shutil.copyfile(cca_path, output/'tables'/cca_path.name)
    contrast.to_csv(output/'tables'/contrast_path.name, index=False)
    _forest_figures(cca, contrast, output/'figures', [42,202,340])
    for path in files:
        shutil.copyfile(path, output/'provenance'/path.name)
    if any(sha256(Path(path)) != digest for path,digest in before.items()):
        raise RuntimeError('Input/source changed during reporting correction')
    metadata = dict(status='GENERATED_PENDING_INDEPENDENT_REVIEW',
        scientific_completion=False, operation='reporting_only_no_statistical_refit',
        inherited_evidence=str(source), input_source_sha256=before,
        removed_columns=['delta_p_value','delta_q_value'],
        cca_rows=len(cca), contrast_rows=len(contrast), figures=6,
        scope='Supersedes original delta p/q interpretation and six forest renderings only',
        limitations=['No null-calibrated hypothesis test for the paired Cramer V difference.',
                    'CCA CI is conditional on fixed learned directions; loading CI is not provided.',
                    'All original source/data authority and historical isolation limitations remain.'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'report.md').write_text(
        '# CCA and association reporting correction\n\n'
        'This addendum reuses the original estimates and paired percentile intervals; no model or '
        'bootstrap was refitted. CCA retains its permutation p-values and BH q-values, now printed '
        'beside effects in the forest figures. Interval endpoints are plotted directly, without clipping.\n\n'
        'The original Cramer V delta_p_value was an uncentered bootstrap sign-tail fraction, '
        'not a demonstrated null-calibrated p-value. It is renamed as a descriptive fraction and '
        'delta_q_value is removed. Do not use the original delta p/q columns for hypothesis testing. '
        'The paired Cramer V difference and its interval remain descriptive. The original evidence '
        'is retained for traceability; full Stage 05 completion is not claimed. Independent review pending.\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.source,args.output)
