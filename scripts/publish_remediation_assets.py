"""Copy small reviewed remediation evidence into the versioned results hub."""
import hashlib
import json
from pathlib import Path
import shutil


def main():
    source=Path('artifacts/poi-uncertainty-20261001-001/reports')
    target=Path('docs/assets/poi-remediation-20261001')
    files=[]
    selected=[source/'06_uncertainty'/name for name in ['report.md','metadata.json','config.yaml']]
    selected+=sorted((source/'06_uncertainty'/'tables').glob('*.csv'))
    selected+=sorted((source/'06_uncertainty'/'figures').glob('*.png'))
    selected+=[source/'99_independent_review'/'review.md',source/'99_independent_review'/'verdict.json']
    for original in selected:
        relative=original.relative_to(source)
        # Preserve the independent review verbatim without repo-relative Markdown links.
        if str(relative)=='99_independent_review/review.md':
            relative=relative.with_suffix('.txt')
        destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,destination)
        files.append(dict(path=str(relative),source=str(original),bytes=original.stat().st_size,
            sha256=hashlib.sha256(original.read_bytes()).hexdigest()))
    review_source=Path('artifacts/poi-remediation-20261001/reports/99_independent_review')
    report_source=review_source.parent
    # Generated evidence stays immutable; separate review files supply its current verdict.
    for stage in ('05_correlation_changes', '06_normalized_confusion',
                  '05_cca_association_reporting_revision2', '09_attack_detection_analysis',
                  '09_attack_detection_visual_supplement', '13_odds_ratio_attack'):
        directory=report_source/stage
        if not (directory/'metadata.json').is_file():
            continue
        selected=[directory/'report.md',directory/'metadata.json']
        selected+=sorted((directory/'tables').glob('*.csv'))
        selected+=sorted((directory/'figures').glob('*.png'))
        for original in selected:
            if not original.is_file():
                continue
            relative=original.relative_to(report_source)
            destination=target/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original,destination)
            files.append(dict(path=str(relative),source=str(original),bytes=original.stat().st_size,
                sha256=hashlib.sha256(original.read_bytes()).hexdigest()))
    for original in sorted(review_source.glob('*')):
        if original.suffix not in ('.md', '.json'):
            continue
        relative=Path('99_independent_review')/original.name
        if relative.suffix == '.md':
            relative=relative.with_suffix('.txt')
        destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,destination)
        files.append(dict(path=str(relative),source=str(original),bytes=original.stat().st_size,
            sha256=hashlib.sha256(original.read_bytes()).hexdigest()))
    (target/'manifest.json').write_text(json.dumps(dict(
        conditional_uncertainty_verdict='PASS',overall_experiment_complete=False,files=files),indent=2)+'\n')


if __name__=='__main__':
    main()
