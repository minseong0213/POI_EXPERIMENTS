from types import SimpleNamespace
import pytest
from poi.feature_ablation import fit_provenance


def factory():
    return SimpleNamespace(tabpfn_runtime={'torch': 'actual-runtime'}, last_fit_attestation={
        'model_id': 'tabpfn_v3', 'seed': 42, 'feature_order': ['X_COORD'],
        'train_matrix_sha256': 'a'*64, 'truth_sha256': 'b'*64,
        'probe_prediction_sha256': 'c'*64, 'official_checkpoint_sha256': 'd'*64})


def test_actual_fit_attestation_is_recorded_without_claiming_saved_state_reload():
    result = fit_provenance(factory(), 'tabpfn_v3', ['X_COORD'], 42, 'Seoul')
    assert result['fit_attestation']['official_checkpoint_sha256'] == 'd'*64
    assert result['runtime']['torch'] == 'actual-runtime'
    assert result['independent_reloaded_state_check'] is False


@pytest.mark.parametrize('mutation', ['missing', 'seed', 'checkpoint', 'features'])
def test_missing_or_misaligned_fit_evidence_rejected(mutation):
    model = factory()
    if mutation == 'missing':
        del model.last_fit_attestation
    elif mutation == 'seed':
        model.last_fit_attestation['seed'] = 202
    elif mutation == 'checkpoint':
        model.last_fit_attestation['official_checkpoint_sha256'] = None
    else:
        model.last_fit_attestation['feature_order'] = ['Y_COORD']
    with pytest.raises(ValueError):
        fit_provenance(model, 'tabpfn_v3', ['X_COORD'], 42, 'Seoul')
