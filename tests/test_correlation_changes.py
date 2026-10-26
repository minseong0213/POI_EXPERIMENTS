import pandas as pd
import pytest
from poi.correlation_changes import paired_changes


def sample():
    return pd.DataFrame([dict(region='Seoul', attack_condition='pgd', generation_seed=42,
                             method='pearson', condition=condition, correlation=value)
                         for condition,value in [('clean', .8), ('adversarial', .3)]])


def test_paired_change_sign_and_key_alignment():
    result = paired_changes(sample().iloc[::-1])
    assert result.correlation_change_attack_minus_clean.iloc[0] == pytest.approx(-.5)


def test_unpaired_or_duplicate_rows_rejected():
    for frame in [sample().iloc[:1], pd.concat([sample(), sample().iloc[:1]])]:
        with pytest.raises(ValueError):
            paired_changes(frame)
