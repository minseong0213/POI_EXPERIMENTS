import pytest
from poi.final_test import run


def test_legacy_test_entrypoint_fails_before_config_or_data_access():
    with pytest.raises(RuntimeError, match='Locked test remains disabled'):
        run('/a/config/that/must/not/be/read.yaml')
