import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from poi.geospatial import AdministrativeBoundaryValidator


def _write_source(path):
    document = {
        'type': 'FeatureCollection',
        'features': [
            {'type': 'Feature', 'properties': {'name': 'left'},
             'geometry': {'type': 'Polygon', 'coordinates': [[
                 [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}},
            {'type': 'Feature', 'properties': {'name': 'right'},
             'geometry': {'type': 'Polygon', 'coordinates': [[
                 [1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]]}},
        ],
    }
    path.write_text(json.dumps(document))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path, second_mapping=None):
    sources = []
    for index in range(2):
        path = tmp_path / f'boundary-{index}.geojson'
        digest = _write_source(path)
        mapping = ({'left': 'A', 'right': 'B'} if index == 0 or second_mapping is None
                   else second_mapping)
        sources.append({'id': f'source_{index}', 'path': str(path), 'sha256': digest,
                        'region_field': 'name', 'region_map': mapping})
    config = {'policy': 'unanimous_expected_region', 'coordinate_crs': 'OGC:CRS84',
              'original_poi_label_boundary_vintage': 'unknown_not_supplied',
              'provenance_limitations': ['synthetic third-party test fixture'],
              'expected_regions': ['A', 'B'], 'sources': sources}
    config_path = tmp_path / 'boundaries.yaml'
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_consensus_polygon_validation_is_fail_closed(tmp_path):
    validator = AdministrativeBoundaryValidator.from_config(
        _config(tmp_path), project_root=Path('/'))
    result = validator.validate(
        np.array([[0.5, 0.5], [1.5, 0.5], [4.0, 4.0]]),
        np.array(['A', 'B', 'A']))

    assert result['matches'].tolist() == [True, True, False]
    assert result['status'].tolist() == [
        'consensus_match', 'consensus_match', 'unclassified']
    assert result['consensus_region'].tolist() == ['A', 'B', '']


def test_boundary_source_disagreement_is_not_label_preservation(tmp_path):
    validator = AdministrativeBoundaryValidator.from_config(
        _config(tmp_path, {'left': 'B', 'right': 'A'}), project_root=Path('/'))
    result = validator.validate(np.array([[0.5, 0.5]]), np.array(['A']))

    assert not result['matches'][0]
    assert result['status'][0] == 'source_disagreement'
    assert result['consensus_region'][0] == ''


def test_checked_in_real_boundaries_have_two_vintages_and_known_capitals():
    validator = AdministrativeBoundaryValidator.from_config(
        'configs/geospatial/korea_adm1_consensus.yaml')
    result = validator.validate(
        np.array([[126.9780, 37.5665], [126.5312, 33.4996]]),
        np.array(['Seoul', 'Jeju']))

    assert len(validator.sources) == 2
    assert validator.original_poi_label_boundary_vintage == 'unknown_not_supplied'
    assert validator.provenance()['provenance_limitations']
    assert result['matches'].tolist() == [True, True]
    assert set(result['source_labels']) == {'sgis_adm1_2020', 'vworld_adm1_2023'}


def test_boundary_checksum_mismatch_is_rejected(tmp_path):
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config['sources'][0]['sha256'] = '0' * 64
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match='checksum mismatch'):
        AdministrativeBoundaryValidator.from_config(config_path, project_root=Path('/'))
