"""Fail-closed administrative polygon validation for POI coordinates."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath
import yaml


AMBIGUOUS = '__AMBIGUOUS__'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_geojson(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


class BoundarySource:
    """One checksummed ADM1 polygon source in longitude/latitude order."""

    def __init__(self, config, expected_regions, root):
        self.id = config['id']
        self.path = (root / config['path']).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f'Boundary source is missing: {self.path}')
        observed_hash = _sha256(self.path)
        if observed_hash != config['sha256']:
            raise ValueError(
                f'Boundary checksum mismatch for {self.id}: {observed_hash}')
        document = _read_geojson(self.path)
        if document.get('type') != 'FeatureCollection':
            raise ValueError(f'{self.id} is not a GeoJSON FeatureCollection')
        field = config['region_field']
        mapping = config['region_map']
        geometries = {}
        for feature in document.get('features', []):
            raw_name = feature.get('properties', {}).get(field)
            if raw_name not in mapping:
                raise ValueError(f'Unmapped region {raw_name!r} in {self.id}')
            region = mapping[raw_name]
            if region in geometries:
                raise ValueError(f'Duplicate mapped region {region!r} in {self.id}')
            geometry = feature.get('geometry') or {}
            coordinates = geometry.get('coordinates')
            if geometry.get('type') == 'Polygon':
                coordinates = [coordinates]
            elif geometry.get('type') != 'MultiPolygon':
                raise ValueError(
                    f'Unsupported geometry {geometry.get("type")!r} in {self.id}')
            geometries[region] = self._compile_polygons(coordinates)
        if set(geometries) != set(expected_regions):
            raise ValueError(
                f'{self.id} regions differ: expected={sorted(expected_regions)}, '
                f'observed={sorted(geometries)}')
        self.geometries = geometries
        self.sha256 = observed_hash
        self.metadata = {key: value for key, value in config.items()
                         if key not in {'region_map'}}

    @staticmethod
    def _compile_polygons(polygons):
        compiled = []
        for polygon in polygons:
            if not polygon:
                continue
            rings = [np.asarray(ring, dtype=np.float64) for ring in polygon]
            for ring in rings:
                if ring.ndim != 2 or ring.shape[1] != 2 or len(ring) < 4:
                    raise ValueError('Invalid polygon ring')
                if not np.isfinite(ring).all():
                    raise ValueError('Non-finite polygon coordinate')
                if (ring[:, 0].min() < -180 or ring[:, 0].max() > 180 or
                        ring[:, 1].min() < -90 or ring[:, 1].max() > 90):
                    raise ValueError('Boundary coordinates are not plausible CRS84 lon/lat')
            outer = rings[0]
            compiled.append({
                'outer': PolygonPath(outer, closed=True),
                'holes': [PolygonPath(ring, closed=True) for ring in rings[1:]],
                'bounds': (outer[:, 0].min(), outer[:, 1].min(),
                           outer[:, 0].max(), outer[:, 1].max()),
            })
        if not compiled:
            raise ValueError('Region has no polygon geometry')
        return compiled

    def contains(self, region, coordinates):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError('coordinates must have shape (rows, 2)')
        result = np.zeros(len(coordinates), dtype=bool)
        finite = np.isfinite(coordinates).all(axis=1)
        for polygon in self.geometries[region]:
            xmin, ymin, xmax, ymax = polygon['bounds']
            candidates = (finite & (coordinates[:, 0] >= xmin) &
                          (coordinates[:, 0] <= xmax) &
                          (coordinates[:, 1] >= ymin) &
                          (coordinates[:, 1] <= ymax))
            positions = np.flatnonzero(candidates)
            if not len(positions):
                continue
            values = coordinates[positions]
            inside = polygon['outer'].contains_points(values, radius=1e-12)
            for hole in polygon['holes']:
                inside &= ~hole.contains_points(values, radius=-1e-12)
            result[positions[inside]] = True
        return result

    def classify(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        observed = np.full(len(coordinates), '', dtype=object)
        for region in sorted(self.geometries):
            matched = self.contains(region, coordinates)
            collision = matched & (observed != '')
            observed[collision] = AMBIGUOUS
            observed[matched & ~collision] = region
        return observed


class AdministrativeBoundaryValidator:
    """Require all configured real boundary sources to agree with the expected label."""

    def __init__(self, config, root):
        if config.get('policy') != 'unanimous_expected_region':
            raise ValueError('Only unanimous_expected_region is supported')
        if config.get('coordinate_crs') != 'OGC:CRS84':
            raise ValueError('POI coordinates must be explicitly declared OGC:CRS84')
        self.expected_regions = tuple(config['expected_regions'])
        if len(set(self.expected_regions)) != len(self.expected_regions):
            raise ValueError('expected_regions contains duplicates')
        self.sources = tuple(BoundarySource(source, self.expected_regions, root)
                             for source in config['sources'])
        if len(self.sources) < 2:
            raise ValueError('Consensus validation requires at least two boundary sources')
        self.policy = config['policy']
        self.coordinate_crs = config['coordinate_crs']
        self.original_poi_label_boundary_vintage = config.get(
            'original_poi_label_boundary_vintage', 'unknown_not_supplied')
        self.provenance_limitations = list(config.get('provenance_limitations', ()))
        if self.original_poi_label_boundary_vintage != 'unknown_not_supplied':
            raise ValueError(
                'Original POI label boundary vintage has not been supplied or authenticated')
        if not self.provenance_limitations:
            raise ValueError('Boundary provenance limitations must be explicit')

    @classmethod
    def from_config(cls, config_path, project_root=None):
        config_path = Path(config_path).resolve()
        config = yaml.safe_load(config_path.read_text())
        root = Path(project_root).resolve() if project_root else config_path.parents[2]
        return cls(config, root)

    def validate(self, coordinates, expected_regions):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        expected = np.asarray(expected_regions, dtype=object)
        if len(coordinates) != len(expected):
            raise ValueError('Coordinate and expected-label lengths differ')
        unknown = set(expected) - set(self.expected_regions)
        if unknown:
            raise ValueError(f'Unknown expected regions: {sorted(unknown)}')
        source_labels = {source.id: source.classify(coordinates)
                         for source in self.sources}
        matrix = np.column_stack([source_labels[source.id] for source in self.sources])
        unanimous = np.all(matrix == matrix[:, [0]], axis=1)
        classified = np.all((matrix != '') & (matrix != AMBIGUOUS), axis=1)
        matches = unanimous & classified & (matrix[:, 0] == expected)
        status = np.full(len(expected), 'source_disagreement', dtype=object)
        unclassified = np.any(matrix == '', axis=1)
        ambiguous = np.any(matrix == AMBIGUOUS, axis=1)
        status[unclassified] = 'unclassified'
        status[ambiguous] = 'ambiguous'
        agreed_other = unanimous & classified & (matrix[:, 0] != expected)
        status[agreed_other] = 'label_mismatch'
        status[matches] = 'consensus_match'
        return {
            'matches': matches,
            'status': status,
            'consensus_region': np.where(unanimous & classified, matrix[:, 0], ''),
            'source_labels': source_labels,
        }

    def provenance(self):
        result = {
            'policy': self.policy,
            'coordinate_crs': self.coordinate_crs,
            'original_poi_label_boundary_vintage': self.original_poi_label_boundary_vintage,
            'provenance_limitations': self.provenance_limitations,
            'sources': [dict(source.metadata, resolved_path=str(source.path),
                             observed_sha256=source.sha256)
                        for source in self.sources],
        }
        return json.loads(json.dumps(result, default=str))
