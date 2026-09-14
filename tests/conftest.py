import pandas as pd
import pytest


@pytest.fixture
def dataset(tmp_path):
    rows = []
    for region, offset in [('A', 0), ('B', 100)]:
        for split, size in [('train', 8), ('validation', 3), ('test', 3)]:
            for i in range(size):
                rows.append({'POI_ID': f'{region}-{split}-{i}', 'region': region, 'split': split,
                             'X_COORD': str(offset + i), 'Y_COORD': str(offset + i),
                             'ASORT_LCLASDC': '001', 'ASORT_MLSFCDC': '002', 'ASORT_SDASDC': '003'})
    frame = pd.DataFrame(rows)
    frame[['POI_ID', 'region', 'split']].to_csv(tmp_path / 'sample_manifest.csv', index=False)
    frame.drop(columns='split').to_csv(tmp_path / 'poi_data_region.csv', index=False)
    attack = frame.drop(columns='split').copy()
    attack['X_COORD'] = (attack.X_COORD.astype(float) + 1).astype(str)
    attack.iloc[::-1].to_csv(tmp_path / 'poi_adversarial_data_final.csv', index=False)
    return tmp_path
