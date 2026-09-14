"""CPU random-forest reference model with train-only preprocessing."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL, NUMERIC


def build_model(config, seed):
    if config['name'] != 'random_forest':
        raise ValueError(f"Unsupported model: {config['name']}")
    preprocessor = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', keep_empty_features=True), NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])
    return Pipeline([
        ('preprocess', preprocessor),
        ('classifier', RandomForestClassifier(random_state=seed, **config['params'])),
    ])
