"""Structural POI defense built around the official TabPFN v2.5 architecture.

The classes in this package are the proposed model.  External attack detectors live in
``poi.ure`` and are deliberately not exposed as TabPFN variants.
"""

from .model import (
    ProposedDefenseOutput,
    ProposedTabPFNDefense,
    load_official_tabpfn25,
)
from .predictor import ProposedDefensePredictor, load_proposed_predictor
from .schema import POIFeatureCodebook
from .structural import StructuralDefenseAdapter, StructuralDefenseOutput

__all__ = [
    "POIFeatureCodebook",
    "ProposedDefenseOutput",
    "ProposedDefensePredictor",
    "ProposedTabPFNDefense",
    "StructuralDefenseAdapter",
    "StructuralDefenseOutput",
    "load_official_tabpfn25",
    "load_proposed_predictor",
]
