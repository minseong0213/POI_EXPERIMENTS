"""Portable adaptation of Kireev et al.'s Universal Robust Embeddings.

This package is an external robust-embedding baseline.  It is never presented as a
TabPFN modification.  The downstream attack detector is explicitly named URE-RF.
"""

from .detector import UREAttackDetector, load_ure_attack_detector
from .model import UREEmbedding, URERegionModel
from .preprocessing import UREPreprocessor
from .training import URETrainingConfig, train_universal_robust_embeddings

__all__ = [
    "UREAttackDetector",
    "UREEmbedding",
    "UREPreprocessor",
    "URERegionModel",
    "URETrainingConfig",
    "load_ure_attack_detector",
    "train_universal_robust_embeddings",
]
