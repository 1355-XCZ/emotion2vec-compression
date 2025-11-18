"""
Emotion Recognition Robustness under RVQ Information Bottleneck

This package implements grouped RVQ with entropy coding for studying
emotion recognition robustness under information bottleneck constraints.
"""

__version__ = "1.0.0"

from .config import (
    GroupedRVQConfig,
    EntropyModelConfig,
    RateControlConfig,
    TrainingConfig,
    EvaluationConfig,
)
from .grouped_rvq import GroupedRVQ
from .entropy_model import EntropyModel
from .rate_controller import RateController

__all__ = [
    "GroupedRVQConfig",
    "EntropyModelConfig",
    "RateControlConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "GroupedRVQ",
    "EntropyModel",
    "RateController",
]

