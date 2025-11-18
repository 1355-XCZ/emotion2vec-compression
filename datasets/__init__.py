"""
Dataset abstraction layer - Object-oriented design
Support flexible emotion labels mapping and filtering
"""

from .base_dataset import EmotionDataset
from .iemocap_dataset import IEMOCAPDataset
from .ravdess_dataset import RAVDESSDataset
from .esd_dataset import ESDDataset

__all__ = [
    'EmotionDataset',
    'IEMOCAPDataset',
    'RAVDESSDataset',
    'ESDDataset',
]

