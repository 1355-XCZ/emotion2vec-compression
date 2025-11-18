"""
RAVDESS dataset implementation
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

try:
    from .base_dataset import EmotionDataset
except ImportError:
    from base_dataset import EmotionDataset

logger = logging.getLogger(__name__)


class RAVDESSDataset(EmotionDataset):
    """
    RAVDESS dataset
    
    Characteristics:
    - 8 emotion classes
    - Fully balanced (192 speech samples per class)
    - Filename encoding contains all information
    """
    
    def __init__(self, data_root: str, samples_per_emotion: int = 100):
        super().__init__(data_root)
        self._num_classes = 8
        self.samples_per_emotion = samples_per_emotion
        
        # RAVDESS emotion codes
        self.emotion_codes = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        # Load samples
        self.samples = self.load_samples()
    
    @property
    def name(self) -> str:
        return "RAVDESS"
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    def get_emotion_mapping(self) -> Dict[str, str]:
        """
        RAVDESS label -> emotion2vec 9 classes mapping
        
        Mapping strategy:
        - neutral → neutral ✓
        - calm → neutral     # TODO: User confirmation (currently calm→neutral, analysis shows correct)
        - happy → happy ✓
        - sad → sad ✓
        - angry → angry ✓
        - fearful → fearful ✓
        - disgust → disgusted ✓
        - surprised → surprised ✓
        """
        return {
            'neutral': 'neutral',
            'calm': 'neutral',        # TODO: User confirm mapping strategy
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'fearful',
            'disgust': 'disgusted',
            'disgusted': 'disgusted',  # Already mapped label
            'surprised': 'surprised',
        }
    
    def get_emotion_filter(self) -> Optional[List[str]]:
        """
        Return enabled emotion categories
        
        Default: None (use all)
        TODO: User may only use some emotions later
        """
        return None  # TODO: User configuration later
    
    def load_samples(self) -> List[Dict]:
        """
        Load RAVDESS samples (from already extracted features files)
        
        evaluation_features/RAVDESS/ directory structure:
            03-01-06-01-02-01-12_ev2_frame.npy
            03-01-06-01-02-01-12_emotion.txt
        """
        samples = []
        
        # Directly search for all *_ev2_frame.npy files
        feature_files = list(self.data_root.glob("*_ev2_frame.npy"))
        
        for feat_file in feature_files:
            label_file = feat_file.parent / (feat_file.stem.replace('_ev2_frame', '_emotion') + '.txt')
            
            if not label_file.exists():
                continue
            
            with open(label_file) as f:
                emotion = f.read().strip()
            
            # Parse filename to get metadata
            parts = feat_file.stem.replace('_ev2_frame', '').split('-')
            if len(parts) == 7:
                actor = parts[6]
                intensity = parts[3]
            else:
                actor = 'unknown'
                intensity = '01'
            
            # Compatible with method_rate_sweep.py
            fake_audio_path = str(feat_file).replace('_ev2_frame.npy', '.wav')
            
            sample = {
                'audio_path': fake_audio_path,
                'emotion': emotion,
                'speaker_id': f'Actor_{actor}',
                'intensity': intensity,
            }
            samples.append(sample)
        
        # Apply emotion_filter
        samples = self.filter_samples_by_emotion(samples)
        
        logger.info(f"✅ RAVDESS: Loaded {len(samples)} samples")
        
        # Sample (random seed 1344871 ensures reproducibility)
        samples = self.sample_balanced(samples, samples_per_emotion=self.samples_per_emotion, seed=1344871)
        logger.info(f"   After sampling: {len(samples)} samples ({self.samples_per_emotion} per class)")
        
        return samples

