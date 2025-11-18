"""
IEMOCAP dataset implementation
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

try:
    from .base_dataset import EmotionDataset
except ImportError:
    from base_dataset import EmotionDataset

logger = logging.getLogger(__name__)


class IEMOCAPDataset(EmotionDataset):
    """
    IEMOCAP dataset
    
    Characteristics:
    - 10 emotion classes (after excluding xxx)
    - Has special labels like frustrated, excited
    - Sample count: ~7,266 (after excluding xxx)
    """
    
    def __init__(self, data_root: str, samples_per_emotion: int = 100):
        super().__init__(data_root)
        self._num_classes = 10  # Valid emotion categories (excluding xxx)
        self.samples_per_emotion = samples_per_emotion
        
        # Load samples
        self.samples = self.load_samples()
    
    @property
    def name(self) -> str:
        return "IEMOCAP"
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    def get_emotion_mapping(self) -> Dict[str, str]:
        """
        IEMOCAP label -> emotion2vec 9 classes mapping
        
        Mapping strategy:
        - ang → angry ✓
        - hap → happy ✓
        - sad → sad ✓
        - neu → neutral ✓
        - fru → angry  # TODO: User confirmation (currently frustrated→angry, but analysis suggests should map to neutral)
        - exc → happy  # TODO: User confirmation (currently excited→happy, analysis shows correct)
        - sur → surprised
        - fea → fearful
        - dis → disgusted
        - oth → other
        """
        return {
            'ang': 'angry',
            'angry': 'angry',      # Already mapped label
            'hap': 'happy',
            'happy': 'happy',      # Already mapped label
            'sad': 'sad',
            'neu': 'neutral',
            'neutral': 'neutral',  # Already mapped label
            'fru': 'angry',       # TODO: User confirm mapping strategy
            'exc': 'happy',       # TODO: User confirm mapping strategy
            'sur': 'surprised',
            'surprised': 'surprised',  # Already mapped label
            'fea': 'fearful',
            'fearful': 'fearful',  # Already mapped label
            'dis': 'disgusted',
            'disgusted': 'disgusted',  # Already mapped label
            'oth': 'other',
            'other': 'other',      # Already mapped label
        }
    
    def get_emotion_filter(self) -> Optional[List[str]]:
        """
        Return enabled emotion categories
        
        Default: None (use all, but exclude 'xxx')
        TODO: User may only use some emotions later (e.g. only 4 core emotions)
        """
        # Default use all, exclude 'xxx' (unknown label)
        return None  # TODO: User configuration later
    
    def load_samples(self) -> List[Dict]:
        """
        Load IEMOCAP samples (from already extracted features files)
        
        evaluation_features/IEMOCAP/ directory structure:
            Session1/
                Ses01F_impro01/
                    Ses01F_impro01_F000_ev2_frame.npy
                    Ses01F_impro01_F000_emotion.txt
        """
        samples = []
        
        # Directly search for all *_ev2_frame.npy files
        feature_files = list(self.data_root.glob("**/*_ev2_frame.npy"))
        
        for feat_file in feature_files:
            label_file = feat_file.parent / (feat_file.stem.replace('_ev2_frame', '_emotion') + '.txt')
            
            if not label_file.exists():
                continue
            
            with open(label_file) as f:
                emotion = f.read().strip()
            
            # Parse speaker from filename
            utterance_id = feat_file.stem.replace('_ev2_frame', '')
            speaker_id = utterance_id.split('_')[0]  # e.g. Ses01F
            
            # Compatible with method_rate_sweep.py
            fake_audio_path = str(feat_file).replace('_ev2_frame.npy', '.wav')
            
            sample = {
                'audio_path': fake_audio_path,
                'emotion': emotion,
                'speaker_id': speaker_id,
            }
            samples.append(sample)
        
        # Apply emotion_filter
        samples = self.filter_samples_by_emotion(samples)
        
        logger.info(f"✅ IEMOCAP: Loaded {len(samples)} samples")
        
        # Sample (random seed 1344871 ensures reproducibility)
        samples = self.sample_balanced(samples, samples_per_emotion=self.samples_per_emotion, seed=1344871)
        logger.info(f"   After sampling: {len(samples)} samples ({self.samples_per_emotion} per class)")
        
        return samples
    
    def _parse_emotion_from_filename(self, filename: str) -> str:
        """
        Parse emotion label from filename
        
        IEMOCAP filename format usually contains emotion information
        This is a simplified implementation, actual implementation needs to be adjusted according to dataset specific format
        """
        # Simplified implementation: return placeholder
        # TODO: Actual implementation needs to parse IEMOCAP annotation file
        return 'neu'  # placeholder

