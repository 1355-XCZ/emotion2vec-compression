"""
ESD (Emotional Speech Dataset) dataset implementation
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

try:
    from .base_dataset import EmotionDataset
except ImportError:
    from base_dataset import EmotionDataset

logger = logging.getLogger(__name__)


class ESDDataset(EmotionDataset):
    """
    ESD dataset
    
    Characteristics:
    - 5 emotion classes
    - Fully balanced (7,000 samples per class, 3,500 each for English and Chinese)
    - Clear directory structure
    """
    
    def __init__(self, data_root: str, languages: Optional[List[str]] = None, samples_per_emotion: int = 100):
        """
        Args:
            data_root: ESD dataset root directory
            languages: List of languages to use, None means use all (['english', 'chinese'])
            samples_per_emotion: number of samples to sample per emotion
        """
        super().__init__(data_root)
        self._num_classes = 5
        self.languages = languages or ['english', 'chinese']
        self.samples_per_emotion = samples_per_emotion
        
        # Load samples
        self.samples = self.load_samples()
        
    @property
    def name(self) -> str:
        return "ESD"
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    def get_emotion_mapping(self) -> Dict[str, str]:
        """
        ESD label -> emotion2vec 9 classes mapping
        
        ESD labels (Chinese and English):
        - English: Angry, Happy, Neutral, Sad, Surprise
        - Chinese: 生气, 快乐, 中立, 伤心, 惊喜
        
        Mapping strategy:
        - Direct correspondence, only handle spelling differences (surprise → surprised)
        """
        return {
            # English labels
            'Angry': 'angry',
            'angry': 'angry',
            'Happy': 'happy',
            'happy': 'happy',
            'Neutral': 'neutral',
            'neutral': 'neutral',
            'Sad': 'sad',
            'sad': 'sad',
            'Surprise': 'surprised',  # Note spelling conversion
            'surprise': 'surprised',
            # Chinese labels
            '生气': 'angry',
            '快乐': 'happy',
            '中立': 'neutral',
            '伤心': 'sad',
            '惊喜': 'surprised',
        }
    
    def get_emotion_filter(self) -> Optional[List[str]]:
        """
        Return enabled emotion categories
        
        Default: None (use all 5 classes)
        TODO: User may only use some emotions later
        """
        return None  # TODO: User configuration later
    
    def load_samples(self) -> List[Dict]:
        """
        Load ESD samples (from already extracted features files)
        
        evaluation_features/ESD/ directory structure:
            0001/
                Angry/
                    0001_000351_ev2_frame.npy
                    0001_000351_emotion.txt
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
            
            # Infer speaker ID from path
            speaker_id = feat_file.parent.parent.name  # e.g. 0001
            
            # Determine language (0001-0010 Chinese, 0011-0020 English)
            if speaker_id.isdigit():
                sid_num = int(speaker_id)
                lang = 'chinese' if sid_num <= 10 else 'english'
            else:
                lang = 'unknown'
            
            # Filter language
            if lang not in self.languages:
                continue
            
            # For compatibility with method_rate_sweep.py, use features_path as audio_path
            # method_rate_sweep.py will use .replace('.wav', '_ev2_frame.npy') to find features
            # We directly give it the correct features path
            fake_audio_path = str(feat_file).replace('_ev2_frame.npy', '.wav')
            
            sample = {
                'audio_path': fake_audio_path,  # compatible with method_rate_sweep
                'emotion': emotion,
                'speaker_id': f'{lang}_{speaker_id}',
                'language': lang,
            }
            samples.append(sample)
        
        # Apply emotion_filter
        samples = self.filter_samples_by_emotion(samples)
        
        logger.info(f"✅ ESD: Loaded {len(samples)} samples (languages: {self.languages})")
        
        # Sample (random seed 1344871 ensures reproducibility)
        samples = self.sample_balanced(samples, samples_per_emotion=self.samples_per_emotion, seed=1344871)
        logger.info(f"   After sampling: {len(samples)} samples ({self.samples_per_emotion} per class)")
        
        return samples

