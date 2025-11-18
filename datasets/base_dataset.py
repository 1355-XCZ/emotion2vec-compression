"""
Dataset base class - Object-oriented design
Define unified interface, support flexible emotion labels mapping and filtering
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path
import random


class EmotionDataset(ABC):
    """
    Emotion dataset abstract base class
    
    Each concrete dataset needs to implement:
    1. load_samples() - Load samples list
    2. get_emotion_mapping() - Emotion labels mapping to emotion2vec 9 classes
    3. get_emotion_filter() - Enabled emotion categories (for user configuration)
    """
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: dataset root directory
        """
        self.data_root = Path(data_root)
        self.samples = []
        
    @abstractmethod
    def load_samples(self) -> List[Dict]:
        """
        Load data samples
        
        Returns:
            List of dicts, each containing:
                - 'audio_path': str, audio file path
                - 'features_path': str (optional), ev2 features file path
                - 'emotion': str, original emotion label
                - 'speaker_id': str (optional), speaker ID
                - ... other metadata
        """
        pass
    
    @abstractmethod
    def get_emotion_mapping(self) -> Dict[str, str]:
        """
        Return emotion labels mapping: dataset label -> emotion2vec 9 classes
        
        emotion2vec 9 classes:
        ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 
         'other', 'sad', 'surprised', 'unknown']
        
        Returns:
            Dict[original label, emotion2vec label]
        
        Note:
            - TODO: User confirms mapping strategy later
            - Mapping should be based on semantic similarity
        """
        pass
    
    @abstractmethod
    def get_emotion_filter(self) -> Optional[List[str]]:
        """
        Return enabled emotion categories (dataset original labels)
        
        Returns:
            List of str: enabled emotion categories, None means use all
        
        Note:
            - TODO: User configuration (e.g. only use certain emotion categories)
            - Default return None (use all)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name"""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Original emotion category count"""
        pass
    
    def filter_samples_by_emotion(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter samples by emotion_filter
        
        Args:
            samples: samples list
        
        Returns:
            filtered samples list
        """
        emotion_filter = self.get_emotion_filter()
        
        if emotion_filter is None:
            return samples
        
        filtered = [s for s in samples if s['emotion'] in emotion_filter]
        
        return filtered
    
    def sample_balanced(self, samples: List[Dict], samples_per_emotion: int = 500, seed: int = 1344871) -> List[Dict]:
        """
        Sample fixed number of samples per emotion category
        
        Args:
            samples: samples list
            samples_per_emotion: number of samples to sample per emotion
            seed: random seed
        
        Returns:
            sampled samples list
        """
        # Group by emotion
        emotion_groups = {}
        for sample in samples:
            emotion = sample['emotion']
            if emotion not in emotion_groups:
                emotion_groups[emotion] = []
            emotion_groups[emotion].append(sample)
        
        # Sample per emotion
        random.seed(seed)
        sampled = []
        for emotion, group in emotion_groups.items():
            if len(group) <= samples_per_emotion:
                # Insufficient samples, keep all
                sampled.extend(group)
            else:
                # Random sampling
                sampled.extend(random.sample(group, samples_per_emotion))
        
        # Shuffle sample order, avoid grouping by emotion
        # This prevents certain emotions from concentrating in specific intervals, causing systematic bias during evaluation
        random.shuffle(sampled)
        
        return sampled
    
    def map_emotion_to_ev2(self, emotion: str) -> str:
        """
        Map dataset label to emotion2vec label
        
        Args:
            emotion: dataset original label
        
        Returns:
            emotion2vec label
        """
        mapping = self.get_emotion_mapping()
        
        # Try direct mapping
        if emotion in mapping:
            return mapping[emotion]
        
        # Try lowercase mapping
        emotion_lower = emotion.lower()
        if emotion_lower in mapping:
            return mapping[emotion_lower]
        
        # No mapping found, return 'other'
        return 'other'
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics
        
        Returns:
            statistics dictionary
        """
        if not self.samples:
            self.samples = self.load_samples()
        
        from collections import Counter
        
        emotion_counts = Counter([s['emotion'] for s in self.samples])
        
        stats = {
            'name': self.name,
            'num_samples': len(self.samples),
            'num_classes': self.num_classes,
            'emotion_distribution': dict(emotion_counts),
            'emotion_mapping': self.get_emotion_mapping(),
            'emotion_filter': self.get_emotion_filter(),
        }
        
        return stats
    
    def __len__(self) -> int:
        """Return number of samples"""
        if not self.samples:
            self.samples = self.load_samples()
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get single sample"""
        if not self.samples:
            self.samples = self.load_samples()
        return self.samples[idx]

