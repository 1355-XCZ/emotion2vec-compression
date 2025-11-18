"""
data loader - Minimal Version
Support loading emotion2vec featuresandemotion labels
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
import random

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id):
    """
    DataLoader worker initialization function
    Ensure controlled randomness in multiprocessing without conflicts
    """
    # Get PyTorch random seed
    worker_seed = torch.initial_seed() % 2**32
    
    # Set numpy and python random seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EmotionDataset(Dataset):
    """
    emotion2vec frame-level features + emotion labels dataset
    """
    
    def __init__(
        self,
        data_dir: str,
        max_samples: Optional[int] = None,
        max_frames: int = 200,
        min_frames: int = 10,
        feature_dim: int = 768,
        supported_languages: Optional[list] = None,
        normalize: bool = True,
        mean_std_path: Optional[str] = None,
        feature_suffix: str = "_ev2_frame.npy",
        label_suffix: str = "_emotion.txt",
        emotion_label_map: Optional[dict] = None,
        seed: int = 42
    ):
        """
        Args:
            data_dir: data directory (e.g. emilia_vevo_training_50h)
            max_samples: maximum number of samples (for quick testing)
            max_frames: maximum number of frames (truncate)
            min_frames: minimum number of frames (filter)
            feature_dim: feature dimension (default 768)
            supported_languages: supported language list (default ['EN', 'ZH'])
            normalize: whether to normalize features
            mean_std_path: normalization parameters path
            feature_suffix: features file suffix
            label_suffix: label file suffix
            emotion_label_map: emotion labels mapping {label_str: class_id}
            seed: random seed
        """
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.feature_dim = feature_dim
        self.normalize = normalize
        self.feature_suffix = feature_suffix
        self.label_suffix = label_suffix
        
        # Emotion labels mapping (support aliases and case insensitive)
        if emotion_label_map is None:
            # Default mapping (5 classes, compatible with ESD)
            self.emotion_label_map = {
                'angry': 0, 'anger': 0, 'ang': 0,
                'happy': 1, 'happiness': 1, 'excited': 1, 'hap': 1,
                'neutral': 2, 'neu': 2, 'frustrated': 2,  # keep consistent with DataConfig
                'sad': 3, 'sadness': 3,
                'surprise': 4, 'surprised': 4, 'sur': 4,
            }
        else:
            self.emotion_label_map = emotion_label_map
        
        self.unknown_labels_count = 0  # Count unknown labels
        
        # Features normalization parameters
        self.mean = None
        self.std = None
        if normalize and mean_std_path and Path(mean_std_path).exists():
            data = np.load(mean_std_path)
            self.mean = torch.from_numpy(data['mean']).float()  # (768,)
            self.std = torch.from_numpy(data['std']).float().clamp_min(1e-5)
            logger.info(f"✅ Loaded normalization parameters: {mean_std_path}")
        elif normalize and not mean_std_path:
            logger.warning("⚠️ normalize_features=True but mean_std_path not set")
            logger.warning("   Skipping normalization. Please run compute_normalization.py first and set path.")
        elif normalize and mean_std_path and not Path(mean_std_path).exists():
            logger.warning(f"⚠️ normalize_features=True but file does not exist: {mean_std_path}")
            logger.warning("   Skipping normalization.")
        
        # Collect all features files and cache lengths
        self.feature_files = []
        self.lengths = []  # Cache frame length per sample
        
        if supported_languages is None:
            # No language subdirectory, recursively search entire data_dir
            frame_files = sorted(self.data_dir.rglob(f"*{self.feature_suffix}"))
            self.feature_files.extend(frame_files)
        else:
            # Has language subdirectory
            for lang in supported_languages:
                lang_dir = self.data_dir / lang
                if not lang_dir.exists():
                    logger.warning(f"Directory does not exist: {lang_dir}")
                    continue
                
                # Frame-level features files
                frame_files = sorted(lang_dir.glob(f"*{self.feature_suffix}"))
                self.feature_files.extend(frame_files)
        
        if len(self.feature_files) == 0:
            raise ValueError(f"No features files found (suffix={self.feature_suffix}): {data_dir}")
        
        # Filter: Check frame count range and cache length (keep overlong samples, truncate in __getitem__)
        valid_files = []
        valid_lengths = []
        for fpath in self.feature_files:
            try:
                feat = np.load(fpath, mmap_mode='r')  # memory mapping, avoid reading entire array
                T = feat.shape[0]
                if T >= self.min_frames:  # only filter by lower limit
                    valid_files.append(fpath)
                    # Bucketing length: truncate to max_frames, consistent with actual batch length
                    valid_lengths.append(min(T, self.max_frames))
            except:
                continue
        
        self.feature_files = valid_files
        self.lengths = valid_lengths
        
        # Shuffle (fixed random seed) - shuffle file and length synchronously
        np.random.seed(seed)
        perm = np.random.permutation(len(self.feature_files))
        self.feature_files = [self.feature_files[i] for i in perm]
        self.lengths = [self.lengths[i] for i in perm]
        
        # Limit sample count
        if max_samples is not None:
            self.feature_files = self.feature_files[:max_samples]
            self.lengths = self.lengths[:max_samples]
        
        logger.info(f"✅ EmotionDataset: {len(self.feature_files)} samples")
    
    def __len__(self):
        return len(self.feature_files)
    
    def get_length(self, idx: int) -> int:
        """Get sample length (for bucket sampler, avoid second IO)"""
        return int(self.lengths[idx])
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'features': (T, 768) emotion2vec frame-level features
                'label': (1,) emotion labels (if available)
                'length': (1,) actual frame count
            }
        """
        feature_file = self.feature_files[idx]
        
        try:
            # Load features (memory mapping, avoid reading entire array)
            arr = np.load(feature_file)  # (T, 768) - Not using memory mapping to improve performance
            T = arr.shape[0]

            # Truncate or pad
            if T > self.max_frames:
                # Random crop
                start = np.random.randint(0, T - self.max_frames + 1)
                arr = arr[start:start + self.max_frames]  # slice first
                T = self.max_frames

            features = torch.from_numpy(np.array(arr, copy=True)).float()
            
            # Features normalization
            if self.normalize and self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
            
            # Try to load emotion labels (if exists) - use unified mapping
            label_file = feature_file.with_name(
                feature_file.name.replace(self.feature_suffix, self.label_suffix)
            )
            if label_file.exists():
                with open(label_file, 'r') as f:
                    label_str = f.read().strip().lower()  # to lowercase
                    
                    # Use unified mapping (support aliases)
                    if label_str in self.emotion_label_map:
                        label_id = self.emotion_label_map[label_str]
                    else:
                        # Unknown label: record and use neutral
                        self.unknown_labels_count += 1
                        label_id = 2  # Neutral
                        if self.unknown_labels_count <= 10:  # only print first 10
                            logger.warning(f"Unknown emotion label: '{label_str}' in file {feature_file.name}, using Neutral")
                    
                    label = torch.tensor([label_id], dtype=torch.long)
            else:
                label = torch.tensor([2], dtype=torch.long)  # default Neutral
            
            return {
                'features': features,  # (T, 768)
                'label': label,        # (1,)
                'length': torch.tensor([T], dtype=torch.long)
            }
        
        except Exception as e:
            logger.error(f"Load failed: {feature_file}, Error: {e}")
            # Return zero vector as fallback
            return {
                'features': torch.zeros(self.min_frames, self.feature_dim, dtype=torch.float32),
                'label': torch.tensor([2], dtype=torch.long),
                'length': torch.tensor([self.min_frames], dtype=torch.long)
            }


def collate_fn(batch):
    """
    Handle variable-length sequences: padding
    
    Args:
        batch: List of dicts
    
    Returns:
        {
            'features': (B, T_max, 768) padded features
            'labels': (B,) emotion labels
            'lengths': (B,) actual lengths
        }
    """
    # Get max length and dtype
    lengths = torch.cat([item['length'] for item in batch])
    max_len = lengths.max().item()
    feature_dim = batch[0]['features'].shape[1]
    dtype = batch[0]['features'].dtype  # preserve original dtype
    
    # Padding
    batch_size = len(batch)
    features = torch.zeros(batch_size, max_len, feature_dim, dtype=dtype)
    labels = torch.cat([item['label'] for item in batch])  # (B,)
    
    for i, item in enumerate(batch):
        T = item['features'].shape[0]
        features[i, :T] = item['features']
    
    return {
        'features': features,  # (B, T_max, 768)
        'labels': labels,      # (B,)
        'lengths': lengths     # (B,)
    }


def create_dataloaders(
    data_config,
    training_config,
    rvq_config,
    use_bucketing: bool = False,
    num_buckets: int = 10
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        data_config: DataConfig instance
        training_config: TrainingConfig instance
        rvq_config: GroupedRVQConfig instance
        use_bucketing: whether to use length bucketing (reduce padding)
        num_buckets: number of buckets
    
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    full_dataset = EmotionDataset(
        data_dir=data_config.train_data_path,
        max_samples=data_config.max_samples,
        max_frames=data_config.max_frames,
        min_frames=data_config.min_frames,
        feature_dim=rvq_config.feature_dim,
        supported_languages=data_config.supported_languages,
        normalize=data_config.normalize_features,
        mean_std_path=data_config.mean_std_path,
        feature_suffix=data_config.feature_suffix,
        label_suffix=data_config.label_suffix,
        emotion_label_map=data_config.emotion_label_map,  # use unified label mapping
        seed=data_config.seed
    )
    
    # Calculate split size
    total_size = len(full_dataset)
    train_size = int(data_config.train_split * total_size)
    val_size = total_size - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(data_config.seed)
    )
    
    logger.info(f"✅ Training set: {len(train_dataset)} samples")
    logger.info(f"✅ Validation set: {len(val_dataset)} samples")
    
    # DataLoader common parameters (avoid prefetch_factor=None error in single process)
    persistent_workers = training_config.num_workers > 0
    
    # Build common kwargs
    common_kwargs = {
        'num_workers': training_config.num_workers,
        'pin_memory': training_config.pin_memory and torch.cuda.is_available(),  # only beneficial in CUDA scenario
        'collate_fn': collate_fn,
        'worker_init_fn': _worker_init_fn if training_config.num_workers > 0 else None,
        'persistent_workers': persistent_workers,
    }
    
    # Only add prefetch_factor in multiprocessing
    if persistent_workers:
        common_kwargs['prefetch_factor'] = 2  # can adjust to 3 based on I/O capability
    
    # Create data loader
    if use_bucketing:
        # Use Bucket Batch Sampler
        try:
            from .bucket_sampler import BucketBatchSampler
        except ImportError:
            from bucket_sampler import BucketBatchSampler
        
        train_sampler = BucketBatchSampler(
            train_dataset,
            batch_size=training_config.batch_size,
            num_buckets=num_buckets,
            shuffle=True,
            drop_last=True,
            seed=data_config.seed
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            **common_kwargs
        )
        
        logger.info(f"✅ Using Bucket Batch Sampler (num_buckets={num_buckets})")
    else:
        # Standard DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            drop_last=True,
            **common_kwargs
        )
    
    # Validation set doesn't need bucketing
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs
    )
    
    return train_loader, val_loader

