"""
Bucket Batch Sampler
Reduces padding and improves codebook training efficiency
"""

import torch
from torch.utils.data import Sampler, Subset
import numpy as np
from typing import Iterator, List


class BucketBatchSampler(Sampler):
    """
    Batch sampler that buckets sequences by length
    
    Principle:
    1. Sort dataset by length
    2. Divide into buckets
    3. Random sample batches within each bucket
    4. Ensure similar sequence lengths in the same batch, reducing padding
    
    Usage example:
        sampler = BucketBatchSampler(
            dataset,
            batch_size=32,
            num_buckets=10,
            shuffle=True
        )
        loader = DataLoader(dataset, batch_sampler=sampler, ...)
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_buckets: int = 10,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42
    ):
        """
        Args:
            dataset: Dataset (must have length information)
            batch_size: Batch size
            num_buckets: Number of buckets (more buckets = closer lengths, but less randomness)
            shuffle: Whether to shuffle within buckets
            drop_last: Whether to drop last incomplete batch
            seed: Random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        # Collect all sample lengths
        self.lengths = self._get_lengths()
        
        # Create buckets by length
        self.buckets = self._create_buckets()
        
        # Calculate total number of batches
        self.num_batches = sum(
            len(bucket) // batch_size if drop_last else (len(bucket) + batch_size - 1) // batch_size
            for bucket in self.buckets
        )
    
    def _get_lengths(self) -> np.ndarray:
        """Get all sample lengths (optimization: avoid double I/O for Subset)"""
        # Priority: read cached lengths from Subset's base dataset
        if isinstance(self.dataset, Subset) and hasattr(self.dataset.dataset, 'get_length'):
            base = self.dataset.dataset
            idxs = self.dataset.indices
            return np.array([base.get_length(j) for j in idxs], dtype=np.int64)
        
        # Fallback method
        lengths = []
        for i in range(len(self.dataset)):
            try:
                if hasattr(self.dataset, 'get_length'):
                    length = int(self.dataset.get_length(i))
                else:
                    # Load sample and get length
                    sample = self.dataset[i]
                    if isinstance(sample, dict) and 'length' in sample:
                        length = int(sample['length'].item())
                    elif isinstance(sample, dict) and 'features' in sample:
                        length = int(sample['features'].shape[0])
                    else:
                        length = 100  # Default value
            except Exception:
                length = 100  # Fallback
            
            lengths.append(length)
        
        return np.array(lengths, dtype=np.int64)
    
    def _create_buckets(self) -> List[List[int]]:
        """Create buckets by length (using linspace for uniform splitting, more robust)"""
        # Sort indices by length
        sorted_indices = np.argsort(self.lengths)
        
        # Use linspace for uniform splitting, avoid empty buckets
        edges = np.linspace(0, len(sorted_indices), num=self.num_buckets + 1, dtype=np.int64)
        buckets = []
        
        for i in range(len(edges) - 1):
            start, end = edges[i], edges[i + 1]
            if start < end:
                bucket = sorted_indices[start:end].tolist()
                buckets.append(bucket)
        
        return buckets
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches"""
        # Set random seed (ensure reproducibility across epochs)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        all_batches = []
        
        # Generate batches per bucket
        for bucket in self.buckets:
            # Shuffle within bucket (if needed)
            if self.shuffle:
                indices = torch.randperm(len(bucket), generator=g).tolist()
                bucket_shuffled = [bucket[i] for i in indices]
            else:
                bucket_shuffled = bucket
            
            # Split into batches
            for i in range(0, len(bucket_shuffled), self.batch_size):
                batch = bucket_shuffled[i:i + self.batch_size]
                
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle batch order (maintain bucket cohesion, but randomize between batches)
        if self.shuffle:
            batch_indices = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_indices]

        # DDP support: let each rank take different subsequence (stride slicing)
        # Fix: pad to world_size multiple, avoid inconsistent batch count between ranks causing deadlock
        rank, world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            
            # Calculate number of batches to pad
            n = len(all_batches)
            pad = (-n) % world_size  # Number of batches to pad
            
            if pad > 0:
                # Copy first pad batches for padding (circular reuse)
                all_batches += all_batches[:pad]
            
            # Now each rank gets the same number of batches
            all_batches = all_batches[rank::world_size]

        return iter(all_batches)
    
    def __len__(self) -> int:
        """Return total batch count (DDP friendly: return batch count from current rank's perspective)"""
        import math
        n = self.num_batches
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            # Keep consistent with __iter__ padding logic
            # Total batch count after padding: n + ((-n) % world_size)
            n_padded = n + ((-n) % world_size)
            # Each rank gets exactly the same number of batches
            return n_padded // world_size
        return n
    
    def set_epoch(self, epoch: int):
        """Set epoch (for distributed training and randomness control)"""
        self.epoch = epoch


def test_bucket_sampler():
    """Test BucketBatchSampler"""
    from torch.utils.data import Dataset, DataLoader
    
    # Create mock dataset
    class DummyDataset(Dataset):
        def __init__(self, sizes):
            self.sizes = sizes
        
        def __len__(self):
            return len(self.sizes)
        
        def __getitem__(self, idx):
            return {
                'features': torch.zeros(self.sizes[idx], 10),
                'length': torch.tensor([self.sizes[idx]])
            }
    
    # Create dataset with uneven length distribution
    np.random.seed(42)
    sizes = np.random.randint(50, 200, size=100)
    dataset = DummyDataset(sizes)
    
    # Use BucketBatchSampler
    sampler = BucketBatchSampler(
        dataset,
        batch_size=8,
        num_buckets=5,
        shuffle=True
    )
    
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda x: x
    )
    
    print(f"Total batches: {len(loader)}")
    
    # Check length distribution of first few batches
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        lengths = [item['length'].item() for item in batch]
        print(f"Batch {i}: length range [{min(lengths)}, {max(lengths)}], "
              f"std {np.std(lengths):.1f}")
    
    # Compare with standard DataLoader
    print("\nComparison: standard DataLoader (no bucketing)")
    standard_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: x)
    for i, batch in enumerate(standard_loader):
        if i >= 5:
            break
        lengths = [item['length'].item() for item in batch]
        print(f"Batch {i}: length range [{min(lengths)}, {max(lengths)}], "
              f"std {np.std(lengths):.1f}")


if __name__ == "__main__":
    test_bucket_sampler()

