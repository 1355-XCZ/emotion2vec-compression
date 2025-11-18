#!/usr/bin/env python3
"""
Example: Training RVQ with new configuration system

Usage:
    # Default config
    python examples/train_example.py
    
    # With dataset config
    python examples/train_example.py --dataset configs/datasets/iemocap.yaml
    
    # With overrides
    python examples/train_example.py --batch-size 64 --num-epochs 10
    
    # Quick test
    python examples/train_example.py --experiment configs/experiments/quick_test.yaml
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import setup_config
import torch


def main():
    # Load configuration
    config, args = setup_config()
    
    print("="*60)
    print("Configuration Loaded")
    print("="*60)
    
    # Access model configuration
    print("\n[Model Configuration]")
    print(f"  Feature dim: {config.get('model.grouped_rvq.feature_dim')}")
    print(f"  Num groups: {config.get('model.grouped_rvq.num_groups')}")
    print(f"  Layers per group: {config.get('model.grouped_rvq.num_fine_layers')}")
    print(f"  Codebook size: {config.get('model.grouped_rvq.fine_codebook_size')}")
    print(f"  Enable SKIP: {config.get('model.grouped_rvq.enable_skip')}")
    
    # Access training configuration
    print("\n[Training Configuration]")
    print(f"  Num epochs: {config.get('training.rvq.num_epochs')}")
    print(f"  Batch size: {config.get('training.rvq.batch_size')}")
    print(f"  Learning rate: {config.get('training.rvq.learning_rate')}")
    print(f"  Early stopping: {config.get('training.rvq.early_stopping.enabled')}")
    
    # Access dataset configuration (if specified)
    dataset_name = config.get('dataset.name')
    if dataset_name:
        print("\n[Dataset Configuration]")
        print(f"  Dataset: {dataset_name}")
        print(f"  Emotions: {config.get('dataset.emotions')}")
        print(f"  Samples per emotion: {config.get('dataset.samples_per_emotion')}")
    
    # Access evaluation configuration
    print("\n[Evaluation Configuration]")
    print(f"  Rate sweep enabled: {config.get('evaluation.rate_sweep.enabled')}")
    print(f"  Rate points (BPF): {config.get('evaluation.rate_sweep.rates_bpf')}")
    print(f"  Layer sweep enabled: {config.get('evaluation.layer_sweep.enabled')}")
    print(f"  Layer points: {config.get('evaluation.layer_sweep.layers')}")
    
    # Device setup
    device = args.device if hasattr(args, 'device') else 'cuda'
    print(f"\n[Device]")
    print(f"  Using device: {device}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    print("\n" + "="*60)
    print("✅ Configuration system working correctly!")
    print("="*60)
    
    # Example: Create model from config
    print("\n[Example: Creating Model]")
    feature_dim = config.get('model.grouped_rvq.feature_dim')
    num_groups = config.get('model.grouped_rvq.num_groups')
    print(f"  Would create GroupedRVQ with:")
    print(f"    feature_dim={feature_dim}, num_groups={num_groups}, ...")
    
    return config


if __name__ == "__main__":
    config = main()

