#!/usr/bin/env python3
"""
Prepare evaluation subset data

Randomly sample specified number of samples from complete dataset for paper experiment reproduction
- Randomly sample 100 samples per emotion category
- Keep data balanced
- Can set random seed to ensure reproducibility
"""

import os
import sys
import json
import random
import shutil
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_samples_by_emotion(data_root: Path, dataset_name: str):
    """
    Collect all samples classified by emotion in dataset
    
    Returns:
        dict: {emotion: [sample_paths]}
    """
    samples_by_emotion = defaultdict(list)
    
    if dataset_name == 'ESD':
        # ESD structure: data_root/speaker_id/emotion/xxx_ev2_frame.npy
        for feat_file in data_root.glob("**/*_ev2_frame.npy"):
            label_file = feat_file.parent / (feat_file.stem.replace('_ev2_frame', '_emotion') + '.txt')
            
            if not label_file.exists():
                continue
            
            with open(label_file) as f:
                emotion = f.read().strip()
            
            # Normalize emotion labels
            emotion_normalized = emotion.lower()
            if emotion_normalized == 'surprise':
                emotion_normalized = 'surprised'
            
            samples_by_emotion[emotion_normalized].append({
                'feature_file': feat_file,
                'label_file': label_file,
                'emotion': emotion
            })
    
    elif dataset_name == 'IEMOCAP':
        # IEMOCAP structure: data_root/SessionX/xxx_ev2_frame.npy
        for feat_file in data_root.glob("**/*_ev2_frame.npy"):
            label_file = feat_file.parent / (feat_file.stem.replace('_ev2_frame', '_emotion') + '.txt')
            
            if not label_file.exists():
                continue
            
            with open(label_file) as f:
                emotion = f.read().strip()
            
            emotion_normalized = emotion.lower()
            samples_by_emotion[emotion_normalized].append({
                'feature_file': feat_file,
                'label_file': label_file,
                'emotion': emotion
            })
    
    elif dataset_name == 'RAVDESS':
        # RAVDESS structure: data_root/xxx_ev2_frame.npy
        for feat_file in data_root.glob("*_ev2_frame.npy"):
            label_file = feat_file.parent / (feat_file.stem.replace('_ev2_frame', '_emotion') + '.txt')
            
            if not label_file.exists():
                continue
            
            with open(label_file) as f:
                emotion = f.read().strip()
            
            emotion_normalized = emotion.lower()
            samples_by_emotion[emotion_normalized].append({
                'feature_file': feat_file,
                'label_file': label_file,
                'emotion': emotion
            })
    
    return samples_by_emotion


def sample_subset(samples_by_emotion, samples_per_emotion: int, seed: int = 1344871):
    """
    Randomly sample specified number of samples from each emotion category
    
    Args:
        samples_by_emotion: Samples dictionary classified by emotion
        samples_per_emotion: Number of samples to extract per emotion
        seed: Random seed
    
    Returns:
        dict: {emotion: [selected_samples]}
    """
    random.seed(seed)
    subset = {}
    
    for emotion, samples in samples_by_emotion.items():
        available = len(samples)
        
        if available < samples_per_emotion:
            logger.warning(f"⚠️  {emotion}: Insufficient available samples ({available} < {samples_per_emotion}), using all samples")
            subset[emotion] = samples
        else:
            # Random sampling
            subset[emotion] = random.sample(samples, samples_per_emotion)
            logger.info(f"✅ {emotion}: Randomly sampled {samples_per_emotion} from {available} samples")
    
    return subset


def copy_subset(subset, source_root: Path, target_root: Path, dataset_name: str):
    """
    Copy subset files to target directory
    
    Args:
        subset: Selected samples subset
        source_root: Source data root directory
        target_root: Target data root directory
        dataset_name: Dataset name
    """
    target_root.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    
    for emotion, samples in subset.items():
        for sample in tqdm(samples, desc=f"Copying {emotion}"):
            feat_file = sample['feature_file']
            label_file = sample['label_file']
            
            # Calculate relative path
            rel_path = feat_file.relative_to(source_root)
            
            # Target path
            target_feat = target_root / rel_path
            target_label = target_feat.parent / label_file.name
            
            # Create directory
            target_feat.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            shutil.copy2(feat_file, target_feat)
            shutil.copy2(label_file, target_label)
            
            copied_count += 2
    
    logger.info(f"✅ Total {copied_count} files copied")


def save_subset_info(subset, output_file: Path, dataset_name: str, samples_per_emotion: int, seed: int):
    """
    Save subset information to JSON file
    
    Args:
        subset: Selected samples subset
        output_file: Output JSON file path
        dataset_name: Dataset name
        samples_per_emotion: Number of samples per emotion
        seed: Random seed
    """
    info = {
        'dataset': dataset_name,
        'samples_per_emotion': samples_per_emotion,
        'random_seed': seed,
        'emotions': {}
    }
    
    for emotion, samples in subset.items():
        info['emotions'][emotion] = {
            'count': len(samples),
            'files': [str(s['feature_file'].name) for s in samples]
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Subset info saved to: {output_file}")


def prepare_dataset_subset(dataset_name: str, source_root: str, target_root: str, 
                          samples_per_emotion: int = 100, seed: int = 1344871):
    """
    Prepare evaluation subset for single dataset
    
    Args:
        dataset_name: Dataset name (ESD/IEMOCAP/RAVDESS)
        source_root: Source data root directory
        target_root: Target data root directory
        samples_per_emotion: Number of samples per emotion
        seed: Random seed
    """
    logger.info("=" * 80)
    logger.info(f"Preparing {dataset_name} evaluation subset")
    logger.info("=" * 80)
    
    source_path = Path(source_root)
    target_path = Path(target_root)
    
    if not source_path.exists():
        logger.error(f"❌ Source data does not exist: {source_path}")
        return False
    
    # 1. Collect all samples
    logger.info(f"📂 Scanning source data directory: {source_path}")
    samples_by_emotion = collect_samples_by_emotion(source_path, dataset_name)
    
    if not samples_by_emotion:
        logger.error(f"❌ No samples found")
        return False
    
    logger.info(f"✅ Found {len(samples_by_emotion)} emotion categories:")
    for emotion, samples in sorted(samples_by_emotion.items()):
        logger.info(f"  - {emotion}: {len(samples)} samples")
    
    # 2. Random sampling subset
    logger.info(f"\n🎲 Random sampling subset (seed={seed}, {samples_per_emotion} per class)")
    subset = sample_subset(samples_by_emotion, samples_per_emotion, seed)
    
    total_samples = sum(len(samples) for samples in subset.values())
    logger.info(f"✅ Total samples in subset: {total_samples}")
    
    # 3. Copy files
    logger.info(f"\n📋 Copying files to: {target_path}")
    copy_subset(subset, source_path, target_path, dataset_name)
    
    # 4. Save subset info
    info_file = target_path.parent / f"{dataset_name}_subset_info.json"
    save_subset_info(subset, info_file, dataset_name, samples_per_emotion, seed)
    
    logger.info(f"\n✅ {dataset_name} subset preparation complete")
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare evaluation subset data')
    parser.add_argument('--source-data', type=str, default='data',
                       help='Source data root directory (default: data)')
    parser.add_argument('--target-data', type=str, default='data_subset',
                       help='Target data root directory (default: data_subset)')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples per emotion (default: 100)')
    parser.add_argument('--seed', type=int, default=1344871,
                       help='Random seed (default: 1344871)')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['ESD', 'IEMOCAP', 'RAVDESS'],
                       help='Datasets to process (default: ESD IEMOCAP RAVDESS)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Evaluation Subset Data Preparation Tool")
    logger.info("=" * 80)
    logger.info(f"Source data directory: {args.source_data}")
    logger.info(f"Target directory: {args.target_data}")
    logger.info(f"Samples per emotion: {args.samples}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    
    success_count = 0
    
    for dataset in args.datasets:
        # Uniform conversion to uppercase (data directory is uppercase)
        dataset_upper = dataset.upper()
        source_root = Path(args.source_data) / dataset_upper
        target_root = Path(args.target_data) / dataset_upper
        
        success = prepare_dataset_subset(
            dataset_name=dataset_upper,
            source_root=str(source_root),
            target_root=str(target_root),
            samples_per_emotion=args.samples,
            seed=args.seed
        )
        
        if success:
            success_count += 1
        
        print()  # Empty line separator
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Success: {success_count}/{len(args.datasets)}")
    logger.info(f"Subset data saved to: {args.target_data}/")
    logger.info(f"Subset info files: {args.target_data}/*_subset_info.json")
    
    if success_count == len(args.datasets):
        logger.info("\n🎉 All subset preparation complete!")
        logger.info("\nNext steps:")
        logger.info("  1. Check subset info: cat data_subset/*_subset_info.json")
        logger.info("  2. Run experiment: python reproduce_experiments.py --mode all")
    else:
        logger.error("\n❌ Some datasets failed to process")
        sys.exit(1)


if __name__ == '__main__':
    main()

