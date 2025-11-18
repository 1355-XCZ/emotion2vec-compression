#!/usr/bin/env python3
"""
Paper experiment complete reproduction workflow script

Usage:
    python reproduce_experiments.py --mode all  # Run evaluation + plotting
    python reproduce_experiments.py --mode eval  # Only run evaluation
    python reproduce_experiments.py --mode plot  # Only plotting (requires existing evaluation results)
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== Experiment Configuration ==========
# Rate sampling points: 10-200 (step 5), 200-300 (step 20)
RATE_POINTS = list(range(10, 201, 5)) + list(range(220, 301, 20))

# Dataset configuration
DATASETS = ['esd', 'iemocap', 'ravdess']
SAMPLES_PER_EMOTION = 100

# Data directory (use subset data)
DATA_ROOT = 'data_subset'  # Use prepared evaluation subset

# Model checkpoints
RVQ_CHECKPOINT = 'checkpoints/grouped_rvq_best.pt'
ENTROPY_CHECKPOINT = 'checkpoints/entropy_model_best.pt'

# Output directory
OUTPUT_DIR = 'evaluation_results'
PLOT_OUTPUT_DIR = 'evaluation_results/figures'


def check_prerequisites():
    """Check prerequisites"""
    logger.info("=" * 80)
    logger.info("Checking prerequisites")
    logger.info("=" * 80)
    
    # Check model files
    if not Path(RVQ_CHECKPOINT).exists():
        logger.error(f"❌ RVQ model does not exist: {RVQ_CHECKPOINT}")
        return False
    logger.info(f"✅ RVQ model: {RVQ_CHECKPOINT}")
    
    if not Path(ENTROPY_CHECKPOINT).exists():
        logger.error(f"❌ Entropy model does not exist: {ENTROPY_CHECKPOINT}")
        return False
    logger.info(f"✅ Entropy model: {ENTROPY_CHECKPOINT}")
    
    # Check datasets
    data_root = Path(DATA_ROOT)
    if not data_root.exists():
        logger.error(f"❌ Data directory does not exist: {data_root}")
        logger.error(f"   Please run first: python prepare_evaluation_subset.py")
        return False
    
    for dataset in DATASETS:
        dataset_path = data_root / dataset.upper()
        if not dataset_path.exists():
            logger.error(f"❌ Dataset does not exist: {dataset_path}")
            logger.error(f"   Please run first: python prepare_evaluation_subset.py")
            return False
        
        # Check subset info file
        subset_info = data_root / f"{dataset.upper()}_subset_info.json"
        if subset_info.exists():
            with open(subset_info) as f:
                info = json.load(f)
            total_samples = sum(e['count'] for e in info['emotions'].values())
            logger.info(f"✅ Dataset: {dataset.upper()} ({total_samples} samples)")
        else:
            logger.info(f"✅ Dataset: {dataset_path}")
    
    # Check Python dependencies
    try:
        import torch
        import numpy as np
        import matplotlib
        import seaborn
        import sklearn
        import tqdm
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False
    
    return True


def run_evaluation():
    """Run evaluation for all datasets"""
    logger.info("\n" + "=" * 80)
    logger.info("Starting evaluation experiment")
    logger.info("=" * 80)
    
    rates_str = ",".join(map(str, RATE_POINTS))
    
    for dataset in DATASETS:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating dataset: {dataset.upper()}")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Sample count: {SAMPLES_PER_EMOTION}/emotion")
        logger.info(f"  Rate points: {len(RATE_POINTS)} ({RATE_POINTS[0]}-{RATE_POINTS[-1]} BPF)")
        
        cmd = [
            sys.executable, 'run_evaluation.py',
            '--dataset', dataset,
            '--samples', str(SAMPLES_PER_EMOTION),
            '--rates', rates_str,
            '--data-root', DATA_ROOT,  # Use subset data
            '--output-dir', OUTPUT_DIR,
            '--rvq-checkpoint', RVQ_CHECKPOINT,
            '--entropy-checkpoint', ENTROPY_CHECKPOINT
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            logger.info(f"✅ {dataset.upper()} evaluation complete")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {dataset.upper()} evaluation failed: {e}")
            return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ All dataset evaluations complete")
    logger.info("=" * 80)
    return True


def run_plotting():
    """Generate all paper figures using main plotting script"""
    logger.info("\n" + "=" * 80)
    logger.info("Generating paper figures")
    logger.info("=" * 80)
    
    # Ensure output directory exists
    Path(PLOT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating 5 key paper figures...")
    
    cmd = [sys.executable, 'generate_paper_figures.py']
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"✅ All figures generated successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Figure generation failed: {e}")
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ All figures generation complete")
    logger.info("=" * 80)
    logger.info(f"Figures saved to: {PLOT_OUTPUT_DIR}")
    return True


def summarize_results():
    """Summarize experiment results"""
    logger.info("\n" + "=" * 80)
    logger.info("Experiment Results Summary")
    logger.info("=" * 80)
    
    for dataset in DATASETS:
        result_file = Path(OUTPUT_DIR) / f"{dataset.upper()}_evaluation_results.json"
        
        if not result_file.exists():
            logger.warning(f"⚠️  {dataset.upper()}: Result file does not exist")
            continue
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"\n{dataset.upper()}:")
            logger.info(f"  Sample count: {data.get('samples', 'N/A')}")
            logger.info(f"  Rate points: {len(data.get('rate_points', {}))} points")
            
            # Find highest and lowest accuracy points
            rate_points = data.get('rate_points', {})
            if rate_points:
                accuracies = [(rp['target_rate_bpf'], rp['accuracy']) 
                             for rp in rate_points.values()]
                accuracies.sort(key=lambda x: x[1])
                
                min_acc = accuracies[0]
                max_acc = accuracies[-1]
                
                logger.info(f"  Accuracy range: {min_acc[1]*100:.2f}% (@{min_acc[0]}BPF) - {max_acc[1]*100:.2f}% (@{max_acc[0]}BPF)")
        
        except Exception as e:
            logger.error(f"❌ Failed to read {dataset.upper()} results: {e}")
    
    # List generated figures
    logger.info(f"\nGenerated figures:")
    if Path(PLOT_OUTPUT_DIR).exists():
        plot_files = list(Path(PLOT_OUTPUT_DIR).glob("*.png"))
        for pf in sorted(plot_files):
            logger.info(f"  - {pf.name}")
    
    logger.info("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Reproduce paper experiments')
    parser.add_argument('--mode', type=str, choices=['all', 'eval', 'plot'], default='all',
                       help='Run mode: all=evaluation+plotting, eval=only evaluation, plot=only plotting')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Paper Experiment Reproduction Script")
    logger.info("=" * 80)
    logger.info(f"Run mode: {args.mode}")
    logger.info(f"Datasets: {', '.join([d.upper() for d in DATASETS])}")
    logger.info(f"Rate points: {len(RATE_POINTS)} points ({RATE_POINTS[0]}-{RATE_POINTS[-1]} BPF)")
    logger.info(f"Sample count: {SAMPLES_PER_EMOTION}/emotion")
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("❌ Prerequisite check failed, please install dependencies and prepare data first")
        sys.exit(1)
    
    # Execute corresponding mode
    success = True
    
    if args.mode in ['all', 'eval']:
        success = run_evaluation()
        if not success:
            logger.error("❌ Evaluation failed")
            sys.exit(1)
    
    if args.mode in ['all', 'plot']:
        success = run_plotting()
        if not success:
            logger.error("❌ Plotting failed")
            sys.exit(1)
    
    # Summarize results
    summarize_results()
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 Experiment reproduction complete!")
    logger.info("=" * 80)
    logger.info(f"Evaluation results: {OUTPUT_DIR}/")
    logger.info(f"Paper figures: {PLOT_OUTPUT_DIR}/")


if __name__ == '__main__':
    main()

