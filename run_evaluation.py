"""
Unified evaluation script - Support all datasets
Usage:
    python run_evaluation.py --dataset iemocap
    python run_evaluation.py --dataset ravdess --samples 50
    python run_evaluation.py --dataset esd --rates "10,30,50,100"
"""

import torch
import logging
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# Project is now self-contained, no need to reference outer Amphion code

from config import get_default_config
from grouped_rvq import GroupedResidualVQ
from entropy_model import create_entropy_model

eval_module_path = Path(__file__).parent / 'evaluation'
if str(eval_module_path) not in sys.path:
    sys.path.insert(0, str(eval_module_path))

from emotion_classifier import EmotionClassifierV2
from method_rate_sweep import rate_sweep_evaluation
# analyzer lazy import (avoid matplotlib dependency issues)
# from analyzer import ResultAnalyzer

dataset_module_path = Path(__file__).parent / 'datasets'
if str(dataset_module_path) not in sys.path:
    sys.path.insert(0, str(dataset_module_path))

from iemocap_dataset import IEMOCAPDataset
from ravdess_dataset import RAVDESSDataset
from esd_dataset import ESDDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASETS = {
    'iemocap': IEMOCAPDataset,
    'ravdess': RAVDESSDataset,
    'esd': ESDDataset,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate emotion recognition')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys(),
                       help='Dataset name (iemocap/ravdess/esd)')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples per emotion (default: 100)')
    parser.add_argument('--rates', type=str, default=None,
                       help='Rate points (comma separated, e.g. "10,30,50,100")')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory (default: evaluation_results)')
    parser.add_argument('--rvq-checkpoint', type=str, default='checkpoints/grouped_rvq_best.pt',
                       help='RVQ model path')
    parser.add_argument('--entropy-checkpoint', type=str, default='checkpoints/entropy_model_best.pt',
                       help='Entropy model path')
    parser.add_argument('--data-root', type=str, default='data',
                       help='Data root directory (default: data)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("="*80)
    logger.info(f"{args.dataset.upper()} Evaluation ({args.samples} samples/emotion)")
    logger.info("="*80)
    
    config = get_default_config()
    
    # Override configuration
    if args.rates:
        rate_list = [float(r.strip()) for r in args.rates.split(',')]
        config['evaluation'].rate_sweep_rates_bpf = rate_list
    elif config['evaluation'].rate_sweep_rates_bpf is None:
        config['evaluation'].rate_sweep_rates_bpf = config['entropy_model'].target_bpf_grid
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Print full configuration info
    logger.info("\n" + "="*80)
    logger.info("Configuration")
    logger.info("="*80)
    logger.info(f"Lambda range: [{config['rate_control'].lambda_min}, {config['rate_control'].lambda_max}]")
    logger.info(f"Lambda initial value: {config['rate_control'].lambda_init}")
    logger.info(f"Binary search max iterations: {config['rate_control'].max_binary_search_iters}")
    logger.info(f"Rate tolerance: {config['rate_control'].rate_tolerance_bpf} bpf")
    logger.info(f"Target rates: {config['evaluation'].rate_sweep_rates_bpf}")
    logger.info(f"RVQ configuration: {config['grouped_rvq'].num_groups} groups × {config['grouped_rvq'].num_fine_layers} layers")
    logger.info(f"Using Full Ranking ECVQ: {config['grouped_rvq'].use_full_ranking_ecvq}")
    logger.info("="*80)
    
    # Load models
    logger.info("\n" + "="*80)
    logger.info("Loading models")
    logger.info("="*80)
    
    rvq_model = GroupedResidualVQ(config['grouped_rvq'])
    rvq_checkpoint = torch.load(args.rvq_checkpoint, map_location=device)
    rvq_model.load_state_dict(rvq_checkpoint['model_state_dict'])
    rvq_model = rvq_model.to(device)
    rvq_model.eval()
    logger.info(f"✓ RVQ model loaded: {args.rvq_checkpoint}")
    
    entropy_model = create_entropy_model(config['entropy_model'], config['grouped_rvq'])
    entropy_checkpoint = torch.load(args.entropy_checkpoint, map_location=device)
    entropy_model.load_state_dict(entropy_checkpoint['model_state_dict'])
    entropy_model = entropy_model.to(device)
    entropy_model.eval()
    logger.info(f"✓ Entropy model loaded: {args.entropy_checkpoint}")
    
    classifier = EmotionClassifierV2(
        model_name="iic/emotion2vec_plus_base",  # Use plus_base (768-dim) to match extracted features
        hub="modelscope",
        device=device
    )
    logger.info(f"✓ Classifier loaded: emotion2vec_plus_base (768-dim)")
    
    # Load dataset
    logger.info("\n" + "="*80)
    logger.info("Loading dataset")
    logger.info("="*80)
    
    DatasetClass = DATASETS[args.dataset]
    # Dataset path: project root directory/data/DATASET_NAME/
    project_root = Path(__file__).parent
    dataset_data_root = project_root / args.data_root / args.dataset.upper()
    dataset = DatasetClass(
        data_root=str(dataset_data_root),
        samples_per_emotion=args.samples
    )
    
    logger.info(f"✓ {args.dataset.upper()} dataset loaded")
    logger.info(f"  Dataset: {dataset.name}")
    logger.info(f"  Total samples: {len(dataset)}")
    
    # Run evaluation
    logger.info("\n" + "="*80)
    logger.info("Starting rate sweep evaluation")
    logger.info("="*80)
    
    results = rate_sweep_evaluation(
        rvq_model=rvq_model,
        entropy_model=entropy_model,
        dataset=dataset,
        target_rates_bpf=config['evaluation'].rate_sweep_rates_bpf,
        classifier=classifier,
        output_dir=str(Path(args.output_dir)),
        device=device
    )
    
    # Analyze results
    if results:
        logger.info("\n" + "="*80)
        logger.info("Analyzing results")
        logger.info("="*80)
        
        # analyzer = ResultAnalyzer(output_dir=Path(args.output_dir))
        # analyzer.analyze_rate_sweep(results, dataset.name)  # Temporarily commented, feature not implemented
        logger.info(f"✓ Evaluation results saved: {args.output_dir}/rate_sweep_{dataset.name}.json")
    
    logger.info("\n" + "="*80)
    logger.info("Evaluation complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()

