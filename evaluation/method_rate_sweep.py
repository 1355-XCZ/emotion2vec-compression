"""
Method 1: Rate Sweep Evaluation
Control bitrate through λ parameter, evaluate emotion2vec classification performance
"""

import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import logging
import json
from typing import Dict, List, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rate_controller import RateController

logger = logging.getLogger(__name__)


def rate_sweep_evaluation(
    rvq_model,
    entropy_model,
    dataset,  # EmotionDataset instance
    target_rates_bpf: List[float],
    classifier,  # EmotionClassifierV2 instance
    output_dir: str,
    device: str = 'cuda',
    frame_rate_hz: float = 50.0,
    batch_size: int = 8,  # ChatGPT suggestion: batch processing
    num_workers: int = 4  # ChatGPT suggestion: multi-process I/O
) -> Dict:
    """
    Rate sweep evaluation main function
    
    Args:
        rvq_model: trained GroupedRVQ model
        entropy_model: trained unconditional entropy model q(z)
        dataset: EmotionDataset instance (IEMOCAP/RAVDESS/ESD)
        target_rates_bpf: target bitrate list (bits per frame)
        classifier: emotion2vec classifier
        output_dir: output directory
        device: device
        frame_rate_hz: frame rate
    
    Returns:
        results: evaluation results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rvq_model.eval()
    entropy_model.eval()
    
    # Get emotion mapping
    emotion_mapping = dataset.get_emotion_mapping()
    
    logger.info(f"Starting rate sweep evaluation: {dataset.name}")
    logger.info(f"  Target rates: {target_rates_bpf} bpf")
    logger.info(f"  Number of samples: {len(dataset)}")
    logger.info(f"  Emotion mapping: {emotion_mapping}")
    
    # Results storage
    results = {
        'dataset': dataset.name,
        'target_rates_bpf': target_rates_bpf,
        'emotion_mapping': emotion_mapping,
        'ev2_emotion_labels': classifier.ev2_emotions,  # emotion2vec 9 category labels (for confusion matrix)
        'samples': {},
        'rate_points': {}
    }
    
    # Rate controller configuration
    from config import RateControlConfig
    rate_config = RateControlConfig()
    rate_controller = RateController(rate_config)
    
    # Load samples
    if not dataset.samples:
        dataset.samples = dataset.load_samples()
    
    logger.info(f"Total samples: {len(dataset.samples)}")
    
    # Evaluate for each target rate
    for target_rate_bpf in tqdm(target_rates_bpf, desc="Rate points"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Target rate: {target_rate_bpf} bpf ({target_rate_bpf * frame_rate_hz:.1f} bps)")
        
        # ⭐ Reset rate_controller, ensure each rate point starts from lambda_init
        rate_controller.reset()
        
        # Initialize results for this rate point
        rate_results = {
            'target_rate_bpf': target_rate_bpf,
            # ===== Aggregated statistics =====
            'accuracy': None,  # Calculate later
            'avg_confidence': None,  # Calculate later
            'avg_rate_bpf': None,  # Calculate later
            'num_samples': 0,
            # ===== Complete information for each sample (list form, ordered correspondence) =====
            'samples': []  # Complete information dictionary for each sample
        }
        
        # Encode and classify for each sample
        for idx, sample in enumerate(tqdm(dataset.samples, desc=f"Samples @ {target_rate_bpf} bpf")):
            audio_path = sample['audio_path']
            emotion_original = sample['emotion']
            
            # Map emotion label to emotion2vec
            emotion_ev2 = dataset.map_emotion_to_ev2(emotion_original)
            
            # Load or extract emotion2vec features
            # TODO: Here we assume pre-extracted features, actual implementation needs audio extraction
            features_path = audio_path.replace('.wav', '_ev2_frame.npy')
            
            if not Path(features_path).exists():
                # If no pre-extracted features, extract from audio
                logger.warning(f"Features file does not exist, skipping: {features_path}")
                continue
            
            try:
                features = np.load(features_path)  # (T, 768)
                features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, T, 768)
                valid_mask = torch.ones(1, features_tensor.size(1), dtype=torch.bool, device=device)
                
                # Search for λ independently for each sample (ensure reaching target rate)
                def encoder_fn(lambda_val):
                    with torch.inference_mode():
                        _, indices, _, _ = rvq_model(
                            features_tensor,
                            lambda_rate=torch.tensor(lambda_val, device=device),
                            entropy_model=entropy_model,
                            valid_mask=valid_mask
                        )
                        bits = entropy_model.compute_bits(indices, valid_mask=valid_mask).item()
                        num_frames = valid_mask.sum().item()
                        actual_rate_bpf = bits / num_frames if num_frames > 0 else 0.0
                        return indices, actual_rate_bpf
                
                # ⭐ Completely disable hint, independent full-range search for each sample
                sample_lambda, sample_rate_bpf = rate_controller.binary_search(
                    encoder_fn,
                    target_rate_bpf,
                    tolerance_bpf=rate_config.rate_tolerance_bpf,
                    lambda_hint=None  # Force full-range search
                )
                
                # Record first sample result (for logging only)
                if idx == 0:
                    logger.info(f"  First sample: λ={sample_lambda:.4f}, R={sample_rate_bpf:.2f} bpf")
                
                # Quantize using found λ
                with torch.inference_mode():
                    quantized, indices, _, stats = rvq_model(
                        features_tensor,
                        lambda_rate=torch.tensor(sample_lambda, device=device),
                        entropy_model=entropy_model,
                        valid_mask=valid_mask
                    )
                    
                    # Calculate actual bitrate
                    bits = entropy_model.compute_bits(indices, valid_mask=valid_mask).item()
                    num_frames = valid_mask.sum().item()
                    actual_rate_bpf = bits / num_frames
                
                # Classify quantized features using emotion2vec
                quantized_np = quantized[0].cpu().numpy()  # (T, 768)
                
                # Call classifier
                result = classifier.classify_from_features(
                    quantized_np,
                    esd_ground_truth=emotion_ev2
                )
                
                # ===== Record complete sample information =====
                sample_info = {
                    # Sample identification
                    'sample_id': idx,
                    'audio_path': str(audio_path),
                    'original_emotion': emotion_original,  # Dataset native label
                    
                    # Encoding information
                    'lambda': float(sample_lambda),
                    'achieved_rate_bpf': float(actual_rate_bpf),
                    'bits_total': float(bits),
                    'num_frames': int(num_frames),
                    'skip_rate': float(stats.get('skip_rate', 0.0)),
                    
                    # Classification information
                    'ground_truth': emotion_ev2,  # Mapped emotion2vec label
                    'prediction': result['ev2_predicted'],
                    'confidence': float(result['ev2_confidence']),
                    'is_correct': (result['ev2_predicted'] == emotion_ev2),
                    
                    # Complete classification probabilities (all 9 classes)
                    'all_class_probs': result.get('ev2_probs', []).tolist() if hasattr(result.get('ev2_probs', []), 'tolist') else list(result.get('ev2_probs', [])),
                    'class_labels': result.get('ev2_all_labels', classifier.ev2_emotions),
                    
                    # Mapping information (if needed)
                    'esd_mapped': result.get('esd_mapped'),
                    'is_consistent_with_esd': result.get('is_consistent', False)
                }
                
                rate_results['samples'].append(sample_info)
                
            except Exception as e:
                logger.error(f"Failed to process sample: {audio_path}, Error: {e}")
                continue
        
        # Calculate statistics for this rate point
        if len(rate_results['samples']) > 0:
            # Extract data from samples for statistics calculation
            predictions = [s['prediction'] for s in rate_results['samples']]
            ground_truths = [s['ground_truth'] for s in rate_results['samples']]
            confidences = [s['confidence'] for s in rate_results['samples']]
            achieved_rates = [s['achieved_rate_bpf'] for s in rate_results['samples']]
            
            accuracy = sum(1 for s in rate_results['samples'] if s['is_correct']) / len(rate_results['samples'])
            avg_confidence = np.mean(confidences)
            avg_rate_bpf = np.mean(achieved_rates)
            
            rate_results['accuracy'] = float(accuracy)
            rate_results['avg_confidence'] = float(avg_confidence)
            rate_results['avg_rate_bpf'] = float(avg_rate_bpf)
            rate_results['num_samples'] = len(rate_results['samples'])
            
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Average confidence: {avg_confidence:.4f}")
            logger.info(f"  Average rate: {avg_rate_bpf:.2f} bpf")
        
        # Save results for this rate point
        rate_key = "original" if target_rate_bpf == float('inf') else f'{target_rate_bpf}_bpf'
        results['rate_points'][rate_key] = rate_results
    
    # Save complete results
    output_file = output_dir / f'rate_sweep_{dataset.name}.json'
    os.makedirs(output_file.parent, exist_ok=True)  # Ensure directory exists
    
    # Handle inf values: replace with "original" in JSON
    results_serializable = results.copy()
    results_serializable['target_rates_bpf'] = [
        "original" if rate == float('inf') else rate 
        for rate in target_rates_bpf
    ]
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"\n✅ Rate sweep evaluation complete: {dataset.name}")
    logger.info(f"  Results saved: {output_file}")
    
    return results

