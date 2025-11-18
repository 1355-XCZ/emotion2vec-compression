"""
Method 2: Layer Sweep Evaluation
Evaluate emotion2vec classification performance by controlling the number of quantization layers used
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json
from typing import Dict, List, Optional
import sys

logger = logging.getLogger(__name__)


def layer_sweep_evaluation(
    rvq_model,
    dataset,  # EmotionDataset instance
    num_layers_list: List[int],
    classifier,  # EmotionClassifierV2 instance
    output_dir: str,
    device: str = 'cuda'
) -> Dict:
    """
    Layer sweep evaluation main function
    
    Similar to emotion_information_bottleneck, but uses grouped RVQ
    
    Args:
        rvq_model: trained GroupedRVQ model
        dataset: EmotionDataset instance
        num_layers_list: list of layer numbers to use (total layers = 12 groups × 3 layers = 36)
        classifier: emotion2vec classifier
        output_dir: output directory
        device: device
    
    Returns:
        results: evaluation results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rvq_model.eval()
    
    # Get emotion mapping
    emotion_mapping = dataset.get_emotion_mapping()
    
    logger.info(f"Starting layer sweep evaluation: {dataset.name}")
    logger.info(f"  Layer list: {num_layers_list}")
    logger.info(f"  Number of samples: {len(dataset)}")
    
    # Results storage
    results = {
        'dataset': dataset.name,
        'num_layers_list': num_layers_list,
        'emotion_mapping': emotion_mapping,
        'layer_points': {}
    }
    
    # Load samples
    if not dataset.samples:
        dataset.samples = dataset.load_samples()
    
    # Total layers
    total_layers = rvq_model.num_groups * rvq_model.config.num_fine_layers
    
    logger.info(f"Total samples: {len(dataset.samples)}")
    logger.info(f"Total layers: {total_layers}")
    
    # Evaluate for each layer number
    for num_layers in tqdm(num_layers_list, desc="Layer points"):
        if num_layers > total_layers:
            logger.warning(f"Layer number {num_layers} exceeds total layers {total_layers}, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Using layers: {num_layers}/{total_layers}")
        
        # Initialize results for this layer point
        layer_results = {
            'num_layers': num_layers,
            'predictions': [],
            'ground_truths': [],
            'confidences': [],
        }
        
        # Quantize and classify for each sample
        for sample in tqdm(dataset.samples, desc=f"Samples @ {num_layers} layers"):
            audio_path = sample['audio_path']
            emotion_original = sample['emotion']
            
            # Map emotion label
            emotion_ev2 = dataset.map_emotion_to_ev2(emotion_original)
            
            # Load features
            features_path = audio_path.replace('.wav', '_ev2_frame.npy')
            
            if not Path(features_path).exists():
                logger.warning(f"Features file does not exist, skipping: {features_path}")
                continue
            
            try:
                features = np.load(features_path)  # (T, 768)
                features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, T, 768)
                
                # Quantize with specified number of layers (not using ECVQ, direct quantization)
                with torch.no_grad():
                    # Disable ECVQ, direct reconstruction
                    quantized = quantize_with_num_layers(
                        rvq_model,
                        features_tensor,
                        num_layers,
                        device
                    )
                
                # Classify quantized features using emotion2vec
                quantized_np = quantized[0].cpu().numpy()  # (T, 768)
                
                # Call classifier
                result = classifier.classify_from_features(
                    quantized_np,
                    esd_ground_truth=emotion_ev2
                )
                
                # Record results
                layer_results['predictions'].append(result['ev2_predicted'])
                layer_results['ground_truths'].append(emotion_ev2)
                layer_results['confidences'].append(result['ev2_confidence'])
                
            except Exception as e:
                logger.error(f"Failed to process sample: {audio_path}, Error: {e}")
                continue
        
        # Calculate statistics for this layer point
        if len(layer_results['predictions']) > 0:
            predictions = np.array(layer_results['predictions'])
            ground_truths = np.array(layer_results['ground_truths'])
            confidences = np.array(layer_results['confidences'])
            
            accuracy = (predictions == ground_truths).mean()
            avg_confidence = confidences.mean()
            
            layer_results['accuracy'] = float(accuracy)
            layer_results['avg_confidence'] = float(avg_confidence)
            
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Average confidence: {avg_confidence:.4f}")
        
        # Save results for this layer point
        results['layer_points'][f'{num_layers}_layers'] = layer_results
    
    # Save complete results
    output_file = output_dir / f'layer_sweep_{dataset.name}.json'
    os.makedirs(output_file.parent, exist_ok=True)  # Ensure directory exists
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Layer sweep evaluation complete: {dataset.name}")
    logger.info(f"  Results saved: {output_file}")
    
    return results


def quantize_with_num_layers(rvq_model, features, num_layers, device):
    """
    Quantize with specified number of layers (helper function)
    
    Implementation:
    1. Use RVQ to quantize normally
    2. Only use first num_layers layers for reconstruction
    
    Args:
        rvq_model: GroupedRVQ model
        features: (B, T, 768) features
        num_layers: number of layers to use
        device: device
    
    Returns:
        quantized: (B, T, 768) quantized features
    """
    B, T, D = features.shape
    
    # Reshape to grouped form
    features_grouped = features.view(B, T, rvq_model.num_groups, rvq_model.group_dim)
    
    # Quantize per group (only use first num_layers_per_group layers)
    total_layers = rvq_model.num_groups * rvq_model.config.num_fine_layers
    layers_per_group = num_layers // rvq_model.num_groups
    remaining_layers = num_layers % rvq_model.num_groups
    
    reconstructed_grouped = torch.zeros_like(features_grouped)
    
    for g in range(rvq_model.num_groups):
        # Determine how many layers to use for this group
        if g < remaining_layers:
            group_layers = layers_per_group + 1
        else:
            group_layers = layers_per_group
        
        group_layers = min(group_layers, rvq_model.config.num_fine_layers)
        
        # Residual quantization
        residual = features_grouped[:, :, g, :]  # (B, T, group_dim)
        
        for m in range(group_layers):
            # Quantize
            vq = rvq_model.fine_vqs[g][m]
            quantized_layer, indices, commit_loss = vq(residual)
            
            # Accumulate
            reconstructed_grouped[:, :, g, :] += quantized_layer
            
            # Update residual
            residual = residual - quantized_layer
    
    # Reshape back to original form
    quantized = reconstructed_grouped.view(B, T, D)
    
    return quantized


