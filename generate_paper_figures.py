#!/usr/bin/env python3
"""
Generate 5 key figures required for paper

Usage:
    python generate_paper_figures.py

Generated figures:
    1. Overall Weighted F1 Score Comparison (3 datasets comparison)
    2. Overall Model Confidence Comparison (3 datasets comparison)
    3. Per-class Accuracy by Emotion (3 datasets vertical layout)
    4. Model Confidence by Emotion (3 datasets vertical layout)
    5. Confusion Matrices 3×4 Grid (3 datasets × 4 rate points)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem, t
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import defaultdict
import sys

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# Unified color scheme
DATASET_COLORS = {
    'ESD': '#E53935',      # Red
    'IEMOCAP': '#1E88E5',  # Blue
    'RAVDESS': '#43A047'   # Green
}

EMOTION_COLORS = {
    'angry': '#d32f2f',
    'happy': '#ffa726',
    'neutral': '#9e9e9e',
    'sad': '#42a5f5',
    'surprised': '#ab47bc',
    'disgusted': '#8d6e63',
    'fearful': '#66bb6a',
    'calm': '#26c6da',
    'excited': '#ffee58'
}

# Dataset configuration
DATASETS = ['ESD', 'IEMOCAP', 'RAVDESS']
DATA_DIR = Path('evaluation_results')
OUTPUT_DIR = Path('evaluation_results/figures')

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_results(dataset_name):
    """Load dataset evaluation results"""
    json_file = DATA_DIR / f"{dataset_name}_evaluation_results.json"
    if not json_file.exists():
        print(f"⚠️  Could not find {dataset_name} evaluation results: {json_file}")
        return None
    
    with open(json_file, 'r') as f:
        return json.load(f)


def calculate_metrics_with_ci(data, metric_key='accuracy', confidence_level=0.95):
    """Calculate metrics and confidence intervals"""
    results = {'rates': [], 'values': [], 'lower_bounds': [], 'upper_bounds': []}
    
    for rate_key, rate_point in sorted(data['rate_points'].items(), 
                                       key=lambda x: x[1]['target_rate_bpf']):
        rate = rate_point['target_rate_bpf']
        samples = rate_point['samples']
        
        if metric_key == 'accuracy':
            values = [1.0 if s['is_correct'] else 0.0 for s in samples]
        elif metric_key == 'confidence':
            values = [s['confidence'] for s in samples]
        elif metric_key == 'f1':
            # Get F1 score from rate_point
            values = [rate_point.get('f1_score', rate_point.get('accuracy', 0))]
        else:
            continue
        
        if not values:
            continue
        
        mean_val = np.mean(values)
        if len(values) > 1:
            std_error = sem(values)
            df = len(values) - 1
            t_value = t.ppf((1 + confidence_level) / 2, df)
            margin = t_value * std_error
            lower = max(0, mean_val - margin)
            upper = min(1, mean_val + margin)
        else:
            lower = upper = mean_val
        
        results['rates'].append(rate)
        results['values'].append(mean_val * 100)
        results['lower_bounds'].append(lower * 100)
        results['upper_bounds'].append(upper * 100)
    
    return results


def figure1_weighted_f1_comparison():
    """Figure 1: Overall Weighted F1 Score Comparison"""
    print("\n" + "="*70)
    print("Generating Figure 1: Overall Weighted F1 Score Comparison")
    print("="*70)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for dataset in DATASETS:
        data = load_dataset_results(dataset)
        if not data:
            continue
        
        results = calculate_metrics_with_ci(data, 'f1')
        
        color = DATASET_COLORS[dataset]
        ax.plot(results['rates'], results['values'], 
               label=dataset, color=color, linewidth=2.5, marker='o', markersize=4)
        ax.fill_between(results['rates'], results['lower_bounds'], results['upper_bounds'],
                       alpha=0.2, color=color, label=f'{dataset} 95% CI')
    
    ax.set_xlabel('Bitrate (bpf)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Weighted F1 Score (%)', fontsize=16, fontweight='bold')
    ax.set_title('Overall Weighted F1 Score Comparison with 95% Confidence Intervals',
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / '1_Overall_Weighted_F1_Comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def figure2_model_confidence_comparison():
    """Figure 2: Overall Model Confidence Comparison"""
    print("\n" + "="*70)
    print("Generating Figure 2: Overall Model Confidence Comparison")
    print("="*70)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for dataset in DATASETS:
        data = load_dataset_results(dataset)
        if not data:
            continue
        
        results = calculate_metrics_with_ci(data, 'confidence')
        
        color = DATASET_COLORS[dataset]
        ax.plot(results['rates'], results['values'],
               label=dataset, color=color, linewidth=2.5, marker='o', markersize=4)
        ax.fill_between(results['rates'], results['lower_bounds'], results['upper_bounds'],
                       alpha=0.2, color=color, label=f'{dataset} 95% CI')
    
    ax.set_xlabel('Bitrate (bpf)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Overall Model Confidence (%)', fontsize=16, fontweight='bold')
    ax.set_title('Overall Model Confidence Comparison with 95% Confidence Intervals\n(Average probability assigned to true emotion label)',
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / '2_Overall_Model_Confidence_Comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def calculate_emotion_accuracy_with_ci(data, confidence_level=0.95):
    """Calculate accuracy and confidence intervals for each emotion"""
    emotion_labels = data.get('ev2_emotion_labels', [])
    target_rates = sorted(set(rp['target_rate_bpf'] for rp in data['rate_points'].values()))
    
    emotion_stats = {emotion: {rate: {'correct': 0, 'total': 0} 
                              for rate in target_rates} 
                    for emotion in emotion_labels}
    
    for rate_point in data['rate_points'].values():
        rate = rate_point['target_rate_bpf']
        for sample in rate_point['samples']:
            true_emotion = sample.get('ground_truth') or sample.get('original_emotion')
            pred_emotion = sample.get('prediction') or sample.get('predicted_emotion')
            
            if true_emotion in emotion_stats:
                emotion_stats[true_emotion][rate]['total'] += 1
                if true_emotion == pred_emotion:
                    emotion_stats[true_emotion][rate]['correct'] += 1
    
    results = {}
    for emotion in emotion_labels:
        results[emotion] = {'rates': [], 'accuracy': [], 'lower': [], 'upper': []}
        for rate in target_rates:
            stats = emotion_stats[emotion][rate]
            if stats['total'] > 0:
                correct_list = [1] * stats['correct'] + [0] * (stats['total'] - stats['correct'])
                accuracy = stats['correct'] / stats['total']
                
                if len(correct_list) > 1:
                    std_error = sem(correct_list)
                    df = len(correct_list) - 1
                    t_value = t.ppf((1 + confidence_level) / 2, df)
                    margin = t_value * std_error
                    lower = max(0, accuracy - margin)
                    upper = min(1, accuracy + margin)
                else:
                    lower = upper = accuracy
                
                results[emotion]['rates'].append(rate)
                results[emotion]['accuracy'].append(accuracy * 100)
                results[emotion]['lower'].append(lower * 100)
                results[emotion]['upper'].append(upper * 100)
    
    return results, emotion_labels


def figure3_emotion_accuracy_vertical():
    """Figure 3: Per-class Accuracy by Emotion (Vertical Layout)"""
    print("\n" + "="*70)
    print("Generating Figure 3: Per-class Accuracy by Emotion (Vertical Layout)")
    print("="*70)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
    
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        data = load_dataset_results(dataset)
        if not data:
            ax.text(0.5, 0.5, f'No data for {dataset}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        results, emotion_labels = calculate_emotion_accuracy_with_ci(data)
        
        for emotion in emotion_labels:
            if emotion not in results or not results[emotion]['rates']:
                continue
            color = EMOTION_COLORS.get(emotion.lower(), '#666666')
            ax.plot(results[emotion]['rates'], results[emotion]['accuracy'],
                   label=emotion, color=color, linewidth=2, marker='o', markersize=3)
            ax.fill_between(results[emotion]['rates'], 
                           results[emotion]['lower'], results[emotion]['upper'],
                           alpha=0.15, color=color)
        
        ax.set_ylabel(f'{dataset}\nPer-class Accuracy (%)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)
    
    axes[-1].set_xlabel('Bitrate (bits per frame)', fontsize=14, fontweight='bold')
    fig.suptitle('Per-class Accuracy (Recall) by Emotion across Datasets (with 95% CI)',
                fontsize=20, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    output_file = OUTPUT_DIR / '3_Emotion_Accuracy_Vertical.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def calculate_emotion_confidence_with_ci(data, confidence_level=0.95):
    """Calculate model confidence for each emotion"""
    emotion_labels = data.get('ev2_emotion_labels', [])
    target_rates = sorted(set(rp['target_rate_bpf'] for rp in data['rate_points'].values()))
    
    emotion_confidences = {emotion: {rate: [] for rate in target_rates} 
                          for emotion in emotion_labels}
    
    for rate_point in data['rate_points'].values():
        rate = rate_point['target_rate_bpf']
        for sample in rate_point['samples']:
            true_emotion = sample.get('ground_truth') or sample.get('original_emotion')
            all_probs = sample.get('all_class_probs', [])
            class_labels = sample.get('class_labels', emotion_labels)
            
            if true_emotion and all_probs and true_emotion in class_labels:
                emotion_idx = class_labels.index(true_emotion)
                if emotion_idx < len(all_probs):
                    confidence = all_probs[emotion_idx]
                    if true_emotion in emotion_confidences:
                        emotion_confidences[true_emotion][rate].append(confidence)
    
    results = {}
    for emotion in emotion_labels:
        results[emotion] = {'rates': [], 'confidence': [], 'lower': [], 'upper': []}
        for rate in target_rates:
            conf_list = emotion_confidences[emotion][rate]
            if conf_list:
                mean_conf = np.mean(conf_list)
                
                if len(conf_list) > 1:
                    std_error = sem(conf_list)
                    df = len(conf_list) - 1
                    t_value = t.ppf((1 + confidence_level) / 2, df)
                    margin = t_value * std_error
                    lower = max(0, mean_conf - margin)
                    upper = min(1, mean_conf + margin)
                else:
                    lower = upper = mean_conf
                
                results[emotion]['rates'].append(rate)
                results[emotion]['confidence'].append(mean_conf * 100)
                results[emotion]['lower'].append(lower * 100)
                results[emotion]['upper'].append(upper * 100)
    
    return results, emotion_labels


def figure4_emotion_confidence_vertical():
    """Figure 4: Model Confidence by Emotion (Vertical Layout)"""
    print("\n" + "="*70)
    print("Generating Figure 4: Model Confidence by Emotion (Vertical Layout)")
    print("="*70)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
    
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        data = load_dataset_results(dataset)
        if not data:
            ax.text(0.5, 0.5, f'No data for {dataset}',
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        results, emotion_labels = calculate_emotion_confidence_with_ci(data)
        
        for emotion in emotion_labels:
            if emotion not in results or not results[emotion]['rates']:
                continue
            color = EMOTION_COLORS.get(emotion.lower(), '#666666')
            ax.plot(results[emotion]['rates'], results[emotion]['confidence'],
                   label=emotion, color=color, linewidth=2, marker='o', markersize=3)
            ax.fill_between(results[emotion]['rates'],
                           results[emotion]['lower'], results[emotion]['upper'],
                           alpha=0.15, color=color)
        
        ax.set_ylabel(f'{dataset}\nModel Confidence (%)',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)
    
    axes[-1].set_xlabel('Bitrate (bits per frame)', fontsize=14, fontweight='bold')
    fig.suptitle('Model Confidence by Emotion across Datasets (with 95% CI)\n(Probability assigned to true emotion label)',
                fontsize=20, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    output_file = OUTPUT_DIR / '4_Emotion_Confidence_Vertical.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def get_confusion_matrix_at_rate(data, target_bpf):
    """Get confusion matrix at specified bitrate point"""
    for rate_key, rate_point in data['rate_points'].items():
        if abs(rate_point['target_rate_bpf'] - target_bpf) < 0.1:
            samples = rate_point['samples']
            y_true = [s.get('ground_truth') or s.get('original_emotion') for s in samples]
            y_pred = [s.get('prediction') or s.get('predicted_emotion') for s in samples]
            
            emotion_labels = data.get('ev2_emotion_labels', sorted(set(y_true)))
            cm = confusion_matrix(y_true, y_pred, labels=emotion_labels)
            accuracy = rate_point.get('accuracy', 0)
            
            return cm, emotion_labels, accuracy
    
    return None, None, None


def format_value(value):
    """Format values in confusion matrix"""
    return "0" if value == 0 else f"{value:.2f}"


def figure5_confusion_matrices_3x4():
    """Figure 5: Confusion Matrices 3×4 Grid"""
    print("\n" + "="*70)
    print("Generating Figure 5: Confusion Matrices 3×4 Grid")
    print("="*70)
    
    RATES = [10, 20, 100, 200]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    im_for_colorbar = None
    
    for row_idx, dataset in enumerate(DATASETS):
        data = load_dataset_results(dataset)
        
        for col_idx, rate in enumerate(RATES):
            ax = axes[row_idx, col_idx]
            
            if data is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            cm, labels, accuracy = get_confusion_matrix_at_rate(data, rate)
            
            if cm is None:
                ax.text(0.5, 0.5, f'No data\nat {rate} BPF',
                       ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Row normalization
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            # Draw heatmap
            im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            if im_for_colorbar is None:
                im_for_colorbar = im
            
            # Add text annotations
            for i in range(len(labels)):
                for j in range(len(labels)):
                    value = cm_normalized[i, j]
                    text_color = 'white' if value > 0.5 else 'black'
                    ax.text(j, i, format_value(value),
                           ha='center', va='center', color=text_color,
                           fontsize=9, fontweight='bold')
            
            # Set labels
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            
            if row_idx == 2:  # Last row
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            else:
                ax.set_xticklabels([])
            
            if col_idx == 0:  # First column
                ax.set_yticklabels(labels, fontsize=10)
            else:
                ax.set_yticklabels([])
            
            # Title
            title = f"{rate} BPF"
            if accuracy is not None:
                title += f"\n(Acc: {accuracy*100:.1f}%)"
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # Row labels
        axes[row_idx, 0].set_ylabel(f'{dataset}\n\nTrue Label',
                                     fontsize=13, fontweight='bold', labelpad=20)
    
    # Column labels
    for col_idx in range(4):
        axes[2, col_idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold', labelpad=10)
    
    fig.suptitle('Confusion Matrices across Datasets and Bitrates\n(Row-normalized, showing Recall)',
                fontsize=18, fontweight='bold', y=0.995)
    
    # Add colorbar
    if im_for_colorbar is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im_for_colorbar, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Recall (Row-normalized)', fontsize=12, fontweight='bold',
                      rotation=270, labelpad=20)
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.98])
    output_file = OUTPUT_DIR / '5_Confusion_Matrices_3x4.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def main():
    print("="*70)
    print("Generate 5 key figures for paper")
    print("="*70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Check data files
    missing_datasets = []
    for dataset in DATASETS:
        json_file = DATA_DIR / f"{dataset}_evaluation_results.json"
        if not json_file.exists():
            missing_datasets.append(dataset)
    
    if missing_datasets:
        print(f"⚠️  Warning: Missing evaluation results for datasets: {', '.join(missing_datasets)}")
        print(f"   Skipping plotting for these datasets")
        print()
    
    # Generate 5 figures
    try:
        figure1_weighted_f1_comparison()
        figure2_model_confidence_comparison()
        figure3_emotion_accuracy_vertical()
        figure4_emotion_confidence_vertical()
        figure5_confusion_matrices_3x4()
        
        print("\n" + "="*70)
        print("✅ All 5 figures generated successfully!")
        print("="*70)
        print(f"\nFigures saved to: {OUTPUT_DIR}/")
        print("\nGenerated files:")
        for i in range(1, 6):
            files = list(OUTPUT_DIR.glob(f"{i}_*.png"))
            if files:
                print(f"  {i}. {files[0].name}")
        
    except Exception as e:
        print(f"\n❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

