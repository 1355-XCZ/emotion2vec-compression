"""
Result analyzer
Generate visualization charts and statistical reports
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Result analyzer"""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def plot_rate_vs_accuracy(self, results_dict: Dict, save_name: str = 'rate_vs_accuracy.png'):
        """
        Plot accuracy vs bitrate curve (by emotion)
        
        Args:
            results_dict: multiple datasets result dictionary
            save_name: save file name
        """
        for dataset_name, results in results_dict.items():
            # Extract rate point data and sort
            rate_data = []
            for key, rate_point in results['rate_points'].items():
                if 'target_rate_bpf' in rate_point and len(rate_point.get('predictions', [])) > 0:
                    target_rate = rate_point['target_rate_bpf']
                    # Skip inf/original (no quantization baseline)
                    if target_rate != float('inf') and key != "original":
                        rate_data.append({
                            'rate_bpf': target_rate,  # Use target rate, not average rate
                            'predictions': rate_point['predictions'],
                            'ground_truths': rate_point['ground_truths']
                        })
            
            # Sort by rate
            rate_data.sort(key=lambda x: x['rate_bpf'])
            
            if not rate_data:
                continue
            
            # Calculate accuracy per emotion
            emotion_mapping = results.get('emotion_mapping', {})
            all_emotions = set()
            for gt in rate_data[0]['ground_truths']:
                all_emotions.add(gt)
            all_emotions = sorted(all_emotions)
            
            # Create chart
            fig, ax = plt.subplots(figsize=(12, 7))
            
            for emotion in all_emotions:
                rates = []
                accuracies = []
                
                for rd in rate_data:
                    # Calculate accuracy for this emotion
                    emotion_preds = [p for p, g in zip(rd['predictions'], rd['ground_truths']) if g == emotion]
                    emotion_gts = [g for g in rd['ground_truths'] if g == emotion]
                    
                    if len(emotion_gts) > 0:
                        correct = sum([1 for p, g in zip(emotion_preds, emotion_gts) if p == g])
                        acc = correct / len(emotion_gts)
                        rates.append(rd['rate_bpf'])
                        accuracies.append(acc)
                
                if rates:
                    ax.plot(rates, accuracies, marker='o', label=emotion, linewidth=2, markersize=6)
            
            ax.set_xlabel('Rate (BPF)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
            ax.set_title(f'{dataset_name}: Accuracy vs Rate (Per-Emotion)', fontsize=15, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
            
            save_path = self.output_dir / f'{dataset_name}_rate_vs_accuracy.png'
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved chart: {save_path}")
    
    def plot_rate_vs_confidence(self, results_dict: Dict, save_name: str = 'rate_vs_confidence.png'):
        """
        Plot confidence vs bitrate curve
        
        Args:
            results_dict: multiple datasets result dictionary
            save_name: save file name
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for dataset_name, results in results_dict.items():
            rates = []
            confidences = []
            
            for key, rate_point in results['rate_points'].items():
                if 'avg_rate_bpf' in rate_point and 'avg_confidence' in rate_point:
                    rates.append(rate_point['avg_rate_bpf'])
                    confidences.append(rate_point['avg_confidence'])
            
            if rates:
                ax.plot(rates, confidences, marker='s', label=dataset_name, linewidth=2)
        
        ax.set_xlabel('Rate (bits/frame)', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title('Confidence vs Rate', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"✅ Saved chart: {save_path}")
    
    def plot_layer_vs_accuracy(self, results_dict: Dict, save_name: str = 'layer_vs_accuracy.png'):
        """
        Plot accuracy vs layer curve
        
        Args:
            results_dict: multiple datasets result dictionary
            save_name: save file name
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for dataset_name, results in results_dict.items():
            # Extract layer point data and sort
            layer_data = []
            for key, layer_point in results['layer_points'].items():
                if 'num_layers' in layer_point and len(layer_point.get('predictions', [])) > 0:
                    layer_data.append({
                        'num_layers': layer_point['num_layers'],
                        'predictions': layer_point['predictions'],
                        'ground_truths': layer_point['ground_truths']
                    })
            
            # Sort by layer
            layer_data.sort(key=lambda x: x['num_layers'])
            
            if not layer_data:
                continue
            
            # Extract all emotion categories
            all_emotions = set()
            for gt in layer_data[0]['ground_truths']:
                all_emotions.add(gt)
            all_emotions = sorted(all_emotions)
            
            # Create chart
            fig, ax = plt.subplots(figsize=(12, 7))
            
            for emotion in all_emotions:
                layers = []
                accuracies = []
                
                for ld in layer_data:
                    # Calculate accuracy for this emotion
                    emotion_preds = [p for p, g in zip(ld['predictions'], ld['ground_truths']) if g == emotion]
                    emotion_gts = [g for g in ld['ground_truths'] if g == emotion]
                    
                    if len(emotion_gts) > 0:
                        correct = sum([1 for p, g in zip(emotion_preds, emotion_gts) if p == g])
                        acc = correct / len(emotion_gts)
                        layers.append(ld['num_layers'])
                        accuracies.append(acc)
                
                if layers:
                    ax.plot(layers, accuracies, marker='s', label=emotion, linewidth=2, markersize=6)
            
            ax.set_xlabel('Number of Layers (12 Groups × M Layers/Group)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
            ax.set_title(f'{dataset_name}: Accuracy vs Layers (Per-Emotion)', fontsize=15, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
            
            save_path = self.output_dir / f'{dataset_name}_layer_vs_accuracy.png'
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved chart: {save_path}")
    
    def plot_combined_comparison(self, rate_results: Dict, layer_results: Dict, 
                                save_name: str = 'combined_comparison.png'):
        """
        Plot comparison between rate method and layer method
        
        Args:
            rate_results: rate sweep results
            layer_results: layer sweep results
            save_name: save file name
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left chart: rate method
        for dataset_name, results in rate_results.items():
            rates = []
            accuracies = []
            
            for key, rate_point in results['rate_points'].items():
                if 'avg_rate_bpf' in rate_point and 'accuracy' in rate_point:
                    rates.append(rate_point['avg_rate_bpf'])
                    accuracies.append(rate_point['accuracy'])
            
            if rates:
                ax1.plot(rates, accuracies, marker='o', label=dataset_name, linewidth=2)
        
        ax1.set_xlabel('Rate (bits/frame)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Method 1: Rate Sweep', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right chart: layer method
        for dataset_name, results in layer_results.items():
            layers = []
            accuracies = []
            
            for key, layer_point in results['layer_points'].items():
                if 'num_layers' in layer_point and 'accuracy' in layer_point:
                    layers.append(layer_point['num_layers'])
                    accuracies.append(layer_point['accuracy'])
            
            if layers:
                sorted_indices = np.argsort(layers)
                layers = [layers[i] for i in sorted_indices]
                accuracies = [accuracies[i] for i in sorted_indices]
                
                ax2.plot(layers, accuracies, marker='s', label=dataset_name, linewidth=2)
        
        ax2.set_xlabel('Number of Layers', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Method 2: Layer Sweep', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"✅ Saved chart: {save_path}")
    
    def generate_summary_report(self, rate_results: Dict, layer_results: Dict, 
                               save_name: str = 'summary_report.txt'):
        """
        Generate text summary report
        
        Args:
            rate_results: rate sweep results
            layer_results: layer sweep results
            save_name: save file name
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("Emotion RVQ Information Bottleneck Experiment - Evaluation Summary Report")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Rate sweep results
        report_lines.append("Method 1: Rate Sweep")
        report_lines.append("-"*80)
        
        for dataset_name, results in rate_results.items():
            report_lines.append(f"\nDataset: {dataset_name}")
            report_lines.append(f"  Emotion mapping: {results['emotion_mapping']}")
            report_lines.append(f"\n  Rate Point | Accuracy | Confidence")
            report_lines.append(f"  " + "-"*40)
            
            for key, rate_point in sorted(results['rate_points'].items()):
                if 'accuracy' in rate_point:
                    rate_bpf = rate_point.get('avg_rate_bpf', 0)
                    acc = rate_point['accuracy']
                    conf = rate_point.get('avg_confidence', 0)
                    report_lines.append(f"  {rate_bpf:7.2f} | {acc:6.2%} | {conf:6.4f}")
        
        # Layer sweep results
        report_lines.append("\n\nMethod 2: Layer Sweep")
        report_lines.append("-"*80)
        
        for dataset_name, results in layer_results.items():
            report_lines.append(f"\nDataset: {dataset_name}")
            report_lines.append(f"\n  Layers | Accuracy | Confidence")
            report_lines.append(f"  " + "-"*30)
            
            # Sort by layer number
            layer_points_sorted = sorted(results['layer_points'].items(), 
                                        key=lambda x: x[1].get('num_layers', 0))
            
            for key, layer_point in layer_points_sorted:
                if 'accuracy' in layer_point:
                    num_layers = layer_point['num_layers']
                    acc = layer_point['accuracy']
                    conf = layer_point.get('avg_confidence', 0)
                    report_lines.append(f"  {num_layers:4d} | {acc:6.2%} | {conf:6.4f}")
        
        report_lines.append("\n" + "="*80)
        
        # Save report
        report_text = "\n".join(report_lines)
        save_path = self.output_dir / save_name
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"✅ Saved summary report: {save_path}")
        
        # Also print to log
        logger.info(f"\n{report_text}")
    
    def plot_all(self, rate_results: Dict, layer_results: Dict):
        """
        Generate all charts and reports
        
        Args:
            rate_results: rate sweep results
            layer_results: layer sweep results
        """
        logger.info("Starting to generate analysis charts and reports...")
        
        # Rate method charts
        if rate_results:
            self.plot_rate_vs_accuracy(rate_results)
            self.plot_rate_vs_confidence(rate_results)
        
        # Layer method charts
        if layer_results:
            self.plot_layer_vs_accuracy(layer_results)
        
        # Comparison charts
        if rate_results and layer_results:
            self.plot_combined_comparison(rate_results, layer_results)
        
        # Summary report
        self.generate_summary_report(rate_results, layer_results)
        
        logger.info(f"✅ All analysis results saved to: {self.output_dir}")


