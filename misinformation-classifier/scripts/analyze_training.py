#!/usr/bin/env python3
"""
Training Analysis Script - Analyze epoch-by-epoch training results and generate plots and reports.
"""

import json
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List


def load_training_log(log_path: str) -> Dict:
    with open(log_path, 'r') as f:
        return json.load(f)

# create plots for training and validation metrics over epochs
def plot_training_curves(log_data: Dict, output_dir: str):
    epochs = [ep['epoch'] for ep in log_data['epochs']]
    train_loss = [ep['train_loss'] for ep in log_data['epochs']]
    val_loss = [ep['val_metrics']['loss'] for ep in log_data['epochs']]
    train_f1 = [ep['train_metrics']['macro_f1'] for ep in log_data['epochs']]
    val_f1 = [ep['val_metrics']['macro_f1'] for ep in log_data['epochs']]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 Score curves
    ax2.plot(epochs, train_f1, 'b-', label='Training F1', linewidth=2)
    ax2.plot(epochs, val_f1, 'r-', label='Validation F1', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1 Score')
    ax2.set_title('Training vs Validation F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    learning_rates = [ep['learning_rate'] for ep in log_data['epochs']]
    ax3.plot(epochs, learning_rates, 'g-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Epoch duration
    durations = [ep['duration_seconds'] for ep in log_data['epochs']]
    ax4.bar(epochs, durations, color='orange', alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Duration (seconds)')
    ax4.set_title('Training Time per Epoch')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

# create plots comparing multiple metrics over epochs
def plot_metrics_comparison(log_data: Dict, output_dir: str):
    epochs = [ep['epoch'] for ep in log_data['epochs']]
    
    # Metric names from first epoch
    train_metrics = list(log_data['epochs'][0]['train_metrics'].keys())
    val_metrics = list(log_data['epochs'][0]['val_metrics'].keys())
    
    # Key metrics
    key_metrics = ['macro_f1', 'micro_f1', 'hamming_accuracy', 'roc_auc']
    available_metrics = [m for m in key_metrics if m in train_metrics]
    
    if not available_metrics:
        available_metrics = ['macro_f1', 'loss']
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots((n_metrics + 1) // 2, 2, figsize=(15, 4 * ((n_metrics + 1) // 2)))
    if n_metrics == 1:
        axes = [axes]
    elif n_metrics <= 2:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics):
        if metric == 'loss':
            train_vals = [ep['train_loss'] for ep in log_data['epochs']]
            val_vals = [ep['val_metrics']['loss'] for ep in log_data['epochs']]
        else:
            train_vals = [ep['train_metrics'][metric] for ep in log_data['epochs']]
            val_vals = [ep['val_metrics'][metric] for ep in log_data['epochs']]
        
        axes[i].plot(epochs, train_vals, 'b-', label=f'Train {metric}', linewidth=2)
        axes[i].plot(epochs, val_vals, 'r-', label=f'Val {metric}', linewidth=2)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} Over Time')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--log-path', required=True, help='Path to training_log_final.json')
    parser.add_argument('--output-dir', default='analysis', help='Output directory for analysis')
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        print(f"Error: Log file not found: {args.log_path}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training log
    log_data = load_training_log(args.log_path)
    
    # Generate plots
    print("Generating training curves...")
    plot_training_curves(log_data, args.output_dir)
    
    print("Generating metrics comparison...")
    plot_metrics_comparison(log_data, args.output_dir)
    
    
    print(f"Analysis complete! Files saved to: {args.output_dir}")
    print(f"- Training curves: training_curves.png")
    print(f"- Metrics comparison: metrics_comparison.png")


if __name__ == '__main__':
    main()