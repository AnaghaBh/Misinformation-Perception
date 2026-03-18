#!/usr/bin/env python3
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_log(log_path: str) -> dict:
    """Load training log from JSON file."""
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_loss_curves(epochs_data: list, save_dir: Path):
    epochs = [e['epoch'] for e in epochs_data]
    train_loss = [e['train_loss'] for e in epochs_data]
    val_loss = [e['val_metrics']['loss'] for e in epochs_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', markersize=4)
    plt.plot(epochs, val_loss, 'r-o', label='Validation Loss', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'loss_curves.png'}")


def plot_f1_curves(epochs_data: list, save_dir: Path):
    epochs = [e['epoch'] for e in epochs_data]
    train_macro_f1 = [e['train_metrics']['macro_f1'] for e in epochs_data]
    val_macro_f1 = [e['val_metrics']['macro_f1'] for e in epochs_data]
    train_micro_f1 = [e['train_metrics']['micro_f1'] for e in epochs_data]
    val_micro_f1 = [e['val_metrics']['micro_f1'] for e in epochs_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Macro F1
    axes[0].plot(epochs, train_macro_f1, 'b-o', label='Train Macro F1', markersize=4)
    axes[0].plot(epochs, val_macro_f1, 'r-o', label='Val Macro F1', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Macro F1 Score')
    axes[0].set_title('Macro F1 Score Over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Micro F1
    axes[1].plot(epochs, train_micro_f1, 'b-o', label='Train Micro F1', markersize=4)
    axes[1].plot(epochs, val_micro_f1, 'r-o', label='Val Micro F1', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Micro F1 Score')
    axes[1].set_title('Micro F1 Score Over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'f1_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'f1_curves.png'}")


def plot_precision_recall(epochs_data: list, save_dir: Path):
    epochs = [e['epoch'] for e in epochs_data]
    
    train_precision = [e['train_metrics']['macro_precision'] for e in epochs_data]
    val_precision = [e['val_metrics']['macro_precision'] for e in epochs_data]
    train_recall = [e['train_metrics']['macro_recall'] for e in epochs_data]
    val_recall = [e['val_metrics']['macro_recall'] for e in epochs_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Precision
    axes[0].plot(epochs, train_precision, 'b-o', label='Train Precision', markersize=4)
    axes[0].plot(epochs, val_precision, 'r-o', label='Val Precision', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Macro Precision')
    axes[0].set_title('Macro Precision Over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Recall
    axes[1].plot(epochs, train_recall, 'b-o', label='Train Recall', markersize=4)
    axes[1].plot(epochs, val_recall, 'r-o', label='Val Recall', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro Recall')
    axes[1].set_title('Macro Recall Over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'precision_recall_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'precision_recall_curves.png'}")


def plot_accuracy_curves(epochs_data: list, save_dir: Path):
    epochs = [e['epoch'] for e in epochs_data]
    
    train_hamming = [e['train_metrics']['hamming_accuracy'] for e in epochs_data]
    val_hamming = [e['val_metrics']['hamming_accuracy'] for e in epochs_data]
    train_exact = [e['train_metrics']['exact_match_accuracy'] for e in epochs_data]
    val_exact = [e['val_metrics']['exact_match_accuracy'] for e in epochs_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hamming Accuracy
    axes[0].plot(epochs, train_hamming, 'b-o', label='Train Hamming Acc', markersize=4)
    axes[0].plot(epochs, val_hamming, 'r-o', label='Val Hamming Acc', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Hamming Accuracy')
    axes[0].set_title('Hamming Accuracy (Per-Label) Over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Exact Match Accuracy
    axes[1].plot(epochs, train_exact, 'b-o', label='Train Exact Match', markersize=4)
    axes[1].plot(epochs, val_exact, 'r-o', label='Val Exact Match', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Exact Match Accuracy')
    axes[1].set_title('Exact Match Accuracy Over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'accuracy_curves.png'}")


def plot_per_label_f1(epochs_data: list, label_names: list, save_dir: Path):
    epochs = [e['epoch'] for e in epochs_data]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_names)))
    
    for idx, label in enumerate(label_names):
        train_f1 = [e['train_metrics'].get(f'{label}_f1', 0) for e in epochs_data]
        val_f1 = [e['val_metrics'].get(f'{label}_f1', 0) for e in epochs_data]
        
        axes[idx].plot(epochs, train_f1, 'b-o', label='Train', markersize=4)
        axes[idx].plot(epochs, val_f1, 'r-o', label='Val', markersize=4)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('F1 Score')
        axes[idx].set_title(f'{label}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1])
    
    plt.suptitle('Per-Label F1 Scores Over Epochs', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'per_label_f1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'per_label_f1.png'}")


def plot_per_label_accuracy(epochs_data: list, label_names: list, save_dir: Path):
    epochs = [e['epoch'] for e in epochs_data]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, label in enumerate(label_names):
        train_acc = [e['train_metrics'].get(f'{label}_accuracy', 0) for e in epochs_data]
        val_acc = [e['val_metrics'].get(f'{label}_accuracy', 0) for e in epochs_data]
        
        axes[idx].plot(epochs, train_acc, 'b-o', label='Train', markersize=4)
        axes[idx].plot(epochs, val_acc, 'r-o', label='Val', markersize=4)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Accuracy')
        axes[idx].set_title(f'{label}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1])
    
    plt.suptitle('Per-Label Accuracy Over Epochs', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'per_label_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'per_label_accuracy.png'}")


def plot_roc_auc(epochs_data: list, save_dir: Path):
    epochs = [e['epoch'] for e in epochs_data]
    train_auc = [e['train_metrics'].get('roc_auc', 0) for e in epochs_data]
    val_auc = [e['val_metrics'].get('roc_auc', 0) for e in epochs_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_auc, 'b-o', label='Train ROC AUC', markersize=4)
    plt.plot(epochs, val_auc, 'r-o', label='Val ROC AUC', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC Score Over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_auc_curve.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'roc_auc_curve.png'}")


def plot_learning_rate(epochs_data: list, save_dir: Path):
    epochs = [e['epoch'] for e in epochs_data]
    lrs = [e.get('learning_rate', 0) for e in epochs_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs, 'g-o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_dir / 'learning_rate.png', dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'learning_rate.png'}")


def create_summary_table(epochs_data: list, label_names: list, save_dir: Path):
    # Find best epoch by validation macro F1
    best_epoch = max(epochs_data, key=lambda x: x['val_metrics']['macro_f1'])
    
    summary = []
    summary.append("=" * 80)
    summary.append("TRAINING SUMMARY")
    summary.append("=" * 80)
    summary.append(f"\nTotal Epochs: {len(epochs_data)}")
    summary.append(f"Best Epoch: {best_epoch['epoch']} (Val Macro F1: {best_epoch['val_metrics']['macro_f1']:.4f})")
    
    summary.append("\n" + "-" * 80)
    summary.append("BEST EPOCH METRICS (Epoch {})".format(best_epoch['epoch']))
    summary.append("-" * 80)
    
    summary.append("\nOverall Metrics:")
    summary.append(f"  {'Metric':<25} {'Train':>12} {'Validation':>12}")
    summary.append(f"  {'-'*25} {'-'*12} {'-'*12}")
    
    metrics_to_show = ['macro_f1', 'micro_f1', 'macro_precision', 'macro_recall', 
                       'hamming_accuracy', 'exact_match_accuracy', 'roc_auc']
    
    for metric in metrics_to_show:
        train_val = best_epoch['train_metrics'].get(metric, 0)
        val_val = best_epoch['val_metrics'].get(metric, 0)
        summary.append(f"  {metric:<25} {train_val:>12.4f} {val_val:>12.4f}")
    
    summary.append("\nPer-Label F1 Scores:")
    summary.append(f"  {'Label':<25} {'Train':>12} {'Validation':>12}")
    summary.append(f"  {'-'*25} {'-'*12} {'-'*12}")
    
    for label in label_names:
        train_f1 = best_epoch['train_metrics'].get(f'{label}_f1', 0)
        val_f1 = best_epoch['val_metrics'].get(f'{label}_f1', 0)
        summary.append(f"  {label:<25} {train_f1:>12.4f} {val_f1:>12.4f}")
    
    summary.append("\nPer-Label Accuracy:")
    summary.append(f"  {'Label':<25} {'Train':>12} {'Validation':>12}")
    summary.append(f"  {'-'*25} {'-'*12} {'-'*12}")
    
    for label in label_names:
        train_acc = best_epoch['train_metrics'].get(f'{label}_accuracy', 0)
        val_acc = best_epoch['val_metrics'].get(f'{label}_accuracy', 0)
        summary.append(f"  {label:<25} {train_acc:>12.4f} {val_acc:>12.4f}")
    
    summary.append("\n" + "-" * 80)
    summary.append("EPOCH-BY-EPOCH SUMMARY")
    summary.append("-" * 80)
    summary.append(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Train F1':>12} {'Val F1':>12} {'Val Acc':>12}")
    summary.append(f"{'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for e in epochs_data:
        marker = " *" if e['is_best'] else ""
        summary.append(f"{e['epoch']:>6} {e['train_loss']:>12.4f} {e['val_metrics']['loss']:>12.4f} "
                      f"{e['train_metrics']['macro_f1']:>12.4f} {e['val_metrics']['macro_f1']:>12.4f} "
                      f"{e['val_metrics']['hamming_accuracy']:>12.4f}{marker}")
    
    summary.append("\n* = Best epoch")
    summary.append("=" * 80)
    
    summary_text = "\n".join(summary)
    
    # Save to file
    with open(save_dir / 'training_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"Saved: {save_dir / 'training_summary.txt'}")
    print("\n" + summary_text)


def save_metrics_csv(epochs_data: list, label_names: list, save_dir: Path):
    import csv
    
    # Prepare headers
    headers = ['epoch', 'train_loss', 'val_loss', 
               'train_macro_f1', 'val_macro_f1',
               'train_micro_f1', 'val_micro_f1',
               'train_macro_precision', 'val_macro_precision',
               'train_macro_recall', 'val_macro_recall',
               'train_hamming_accuracy', 'val_hamming_accuracy',
               'train_exact_match_accuracy', 'val_exact_match_accuracy',
               'train_roc_auc', 'val_roc_auc',
               'learning_rate', 'is_best']
    
    # Add per-label headers
    for label in label_names:
        headers.extend([f'train_{label}_f1', f'val_{label}_f1',
                       f'train_{label}_precision', f'val_{label}_precision',
                       f'train_{label}_recall', f'val_{label}_recall',
                       f'train_{label}_accuracy', f'val_{label}_accuracy'])
    
    # Write CSV
    with open(save_dir / 'all_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for e in epochs_data:
            row = [
                e['epoch'],
                e['train_loss'],
                e['val_metrics']['loss'],
                e['train_metrics']['macro_f1'],
                e['val_metrics']['macro_f1'],
                e['train_metrics']['micro_f1'],
                e['val_metrics']['micro_f1'],
                e['train_metrics']['macro_precision'],
                e['val_metrics']['macro_precision'],
                e['train_metrics']['macro_recall'],
                e['val_metrics']['macro_recall'],
                e['train_metrics']['hamming_accuracy'],
                e['val_metrics']['hamming_accuracy'],
                e['train_metrics']['exact_match_accuracy'],
                e['val_metrics']['exact_match_accuracy'],
                e['train_metrics'].get('roc_auc', 0),
                e['val_metrics'].get('roc_auc', 0),
                e.get('learning_rate', 0),
                e['is_best']
            ]
            
            # Add per-label metrics
            for label in label_names:
                row.extend([
                    e['train_metrics'].get(f'{label}_f1', 0),
                    e['val_metrics'].get(f'{label}_f1', 0),
                    e['train_metrics'].get(f'{label}_precision', 0),
                    e['val_metrics'].get(f'{label}_precision', 0),
                    e['train_metrics'].get(f'{label}_recall', 0),
                    e['val_metrics'].get(f'{label}_recall', 0),
                    e['train_metrics'].get(f'{label}_accuracy', 0),
                    e['val_metrics'].get(f'{label}_accuracy', 0)
                ])
            
            writer.writerow(row)
    
    print(f"Saved: {save_dir / 'all_metrics.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--log_path', type=str, required=True, 
                       help='Path to training_log_final.json')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for visualizations (default: same as log)')
    
    args = parser.parse_args()
    
    # Load training log
    log_path = Path(args.log_path)
    training_log = load_training_log(log_path)
    
    # Set output directory
    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        save_dir = log_path.parent / 'visualizations'
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs_data = training_log['epochs']
    
    # Get label names from config or use default
    label_names = training_log.get('config', {}).get('label_names', [
        'no_mechanism',
        'central_route_present',
        'peripheral_route_present',
        'naturalness_bias',
        'availability_bias',
        'illusory_correlation'
    ])
    
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {save_dir}")
    print(f"Epochs: {len(epochs_data)}")
    print(f"Labels: {label_names}")
    print("-" * 50)
    
    # Generate all plots
    plot_loss_curves(epochs_data, save_dir)
    plot_f1_curves(epochs_data, save_dir)
    plot_precision_recall(epochs_data, save_dir)
    plot_accuracy_curves(epochs_data, save_dir)
    plot_per_label_f1(epochs_data, label_names, save_dir)
    plot_per_label_accuracy(epochs_data, label_names, save_dir)
    plot_roc_auc(epochs_data, save_dir)
    plot_learning_rate(epochs_data, save_dir)
    
    # Create summary
    create_summary_table(epochs_data, label_names, save_dir)
    save_metrics_csv(epochs_data, label_names, save_dir)
    
    print("\n" + "=" * 50)
    print("Visualization complete!")
    print(f"All outputs saved to: {save_dir}")


if __name__ == '__main__':
    main()
