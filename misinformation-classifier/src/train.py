"""Training script for misinformation classifier."""

import argparse
import logging
import os
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np

from config import Config
from dataset import create_datasets
from model import build_model
from utils import set_seed, setup_logging, compute_metrics, save_metrics

# Training function for one epoch
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device, config):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs['loss'].item()
            
            predictions = torch.sigmoid(outputs['logits']) # Applying sigmoid to get probabilities
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    metrics = compute_metrics(predictions, labels, config.label_names)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Train misinformation classifier')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--test_path', type=str, help='Path to test data (optional)')
    parser.add_argument('--config_path', type=str, help='Path to config file (optional)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    set_seed(42)
    
    config = Config()
    config.results_dir = args.output_dir
    config.model_save_path = os.path.join(args.output_dir, 'best_model')
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    
    # Datasets
    train_dataset, val_dataset, test_dataset, tokenizer = create_datasets(
        args.data_path, config, args.test_path
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Calculate pos_weight for weighted BCE loss (handles class imbalance)
    # pos_weight = num_negatives / num_positives for each label

    train_labels = np.array(train_dataset.labels)
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-8), dtype=torch.float32)
    
    logging.info(f"Class imbalance - Positive counts: {pos_counts.tolist()}")
    logging.info(f"Calculated pos_weight: {pos_weight.tolist()}")
    
    # Model
    model = build_model(config, pos_weight=pos_weight).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * getattr(config, 'warmup_ratio', 0.1))    
    scheduler_type = getattr(config, 'lr_scheduler_type', 'linear')
    if scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        logging.info(f"Using cosine LR scheduler with {warmup_steps} warmup steps")
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        logging.info(f"Using linear LR scheduler with {warmup_steps} warmup steps")
    

    epochs_dir = os.path.join(config.results_dir, 'epochs')
    os.makedirs(epochs_dir, exist_ok=True)
    
    # Training log
    training_log = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'model_name': config.model_name,
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'max_length': config.max_length,
            'warmup_ratio': getattr(config, 'warmup_ratio', 0.1)
        },
        'epochs': [],
        'device': str(device),
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        }
    }
    
    # Training loop
    best_val_f1 = 0
    
    for epoch in range(config.num_epochs):
        epoch_start_time = datetime.now()
        logging.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logging.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, device, config)
        logging.info(f"Val loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['macro_f1']:.4f}")
        
        # Evaluate on train set
        train_metrics = evaluate_model(model, train_loader, device, config)
        
        epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
        
        # Create epoch log entry
        epoch_log = {
            'epoch': epoch + 1,
            'timestamp': epoch_start_time.isoformat(),
            'duration_seconds': epoch_duration,
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'is_best': val_metrics['macro_f1'] > best_val_f1
        }
        
        # Save epoch weights
        epoch_model_path = os.path.join(epochs_dir, f'epoch_{epoch+1:02d}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'train_metrics': train_metrics,
            'config': config,
            'tokenizer': tokenizer
        }, epoch_model_path)
        
        # Log detailed metrics
        logging.info(f"Train F1: {train_metrics['macro_f1']:.4f}, Train Acc: {train_metrics['hamming_accuracy']:.4f}")
        logging.info(f"Val F1: {val_metrics['macro_f1']:.4f}, Val Acc: {val_metrics['hamming_accuracy']:.4f}")
        logging.info(f"Epoch duration: {epoch_duration:.1f}s, LR: {optimizer.param_groups[0]['lr']:.2e}")
        logging.info(f"Saved epoch weights to: {epoch_model_path}")
        
        # Save best model
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'tokenizer': tokenizer,
                'epoch': epoch + 1,
                'val_metrics': val_metrics
            }, config.model_save_path + '.pt')
            logging.info(f"New best model saved with F1: {best_val_f1:.4f}")
            epoch_log['is_best'] = True
        
        # Add to training log
        training_log['epochs'].append(epoch_log)
        
        # Save training log after each epoch (incremental backup)
        log_path = os.path.join(config.results_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2, default=str)
        
        logging.info(f"Updated training log: {log_path}")
        logging.info("-" * 60)
    
    logging.info("Evaluating on test set...")
    test_start_time = datetime.now()
    test_metrics = evaluate_model(model, test_loader, device, config)
    test_duration = (datetime.now() - test_start_time).total_seconds()
    
    training_log['end_time'] = datetime.now().isoformat()
    training_log['total_duration_seconds'] = (datetime.now() - datetime.fromisoformat(training_log['start_time'])).total_seconds()
    training_log['best_val_f1'] = best_val_f1
    training_log['test_metrics'] = test_metrics
    training_log['test_evaluation_duration'] = test_duration
    
    save_metrics(test_metrics, os.path.join(config.results_dir, 'test_metrics.json'))
    
    final_log_path = os.path.join(config.results_dir, 'training_log_final.json')
    with open(final_log_path, 'w') as f:
        json.dump(training_log, f, indent=2, default=str)
    
    # Summary report
    summary_report = {
        'training_summary': {
            'total_epochs': config.num_epochs,
            'best_epoch': max(training_log['epochs'], key=lambda x: x['val_metrics']['macro_f1'])['epoch'],
            'best_val_f1': best_val_f1,
            'final_test_f1': test_metrics['macro_f1'],
            'total_training_time': training_log['total_duration_seconds']
        },
        'per_epoch_summary': [
            {
                'epoch': ep['epoch'],
                'train_loss': ep['train_loss'],
                'train_f1': ep['train_metrics']['macro_f1'],
                'val_f1': ep['val_metrics']['macro_f1'],
                'val_loss': ep['val_metrics']['loss'],
                'is_best': ep['is_best'],
                'duration': ep['duration_seconds']
            }
            for ep in training_log['epochs']
        ]
    }
    
    summary_path = os.path.join(config.results_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    logging.info("Training completed!")
    logging.info(f"Test F1: {test_metrics['macro_f1']:.4f}")
    logging.info(f"Best validation F1: {best_val_f1:.4f}")
    logging.info(f"Total training time: {training_log['total_duration_seconds']:.1f}s")
    logging.info(f"Saved detailed logs to: {final_log_path}")
    logging.info(f"Saved training summary to: {summary_path}")
    logging.info(f"Epoch weights saved in: {epochs_dir}")

if __name__ == "__main__":
    main()