"""Evaluation script for misinformation classifier."""

import argparse
import logging
import torch
from torch.utils.data import DataLoader
import numpy as np

from config import Config
from dataset import create_datasets, load_data, MisinformationDataset
from model import build_model
from utils import setup_logging, compute_metrics, save_metrics

def load_trained_model(model_path: str, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config, checkpoint['tokenizer']

# Evaluation function 
def evaluate_model(model, dataloader, device, config):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Apply sigmoid to get probabilities
            predictions = torch.sigmoid(outputs['logits'])
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    return compute_metrics(predictions, labels, config.label_names)

def main():
    parser = argparse.ArgumentParser(description='Evaluate misinformation classifier')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    
    # Device (MPS for Mac)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    
    # model loading
    logging.info(f"Loading model from {args.model_path}")
    model, config, tokenizer = load_trained_model(args.model_path, device)
    
    # data loading
    texts, labels = load_data(args.data_path, config.label_names)
    test_dataset = MisinformationDataset(texts, labels, tokenizer, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    logging.info("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device, config)
    
    #Results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nPer-label results:")
    print("-"*30)
    
    for label in config.label_names:
        precision = metrics[f'{label}_precision']
        recall = metrics[f'{label}_recall']
        f1 = metrics[f'{label}_f1']
        print(f"{label:25} P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f}")
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    save_metrics(metrics, os.path.join(args.output_dir, 'evaluation_metrics.json'))
    
    logging.info(f"Metrics saved to {args.output_dir}/evaluation_metrics.json")

if __name__ == "__main__":
    main()