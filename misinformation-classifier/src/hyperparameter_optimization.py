#!/usr/bin/env python3
# Hyperparameter Optimization for Misinformation Classifier using Optuna.


import argparse
import logging
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("Please install optuna: pip install optuna")
    exit(1)

class MisinformationClassifier(nn.Module):    
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.distilbert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}


# Dataset
class MisinformationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.FloatTensor(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }


# Data Loading
LABEL_NAMES = [
    "central_route_present",
    "peripheral_route_present", 
    "naturalness_bias",
    "availability_bias",
    "illusory_correlation"
]

FIELD_MAPPING = {
    'central_route_present': 'framework1_feature1',
    'peripheral_route_present': 'framework1_feature2', 
    'naturalness_bias': 'framework2_feature1',
    'availability_bias': 'framework2_feature2',
    'illusory_correlation': 'framework2_feature3'
}

def load_data(filepath: str):
    with open(filepath, 'r') as f:
        data_list = json.load(f)
    
    texts = [item['text'] for item in data_list]
    labels = []
    
    for item in data_list:
        label_row = []
        for label in LABEL_NAMES:
            field_name = FIELD_MAPPING[label]
            label_row.append(item.get(field_name, 0))
        labels.append(label_row)
    
    return texts, labels


# Metrics
def compute_f1(predictions, labels, threshold=0.5):
    """Compute macro F1 score."""
    from sklearn.metrics import f1_score
    pred_binary = (predictions > threshold).astype(int)
    return f1_score(labels, pred_binary, average='macro', zero_division=0)


# Training Functions
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs['loss'].item()
            predictions = torch.sigmoid(outputs['logits'])
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    f1 = compute_f1(predictions, labels)
    avg_loss = total_loss / len(dataloader)
    
    return f1, avg_loss


# Optuna Objective Function
def objective(trial, train_texts, train_labels, val_texts, val_labels, device):
    
    # Learning rate (log scale)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    
    # Batch size
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    # Number of epochs
    num_epochs = trial.suggest_int("num_epochs", 2, 6)
    
    # Dropout rate
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
    # Weight decay
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    
    # Warmup ratio
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    
    # Max sequence length
    max_length = trial.suggest_categorical("max_length", [64, 128, 256])
    
    
    # Create model and data
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = MisinformationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = MisinformationDataset(val_texts, val_labels, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    model = MisinformationClassifier(
        model_name=model_name,
        num_labels=5,
        dropout_rate=dropout_rate
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    # Training loop with pruning
    
    
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_f1, val_loss = evaluate(model, val_loader, device)
        trial.report(val_f1, epoch)
        
        # Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
    
    return best_val_f1


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for misinformation classifier')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of optimization trials')
    parser.add_argument('--output_dir', type=str, default='optimization_results', help='Output directory')
    parser.add_argument('--study_name', type=str, default='misinformation_hpo', help='Optuna study name')
    
    args = parser.parse_args()
    
    # logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    
    logging.info(f"Loading data from {args.data_path}")
    texts, labels = load_data(args.data_path)
    logging.info(f"Loaded {len(texts)} samples")
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    logging.info(f"Train: {len(train_texts)}, Validation: {len(val_texts)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",  # Maximize F1 score
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    logging.info(f"Starting hyperparameter optimization with {args.n_trials} trials...")
    
    study.optimize(
        lambda trial: objective(trial, train_texts, train_labels, val_texts, val_labels, device),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    
    # Results
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("="*60)
    
    # Best trial
    best_trial = study.best_trial
    print(f"\nBest Trial: #{best_trial.number}")
    print(f"Best Validation F1: {best_trial.value:.4f}")
    
    print("\nBest Hyperparameters:")
    print("-" * 40)
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "best_trial_number": best_trial.number,
        "best_val_f1": best_trial.value,
        "best_params": best_trial.params,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state)
            }
            for t in study.trials
        ]
    }
    
    results_path = os.path.join(args.output_dir, "hpo_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate config for best hyperparameters
    best_config = f"""
# Best Hyperparameters (copy to config.py)
# Validation F1: {best_trial.value:.4f}

learning_rate: float = {best_trial.params['learning_rate']:.2e}
batch_size: int = {best_trial.params['batch_size']}
num_epochs: int = {best_trial.params['num_epochs']}
dropout_rate: float = {best_trial.params['dropout_rate']:.3f}
weight_decay: float = {best_trial.params['weight_decay']:.4f}
warmup_ratio: float = {best_trial.params['warmup_ratio']:.3f}
max_length: int = {best_trial.params['max_length']}
"""
    
    config_path = os.path.join(args.output_dir, "best_config.txt")
    with open(config_path, 'w') as f:
        f.write(best_config)
    print(f"Best config saved to: {config_path}")
    
    # Trial statistics
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    
    print(f"\nTrial Statistics:")
    print(f"  Completed: {len(completed_trials)}")
    print(f"  Pruned: {len(pruned_trials)}")
    print(f"  Total: {len(study.trials)}")
    
    # Top 5 trials
    print("\nTop 5 Trials:")
    print("-" * 60)
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
    for i, trial in enumerate(sorted_trials, 1):
        print(f"  {i}. Trial #{trial.number}: F1 = {trial.value:.4f}")
        print(f"     lr={trial.params['learning_rate']:.2e}, "
              f"batch={trial.params['batch_size']}, "
              f"epochs={trial.params['num_epochs']}, "
              f"dropout={trial.params['dropout_rate']:.2f}")

if __name__ == "__main__":
    main()
