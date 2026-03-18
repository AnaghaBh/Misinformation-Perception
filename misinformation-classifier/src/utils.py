import json
import logging
import random
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from typing import Dict, List, Tuple

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


# Metric computation
def compute_metrics(predictions: np.ndarray, labels: np.ndarray, 
                   label_names: List[str], threshold: float = 0.5) -> Dict:

    # Apply threshold
    pred_binary = (predictions > threshold).astype(int)
    
    # Per label metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_binary, average=None, zero_division=0
    )
    
    # Micro and macro averages
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, pred_binary, average='micro', zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, pred_binary, average='macro', zero_division=0
    )
    
    # ROC AUC score
    try:
        roc_auc = roc_auc_score(labels, predictions, average='macro')
    except ValueError:
        roc_auc = 0.0
    
    # Accuracy metrics

    # Overall accuracy (exact match - all labels correct)
    exact_match_accuracy = accuracy_score(labels, pred_binary)
    
    # Per-label accuracy
    label_accuracies = []
    for i in range(labels.shape[1]):
        acc = accuracy_score(labels[:, i], pred_binary[:, i])
        label_accuracies.append(acc)
    
    # Hamming accuracy (average per-label accuracy)
    hamming_accuracy = np.mean(label_accuracies)
    
    metrics = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'roc_auc': roc_auc,
        'exact_match_accuracy': exact_match_accuracy,
        'hamming_accuracy': hamming_accuracy,
        'overall_accuracy': hamming_accuracy  
    }
    
    # Add per-label metrics
    for i, label in enumerate(label_names):
        metrics[f'{label}_precision'] = precision[i] if isinstance(precision, np.ndarray) else precision
        metrics[f'{label}_recall'] = recall[i] if isinstance(recall, np.ndarray) else recall
        metrics[f'{label}_f1'] = f1[i] if isinstance(f1, np.ndarray) else f1
        metrics[f'{label}_accuracy'] = label_accuracies[i]
    
    return metrics

def save_metrics(metrics: Dict, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)