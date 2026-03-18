"""
Compare trained model against class-prior (probability-based) baseline predictions.
This baseline predicts labels based on their frequency in the training data.
Generates a comprehensive log documenting the entire process.
"""

import json
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, accuracy_score, hamming_loss,
    confusion_matrix
)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Setup logging
log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(msg)

def log_section(title):
    log("")
    log("=" * 80)
    log(title)
    log("=" * 80)

def log_subsection(title):
    log("")
    log("-" * 80)
    log(title)
    log("-" * 80)


# START LOGGING
log_section("CLASS-PRIOR BASELINE COMPARISON LOG")
log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Random Seed: {RANDOM_SEED}")

  
# 1: LOAD AND PREPARE DATA
log_section("1: DATA LOADING AND PREPARATION")

with open('Dataset/Final.json', 'r') as f:
    data = json.load(f)

log(f"Dataset file: Dataset/Final.json")
log(f"Total samples in dataset: {len(data)}")

# Same split as training (80/10/10)
n_total = len(data)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

log("")
log("Data Split Configuration:")
log(f"  Training set:   {n_train} samples (80%)")
log(f"  Validation set: {n_val} samples (10%)")
log(f"  Test set:       {n_test} samples (10%)")

# Get training and test data
train_data = data[:n_train]
test_data = data[n_train + n_val:]
log(f"\nUsing training set to compute class priors: {len(train_data)} samples")
log(f"Using test set for evaluation: {len(test_data)} samples")

# Label mapping
label_mapping = {
    'no_mechanism': 'framework0_feature1',
    'central_route_present': 'framework1_feature1',
    'peripheral_route_present': 'framework1_feature2',
    'naturalness_bias': 'framework2_feature1',
    'availability_bias': 'framework2_feature2',
    'illusory_correlation': 'framework2_feature3'
}
label_names = list(label_mapping.keys())

log("\nLabel Mapping:")
for label_name, field_name in label_mapping.items():
    log(f"  {label_name:30} -> {field_name}")

  
# 2: COMPUTE CLASS PRIORS FROM TRAINING DATA
log_section("2: COMPUTE CLASS PRIORS FROM TRAINING DATA")

log("""
Class-Prior Baseline Method:
  This baseline predicts labels based on their frequency in the training data.
  It represents a classifier that only knows "how often each label appears"
  but doesn't actually look at the text content.""")

class_priors = {}
log("\nClass Prior Probabilities (from training data):")
log(f"{'Label':<30} {'Count':>10} {'Total':>10} {'Prior Prob':>12}")
log("-" * 65)

for label_name, field_name in label_mapping.items():
    positives = sum(1 for item in train_data if item[field_name] == 1)
    prior = positives / len(train_data)
    class_priors[label_name] = prior
    log(f"{label_name:<30} {positives:>10} {len(train_data):>10} {prior:>12.4f}")

log("\nInterpretation:")
log("  - These probabilities will be used to make predictions")
log("  - Higher prior = more likely to predict positive")
log("  - This baseline ignores the actual text content")

  
# 3: EXTRACT TRUE LABELS FROM TEST SET
log_section("3: EXTRACT TRUE LABELS FROM TEST SET")

y_true = []
for item in test_data:
    labels = [item[field] for field in label_mapping.values()]
    y_true.append(labels)
y_true = np.array(y_true)

log(f"True labels shape: {y_true.shape}")
log(f"  - {y_true.shape[0]} samples")
log(f"  - {y_true.shape[1]} labels per sample")

# Count true positives in test set
log("\nTrue Label Distribution in Test Set:")
log(f"{'Label':<30} {'Positive':>10} {'Negative':>10} {'% Positive':>12}")
log("-" * 65)
for i, label in enumerate(label_names):
    positives = int(y_true[:, i].sum())
    negatives = len(test_data) - positives
    pct = (positives / len(test_data)) * 100
    log(f"{label:<30} {positives:>10} {negatives:>10} {pct:>11.1f}%")

  
# 4: GENERATE CLASS-PRIOR PREDICTIONS
log_section("4: CLASS-PRIOR BASELINE PREDICTIONS")

log_subsection("4.1 Prediction Generation Process")

log("""
Prediction Process:
  1. For each sample and each label, draw a random number r ~ Uniform(0, 1)
  2. Compare r to the class prior probability p
  3. If r < p, predict 1 (positive), otherwise predict 0 (negative)
  
  This ensures predictions follow the same distribution as the training data.
""")

# Generate predictions using class priors
y_pred_prior_prob = np.random.uniform(0, 1, y_true.shape)
y_pred_prior = np.zeros_like(y_true)

for i, label in enumerate(label_names):
    prior = class_priors[label]
    y_pred_prior[:, i] = (y_pred_prior_prob[:, i] < prior).astype(int)

log("Class Priors Used for Prediction:")
for label, prior in class_priors.items():
    log(f"  {label:30} p = {prior:.4f} ({prior*100:.1f}%)")

# Show sample predictions
log("\nSample of first 5 predictions:")
log(f"{'Sample':<8} " + " ".join([f"{label[:8]:>10}" for label in label_names]))
log("-" * 75)

log("Random draws (r):")
for i in range(min(5, len(y_pred_prior_prob))):
    probs = " ".join([f"{p:>10.4f}" for p in y_pred_prior_prob[i]])
    log(f"{i+1:<8} {probs}")

log("\nClass priors (p):")
priors_str = " ".join([f"{class_priors[label]:>10.4f}" for label in label_names])
log(f"{'':8} {priors_str}")

log("\nPredictions (1 if r < p):")
for i in range(min(5, len(y_pred_prior))):
    preds = " ".join([f"{int(p):>10}" for p in y_pred_prior[i]])
    log(f"{i+1:<8} {preds}")

log("\nCorresponding true labels:")
for i in range(min(5, len(y_true))):
    true = " ".join([f"{int(p):>10}" for p in y_true[i]])
    log(f"{i+1:<8} {true}")

# Distribution of predictions
log("\nPrediction Distribution (Class-Prior Baseline):")
log(f"{'Label':<30} {'Pred=1':>10} {'Pred=0':>10} {'% Pred=1':>12} {'Prior':>10}")
log("-" * 75)
for i, label in enumerate(label_names):
    pred_pos = int(y_pred_prior[:, i].sum())
    pred_neg = len(test_data) - pred_pos
    pct = (pred_pos / len(test_data)) * 100
    prior_pct = class_priors[label] * 100
    log(f"{label:<30} {pred_pos:>10} {pred_neg:>10} {pct:>11.1f}% {prior_pct:>9.1f}%")

log("\nNote: % Pred=1 should approximately match the Prior (random variation expected)")

log_subsection("4.2 Class-Prior Baseline Metrics Calculation")
# Calculating all metrics for class-prior baseline
prior_metrics = {}

# Per-label metrics
log("\nPer-Label Metrics (Class-Prior Baseline):")
log(f"{'Label':<30} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Accuracy':>12}")
log("-" * 80)

for i, label in enumerate(label_names):
    y_true_label = y_true[:, i]
    y_pred_label = y_pred_prior[:, i]
    
    precision = precision_score(y_true_label, y_pred_label, zero_division=0)
    recall = recall_score(y_true_label, y_pred_label, zero_division=0)
    f1 = f1_score(y_true_label, y_pred_label, zero_division=0)
    acc = accuracy_score(y_true_label, y_pred_label)
    
    prior_metrics[f'{label}_precision'] = precision
    prior_metrics[f'{label}_recall'] = recall
    prior_metrics[f'{label}_f1'] = f1
    prior_metrics[f'{label}_accuracy'] = acc
    
    log(f"{label:<30} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f} {acc:>12.4f}")

# Aggregate metrics
prior_metrics['macro_f1'] = f1_score(y_true, y_pred_prior, average='macro')
prior_metrics['micro_f1'] = f1_score(y_true, y_pred_prior, average='micro')
prior_metrics['macro_precision'] = precision_score(y_true, y_pred_prior, average='macro')
prior_metrics['macro_recall'] = recall_score(y_true, y_pred_prior, average='macro')
prior_metrics['micro_precision'] = precision_score(y_true, y_pred_prior, average='micro')
prior_metrics['micro_recall'] = recall_score(y_true, y_pred_prior, average='micro')
prior_metrics['roc_auc'] = roc_auc_score(y_true, y_pred_prior_prob, average='macro')
prior_metrics['hamming_loss'] = hamming_loss(y_true, y_pred_prior)
prior_metrics['hamming_accuracy'] = 1 - prior_metrics['hamming_loss']

# Exact match
exact_matches = np.all(y_true == y_pred_prior, axis=1).sum()
prior_metrics['exact_match_accuracy'] = exact_matches / len(y_true)

log("\nAggregate Metrics (Class-Prior Baseline):")
log(f"  Macro F1:           {prior_metrics['macro_f1']:.4f}")
log(f"  Micro F1:           {prior_metrics['micro_f1']:.4f}")
log(f"  Macro Precision:    {prior_metrics['macro_precision']:.4f}")
log(f"  Macro Recall:       {prior_metrics['macro_recall']:.4f}")
log(f"  Micro Precision:    {prior_metrics['micro_precision']:.4f}")
log(f"  Micro Recall:       {prior_metrics['micro_recall']:.4f}")
log(f"  ROC-AUC:            {prior_metrics['roc_auc']:.4f}")
log(f"  Hamming Accuracy:   {prior_metrics['hamming_accuracy']:.4f}")
log(f"  Exact Match Acc:    {prior_metrics['exact_match_accuracy']:.4f}")

  
# 5: LOAD TRAINED MODEL RESULTS
log_section("5: TRAINED MODEL RESULTS")

with open('results_optimal/test_metrics.json', 'r') as f:
    model_metrics = json.load(f)

log("Loaded from: results_optimal/test_metrics.json")
log("")

log("Per-Label Metrics (Trained Model):")
log(f"{'Label':<30} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Accuracy':>12}")
log("-" * 80)

for label in label_names:
    precision = model_metrics[f'{label}_precision']
    recall = model_metrics[f'{label}_recall']
    f1 = model_metrics[f'{label}_f1']
    acc = model_metrics[f'{label}_accuracy']
    log(f"{label:<30} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f} {acc:>12.4f}")

log("\nAggregate Metrics (Trained Model):")
log(f"  Macro F1:           {model_metrics['macro_f1']:.4f}")
log(f"  Micro F1:           {model_metrics['micro_f1']:.4f}")
log(f"  Macro Precision:    {model_metrics['macro_precision']:.4f}")
log(f"  Macro Recall:       {model_metrics['macro_recall']:.4f}")
log(f"  Micro Precision:    {model_metrics['micro_precision']:.4f}")
log(f"  Micro Recall:       {model_metrics['micro_recall']:.4f}")
log(f"  ROC-AUC:            {model_metrics['roc_auc']:.4f}")
log(f"  Hamming Accuracy:   {model_metrics['hamming_accuracy']:.4f}")
log(f"  Exact Match Acc:    {model_metrics['exact_match_accuracy']:.4f}")

  
# 6: DETAILED COMPARISON
log_section("6: DETAILED COMPARISON - MODEL vs CLASS-PRIOR BASELINE")

log_subsection("6.1 Per-Label F1 Score Comparison")

log(f"{'Label':<30} {'Model F1':>12} {'Prior F1':>12} {'Difference':>12} {'Ratio':>10}")
log("-" * 80)

for label in label_names:
    model_f1 = model_metrics[f'{label}_f1']
    prior_f1 = prior_metrics[f'{label}_f1']
    diff = model_f1 - prior_f1
    ratio = model_f1 / prior_f1 if prior_f1 > 0 else float('inf')
    log(f"{label:<30} {model_f1:>12.4f} {prior_f1:>12.4f} {diff:>+12.4f} {ratio:>10.2f}x")

log_subsection("6.2 Per-Label Precision Comparison")

log(f"{'Label':<30} {'Model':>12} {'Prior':>12} {'Difference':>12}")
log("-" * 70)

for label in label_names:
    model_val = model_metrics[f'{label}_precision']
    prior_val = prior_metrics[f'{label}_precision']
    diff = model_val - prior_val
    log(f"{label:<30} {model_val:>12.4f} {prior_val:>12.4f} {diff:>+12.4f}")

log_subsection("6.3 Per-Label Recall Comparison")

log(f"{'Label':<30} {'Model':>12} {'Prior':>12} {'Difference':>12}")
log("-" * 70)

for label in label_names:
    model_val = model_metrics[f'{label}_recall']
    prior_val = prior_metrics[f'{label}_recall']
    diff = model_val - prior_val
    log(f"{label:<30} {model_val:>12.4f} {prior_val:>12.4f} {diff:>+12.4f}")

log_subsection("6.4 Per-Label Accuracy Comparison")

log(f"{'Label':<30} {'Model':>12} {'Prior':>12} {'Difference':>12}")
log("-" * 70)

for label in label_names:
    model_val = model_metrics[f'{label}_accuracy']
    prior_val = prior_metrics[f'{label}_accuracy']
    diff = model_val - prior_val
    log(f"{label:<30} {model_val:>12.4f} {prior_val:>12.4f} {diff:>+12.4f}")

log_subsection("6.5 Aggregate Metrics Comparison")

aggregate_metrics = [
    ('Macro F1', 'macro_f1'),
    ('Micro F1', 'micro_f1'),
    ('Macro Precision', 'macro_precision'),
    ('Macro Recall', 'macro_recall'),
    ('Micro Precision', 'micro_precision'),
    ('Micro Recall', 'micro_recall'),
    ('ROC-AUC', 'roc_auc'),
    ('Hamming Accuracy', 'hamming_accuracy'),
    ('Exact Match Accuracy', 'exact_match_accuracy'),
]

log(f"{'Metric':<25} {'Model':>12} {'Prior':>12} {'Difference':>12} {'% Improvement':>15}")
log("-" * 80)

for name, key in aggregate_metrics:
    model_val = model_metrics.get(key, 0)
    prior_val = prior_metrics.get(key, 0)
    diff = model_val - prior_val
    pct_improvement = (diff / prior_val * 100) if prior_val > 0 else float('inf')
    log(f"{name:<25} {model_val:>12.4f} {prior_val:>12.4f} {diff:>+12.4f} {pct_improvement:>+14.1f}%")

  
# 7: COMPARISON WITH UNIFORM RANDOM BASELINE
log_section("7: THREE-WAY COMPARISON (MODEL vs CLASS-PRIOR vs UNIFORM RANDOM)")

# Load uniform random results if available
uniform_metrics = None
try:
    with open('results_optimal/random_baseline_comparison.json', 'r') as f:
        uniform_data = json.load(f)
        uniform_metrics = uniform_data['uniform_random_metrics']
except FileNotFoundError:
    log("Note: Uniform random baseline results not found.")

if uniform_metrics:
    log_subsection("7.1 Aggregate Metrics - Three-Way Comparison")
    
    log(f"{'Metric':<25} {'Model':>12} {'Class-Prior':>12} {'Uniform Rand':>12}")
    log("-" * 65)
    
    for name, key in aggregate_metrics:
        model_val = model_metrics.get(key, 0)
        prior_val = prior_metrics.get(key, 0)
        uniform_val = uniform_metrics.get(key, 0)
        log(f"{name:<25} {model_val:>12.4f} {prior_val:>12.4f} {uniform_val:>12.4f}")
    
    log_subsection("7.2 Per-Label F1 - Three-Way Comparison")
    
    log(f"{'Label':<30} {'Model':>10} {'Prior':>10} {'Uniform':>10}")
    log("-" * 65)
    
    for label in label_names:
        model_f1 = model_metrics[f'{label}_f1']
        prior_f1 = prior_metrics[f'{label}_f1']
        uniform_f1 = uniform_metrics[f'{label}_f1']
        log(f"{label:<30} {model_f1:>10.4f} {prior_f1:>10.4f} {uniform_f1:>10.4f}")
    
    log_subsection("7.3 Baseline Comparison Analysis")
    
    log("\nWhich baseline is harder to beat?")
    log("-" * 65)
    
    prior_better_count = 0
    uniform_better_count = 0
    
    for label in label_names:
        prior_f1 = prior_metrics[f'{label}_f1']
        uniform_f1 = uniform_metrics[f'{label}_f1']
        
        if prior_f1 > uniform_f1:
            prior_better_count += 1
            winner = "Class-Prior"
        elif uniform_f1 > prior_f1:
            uniform_better_count += 1
            winner = "Uniform Random"
        else:
            winner = "Tie"
        
        log(f"  {label:30}: {winner} ({prior_f1:.4f} vs {uniform_f1:.4f})")
    
    log(f"\nClass-Prior baseline wins: {prior_better_count}/6 labels")
    log(f"Uniform Random baseline wins: {uniform_better_count}/6 labels")
    
    # Overall winner
    if prior_metrics['macro_f1'] > uniform_metrics['macro_f1']:
        log("\nOverall: Class-Prior is the stronger baseline (harder to beat)")
    else:
        log("\nOverall: Uniform Random is the stronger baseline (harder to beat)")

  
# 8: SUMMARY AND CONCLUSIONS
log_section("8: SUMMARY AND CONCLUSIONS")

log("""
SUMMARY TABLE
""")

log(f"{'Metric':<35} {'Trained Model':>15} {'Class-Prior':>15}")
log("=" * 70)
log(f"{'Macro F1':<35} {model_metrics['macro_f1']:>15.4f} {prior_metrics['macro_f1']:>15.4f}")
log(f"{'Micro F1':<35} {model_metrics['micro_f1']:>15.4f} {prior_metrics['micro_f1']:>15.4f}")
log(f"{'ROC-AUC':<35} {model_metrics['roc_auc']:>15.4f} {prior_metrics['roc_auc']:>15.4f}")
log(f"{'Hamming Accuracy':<35} {model_metrics['hamming_accuracy']:>15.4f} {prior_metrics['hamming_accuracy']:>15.4f}")
log(f"{'Exact Match Accuracy':<35} {model_metrics['exact_match_accuracy']:>15.4f} {prior_metrics['exact_match_accuracy']:>15.4f}")
log("=" * 70)

log("""
KEY FINDINGS:
""")

macro_f1_improvement = (model_metrics['macro_f1'] - prior_metrics['macro_f1']) * 100
roc_improvement = (model_metrics['roc_auc'] - prior_metrics['roc_auc']) * 100
ratio = model_metrics['macro_f1'] / prior_metrics['macro_f1'] if prior_metrics['macro_f1'] > 0 else float('inf')

log(f"1. Model achieves {model_metrics['macro_f1']:.4f} Macro F1 vs {prior_metrics['macro_f1']:.4f} for class-prior baseline")
log(f"   -> Improvement: +{macro_f1_improvement:.1f} percentage points ({ratio:.1f}x better)")
log("")
log(f"2. Model achieves {model_metrics['roc_auc']:.4f} ROC-AUC vs {prior_metrics['roc_auc']:.4f} for class-prior baseline")
log("")
log("3. Per-Label Analysis:")
# Find best and worst improvements
improvements = []
for label in label_names:
    model_f1 = model_metrics[f'{label}_f1']
    prior_f1 = prior_metrics[f'{label}_f1']
    ratio = model_f1 / prior_f1 if prior_f1 > 0 else float('inf')
    improvements.append((label, model_f1, prior_f1, ratio))

improvements.sort(key=lambda x: x[3], reverse=True)
best = improvements[0]
worst = improvements[-1]

log(f"   - Best improvement:  {best[0]} ({best[3]:.1f}x better, {best[1]:.4f} vs {best[2]:.4f})")
log(f"   - Smallest improvement: {worst[0]} ({worst[3]:.1f}x better, {worst[1]:.4f} vs {worst[2]:.4f})")
log("")
log("4. The class-prior baseline knows the label distribution but ignores text content.")
log("   The trained model effectively leverages text features to outperform this baseline.")

  
# SAVE LOG FILE
log_section("LOG SAVED")

# Create results directory if it doesn't exist
os.makedirs('results_optimal', exist_ok=True)

# Save log file
log_filename = 'results_optimal/class_prior_baseline_comparison_log.txt'
with open(log_filename, 'w') as f:
    f.write('\n'.join(log_lines))

print(f"\nLog saved to: {log_filename}")
comparison_data = {
    'timestamp': datetime.now().isoformat(),
    'random_seed': RANDOM_SEED,
    'test_set_size': len(test_data),
    'class_priors': class_priors,
    'class_prior_metrics': prior_metrics,
    'model_metrics': {k: v for k, v in model_metrics.items()},
    'improvements': {
        'macro_f1_diff': model_metrics['macro_f1'] - prior_metrics['macro_f1'],
        'micro_f1_diff': model_metrics['micro_f1'] - prior_metrics['micro_f1'],
        'roc_auc_diff': model_metrics['roc_auc'] - prior_metrics['roc_auc'],
        'macro_f1_ratio': model_metrics['macro_f1'] / prior_metrics['macro_f1'] if prior_metrics['macro_f1'] > 0 else None,
    }
}
json_filename = 'results_optimal/class_prior_baseline_comparison.json'
with open(json_filename, 'w') as f:
    json.dump(comparison_data, f, indent=2)

print(f"JSON data saved to: {json_filename}")
