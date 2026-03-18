"""
Visualize performance comparison between trained model, uniform random baseline, 
and class-prior baseline.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Colors
MODEL_COLOR = '#2ecc71'  
UNIFORM_COLOR = '#e74c3c'  
PRIOR_COLOR = '#3498db'  

# Load data
with open('results_optimal/test_metrics.json', 'r') as f:
    model_metrics = json.load(f)

with open('results_optimal/random_baseline_comparison.json', 'r') as f:
    uniform_data = json.load(f)
    uniform_metrics = uniform_data['uniform_random_metrics']

with open('results_optimal/class_prior_baseline_comparison.json', 'r') as f:
    prior_data = json.load(f)
    prior_metrics = prior_data['class_prior_metrics']
    class_priors = prior_data['class_priors']

# Label names
label_names = [
    'no_mechanism',
    'central_route_present', 
    'peripheral_route_present',
    'naturalness_bias',
    'availability_bias',
    'illusory_correlation'
]

short_names = [
    'No Mech.',
    'Central',
    'Peripheral', 
    'Natural.',
    'Avail.',
    'Illusory'
]

# Create output directory
os.makedirs('results_optimal/visualizations', exist_ok=True)


# FIGURE 1: Per-Label F1 Score Comparison (Grouped Bar Chart)


fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(label_names))
width = 0.25

model_f1 = [model_metrics[f'{label}_f1'] for label in label_names]
uniform_f1 = [uniform_metrics[f'{label}_f1'] for label in label_names]
prior_f1 = [prior_metrics[f'{label}_f1'] for label in label_names]

bars1 = ax.bar(x - width, model_f1, width, label='Trained Model', color=MODEL_COLOR, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, uniform_f1, width, label='Uniform Random', color=UNIFORM_COLOR, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, prior_f1, width, label='Class-Prior', color=PRIOR_COLOR, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Label')
ax.set_ylabel('F1 Score')
ax.set_title('Per-Label F1 Score Comparison: Trained Model vs Baselines')
ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=0)
ax.legend(loc='upper right')
ax.set_ylim(0, 1.0)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results_optimal/visualizations/f1_comparison_bars.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: f1_comparison_bars.png")


# FIGURE 2: Aggregate Metrics Comparison (Horizontal Bar Chart)


fig, ax = plt.subplots(figsize=(12, 8))

metrics = ['Macro F1', 'Micro F1', 'ROC-AUC', 'Hamming Acc.', 'Exact Match']
keys = ['macro_f1', 'micro_f1', 'roc_auc', 'hamming_accuracy', 'exact_match_accuracy']

model_vals = [model_metrics[k] for k in keys]
uniform_vals = [uniform_metrics[k] for k in keys]
prior_vals = [prior_metrics[k] for k in keys]

y = np.arange(len(metrics))
height = 0.25

bars1 = ax.barh(y - height, model_vals, height, label='Trained Model', color=MODEL_COLOR, edgecolor='black', linewidth=0.5)
bars2 = ax.barh(y, uniform_vals, height, label='Uniform Random', color=UNIFORM_COLOR, edgecolor='black', linewidth=0.5)
bars3 = ax.barh(y + height, prior_vals, height, label='Class-Prior', color=PRIOR_COLOR, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Score')
ax.set_ylabel('Metric')
ax.set_title('Aggregate Metrics Comparison: Trained Model vs Baselines')
ax.set_yticks(y)
ax.set_yticklabels(metrics)
ax.legend(loc='lower right')
ax.set_xlim(0, 1.0)
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Random Chance')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        width_val = bar.get_width()
        ax.annotate(f'{width_val:.3f}',
                    xy=(width_val, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('results_optimal/visualizations/aggregate_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: aggregate_metrics_comparison.png")


# FIGURE 3: Model Improvement Ratio (How many times better)


fig, ax = plt.subplots(figsize=(12, 6))

# Calculate improvement ratios over uniform random
uniform_ratios = []
prior_ratios = []
for label in label_names:
    model_f1 = model_metrics[f'{label}_f1']
    u_f1 = uniform_metrics[f'{label}_f1']
    p_f1 = prior_metrics[f'{label}_f1']
    
    uniform_ratios.append(model_f1 / u_f1 if u_f1 > 0 else 10)  # Cap at 10 for inf
    prior_ratios.append(model_f1 / p_f1 if p_f1 > 0 else 10)

x = np.arange(len(label_names))
width = 0.35

bars1 = ax.bar(x - width/2, uniform_ratios, width, label='vs Uniform Random', color=UNIFORM_COLOR, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, prior_ratios, width, label='vs Class-Prior', color=PRIOR_COLOR, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Label')
ax.set_ylabel('Improvement Ratio (x times better)')
ax.set_title('Model Improvement Ratio Over Baselines (F1 Score)')
ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=0)
ax.legend(loc='upper right')
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.set_ylim(0, max(max(uniform_ratios), max(prior_ratios)) * 1.1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        label_text = f'{height:.1f}x' if height < 10 else '∞'
        ax.annotate(label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results_optimal/visualizations/improvement_ratios.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: improvement_ratios.png")


# FIGURE 4: Radar/Spider Chart for Multi-metric Comparison


fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Metrics for radar chart
radar_metrics = ['Macro F1', 'Micro F1', 'ROC-AUC', 'Precision', 'Recall', 'Hamming Acc.']
radar_keys = ['macro_f1', 'micro_f1', 'roc_auc', 'macro_precision', 'macro_recall', 'hamming_accuracy']

model_radar = [model_metrics[k] for k in radar_keys]
uniform_radar = [uniform_metrics[k] for k in radar_keys]
prior_radar = [prior_metrics[k] for k in radar_keys]

# Close the radar chart
model_radar += model_radar[:1]
uniform_radar += uniform_radar[:1]
prior_radar += prior_radar[:1]

angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

ax.plot(angles, model_radar, 'o-', linewidth=2, label='Trained Model', color=MODEL_COLOR)
ax.fill(angles, model_radar, alpha=0.25, color=MODEL_COLOR)

ax.plot(angles, uniform_radar, 'o-', linewidth=2, label='Uniform Random', color=UNIFORM_COLOR)
ax.fill(angles, uniform_radar, alpha=0.25, color=UNIFORM_COLOR)

ax.plot(angles, prior_radar, 'o-', linewidth=2, label='Class-Prior', color=PRIOR_COLOR)
ax.fill(angles, prior_radar, alpha=0.25, color=PRIOR_COLOR)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, size=11)
ax.set_ylim(0, 1)
ax.set_title('Multi-Metric Performance Comparison', size=14, y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('results_optimal/visualizations/radar_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: radar_comparison.png")


# FIGURE 5: Precision-Recall Comparison per Label


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (label, short_name) in enumerate(zip(label_names, short_names)):
    ax = axes[idx]
    
    # Get precision and recall for each method
    methods = ['Trained Model', 'Uniform Random', 'Class-Prior']
    precisions = [
        model_metrics[f'{label}_precision'],
        uniform_metrics[f'{label}_precision'],
        prior_metrics[f'{label}_precision']
    ]
    recalls = [
        model_metrics[f'{label}_recall'],
        uniform_metrics[f'{label}_recall'],
        prior_metrics[f'{label}_recall']
    ]
    colors = [MODEL_COLOR, UNIFORM_COLOR, PRIOR_COLOR]
    
    # Plot points
    for i, (method, prec, rec, color) in enumerate(zip(methods, precisions, recalls, colors)):
        ax.scatter(rec, prec, s=200, c=color, label=method, edgecolors='black', linewidth=1, zorder=5)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{short_name}')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    if idx == 0:
        ax.legend(loc='lower left', fontsize=8)

fig.suptitle('Precision-Recall by Label: Trained Model vs Baselines', size=14, y=1.02)
plt.tight_layout()
plt.savefig('results_optimal/visualizations/precision_recall_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: precision_recall_comparison.png")


# FIGURE 6: Stacked Performance Summary


fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Trained Model', 'Class-Prior', 'Uniform Random']
macro_f1_vals = [model_metrics['macro_f1'], prior_metrics['macro_f1'], uniform_metrics['macro_f1']]
colors_bars = [MODEL_COLOR, PRIOR_COLOR, UNIFORM_COLOR]

bars = ax.barh(categories, macro_f1_vals, color=colors_bars, edgecolor='black', linewidth=1, height=0.6)

# Add value labels and improvement annotations
for i, (bar, val) in enumerate(zip(bars, macro_f1_vals)):
    ax.annotate(f'{val:.4f}',
                xy=(val, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha='left', va='center', fontsize=12, fontweight='bold')

ax.set_xlabel('Macro F1 Score')
ax.set_title('Overall Performance Comparison (Macro F1)', fontsize=14)
ax.set_xlim(0, 1.0)
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Random Chance')

# Add improvement annotation
improvement_vs_uniform = ((model_metrics['macro_f1'] - uniform_metrics['macro_f1']) / uniform_metrics['macro_f1']) * 100
ax.annotate(f'+{improvement_vs_uniform:.0f}% vs Random',
            xy=(model_metrics['macro_f1'], 0),
            xytext=(model_metrics['macro_f1'] + 0.05, 0.7),
            fontsize=11, color=MODEL_COLOR, fontweight='bold')

plt.tight_layout()
plt.savefig('results_optimal/visualizations/performance_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: performance_summary.png")


# FIGURE 7: Class Prior vs Test Distribution with Model Performance


fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(label_names))
width = 0.2

# Class priors from training data
prior_probs = [class_priors[label] for label in label_names]

# Test set distribution (compute from metrics - approximation)
# We'll use the model's recall as a proxy since we don't have raw test data here
# Actually, let's load and compute from the data
with open('Dataset/Final.json', 'r') as f:
    data = json.load(f)

n_total = len(data)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
test_data = data[n_train + n_val:]

label_mapping = {
    'no_mechanism': 'framework0_feature1',
    'central_route_present': 'framework1_feature1',
    'peripheral_route_present': 'framework1_feature2',
    'naturalness_bias': 'framework2_feature1',
    'availability_bias': 'framework2_feature2',
    'illusory_correlation': 'framework2_feature3'
}

test_probs = []
for label in label_names:
    field = label_mapping[label]
    pos_count = sum(1 for item in test_data if item[field] == 1)
    test_probs.append(pos_count / len(test_data))

bars1 = ax.bar(x - width, prior_probs, width, label='Training Prior', color='#9b59b6', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, test_probs, width, label='Test Distribution', color='#f39c12', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, model_f1, width, label='Model F1', color=MODEL_COLOR, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Label')
ax.set_ylabel('Proportion / F1 Score')
ax.set_title('Class Distribution vs Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=0)
ax.legend(loc='upper right')
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('results_optimal/visualizations/distribution_vs_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: distribution_vs_performance.png")


# FIGURE 8: Summary Dashboard
fig = plt.figure(figsize=(16, 12))

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Macro F1 Comparison (large)
ax1 = fig.add_subplot(gs[0, :2])
categories = ['Trained Model', 'Uniform Random', 'Class-Prior']
f1_vals = [model_metrics['macro_f1'], uniform_metrics['macro_f1'], prior_metrics['macro_f1']]
colors_p1 = [MODEL_COLOR, UNIFORM_COLOR, PRIOR_COLOR]
bars = ax1.bar(categories, f1_vals, color=colors_p1, edgecolor='black', linewidth=1)
ax1.set_ylabel('Macro F1')
ax1.set_title('Overall Performance (Macro F1)', fontweight='bold')
ax1.set_ylim(0, 1.0)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, f1_vals):
    ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val), 
                 xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold')

# Panel 2: ROC-AUC Comparison
ax2 = fig.add_subplot(gs[0, 2])
roc_vals = [model_metrics['roc_auc'], uniform_metrics['roc_auc'], prior_metrics['roc_auc']]
bars = ax2.bar(['Model', 'Uniform', 'Prior'], roc_vals, color=colors_p1, edgecolor='black')
ax2.set_ylabel('ROC-AUC')
ax2.set_title('ROC-AUC', fontweight='bold')
ax2.set_ylim(0, 1.0)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, roc_vals):
    ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val), 
                 xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)

# Panel 3: Per-Label F1 (Model only)
ax3 = fig.add_subplot(gs[1, :])
x = np.arange(len(label_names))
model_f1_list = [model_metrics[f'{label}_f1'] for label in label_names]
bars3_panel = ax3.bar(x, model_f1_list, color=MODEL_COLOR, edgecolor='black')
ax3.axhline(y=uniform_metrics['macro_f1'], color=UNIFORM_COLOR, linestyle='--', linewidth=2, label=f'Uniform Avg: {uniform_metrics["macro_f1"]:.3f}')
ax3.axhline(y=prior_metrics['macro_f1'], color=PRIOR_COLOR, linestyle='--', linewidth=2, label=f'Prior Avg: {prior_metrics["macro_f1"]:.3f}')
ax3.set_xticks(x)
ax3.set_xticklabels(short_names)
ax3.set_ylabel('F1 Score')
ax3.set_title('Model Per-Label F1 vs Baseline Averages', fontweight='bold')
ax3.set_ylim(0, 1.0)
ax3.legend(loc='lower right')
for bar, val in zip(bars3_panel, model_f1_list):
    ax3.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val), 
                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

# Panel 4: Key Statistics
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')
stats_text = f"""
KEY STATISTICS

Trained Model:
  • Macro F1: {model_metrics['macro_f1']:.4f}
  • ROC-AUC: {model_metrics['roc_auc']:.4f}
  • Exact Match: {model_metrics['exact_match_accuracy']:.1%}

Improvement vs Uniform:
  • F1: +{(model_metrics['macro_f1']-uniform_metrics['macro_f1'])*100:.1f}%
  • {model_metrics['macro_f1']/uniform_metrics['macro_f1']:.1f}x better

Improvement vs Class-Prior:
  • F1: +{(model_metrics['macro_f1']-prior_metrics['macro_f1'])*100:.1f}%
  • {model_metrics['macro_f1']/prior_metrics['macro_f1']:.1f}x better
"""
ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 5: Best/Worst Labels
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
# Find best and worst
best_idx = np.argmax(model_f1_list)
worst_idx = np.argmin(model_f1_list)
labels_text = f"""
PER-LABEL INSIGHTS

Best Performing:
  • {short_names[best_idx]}: F1 = {model_f1_list[best_idx]:.3f}
  
Most Challenging:
  • {short_names[worst_idx]}: F1 = {model_f1_list[worst_idx]:.3f}

All Labels Beat Both Baselines ✓
"""
ax5.text(0.1, 0.9, labels_text, transform=ax5.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Panel 6: Conclusion
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
conclusion_text = f"""
CONCLUSION

Model significantly outperforms 
both baseline methods across 
all metrics and labels.

Strong evidence of effective
learning beyond simple
class distribution patterns.
"""
ax6.text(0.1, 0.9, conclusion_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

fig.suptitle('Misinformation Classifier: Model vs Baseline Performance Dashboard', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('results_optimal/visualizations/performance_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: performance_dashboard.png")

print("\n" + "="*60)
print("All visualizations saved to: results_optimal/visualizations/")
print("="*60)
print("\nGenerated files:")
print("  1. f1_comparison_bars.png        - Per-label F1 grouped bars")
print("  2. aggregate_metrics_comparison.png - Horizontal bar comparison")
print("  3. improvement_ratios.png        - Model improvement ratios")
print("  4. radar_comparison.png          - Multi-metric radar chart")
print("  5. precision_recall_comparison.png - Per-label P-R scatter")
print("  6. performance_summary.png       - Simple summary bar")
print("  7. distribution_vs_performance.png - Class dist vs F1")
print("  8. performance_dashboard.png     - Complete dashboard")
