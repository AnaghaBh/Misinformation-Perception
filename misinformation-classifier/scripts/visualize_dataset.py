import os
import json
import argparse
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

LABEL_NAMES: List[str] = [
    "central_route_present",
    "peripheral_route_present",
    "naturalness_bias",
    "availability_bias",
    "illusory_correlation",
]

# Map model labels to dataset fields
FIELD_MAPPING = {
    'central_route_present': 'framework1_feature1',
    'peripheral_route_present': 'framework1_feature2',
    'naturalness_bias': 'framework2_feature1',
    'availability_bias': 'framework2_feature2',
    'illusory_correlation': 'framework2_feature3',
}


def load_df(input_path: str) -> pd.DataFrame:
    """Load JSON list or CSV into DataFrame with expected columns."""
    if input_path.endswith('.json'):
        with open(input_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Input file must be .json or .csv")

    # Ensure required columns exist
    required_cols = ['text'] + list(FIELD_MAPPING.values())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def plot_label_distribution(df: pd.DataFrame, out_dir: str):
    """Bar chart of positive counts for each label."""
    counts = {
        label: int(df[FIELD_MAPPING[label]].sum())
        for label in LABEL_NAMES
    }
    total = len(df)

    plt.figure(figsize=(8, 5))
    labels = list(counts.keys())
    values = list(counts.values())
    bars = plt.bar(labels, values, color='#1f77b4')
    plt.ylabel('Positive count')
    plt.xlabel('Label')
    plt.title(f'Label positives (n={total})')
    plt.xticks(rotation=30, ha='right')
    # Add value labels
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val),
                 ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'label_distribution.png'), dpi=200)
    plt.close()


def plot_label_correlation(df: pd.DataFrame, out_dir: str):
    """Heatmap of label correlations."""
    label_cols = [FIELD_MAPPING[l] for l in LABEL_NAMES]
    arr = df[label_cols].to_numpy(dtype=float)
    corr = np.corrcoef(arr, rowvar=False)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=30, ha='right')
    plt.yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    plt.title('Label correlation heatmap')
    # Annotate cells
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'label_correlation.png'), dpi=200)
    plt.close()


def plot_text_length(df: pd.DataFrame, out_dir: str):
    """Histogram of text lengths (tokens approximated by whitespace split)."""
    lengths = df['text'].astype(str).apply(lambda s: len(s.split()))

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=40, color='teal', alpha=0.8)
    plt.xlabel('Text length (approx. tokens)')
    plt.ylabel('Count')
    plt.title('Text length distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'text_length_distribution.png'), dpi=200)
    plt.close()


def plot_label_cardinality(df: pd.DataFrame, out_dir: str):
    """Distribution of number of positive labels per sample."""
    label_cols = [FIELD_MAPPING[l] for l in LABEL_NAMES]
    cardinality = df[label_cols].to_numpy().sum(axis=1)
    counts = Counter(cardinality)
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]

    plt.figure(figsize=(7, 5))
    plt.bar(xs, ys, color='#9467bd')
    plt.xlabel('Positive labels per sample')
    plt.ylabel('Count')
    plt.title('Label cardinality distribution')
    for x, y in zip(xs, ys):
        plt.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'label_cardinality.png'), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize training dataset')
    parser.add_argument('--input', required=True, help='Path to JSON or CSV dataset')
    parser.add_argument('--output-dir', default='results/visualizations', help='Directory to save plots')
    parser.add_argument('--limit', type=int, default=None, help='Optional sample size for speed')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_df(args.input)

    if args.limit is not None and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)

    # Generate plots
    plot_label_distribution(df, args.output_dir)
    plot_label_correlation(df, args.output_dir)
    plot_text_length(df, args.output_dir)
    plot_label_cardinality(df, args.output_dir)

    print(f"Saved visualizations to {args.output_dir}")


if __name__ == '__main__':
    main()
