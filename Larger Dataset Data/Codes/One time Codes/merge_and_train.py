#!/usr/bin/env python3
"""Script to merge F1 and F2 annotated files and train the model."""

import json
import os
import sys

def merge_datasets(f1_path: str, f2_path: str, output_path: str):
    """Merge F1 and F2 annotated JSON files, removing duplicates by ID."""
    
    print(f"Loading F1 data from: {f1_path}")
    with open(f1_path, 'r') as f:
        f1_data = json.load(f)
    print(f"  Loaded {len(f1_data)} samples")
    
    print(f"Loading F2 data from: {f2_path}")
    with open(f2_path, 'r') as f:
        f2_data = json.load(f)
    print(f"  Loaded {len(f2_data)} samples")
    
    # Merge by ID to avoid duplicates
    merged = {}
    for item in f1_data:
        merged[item['id']] = item
    
    for item in f2_data:
        if item['id'] not in merged:
            merged[item['id']] = item
        else:
            # Update existing entry with F2 data (in case labels differ)
            merged[item['id']].update(item)
    
    merged_list = list(merged.values())
    print(f"\nMerged dataset: {len(merged_list)} unique samples")
    
    # Save merged data
    with open(output_path, 'w') as f:
        json.dump(merged_list, f, indent=2)
    print(f"Saved merged data to: {output_path}")
    
    return output_path

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    f1_path = os.path.join(base_dir, "..", "F1_Annotated.json")
    f2_path = os.path.join(base_dir, "..", "F2_Annotated.json")
    output_path = os.path.join(base_dir, "data", "raw", "merged_annotated.json")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Merge datasets
    merged_path = merge_datasets(f1_path, f2_path, output_path)
    
    print("\n" + "="*50)
    print("To train the model, run:")
    print(f"  python src/train.py --data_path {merged_path} --output_dir results")
    print("="*50)

if __name__ == "__main__":
    main()
