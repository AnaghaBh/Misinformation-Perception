#!/usr/bin/env python3
"""
Script to combine annotated JSON files:
- FinalF1 files + F1_Annotated.json -> Full_Final_F1/combined.json
- FinalF2 files + F2_Annotated.json -> Full_Final_F2/combined.json

Ensures all IDs are unique and non-overlapping.
"""

import json
import os
from pathlib import Path

def load_json(filepath):
    """Load JSON file and return data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} samples to: {filepath}")

def combine_files(main_file, additional_dir, output_file, label):
    """Combine main annotated file with additional files from a directory."""
    print(f"\n{'='*60}")
    print(f"Processing {label}")
    print('='*60)
    
    # Load main file
    print(f"\nLoading main file: {main_file}")
    main_data = load_json(main_file)
    print(f"  Loaded {len(main_data)} samples")
    
    # First, fix any duplicate IDs in the main file by re-assigning sequential IDs
    original_ids = [item['id'] for item in main_data]
    if len(original_ids) != len(set(original_ids)):
        print(f"  Found duplicate IDs in main file, re-assigning sequential IDs...")
        for i, item in enumerate(main_data, start=1):
            item['id'] = i
    
    # Get max ID from main file
    max_id = max(item['id'] for item in main_data)
    print(f"  Max ID in main file: {max_id}")
    
    # Load all additional files
    additional_files = sorted(Path(additional_dir).glob('*.json'))
    print(f"\nFound {len(additional_files)} additional files in {additional_dir}:")
    
    all_additional_data = []
    for filepath in additional_files:
        data = load_json(filepath)
        print(f"  {filepath.name}: {len(data)} samples")
        all_additional_data.extend(data)
    
    print(f"\nTotal additional samples: {len(all_additional_data)}")
    
    # Re-assign IDs to additional data starting from max_id + 1
    next_id = max_id + 1
    for item in all_additional_data:
        item['id'] = next_id
        next_id += 1
    
    print(f"Re-assigned IDs from {max_id + 1} to {next_id - 1}")
    
    # Combine all data
    combined_data = main_data + all_additional_data
    print(f"\nCombined dataset: {len(combined_data)} total samples")
    
    # Verify no duplicate IDs
    all_ids = [item['id'] for item in combined_data]
    if len(all_ids) != len(set(all_ids)):
        print("WARNING: Duplicate IDs still detected! Force re-assigning all...")
        for i, item in enumerate(combined_data, start=1):
            item['id'] = i
        print(f"  Re-assigned all IDs from 1 to {len(combined_data)}")
    else:
        print("✓ All IDs are unique")
    
    # Save combined file
    save_json(combined_data, output_file)
    
    return combined_data

def main():
    # Base paths
    base_dir = Path("/Users/anshmadan/Desktop/Capstone BERT/Capstone-AS02")
    
    # Input paths
    f1_main = base_dir / "F1_Annotated.json"
    f2_main = base_dir / "F2_Annotated.json"
    f1_additional = base_dir / "AdditionalFiles" / "FinalF1"
    f2_additional = base_dir / "AdditionalFiles" / "FinalF2"
    
    # Output paths
    output_dir = base_dir / "Full_Final"
    f1_output = output_dir / "Full_Final_F1.json"
    f2_output = output_dir / "Full_Final_F2.json"
    
    print("="*60)
    print("Combining Annotated Dataset Files")
    print("="*60)
    
    # Process F1
    f1_combined = combine_files(
        main_file=f1_main,
        additional_dir=f1_additional,
        output_file=f1_output,
        label="F1 (Elaboration Likelihood Model)"
    )
    
    # Process F2
    f2_combined = combine_files(
        main_file=f2_main,
        additional_dir=f2_additional,
        output_file=f2_output,
        label="F2 (Cognitive Biases)"
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Full_Final_F1.json: {len(f1_combined)} samples")
    print(f"Full_Final_F2.json: {len(f2_combined)} samples")
    print(f"\nOutput directory: {output_dir}")
    print("\nTo train on these files:")
    print(f"  python src/train.py --data_path {f1_output} --output_dir results_f1")
    print(f"  python src/train.py --data_path {f2_output} --output_dir results_f2")

if __name__ == "__main__":
    main()
