#!/usr/bin/env python3
"""
Preprocess to combine set1.json and set2.json into Final.json for DistilBERT training.
"""

import json
import re
import unicodedata
from pathlib import Path


def clean_text_for_bert(text: str) -> str:
    """
    Clean and preprocess text for DistilBERT training.
    
    Steps:
    1. Normalize unicode characters
    2. Remove or replace special characters
    3. Normalize whitespace
    4. Handle encoding issues
    """
    if not text or not isinstance(text, str):
        return ""
    
    #(NFC normalization)
    text = unicodedata.normalize('NFC', text)
    
    # Replace problematic characters
    text = text.replace('\u2019', "'")  # Right single quotation mark
    text = text.replace('\u2018', "'")  # Left single quotation mark
    text = text.replace('\u201c', '"')  # Left double quotation mark
    text = text.replace('\u201d', '"')  # Right double quotation mark
    text = text.replace('\u2014', '-')  # Em dash
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2026', '...')  # Ellipsis
    text = text.replace('\xa0', ' ')  # Non-breaking space
    
    # Remove control characters
    text = ''.join(char for char in text if unicodedata.category(char) != 'Cc' or char in '\n\t')
    
    # Normalize whitespace (multiple spaces/tabs to single space)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize newlines
    text = re.sub(r'\n+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Handle empty or very short texts
    if len(text) < 3:
        return ""
    
    return text


def validate_features(entry: dict) -> bool:
    """
    Validate that all required features exist and are valid (0 or 1).
    """
    required_features = [
        'framework1_feature1',
        'framework1_feature2',
        'framework2_feature1',
        'framework2_feature2',
        'framework2_feature3'
    ]
    
    for feature in required_features:
        if feature not in entry:
            return False
        if entry[feature] not in [0, 1]:
            return False
    
    return True

def preprocess_entry(entry: dict, new_id: int) -> dict | None:
    # Clean the text
    cleaned_text = clean_text_for_bert(entry.get('text', ''))
    
    # Skip entries with empty text after cleaning
    if not cleaned_text:
        print(f"Warning: Skipping entry with original ID {entry.get('id', 'unknown')} - empty text after cleaning")
        return None
    
    # Validate features
    if not validate_features(entry):
        print(f"Warning: Entry with original ID {entry.get('id', 'unknown')} has invalid features, attempting to fix")
    
    f1_f1 = int(entry.get('framework1_feature1', 0))
    f1_f2 = int(entry.get('framework1_feature2', 0))
    f2_f1 = int(entry.get('framework2_feature1', 0))
    f2_f2 = int(entry.get('framework2_feature2', 0))
    f2_f3 = int(entry.get('framework2_feature3', 0))
    
    # Calculate framework0_feature1: 1 if ALL other features are 0, else 0
    all_features_zero = (f1_f1 == 0 and f1_f2 == 0 and f2_f1 == 0 and f2_f2 == 0 and f2_f3 == 0)
    f0_f1 = 1 if all_features_zero else 0
    
    # Create standardized entry with consistent field order
    preprocessed = {
        'id': new_id,
        'text': cleaned_text,
        'framework0_feature1': f0_f1,
        'framework1_feature1': f1_f1,
        'framework1_feature2': f1_f2,
        'framework2_feature1': f2_f1,
        'framework2_feature2': f2_f2,
        'framework2_feature3': f2_f3
    }
    
    return preprocessed


def load_json_file(filepath: Path) -> list:
    """Load and parse a JSON file."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entries")
    return data


def main():
    # Define paths
    dataset_dir = Path(__file__).parent
    file_100 = dataset_dir / 'Set1.json'
    file_200 = dataset_dir / 'Set2.json'
    output_file = dataset_dir / 'Final.json'
    data_100 = load_json_file(file_100)
    data_200 = load_json_file(file_200)
    
    all_entries = data_100 + data_200
    print(f"\nTotal entries before preprocessing: {len(all_entries)}")
    
    # Preprocess and assign new sequential IDs
    processed_entries = []
    new_id = 1
    skipped = 0
    
    for entry in all_entries:
        processed = preprocess_entry(entry, new_id)
        if processed:
            processed_entries.append(processed)
            new_id += 1
        else:
            skipped += 1
    
    print(f"\nPreprocessing complete:")
    print(f"  Total processed: {len(processed_entries)}")
    print(f"  Skipped (invalid): {skipped}")
    print(f"  New ID range: 1 to {len(processed_entries)}")
    
    # Calculate statistics
    text_lengths = [len(entry['text']) for entry in processed_entries]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    max_length = max(text_lengths) if text_lengths else 0
    min_length = min(text_lengths) if text_lengths else 0
    
    print(f"\nText statistics:")
    print(f"  Average length: {avg_length:.1f} characters")
    print(f"  Min length: {min_length} characters")
    print(f"  Max length: {max_length} characters")
    
    # Feature distributions
    print(f"\nFeature distributions:")
    for feature in ['framework0_feature1', 'framework1_feature1', 'framework1_feature2', 
                    'framework2_feature1', 'framework2_feature2', 'framework2_feature3']:
        count = sum(1 for e in processed_entries if e[feature] == 1)
        pct = (count / len(processed_entries)) * 100 if processed_entries else 0
        print(f"  {feature}: {count} ({pct:.1f}%)")
    
    # Save to Final.json
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_entries, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully created {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
