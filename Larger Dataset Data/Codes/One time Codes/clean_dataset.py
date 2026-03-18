#!/usr/bin/env python3
"""
Script to clean Final.json by removing entries missing any framework features.
"""

import json

def main():
    # Load the combined file
    with open('../Full_Final/Final.json', 'r') as f:
        data = json.load(f)

    print(f'Original dataset: {len(data)} samples')

    # Required fields for all 5 labels
    required_fields = [
        'framework1_feature1',  # central_route_present
        'framework1_feature2',  # peripheral_route_present
        'framework2_feature1',  # naturalness_bias
        'framework2_feature2',  # availability_bias
        'framework2_feature3'   # illusory_correlation
    ]

    # Filter to keep only entries with ALL required fields
    valid_data = []
    missing_counts = {field: 0 for field in required_fields}

    for item in data:
        has_all_fields = True
        for field in required_fields:
            if field not in item:
                missing_counts[field] += 1
                has_all_fields = False
        
        if has_all_fields:
            valid_data.append(item)

    print(f'\nMissing field counts:')
    for field, count in missing_counts.items():
        print(f'  {field}: {count} entries missing')

    print(f'\nRemoved: {len(data) - len(valid_data)} entries')
    print(f'Remaining: {len(valid_data)} valid entries')

    # Re-assign sequential IDs
    for i, item in enumerate(valid_data, start=1):
        item['id'] = i

    print(f'Re-assigned IDs: 1 to {len(valid_data)}')

    # Save cleaned file
    with open('../Full_Final/Final.json', 'w') as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)

    print(f'\n✓ Saved cleaned Final.json with {len(valid_data)} samples')

if __name__ == "__main__":
    main()
