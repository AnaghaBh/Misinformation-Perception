#!/usr/bin/env python3
"""
Stratified Sampling Script for Misinformation Dataset

This script selects a balanced sample from the full dataset while maintaining:
1. Balanced topic distribution (health vs technology)
2. Balanced feature distributions across all 5 labels
"""

import json
import argparse
import random
from collections import defaultdict
from typing import List, Dict, Any
import os

def load_data(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r') as f:
        return json.load(f)

def save_data(data: List[Dict[str, Any]], filepath: str):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to {filepath}")

def compute_label_vector(item: Dict) -> tuple:
    return (
        item.get('framework1_feature1', 0),
        item.get('framework1_feature2', 0),
        item.get('framework2_feature1', 0),
        item.get('framework2_feature2', 0),
        item.get('framework2_feature3', 0)
    )

# print distribution of topics and features
def print_distribution(data: List[Dict], title: str = "Distribution"):
    print(f"\n{'='*60}")
    print(f"{title} (n={len(data)})")
    print('='*60)
    
    # Topic distribution
    topics = defaultdict(int)
    for item in data:
        topics[item.get('topic', 'unknown')] += 1
    
    print("\nTopic Distribution:")
    for topic, count in sorted(topics.items()):
        pct = count / len(data) * 100
        print(f"  {topic}: {count} ({pct:.1f}%)")
    
    # Feature distribution
    features = {
        'central_route (f1_f1)': 'framework1_feature1',
        'peripheral_route (f1_f2)': 'framework1_feature2',
        'naturalness_bias (f2_f1)': 'framework2_feature1',
        'availability_bias (f2_f2)': 'framework2_feature2',
        'illusory_correlation (f2_f3)': 'framework2_feature3'
    }
    
    print("\nFeature Distribution:")
    for name, field in features.items():
        pos = sum(1 for item in data if item.get(field, 0) == 1)
        pct = pos / len(data) * 100
        print(f"  {name}: {pos} ({pct:.1f}%)")
    
    # Label combination distribution
    combos = defaultdict(int)
    for item in data:
        vec = compute_label_vector(item)
        combos[vec] += 1
    
    print(f"\nUnique label combinations: {len(combos)}")

def stratified_sample(
    data: List[Dict[str, Any]], 
    sample_size: int = 200,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Creating a stratified sample that balances:
    1. Topic (health vs technology)
    2. All 5 feature labels
    
    Strategy:
    - First, group by topic
    - Within each topic, group by label combination
    - Sample proportionally from each stratum
    - Use iterative balancing to optimize feature distributions
    """
    random.seed(seed)
    
    # Group data by topic and label combination
    strata = defaultdict(list)
    for item in data:
        topic = item.get('topic', 'unknown')
        label_vec = compute_label_vector(item)
        key = (topic, label_vec)
        strata[key].append(item)
    
    print(f"\nFound {len(strata)} unique strata (topic × label combination)")
    
    # Calculate target samples per topic (balanced 50/50)
    samples_per_topic = sample_size // 2
    
    # Separate strata by topic
    health_strata = {k: v for k, v in strata.items() if k[0] == 'health'}
    tech_strata = {k: v for k, v in strata.items() if k[0] == 'technology'}
    
    def sample_from_strata(topic_strata: Dict, target_size: int) -> List[Dict]:
        """Sample from strata while trying to balance features."""
        sampled = []
        
        # Calculate how many to sample from each stratum (proportional)
        total_in_strata = sum(len(v) for v in topic_strata.values())
        
        # First pass: proportional sampling
        for key, items in topic_strata.items():
            # Proportional allocation
            n_samples = max(1, int(len(items) / total_in_strata * target_size))
            n_samples = min(n_samples, len(items))
            sampled.extend(random.sample(items, n_samples))
        
        # Adjust to exact target size
        if len(sampled) > target_size:
            sampled = random.sample(sampled, target_size)
        elif len(sampled) < target_size:
            # Need more samples - pick from underrepresented features
            remaining = target_size - len(sampled)
            sampled_ids = {item['id'] for item in sampled}
            available = [item for items in topic_strata.values() 
                        for item in items if item['id'] not in sampled_ids]
            
            if available:
                # Score items by how much they help balance features
                current_counts = [0] * 5
                for item in sampled:
                    vec = compute_label_vector(item)
                    for i, v in enumerate(vec):
                        current_counts[i] += v
                
                target_per_feature = target_size * 0.4  
                
                def balance_score(item):
                    """Higher score = helps balance more."""
                    vec = compute_label_vector(item)
                    score = 0
                    for i, v in enumerate(vec):
                        if v == 1 and current_counts[i] < target_per_feature:
                            score += (target_per_feature - current_counts[i])
                        elif v == 0 and current_counts[i] > target_per_feature:
                            score += (current_counts[i] - target_per_feature)
                    return score + random.random() * 0.1 
                
                available.sort(key=balance_score, reverse=True)
                sampled.extend(available[:remaining])
        
        return sampled
    
    # Sample from each topic
    health_samples = sample_from_strata(health_strata, samples_per_topic)
    tech_samples = sample_from_strata(tech_strata, samples_per_topic)
    
    # Combine and shuffle
    final_sample = health_samples + tech_samples
    random.shuffle(final_sample)
    
    return final_sample

def iterative_balance(
    data: List[Dict[str, Any]], 
    sample_size: int = 200,
    seed: int = 42,
    iterations: int = 1000
) -> List[Dict[str, Any]]:
    """
    Use iterative optimization to find a well-balanced sample.
    
    This approach:
    1. Starts with a random sample
    2. Iteratively swaps items to improve balance
    3. Optimizes for both topic and feature balance
    """
    random.seed(seed)
    
    # Separate by topic
    health_items = [item for item in data if item.get('topic') == 'health']
    tech_items = [item for item in data if item.get('topic') == 'technology']
    
    # Start with balanced topic selection
    n_per_topic = sample_size // 2
    current_health = random.sample(health_items, n_per_topic)
    current_tech = random.sample(tech_items, n_per_topic)
    
    def compute_balance_score(health_sample, tech_sample):
        """Lower score = better balance."""
        combined = health_sample + tech_sample
        n = len(combined)
        
        # Feature balance: deviation from 50%
        feature_fields = [
            'framework1_feature1', 'framework1_feature2',
            'framework2_feature1', 'framework2_feature2', 'framework2_feature3'
        ]
        
        score = 0
        for field in feature_fields:
            pos_count = sum(1 for item in combined if item.get(field, 0) == 1)
            ratio = pos_count / n
            # Penalize deviation from 50%
            score += abs(ratio - 0.5) ** 2
        
        return score
    
    best_score = compute_balance_score(current_health, current_tech)
    best_health = current_health.copy()
    best_tech = current_tech.copy()
    
    # Get items not in current sample
    health_ids = {item['id'] for item in current_health}
    tech_ids = {item['id'] for item in current_tech}
    health_pool = [item for item in health_items if item['id'] not in health_ids]
    tech_pool = [item for item in tech_items if item['id'] not in tech_ids]
    
    for i in range(iterations):
        # Randomly decide to swap health or tech
        if random.random() < 0.5 and health_pool:
            # Swap a health item
            swap_out_idx = random.randint(0, len(current_health) - 1)
            swap_in = random.choice(health_pool)
            
            old_item = current_health[swap_out_idx]
            current_health[swap_out_idx] = swap_in
            
            new_score = compute_balance_score(current_health, current_tech)
            
            if new_score < best_score:
                best_score = new_score
                best_health = current_health.copy()
                best_tech = current_tech.copy()
                health_pool.remove(swap_in)
                health_pool.append(old_item)
            else:
                current_health[swap_out_idx] = old_item
                
        elif tech_pool:
            # Swap a tech item
            swap_out_idx = random.randint(0, len(current_tech) - 1)
            swap_in = random.choice(tech_pool)
            
            old_item = current_tech[swap_out_idx]
            current_tech[swap_out_idx] = swap_in
            
            new_score = compute_balance_score(current_health, current_tech)
            
            if new_score < best_score:
                best_score = new_score
                best_health = current_health.copy()
                best_tech = current_tech.copy()
                tech_pool.remove(swap_in)
                tech_pool.append(old_item)
            else:
                current_tech[swap_out_idx] = old_item
    
    final_sample = best_health + best_tech
    random.shuffle(final_sample)
    
    print(f"\nOptimization complete. Final balance score: {best_score:.6f}")
    
    return final_sample

def main():
    parser = argparse.ArgumentParser(
        description='Create a balanced stratified sample from the misinformation dataset'
    )
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        default='../Full_Final/Final.json',
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        default='sampled_200.json',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--size', '-n', 
        type=int, 
        default=200,
        help='Number of samples to select (default: 200)'
    )
    parser.add_argument(
        '--seed', '-s', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['stratified', 'iterative'],
        default='iterative',
        help='Sampling method: stratified or iterative (default: iterative)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5000,
        help='Number of optimization iterations for iterative method (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    # Load data
    print(f"Loading data from {input_path}...")
    data = load_data(input_path)
    print(f"Loaded {len(data)} samples")
    
    # Print original distribution
    print_distribution(data, "Original Dataset Distribution")
    
    print(f"\nCreating balanced sample of {args.size} items using {args.method} method...")
    
    if args.method == 'stratified':
        sample = stratified_sample(data, args.size, args.seed)
    else:
        sample = iterative_balance(data, args.size, args.seed, args.iterations)
    
    print_distribution(sample, "Sampled Dataset Distribution")
    
    # Reassign IDs (1 to n)
    for i, item in enumerate(sample, 1):
        item['original_id'] = item['id']
        item['id'] = i
    
    # Save
    save_data(sample, output_path)
    
    print(f"\n✓ Successfully created balanced sample of {len(sample)} headlines")
    print(f"  Output saved to: {output_path}")

if __name__ == "__main__":
    main()
