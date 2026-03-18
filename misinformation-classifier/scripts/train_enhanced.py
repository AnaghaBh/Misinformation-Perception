#!/usr/bin/env python3
"""
Enhanced Training Launcher - Run training with comprehensive logging and weights saving.
"""

import argparse
import os
import sys
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Launch enhanced training with epoch logging')
    parser.add_argument('--data', required=True, help='Path to training dataset')
    parser.add_argument('--test-data', help='Path to separate test dataset (optional)')
    parser.add_argument('--output-dir', help='Output directory (default: results_TIMESTAMP)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=6.98e-05, help='Learning rate (default: 6.98e-05)')
    
    args = parser.parse_args()
    
    # Create timestamped output directory if not specified
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results_{timestamp}"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    
    cmd_parts = [
        sys.executable, 'src/train.py',
        '--data_path', args.data,
        '--output_dir', args.output_dir
    ]
    
    if args.test_data:
        cmd_parts.extend(['--test_path', args.test_data])
    
    # Temp Config
    if (args.epochs != 20 or args.batch_size != 16 or 
        args.learning_rate != 6.98e-05):
        
        print(f"Creating custom config with:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")  
        print(f"  Learning rate: {args.learning_rate}")
        print("Note: Custom parameters require modifying src/config.py")
    
    print(f"Starting training...")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"Command: {' '.join(cmd_parts)}")
    print("-" * 60)
    
    # Execute training
    os.execv(sys.executable, cmd_parts)


if __name__ == '__main__':
    main()