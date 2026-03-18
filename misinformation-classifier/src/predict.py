"""Prediction script for misinformation classifier."""

import argparse
import logging
import torch
import pandas as pd
import json
from typing import List, Dict

from config import Config
from model import build_model
from utils import setup_logging

def load_trained_model(model_path: str, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config, checkpoint['tokenizer']

def predict_single(text: str, model, tokenizer, config, device, threshold: float = 0.5) -> Dict:
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=config.max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
    
    results = {
        'text': text,
        'probabilities': {},
        'predictions': {}
    }
    
    for i, label in enumerate(config.label_names):
        prob = float(probabilities[i])
        pred = int(prob > threshold)
        
        results['probabilities'][label] = prob
        results['predictions'][label] = pred
    
    return results

def predict_batch(texts: List[str], model, tokenizer, config, device, threshold: float = 0.5) -> List[Dict]:
    results = []
    
    for text in texts:
        result = predict_single(text, model, tokenizer, config, device, threshold)
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Predict with misinformation classifier')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--text', type=str, help='Single text to classify')
    parser.add_argument('--input_file', type=str, help='File with texts to classify')
    parser.add_argument('--output_file', type=str, help='Output file for predictions')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    args = parser.parse_args()
    
    if not args.text and not args.input_file:
        parser.error("Either --text or --input_file must be provided")
    
    setup_logging()
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    logging.info(f"Loading model from {args.model_path}")
    model, config, tokenizer = load_trained_model(args.model_path, device)
    
    if args.text:
        result = predict_single(args.text, model, tokenizer, config, device, args.threshold)
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Text: {result['text']}")
        print("\nProbabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label:25}: {prob:.4f}")
        
        print("\nPredictions (threshold={:.2f}):".format(args.threshold))
        for label, pred in result['predictions'].items():
            status = "✓" if pred else "✗"
            print(f"  {status} {label:25}: {pred}")
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.output_file}")
    
    elif args.input_file:
        logging.info(f"Loading texts from {args.input_file}")
        
        if args.input_file.endswith('.csv'):
            df = pd.read_csv(args.input_file)
            if 'text' not in df.columns:
                raise ValueError("CSV file must have a 'text' column")
            texts = df['text'].tolist()
        elif args.input_file.endswith('.txt'):
            with open(args.input_file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("Input file must be .csv or .txt")
        
        logging.info(f"Predicting for {len(texts)} texts...")
        results = predict_batch(texts, model, tokenizer, config, device, args.threshold)
        
        # Print summary
        print(f"\nProcessed {len(results)} texts")
        
        # Count predictions per label
        label_counts = {label: 0 for label in config.label_names}
        for result in results:
            for label, pred in result['predictions'].items():
                if pred:
                    label_counts[label] += 1
        
        print("\nLabel distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {label:25}: {count:3d} ({percentage:5.1f}%)")
        
        # Save results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output_file}")
        else:
            # Default output file
            output_file = args.input_file.replace('.csv', '_predictions.json').replace('.txt', '_predictions.json')
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()