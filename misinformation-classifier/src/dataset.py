"""Dataset handling for misinformation classification."""

import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Union, Any, Optional
import logging

class MisinformationDataset(Dataset):    
    def __init__(self, texts: List[str], labels: List[List[int]], 
                 tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.FloatTensor(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

#data loading
def load_data(filepath: str, label_names: List[str]) -> tuple:
    logging.info(f"Loading data from {filepath}")
    
    # Field mapping
    field_mapping = {
        'no_mechanism': 'framework0_feature1',
        'central_route_present': 'framework1_feature1',
        'peripheral_route_present': 'framework1_feature2', 
        'naturalness_bias': 'framework2_feature1',
        'availability_bias': 'framework2_feature2',
        'illusory_correlation': 'framework2_feature3'
    }
    
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data_list = json.load(f)
        
        texts = [item['text'] for item in data_list]
        labels = []
        
        for item in data_list:
            label_row = []
            for label in label_names:
                field_name = field_mapping[label]
                label_row.append(item[field_name])
            labels.append(label_row)
            
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        texts = df['text'].tolist()
        
        labels = []
        for _, row in df.iterrows():
            label_row = []
            for label in label_names:
                field_name = field_mapping[label]
                label_row.append(row[field_name])
            labels.append(label_row)
    else:
        raise ValueError("File must be .json or .csv")
    
    logging.info(f"Loaded {len(texts)} samples")
    return texts, labels

def create_datasets(filepath: str, config: Any, test_filepath: Optional[str] = None):
    from sklearn.model_selection import train_test_split
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load training data
    texts, labels = load_data(filepath, config.label_names)
    
    # Split data 
    if test_filepath is None:
        
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=config.test_split, random_state=42
        )
        
        val_size = config.val_split / (config.train_split + config.val_split)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=val_size, random_state=42
        )
    else:
        test_texts, test_labels = load_data(test_filepath, config.label_names)
        
        val_size = config.val_split / (config.train_split + config.val_split)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=val_size, random_state=42
        )
    
    # Create datasets
    train_dataset = MisinformationDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = MisinformationDataset(val_texts, val_labels, tokenizer, config.max_length)
    test_dataset = MisinformationDataset(test_texts, test_labels, tokenizer, config.max_length)
    
    return train_dataset, val_dataset, test_dataset, tokenizer