"""Model definition for misinformation classification."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional

class MisinformationClassifier(nn.Module):
    """DistilBERT-based multi-label classifier for psychological mechanisms."""
    
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.1, 
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_labels = num_labels
        
        # DistilBERT configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.distilbert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Store pos_weight for weighted BCE loss (handles class imbalance)
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None):
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0] 
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Use weighted BCE loss to handle class imbalance
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }


# Model builder function
def build_model(config, pos_weight: Optional[torch.Tensor] = None) -> MisinformationClassifier:
    dropout_rate = getattr(config, 'dropout_rate', 0.1)
    return MisinformationClassifier(
        model_name=config.model_name,
        num_labels=config.num_labels,
        dropout_rate=dropout_rate,
        pos_weight=pos_weight
    )