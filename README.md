# Misinformation-Perception

# Psychological Underpinnings of Misinformation Perception

## Overview
This project investigates how psychological features influence the perceived credibility of misinformation using a combination of human studies and machine learning.

---

## Problem
Standard NLP approaches struggle to capture *why* misinformation is believed.

This project focuses on:
- Psychological mechanisms behind believability
- Feature-driven modelling rather than purely black-box models

---

## Dataset
- 4,000 synthetic misinformation headlines (LLM-augmented)
- 2,050 manually validated samples
- 860 human believability ratings
- Multi-domain: health, technology, etc.

---

## Approach

### Data Pipeline
- LLM-assisted generation and pre-annotation
- Human re-annotation for reliability
- Constraint-based sampling:
  - Balanced topics (20/20)
  - Controlled feature distributions

### Features
- Emotional intensity
- Rhetorical structure
- Linguistic framing
- Psychological triggers

### Model
- DistilBERT (≈66M parameters)
- Multi-label classification (6 psychological mechanisms)
- Weighted loss for class imbalance

---

## Results
- Macro F1-score ≈ 0.73
- Outperformed random and class-prior baselines
- Demonstrated strong predictive power of psychological features

---

## Tech Stack
- Python
- Hugging Face Transformers
- PyTorch
- Pandas, NumPy

---

## Key Insights
- Believability is strongly tied to psychological framing
- Structured features improve interpretability over black-box models
- Human-in-the-loop annotation significantly improves dataset quality

---

## Future Work
- Incorporate network-level features (who shares content)
- Study belief persistence over time
- Extend to multimodal misinformation
