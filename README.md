# Psychological Underpinnings of Misinformation Perception

**Authors:** Ansh Madan (ansh.madan_ug25@ashoka.edu.in) & Anagha Bhavsar (anagha.bhavsar_ug25@ashoka.edu.in)

## Project Overview

This project addresses the critical challenge of automated misinformation detection by focusing on **psychological manipulation techniques** rather than content verification. Our system analyzes headlines and predicts the presence of six psychological mechanisms across three theoretical frameworks.

### Detected Mechanisms

**Framework 0 — Baseline**
- `no_mechanism`: Headlines with no detectable psychological manipulation

**Framework 1 — Elaboration Likelihood Model (ELM)**
- `central_route_present`: Appeals to systematic, logical processing
- `peripheral_route_present`: Appeals to emotional or heuristic processing

**Framework 2 — Cognitive Biases**
- `naturalness_bias`: Exploitation of preference for "natural" solutions
- `availability_bias`: Manipulation through memorable/recent examples
- `illusory_correlation`: Creation of false pattern recognition

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
