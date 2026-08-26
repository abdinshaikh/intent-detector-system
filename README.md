# Amazon MASSIVE Intent Detector

A BERT-based natural language intent classification system trained on the Amazon MASSIVE English dataset.

## Model

- Architecture: BERT for Sequence Classification
- Number of intent classes: 60
- Maximum sequence length: 128 tokens
- Training epochs: 3

## Performance

| Metric | Score |
|---|---:|
| Accuracy | 88.90% |
| Macro F1 | 86.33% |
| Weighted F1 | 88.86% |

## Baseline

TF-IDF + Logistic Regression:

- Accuracy: 80.42%
- Macro F1: 75.45%

## Model

The fine-tuned model is hosted on Hugging Face:

`abdinshaikh/intenr-bert`

## Application

The Streamlit application accepts a natural-language utterance and returns:

- Predicted intent
- Prediction probability
- Top 3 predicted intents
