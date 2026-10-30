
# Customer Review Sentiment Classification using DistilBERT

## 1. Project Overview

This project implements a binary sentiment classification system using a fine-tuned DistilBERT Transformer model.

The objective is to classify customer reviews into two categories:
- 0: Negative
- 1: Positive

The project uses transfer learning with a pretrained DistilBERT model and fine-tunes it on the IMDb movie review dataset.

## 2. Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- DistilBERT
- Scikit-learn
- Matplotlib
- Google Colab (T4 GPU)

## 3. Dataset

Dataset: IMDb Movie Reviews

- Training samples: 20,000
- Validation samples: 5,000
- Test samples: 25,000
- Number of classes: 2
- Dataset distribution: Balanced

The original training dataset was split into training and validation sets using stratified sampling. The original test set was kept separate for final evaluation.

## 4. Model Architecture

Pretrained model: distilbert/distilbert-base-uncased

DistilBERT is a lightweight Transformer encoder model derived from BERT through knowledge distillation.

The pretrained model was fine-tuned for binary sequence classification using a classification head with two output classes.

## 5. Training Configuration

| Parameter | Value |
|---|---|
| Model | DistilBERT |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 3 |
| Maximum Sequence Length | 256 |
| Weight Decay | 0.01 |
| Scheduler | Linear Learning Rate Decay |
| Gradient Clipping | 1.0 |
| Hardware | NVIDIA T4 GPU |

The best checkpoint was selected based on the lowest validation loss.

## 6. Evaluation Results

Final performance on the held-out test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 90.39% |
| Precision | 92.93% |
| Recall | 87.43% |
| F1-score | 90.10% |

### Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 11,668 | 832 |
| Actual Positive | 1,571 | 10,929 |

## 7. Repository Structure

```text
customer-review-sentiment-classification/
│
├── customer_review_sentiment_classification.ipynb
├── requirements.txt
├── README.md
├── .gitattributes
│
└── model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
