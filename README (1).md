
# ✈️ Airline Sentiment Analysis Platform

> **Production-Ready NLP Platform** for airline tweet sentiment classification using **DistilBERT, FastAPI, MLflow, and Streamlit**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-blueviolet)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)

---

# Table of Contents

- Executive Summary
- Features
- Architecture
- Dataset
- Model
- Training Strategy
- Version 13 Improvements
- Results
- Business Insights
- Tech Stack
- Project Structure
- Quick Start
- API
- MLflow Lifecycle
- Dashboard
- Roadmap
- Future Work
- Documentation
- Author

---

# Executive Summary

This repository demonstrates an end-to-end production-oriented NLP system that automatically classifies airline tweets into **Positive, Neutral, and Negative** sentiment.

Unlike a notebook-only project, this repository covers the complete machine learning lifecycle:

- Data preprocessing
- DistilBERT fine-tuning
- Evaluation
- MLflow Experiment Tracking
- Model Registry
- FastAPI REST API
- Streamlit Dashboard
- Business Analytics

---

# Features

- DistilBERT sentiment classifier
- Three-class prediction
- Real-time REST API
- Interactive dashboard
- MLflow experiment tracking
- Model Registry
- Business analytics
- Complaint analysis
- Airline benchmarking
- Production-ready architecture

---

# Architecture

```text
Tweets
   │
   ▼
Preprocessing
   │
   ▼
DistilBERT
   │
   ▼
FastAPI
   │
   ├── MLflow
   └── Streamlit Dashboard
```

---

# Dataset

- 14,604 airline tweets
- 6 US airlines
- 3 sentiment classes

Target labels:

- Positive
- Neutral
- Negative

---

# Model

**DistilBERT (distilbert-base-uncased)**

Chosen because it provides an excellent balance between accuracy and inference speed while remaining lightweight enough for production deployment.

---

# Training Strategy

- Cleaning & normalization
- Tokenization
- Train / Validation / Test split
- Synonym augmentation
- Class-weighted loss
- Early stopping
- Dynamic padding
- Gradient clipping
- Linear warmup scheduler

---

# Version 13 Improvements

- Dynamic Padding
- Lazy Loading Dataset
- Gradient Clipping
- Linear Warmup Scheduler
- Hybrid Imbalance Handling
- MLflow Quality Gate
- Faster inference (~0.08 s)

---

# Performance

| Metric | Value |
|---------|------:|
| Accuracy | **86.00%** |
| Macro F1 | **0.81** |
| Weighted F1 | **0.86** |
| Latency | **0.08 s** |
| Dataset | **14,604 Tweets** |

## Class-wise Results

| Class | Precision | Recall | F1 |
|------|------:|------:|------:|
| Negative |0.91|0.92|0.91|
| Neutral|0.74|0.71|0.72|
| Positive|0.82|0.81|0.81|

---

# Business Insights

- 62.72% of tweets are negative
- Customer Service is the top complaint (31.7%)
- Southwest shows highest positive sentiment
- US Airways records highest negative sentiment
- Morning hours have highest complaint activity

---

# Tech Stack

| Layer | Technology |
|-------|------------|
| NLP | DistilBERT |
| Training | PyTorch |
| Transformers | Hugging Face |
| API | FastAPI |
| Dashboard | Streamlit |
| MLOps | MLflow |
| Analysis | Pandas / NumPy |

---

# Project Structure

```text
project/
│
├── api/
├── app.py
├── training/
├── preprocessing/
├── models/
├── mlruns/
├── docs/
├── requirements.txt
└── README.md
```

---

# Quick Start

```bash
git clone <repository>

cd Airline-Sentiment-Analysis

python -m venv .venv

pip install -r requirements.txt
```

Run API

```bash
uvicorn api:app --reload
```

Run Dashboard

```bash
streamlit run app.py
```

---

# REST API Example

Request

```json
{
  "text":"I missed my flight because customer service was terrible."
}
```

Response

```json
{
 "prediction":"Negative",
 "confidence":0.97
}
```

---

# MLflow Lifecycle

Training → Evaluation → Tracking → Model Packaging → Registry → Staging → Production

---

# Dashboard

The Streamlit interface provides:

- Live prediction
- Sentiment distribution
- Complaint analysis
- Airline comparison
- Historical analytics

---

# Roadmap

- [x] DistilBERT
- [x] FastAPI
- [x] Streamlit
- [x] MLflow
- [ ] Docker
- [ ] CI/CD
- [ ] Kubernetes
- [ ] Monitoring
- [ ] Multilingual Model

---

# Future Work

- RoBERTa / DeBERTa comparison
- Complaint classification
- Drift monitoring
- Automated retraining
- CRM integration
- Live Twitter/X streaming

---

# Documentation

- Technical Documentation
- Business Documentation

---

# Author

**Youssef Mahmoud**

AI & Data Science Student

LinkedIn:
https://www.linkedin.com/in/youssef-mahmoud-63b243361

---

# License

MIT License

---

⭐ If you found this repository useful, consider giving it a star.
