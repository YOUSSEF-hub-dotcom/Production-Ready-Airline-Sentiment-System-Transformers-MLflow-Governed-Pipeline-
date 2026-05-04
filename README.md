# ✈️ Airline Sentiment Analysis

### Production-Grade NLP System | DistilBERT + MLOps(MLflow life Cycle)

---

## 🚀 Overview

This project delivers a **production-ready NLP system** that transforms airline customer tweets into **real-time actionable business intelligence**.

Instead of manually reviewing thousands of tweets, the system automatically classifies sentiment into:

* Positive 🙂
* Neutral 😐
* Negative 😡

With **82.83% accuracy**, **0.78 Macro F1-score**, and **~0.12s inference latency**, enabling real-time monitoring and rapid response.

---

## 🧨 Problem Statement

Airlines receive **thousands of social media mentions daily**, but:

* Manual classification is **slow and unscalable**
* Critical complaints are **missed or delayed**
* Negative content spreads **faster and wider**
* No real-time visibility for decision-makers

📊 **Key Insight:**

> **62.72% of tweets are negative**, and **31.7% of complaints are about customer service** — not operations.

---

## 💡 Solution

We built an **end-to-end sentiment analysis system** powered by DistilBERT that:

* Classifies tweets into 3 sentiment classes
* Detects critical complaints in real-time
* Provides live monitoring dashboards
* Enables competitor benchmarking

---

## 🧠 Model Performance (v13)

| Metric            | Value         |
| ----------------- | ------------- |
| Accuracy          | **82.83%**    |
| Macro F1 Score    | **0.78**      |
| Weighted F1 Score | **0.83**      |
| Latency           | **~0.12 sec** |
| Dataset Size      | 14,604 Tweets |

### 📌 Class-wise Performance

* Negative → **F1 = 0.90** 🔥
* Neutral → **F1 = 0.66**
* Positive → **F1 = 0.78**

---

## 🏗️ System Architecture

```
User / Tweets
     ↓
DistilBERT (PyTorch + Transformers)
     ↓
FastAPI (Real-time REST API)
     ↓
Streamlit Dashboard (Monitoring)
     ↓
MLflow (Tracking + Registry + Deployment)
```

---

## ⚙️ Tech Stack

* **Model**: DistilBERT (HuggingFace Transformers)
* **Backend**: FastAPI
* **Frontend**: Streamlit
* **MLOps**: MLflow
* **Training**: PyTorch
* **NLP Tools**: NLTK

---

## 🔬 Methodology

### 1. Data Processing

* Cleaned and preprocessed tweet text
* Handled class imbalance using:

  * **Synonym-based data augmentation**
  * **Class-weighted loss (1 : 2 : 1.5)**

---

### 2. Model Training

* Fine-tuned **DistilBERT (distilbert-base-uncased)**
* Max sequence length: 512
* Batch size: 16 | Epochs: 3

---

### 3. Key Improvements (Version 13)

* ✅ **Lazy Loading Dataset** → Reduced memory usage
* ✅ **Warmup Scheduler** → Stabilized training
* ✅ **Gradient Clipping** → Prevented exploding gradients
* ✅ **Train/Validation/Test Split** → Eliminated data leakage
* ✅ **Hybrid Optimization** → Class weights + augmentation

---

## ⚙️ MLOps & Production

* MLflow experiment tracking
* Model registry & versioning
* Automated **quality gates** (Accuracy & F1 thresholds)
* Staging → Production lifecycle
* FastAPI deployment for real-time inference

---

## 📊 Key Insights

* ❗ **62.72% Negative Sentiment** (high risk)
* 🧑‍💼 **Customer Service = 31.7%** of complaints (top issue)
* 🏆 Best airline: **Southwest (23.6% positive)**
* ⚠️ Worst airline: **US Airways (77.7% negative)**
* ⏰ Peak complaints: **6 AM – 9 AM**
* 🔥 Negative tweets spread faster (higher retweet rate)

---

## 💼 Business Impact

### 💰 ROI & Efficiency

* Saves **8,000+ hours/year**
* Reduces workload from **730 hours → < 30 minutes**
* Generates **$240K+ annual value**
* Estimated **ROI: ~1200%**

---

### ⚡ Operational Benefits

* Real-time complaint detection
* Faster response (< 1 hour)
* Automated monitoring dashboard
* Data-driven decision making
* Competitive benchmarking

---

## 🚀 Features

* ✅ Real-time sentiment classification
* ✅ High-speed inference (~0.12s)
* ✅ Production-ready pipeline
* ✅ MLflow lifecycle management
* ✅ Scalable API (FastAPI)
* ✅ Interactive dashboard (Streamlit)

---

## 📦 How to Run

```bash

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn api:app --reload

# Run dashboard
streamlit run app.py
```

---

## 🛣️ Roadmap

### Phase 1

* Deploy API + Dashboard
* Real-time alert system

### Phase 2

* CRM integration
* Auto-routing complaints

### Phase 3

* Multi-platform (Instagram, Facebook)
* Multilingual model

---

## ⚠️ Challenges & Future Work

* Improve Neutral class performance
* Handle data drift (evolving language)
* Expand dataset coverage
* Experiment with:

  * RoBERTa / DeBERTa
  * Ensemble models

---

## 🎯 Key Takeaway

> This is not just a sentiment analysis model —
> it's a **production-grade AI system** that converts unstructured data into **real-time business decisions**.

---

## 👨‍💻 Author

**Youssef Mahmoud**
Data Scientist | Machine Learning Engineer

---

## ⭐ Support

If you found this project useful, give it a ⭐ on GitHub!

🔗 LinkedIn:
https://www.linkedin.com/in/youssef-mahmoud-63b243361

---
