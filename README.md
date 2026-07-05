# ✈️ Airline Sentiment Analysis Platform

An end-to-end **NLP sentiment analysis platform** for classifying airline customer tweets into **Positive, Neutral, and Negative** sentiment using **DistilBERT**, with a production-oriented workflow built around **MLflow, FastAPI, and Streamlit**.

The project is designed to transform unstructured customer feedback into **actionable operational insights**, enabling faster complaint monitoring, airline benchmarking, and real-time sentiment analysis.

---

## 📌 Project Overview

Airlines receive a large volume of customer mentions on social media every day. Manually reviewing those tweets is slow, inconsistent, and difficult to scale — especially when negative complaints can spread quickly and damage brand reputation.

This project addresses that problem by building an **end-to-end sentiment analysis system** that:

* classifies airline tweets into **Positive / Neutral / Negative**
* highlights high-risk negative sentiment in real time
* supports interactive sentiment monitoring through a dashboard
* enables airline-level benchmarking and complaint analysis
* demonstrates a production-oriented ML workflow with tracking, versioning, and deployment

The system is built around a fine-tuned **DistilBERT** model and combines **NLP modeling, MLOps, API serving, and business-facing analytics** in one project.

---

## 🎯 Objectives

This project was built to answer a practical business question:

> **How can airline customer sentiment be classified automatically, at scale, and fast enough to support real-time monitoring and operational response?**

### The platform is designed to:

* classify airline tweets into **three sentiment classes**
* identify critical negative tweets quickly
* provide a deployable inference service for real-time sentiment scoring
* surface business insights from customer complaints and sentiment patterns
* demonstrate a full ML workflow from training to serving and monitoring

---

## 🧠 End-to-End System Architecture

The solution is structured as a complete NLP application stack:

```text id="u3e9xn"
Tweets / User Input
        ↓
DistilBERT Sentiment Model
        ↓
FastAPI Prediction Service
        ↓
Streamlit Dashboard / Monitoring Layer
        ↓
MLflow Tracking + Model Registry
```

### Pipeline flow

```text id="e4qv7g"
Raw Tweet Data
      ↓
Text Cleaning & Preprocessing
      ↓
Train / Validation / Test Split
      ↓
DistilBERT Fine-Tuning
      ↓
Evaluation + Error Analysis
      ↓
MLflow Tracking + Model Packaging
      ↓
Model Registry + Quality Gate
      ↓
FastAPI Deployment
      ↓
Streamlit Dashboard & Business Monitoring
```

---

## 📊 Dataset

The project uses an airline Twitter sentiment dataset containing **14,604 tweets** collected across **6 major US airlines**.

### Target classes

* **Negative**
* **Neutral**
* **Positive**

### Airlines represented

* United
* US Airways
* American
* Southwest
* Delta
* Virgin America

This makes the project a realistic example of **short-text sentiment classification** in a customer-experience setting.

---

## 🧹 Data Preparation & Training Strategy

The system prepares tweet data for transformer-based training and includes steps to improve generalization under class imbalance.

### Core preprocessing / training preparation

* tweet cleaning and normalization
* tokenization for transformer input
* explicit **Train / Validation / Test** split to avoid leakage
* class imbalance handling
* training optimization for stable fine-tuning

### Class imbalance strategy

The dataset is skewed toward **negative sentiment**. To reduce majority-class bias, the project uses a hybrid strategy combining:

* **synonym-based data augmentation**
* **class-weighted loss**

This improves performance on underrepresented classes such as **Neutral** and **Positive**, while preserving strong performance on **Negative** tweets.

---

## 🤖 Model

### Model architecture

**DistilBERT (`distilbert-base-uncased`)**

DistilBERT was selected because it provides a strong trade-off between **accuracy and inference speed**, which is important for real-time monitoring pipelines.

### Why DistilBERT?

* lighter and faster than full BERT
* strong performance on short-text sentiment classification
* practical for deployment-oriented NLP systems
* suitable for fine-tuning on domain-specific tweet data without extreme compute cost

---

## ⚙️ Training Configuration

### Final model version

**Version 13**

### Training setup

* **Batch size:** 16
* **Epochs:** 3
* **Max sequence length:** 512
* **Framework:** PyTorch + HuggingFace Transformers

### Key improvements introduced in Version 13

Version 13 focused on improving both **training stability** and **generalization quality**.

#### Improvements

* **Lazy-loading dataset pipeline** to reduce memory overhead during training
* **Linear warmup scheduler** for more stable optimization
* **Gradient clipping** to avoid unstable gradient updates
* **Train / Validation / Test split** to eliminate leakage
* **Hybrid imbalance handling** using augmentation + class-weighted loss

These changes improved the model’s reliability and helped raise the final performance of the sentiment system.

---

## 📊 Final Model Performance

The final DistilBERT model achieved the following results on the held-out test set:

| Metric                |             Score |
| --------------------- | ----------------: |
| **Accuracy**          |        **82.83%** |
| **Macro F1**          |          **0.78** |
| **Weighted F1**       |          **0.83** |
| **Inference Latency** |        **~0.12s** |
| **Dataset Size**      | **14,604 tweets** |

### Class-wise performance

| Class        | Precision | Recall | F1 Score |
| ------------ | --------: | -----: | -------: |
| **Negative** |      0.89 |   0.90 | **0.90** |
| **Neutral**  |      0.68 |   0.65 | **0.66** |
| **Positive** |      0.77 |   0.78 | **0.78** |

### Interpretation

* The model performs especially well on **Negative tweets**, which is useful because these often represent urgent customer complaints
* **Neutral** remains the hardest class, which is common in sentiment tasks due to ambiguity and mixed tone
* The final model is strong enough to support **real-time sentiment monitoring and complaint prioritization**

---

## 🔍 Key Sentiment Insights from the Data

A major value of this project is not only classification, but also extracting **business-facing insight** from customer sentiment patterns.

### Overall sentiment distribution

* **Negative:** 62.72%
* **Neutral:** 21.17%
* **Positive:** 16.12%

This means the dataset is heavily skewed toward dissatisfaction, with roughly a **4:1 negative-to-positive ratio**.

### Complaint analysis

Among negative tweets, the most common complaint categories were:

* **Customer Service Issues** → **31.7%**
* **Late Flight** → **18.1%**
* **Can’t Tell / unclear complaint** → **13.0%**
* **Cancelled Flight** → **9.2%**
* **Lost Luggage** → **7.9%**

A notable insight is that **customer service issues — not flight operations — are the dominant complaint driver** across most airlines.

---

## 🏢 Airline-Level Findings

The analysis also surfaces competitive and operational differences across airlines.

### Examples of findings

* **US Airways** had the highest negative sentiment share (**77.7%**)
* **United** also showed a very high negative share (**68.9%**)
* **Southwest** had the strongest positive sentiment share (**23.6%**)
* Customer service was the top complaint category for **5 of the 6 airlines**

This makes the project useful not only as a classifier, but also as a **benchmarking and monitoring tool**.

---

## ⏰ Behavioral & Temporal Insights

The project also analyzes tweet behavior beyond raw sentiment labels.

### Examples of observed patterns

* Negative tweets are **longer on average** than neutral or positive tweets
* Negative tweets show a **higher retweet rate**, suggesting greater viral/reputational risk
* Tweet activity peaks between **6 AM and 9 AM**, which may be an important staffing window for support teams

These patterns can inform operational decisions such as **when to monitor social channels more aggressively** and **which complaints deserve faster escalation**.

---

## 🔄 MLflow Lifecycle Management

The project includes an MLflow-based lifecycle workflow to support experiment reproducibility and deployment governance.

### MLflow components used

* **Experiment tracking**
* **Metric and parameter logging**
* **Model packaging**
* **Model signature management**
* **Model registry**
* **Staging → Production promotion workflow**
* **quality-gate-based validation**

### Why this matters

This makes the repository more than a notebook experiment.
It demonstrates how an NLP model can be managed through a **repeatable ML lifecycle** with versioning and deployment discipline.

---

## 🌐 FastAPI Inference Layer

The model is exposed through a **FastAPI-based prediction service** for real-time sentiment scoring.

### API role

* receive tweet text
* run DistilBERT inference
* return sentiment prediction and probability-related output
* provide a reusable serving interface for downstream applications

This makes it possible to integrate the model into dashboards, internal monitoring tools, or alerting systems.

---

## 🖥 Streamlit Dashboard

A Streamlit dashboard is included to make the system easier to use from a business or analytics perspective.

### Dashboard goals

* score tweets interactively
* visualize sentiment distributions
* inspect complaint and airline-level patterns
* monitor sentiment trends in a more accessible way than raw model outputs

This helps bridge the gap between **model predictions** and **business consumption**.

---

## 🛠️ Tech Stack

| Layer           | Technology                 | Purpose                                  |
| --------------- | -------------------------- | ---------------------------------------- |
| NLP Model       | **DistilBERT**             | Tweet sentiment classification           |
| Training        | **PyTorch + Transformers** | Fine-tuning and inference                |
| NLP Utilities   | **NLTK**                   | augmentation / preprocessing support     |
| MLOps           | **MLflow**                 | tracking, registry, lifecycle management |
| API             | **FastAPI**                | real-time inference service              |
| Frontend        | **Streamlit**              | interactive monitoring dashboard         |
| Data / Analysis | **Pandas, NumPy**          | data handling and analysis               |

---

## 📁 Project Structure

```bash id="q8hy24"
project/
│
├── train.py / model.py         # DistilBERT training and evaluation
├── preprocessing.py            # text cleaning / preparation logic
├── api.py                      # FastAPI inference service
├── app.py                      # Streamlit dashboard
├── mlflow_pipeline.py          # MLflow tracking / packaging workflow
├── requirements.txt
├── README.md
└── docs/                       # technical + business documentation
```

> Update the structure section to match your actual repository layout if your file names differ.

---

## 🏁 How to Run

### 1) Install dependencies

```bash id="t8g4n1"
pip install -r requirements.txt
```

### 2) Run the FastAPI service

```bash id="eqmq86"
uvicorn api:app --reload
```

### 3) Launch the Streamlit dashboard

```bash id="59v8g1"
streamlit run app.py
```

---

## 📌 Why This Project Is Strong as a Portfolio Project

This repository demonstrates more than a sentiment classifier. It combines:

* **transformer-based NLP modeling**
* **short-text sentiment classification**
* **class imbalance handling**
* **training optimization and evaluation**
* **MLflow-based model lifecycle management**
* **FastAPI deployment**
* **Streamlit-based business-facing analytics**
* **insight extraction from unstructured customer feedback**

In other words, it shows how a transformer model can be turned into a **deployable, end-to-end applied NLP system**.

---

## 🌱 Future Improvements

Potential extensions for the project:

* improve **Neutral** class performance through additional labeling and error analysis
* expand training data with more recent airline tweets
* add **multilingual sentiment support**
* evaluate stronger transformer baselines such as **RoBERTa** or **DeBERTa**
* add complaint-category classification in addition to sentiment classification
* connect the model to a live alerting workflow for critical negative tweets
* add drift monitoring and scheduled retraining

---

## 📚 Documentation

This repository is supported by:

* **Technical Documentation** — methodology, model improvements, architecture, metrics, and deployment design
* **Business Documentation** — business impact, operational insights, complaint analysis, and roadmap recommendations

---

## 👨‍💻 Author

**Youssef Mahmoud**
AI / Data Science Student

[LinkedIn](https://www.linkedin.com/in/youssef-mahmoud-63b243361)

---

## ⭐ Final Note

This project is designed as an **end-to-end airline sentiment analysis platform**, not just a standalone NLP model.
It combines **transformer-based sentiment classification, MLflow lifecycle management, API deployment, and business-facing analytics** to turn unstructured customer tweets into actionable monitoring and decision-support signals.

