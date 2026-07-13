# 🛡️ FakeShield AI

> **An Explainable AI-Powered Fake News Detection Platform**
>
> Detect fake news using NLP, Machine Learning, Transformer models, and Explainable AI (XAI). FakeShield AI analyzes news articles, predicts their authenticity, explains the reasoning behind predictions, and provides insightful visual analytics.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

# 📖 Overview

FakeShield AI is an end-to-end intelligent fake news detection system that combines **Natural Language Processing (NLP)**, **Machine Learning**, **Transformer-based Language Models**, and **Explainable AI** to classify news articles as **Real** or **Fake**.

Unlike traditional fake news classifiers, FakeShield AI not only predicts the authenticity of news but also explains *why* a prediction was made using explainability techniques, making the model more transparent and trustworthy.

This project demonstrates the complete lifecycle of an AI application—from data preprocessing and model training to deployment and visualization.

---

# 🚀 Features

- ✅ Fake News Detection
- ✅ Transformer-based NLP Models (BERT/RoBERTa)
- ✅ Traditional ML Baseline Models
- ✅ Text Cleaning & Preprocessing
- ✅ Exploratory Data Analysis (EDA)
- ✅ POS Tagging
- ✅ Named Entity Recognition (NER)
- ✅ Sentiment Analysis
- ✅ Topic Modeling
- ✅ Explainable AI (LIME / SHAP)
- ✅ Confidence Score Prediction
- ✅ REST API using FastAPI
- ✅ Interactive Streamlit Dashboard
- ✅ Docker Support
- ✅ Modular Project Architecture

---

# 🧠 Project Workflow

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Text Preprocessing
    │
    ▼
EDA & Visualization
    │
    ▼
Feature Engineering
    │
    ▼
Model Training
    │
    ▼
Evaluation
    │
    ▼
Explainability
    │
    ▼
FastAPI Backend
    │
    ▼
Streamlit Dashboard
```

---

# 🛠️ Tech Stack

## Programming

- Python

## Machine Learning

- Scikit-learn
- XGBoost
- LightGBM

## Deep Learning

- PyTorch
- Hugging Face Transformers

## NLP

- spaCy
- NLTK
- Transformers

## Explainable AI

- SHAP
- LIME

## Backend

- FastAPI
- Uvicorn

## Frontend

- Streamlit

## Database

- SQLite
- ChromaDB (Optional)

## Visualization

- Matplotlib
- Seaborn
- Plotly

## DevOps

- Docker
- Git
- GitHub

---

# 📂 Project Structure

```
FakeShield-AI
│
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── services/
│   ├── utils/
│   └── schemas/
│
├── data/
│
├── notebooks/
│
├── model/
│
├── streamlit_app/
│
├── tests/
│
├── docs/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── main.py
```

---

# 📊 Machine Learning Pipeline

### Data Collection

- Kaggle Fake News Dataset

### Data Cleaning

- Remove HTML tags
- Remove punctuation
- Lowercase conversion
- Remove stopwords
- Lemmatization

### Feature Engineering

- TF-IDF
- Word Embeddings
- BERT Embeddings

### Models

Traditional Models

- Logistic Regression
- Naive Bayes
- SVM
- Random Forest

Transformer Models

- BERT
- RoBERTa
- DistilBERT

---

# 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

# 🔍 Explainable AI

FakeShield AI explains every prediction using:

- SHAP Values
- LIME
- Feature Importance
- Highlighted influential words

This helps users understand why a piece of news is classified as fake or real.

---

# 💻 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/FakeShield-AI.git

cd FakeShield-AI
```

## Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Run the backend

```bash
python main.py
```

Run FastAPI

```bash
uvicorn app.main:app --reload
```

Run Streamlit

```bash
streamlit run streamlit_app/app.py
```

---

# 📷 Screenshots

### Dashboard

> *(Coming Soon)*

### Prediction

> *(Coming Soon)*

### Explainability

> *(Coming Soon)*

---

# 🎯 Future Improvements

- 🌍 Multi-language Fake News Detection
- 🤖 LLM-based Fact Verification
- 📰 Live News Verification
- 🔗 News Source Credibility Score
- 🌐 Browser Extension
- 📱 Mobile Application
- ☁️ Cloud Deployment
- 🔍 Real-time Web Search Integration

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```
git checkout -b feature-name
```

3. Commit your changes

```
git commit -m "Added new feature"
```

4. Push to GitHub

```
git push origin feature-name
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Your Name**

MCA (Generative AI)

AI • Machine Learning • NLP • LLMs • Explainable AI

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

---

# ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.

It motivates further development and helps others discover the project!
