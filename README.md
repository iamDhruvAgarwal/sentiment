# 🧠 AI-Powered Sentiment Analysis Web App

A lightweight, modular Streamlit application that analyzes the sentiment of user input using multiple AI/ML models — including on-device inference with Ollama's LLM. Built to compare and visualize model predictions in real time.

This project is developed by Dhruv Agarwal for educational and demonstration purposes.

## 🚀 Features

- 🔍 **Single Text Sentiment Analysis**
  - Supports multiple models:
    - TextBlob (Rule-based)
    - Naive Bayes (Supervised ML)
    - BERT (Transformer)
    - Ollama (On-device LLM)

- 📊 **Model Comparison Visualization**
  - Interactive bar chart to compare predictions from all models side-by-side.

- 📁 **Batch Input (Planned)**
  - (Commented out for now) Upload a file and analyze sentiments in bulk.

- 💾 **Export Results (Planned)**
  - (Optional) Export analysis to CSV or PDF format.

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **ML Libraries**: scikit-learn, TextBlob, Transformers
- **LLM Backend**: Ollama
- **Visualization**: Plotly
- **Other Tools**: pandas, matplotlib

## 📁 Folder Structure

```bash
basic_sentiment/
├── main.py                  # Main entry point
├── pages/                   # Streamlit pages (if enabled)
│   ├── visualize.py         # (Optional) Charts for sentiment comparison
│   └── about.py             # App information
├── components/
│   ├── analyzers.py         # Model logic and sentiment functions
│   └── utils.py             # (Optional) Export or helper functions
└── README.md                # You're here
