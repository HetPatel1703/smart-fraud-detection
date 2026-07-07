# 🕵️‍♂️ Smart Credit Card Fraud Investigation 
**MSoC 2026 Hackathon | Team: Neural Sparkers**

Global card payment networks process hundreds of billions of transactions every year, and a small but relentless fraction of those are fraudulent. Legacy rule-based systems (like "block anything over $5,000") are easy for hackers to bypass and end up blocking legitimate customers.

This project is a next-generation **Explainable AI Fraud Detection System**. It doesn't just guess if a transaction is fraud—it mathematically proves it and translates the reasoning into a plain-English dashboard for human investigators.

---

## ✨ Key Features

* **🧠 High-Accuracy AI Engine:** Powered by XGBoost, the system effectively catches subtle fraud patterns without generating excessive false alarms.
* **🗣️ Plain-English Explainability:** Instead of returning a confusing "Black Box" probability, the AI uses SHAP values to explain *exactly* why a transaction was flagged (e.g., "Anomalous System Checks" or "Suspicious Timing").
* **📊 Investigator Dashboard:** A beautiful, intuitive Streamlit web app that displays simulated customer profiles, historical averages, and custom-built, text-free risk charts.
* **⚙️ Dynamic AI Thresholding:** Instead of guessing a flat "50% = fraud" rule, the pipeline dynamically tests 50 different probability cutoffs during training to find the exact mathematical sweet spot (maximizing the F1-Score).
* **🛑 Built-In Guardrails:** The system automatically intercepts user data-entry errors (like negative transaction amounts) before they even reach the AI, preventing false data from polluting the model.

---

## 📈 System Performance Metrics

Because fraud only happens in about **0.17%** of transactions, standard "Accuracy" is a useless metric. We optimized this system for **Precision and Recall**, achieving the following results on our testing data:

| Metric | Score | What it means (Plain English) |
| :--- | :--- | :--- |
| **ROC-AUC** | **98.04%** | The overall ability of the model to distinguish between a fraudster and a normal customer (100% is perfect). |
| **Precision** | **93.10%** | When our system blocks a card for fraud, we are correct 93.10% of the time (extremely low false alarms). |
| **Recall** | **72.00%** | Out of all the *actual* fraud attacks that occurred, we successfully caught 72%. |
| **F1-Score** | **81.20%** | The balanced mathematical score combining both Recall and Precision. |

*Note: The model inference latency is completely optimized to score single transactions in under 200ms on a standard CPU.*

---

## 🛠️ How It Works (Our Approach)

1. **Data Preprocessing & SMOTE:** Credit card datasets are highly imbalanced. We use a technique called **SMOTE** (Synthetic Minority Oversampling) strictly on the training data so our AI has enough examples to learn what a hacker looks like.
2. **XGBoost Training (`train.py`):** We train a highly efficient gradient-boosted tree model. It is calibrated using Scikit-Learn's `CalibratedClassifierCV` (with a frozen estimator lock) to ensure the probability outputs (0% to 100%) are accurate.
3. **SHAP Explainability (`explainer.py`):** We use SHAP (`TreeExplainer`) to break down the exact weight of every single variable. We map the dataset's hidden PCA variables (`V1-V28`) to realistic banking terms like *"Location Safety Check"* and *"Device Recognition"* for the UI.
4. **The UI (`app.py`):** A Streamlit dashboard pulls the saved models and SHAP explainer to render interactive, beginner-friendly warnings and clean bar charts for human investigators.

---

## 🚀 How to Run the Project

You don't need a data science degree to run this! Just follow these simple terminal commands.

### 1. Set Up Your Environment
First, clone this repository and navigate into the folder. It is highly recommended to use a virtual environment so you don't conflict with other Python projects on your computer.

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
