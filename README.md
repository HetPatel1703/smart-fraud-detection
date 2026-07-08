<<<<<<< HEAD
# Smart Credit Card Fraud Investigation 
**MSoC 2026 Hackathon** | **Developed by: Het Pethani,Khushi Khakhiwala (Team: Neural Sparkers)**

** Live Deployment Link:** [View the Live Investigator Dashboard Here](https://hetpatel1703-smart-fraud-detection-app-906nlv.streamlit.app/)

---

## 1. Problem Statement

Global card payment networks process hundreds of billions of transactions every year, and a small but relentless fraction of those are fraudulent. Financial institutions currently face two major problems:
1. **Legacy Systems:** Traditional rule-based systems (e.g., "block any transaction over $5,000") are easily bypassed by modern hackers and result in thousands of legitimate customers getting their cards falsely declined.
2. **The "Black Box" AI Problem:** While modern machine learning models are highly accurate at catching fraud, they operate as a "black box." Due to strict financial regulations (like GDPR and PSD2), banks cannot block a customer's card without being able to explain *why*. 

There is a critical need for a system that is not only highly accurate at detecting fraud but also completely transparent and explainable to human fraud investigators.

---

## 2. Solution Approach

This project is a next-generation **Explainable AI Fraud Detection System**. It replaces human guesswork with dynamic machine learning and translates complex mathematics into a plain-English dashboard.

* **Data Balancing (SMOTE):** Because fraud only occurs in ~0.17% of transactions, we utilized Synthetic Minority Oversampling Technique (SMOTE) on the training data to ensure the AI could actually learn the minority class patterns without overfitting.
* **Algorithmic Engine:** We trained a lightweight, high-speed **XGBoost Classifier**. It is calibrated using Scikit-Learn's `CalibratedClassifierCV` to output true probabilities rather than raw confidence scores.
* **Dynamic Threshold Optimization:** Instead of guessing a flat "50% = fraud" rule, our pipeline runs a simulation across 50 different probability cutoffs during training. It automatically selects the exact mathematical threshold that maximizes the **F1-Score**, perfectly balancing fraud capture (Recall) with customer experience (Precision).
* **SHAP Explainability:** We integrated `TreeExplainer` from the SHAP library to break down the exact mathematical contribution of every single variable.
* **Investigator UX:** We mapped the dataset's hidden PCA variables (`V1-V28`) to realistic banking terms (e.g., *"Location Safety Check"*, *"Device Recognition"*) and built a Streamlit UI that translates the SHAP math into clean bar charts and plain-English warnings for human investigators.
* **System Guardrails:** The system logic includes custom intercepts for user-errors (e.g., negative transaction amounts), filtering them out as "Invalid Entries" before they can pollute the ML model predictions.

---

## 3. Technology Stack

This project was built entirely in Python, focusing on high-speed inference and clear data visualization:

* **Frontend / UI:** [Streamlit](https://streamlit.io/) (for the interactive investigator dashboard)
* **Machine Learning Engine:** [XGBoost](https://xgboost.readthedocs.io/) (Gradient Boosted Decision Trees)
* **Explainable AI:** [SHAP](https://shap.readthedocs.io/) (SHapley Additive exPlanations)
* **Data Processing:** Pandas, NumPy
* **Model Pipeline & Metrics:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Data Visualization:** Matplotlib
* **Deployment:** Streamlit Community Cloud

---

## 4. Project Structure

```text
smart-fraud-detection-main/
├── app.py                      # Main Streamlit dashboard
├── train.py                    # Model training pipeline
├── make_demo.py                # Creates demo dataset for testing
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── RESULTS.md                  # Model evaluation results
├── .gitignore                  # Git ignored files
│
├── data/
│   ├── creditcard.csv          # Original credit card fraud dataset
│   └── demo_creditcard.csv     # Sample dataset for demo/testing
│
├── models/
│   ├── model.joblib            # Trained XGBoost fraud detection model
│   ├── scaler.joblib           # Feature scaler
│   ├── explainer.joblib        # SHAP explainer object
│   └── threshold.joblib        # Optimal fraud decision threshold
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature engineering and scaling
│   ├── model_trainer.py        # Model training and artifact saving
│   ├── evaluator.py            # Model evaluation metrics
│   ├── explainer.py            # SHAP explainability generation
│   └── utils.py                # Helper utility functions
│
└── .venv/                      # Python virtual environment
```

## 5. Dataset Sources

* **Primary Dataset:** Kaggle Credit Card Fraud Detection Dataset
    * **Source Link:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    * **Description:** This dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred over two days, where there are 492 frauds out of 284,807 transactions (highly unbalanced).
    * **Note on Features:** Due to confidentiality issues, the original features (V1-V28) are numerical principal components obtained via PCA. The only unaltered features are 'Time' and 'Amount'.
* **Deployment Dataset (`demo_creditcard.csv`):** Because the original dataset is over 150MB (exceeding GitHub's limits), a lightweight stratified sample containing all 492 original fraud cases and 1,000 legitimate transactions was generated locally and uploaded to power the live cloud deployment.

---

## 6. Setup Instructions

To run this project locally on your own machine, follow these steps:

### Step 1: Clone and Set Up Environment
It is highly recommended to use a virtual environment to prevent dependency conflicts.
```bash
git clone [https://github.com/hetpatel1703/smart-fraud-detection.git](https://github.com/hetpatel1703/smart-fraud-detection.git)
cd smart-fraud-detection

# For Mac/Linux:
python3 -m venv venv
source venv/bin/activate

# For Windows:
python -m venv venv
.\venv\Scripts\activate
=======
# Smart-Credit-Card-Fraud-Investigation-Project
An intelligent credit card fraud detection system that predicts fraudulent transactions and explains why they are suspicious using Explainable AI, XGBoost, SHAP, and an interactive Streamlit dashboard.
>>>>>>> 1ff2ea6938783b0c089bfd539cad2d16209126cd
