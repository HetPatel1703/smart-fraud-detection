import pandas as pd
import numpy as np
import joblib
import os

def load_artifacts(path='models/'):
    """Load model, scaler, explainer, threshold."""
    model = joblib.load(f'{path}/model.joblib')
    scaler = joblib.load(f'{path}/scaler.joblib')
    explainer = joblib.load(f'{path}/explainer.joblib')
    threshold = joblib.load(f'{path}/threshold.joblib')
    return model, scaler, explainer, threshold

def get_risk_score(prob):
    """Convert probability to 0-100 risk score."""
    return int(round(prob * 100))

if __name__ == "__main__":
    pass