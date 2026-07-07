import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from src.data_loader import load_data, temporal_split
from src.feature_engineering import prepare_features
from src.explainer import create_explainer
from src.model_trainer import save_artifacts
from src.evaluator import evaluate_model

def main():
    # 1. Load and temporally split data
    df = load_data('data/creditcard.csv')
    train_df, test_df = temporal_split(df, test_size=0.2)
    
    X_temp, y_temp, scaler = prepare_features(train_df, fit_scaler=True)
    X_test, y_test, _ = prepare_features(test_df, scaler=scaler, fit_scaler=False)
    
    # 2. Split train into train_sub and val for proper prefit calibration
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
    )
    
    # 3. SMOTE ONLY on the training subset
    smote = SMOTE(random_state=42, sampling_strategy=0.1)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # 4. Train Base XGBoost Model
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_res, y_res)
    
    # 5. Calibrate on the clean validation set (cv='prefit')
    # 5. Calibrate on the clean validation set using FrozenEstimator
    from sklearn.frozen import FrozenEstimator
    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(model), method='sigmoid')
    calibrated.fit(X_val, y_val)
    
    # 6. Find optimal threshold using F1 score on validation set
    y_prob_val = calibrated.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 50)
    f1s = [f1_score(y_val, (y_prob_val >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1s)]
    
    # 7. SHAP Explainer MUST use the base XGBoost model, not the Calibrated wrapper
    background = X_train.sample(n=100, random_state=42)
    explainer = create_explainer(model, background, feature_names=X_train.columns.tolist())
    
    # 8. Save artifacts
    save_artifacts(calibrated, scaler, explainer, best_threshold, path='models/')
    
    # 9. Evaluate on Held-out Test Set
    y_prob_test = calibrated.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= best_threshold).astype(int)
    metrics = evaluate_model(y_test, y_pred_test, y_prob_test)
    
    print("Test Set Metrics:", metrics)
    with open('RESULTS.md', 'w') as f:
        f.write("# Model Performance\n")
        f.write("| Metric | Score |\n")
        f.write("|---|---|\n")
        for k, v in metrics.items():
            f.write(f"| {k.capitalize()} | {v:.4f} |\n")

if __name__ == "__main__":
    main()