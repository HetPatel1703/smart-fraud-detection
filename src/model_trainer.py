import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score 
from sklearn.metrics import make_scorer, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

def train_model(X_train, y_train, random_state=42):
    """
    Train an XGBoost model with SMOTE and hyperparameter tuning.
    Returns the best estimator and the threshold that maximises F1 on validation.
    """
    # Use a small validation set from training for tuning (10% of train)
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=random_state, stratify=y_train
    )
    
    # Base model: XGBoost with scale_pos_weight to handle imbalance (we also use SMOTE)
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Pipeline with SMOTE
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=random_state, sampling_strategy=0.1)),  # balance to 10% fraud
        ('clf', model)
    ])
    
    # Hyperparameter search
    param_grid = {
        'clf__max_depth': [4, 6, 8],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__subsample': [0.7, 0.8, 0.9],
        'clf__colsample_bytree': [0.7, 0.8, 0.9],
        'clf__scale_pos_weight': [1, 5, 10]   # but SMOTE already handles imbalance
    }
    # For speed, we do a small random search
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        scoring='f1',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X_tr, y_tr)
    best_pipeline = random_search.best_estimator_
    
    # Calibrate probabilities on validation set (Platt scaling)
    # We need to get predicted probabilities from the pipeline, then calibrate
    # Calibration will be done on the full training set later, but we can do it here
    # For simplicity, we'll calibrate after tuning using CalibratedClassifierCV on the whole train set.
    # We'll return the best pipeline and the threshold.
    
    # Find optimal threshold on validation set using F1
    y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    for t in thresholds:
        pred = (y_val_prob >= t).astype(int)
        f1 = f1_score(y_val, pred)
        f1_scores.append(f1)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    return best_pipeline, best_threshold

def calibrate_model(pipeline, X_train, y_train, cv='prefit'):
    """
    Calibrate the model using Platt scaling on the training set.
    If cv='prefit', we assume pipeline is already fitted on X_train.
    """
    calibrated = CalibratedClassifierCV(pipeline, cv=cv, method='sigmoid')
    calibrated.fit(X_train, y_train)
    return calibrated

def save_artifacts(model, scaler, explainer, threshold, path='models/'):
    """Save model, scaler, explainer, threshold."""
    import os
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, f'{path}/model.joblib')
    joblib.dump(scaler, f'{path}/scaler.joblib')
    joblib.dump(explainer, f'{path}/explainer.joblib')
    joblib.dump(threshold, f'{path}/threshold.joblib')

if __name__ == "__main__":
    # test (will be called from train.py)
    pass