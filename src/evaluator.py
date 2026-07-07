import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    average_precision_score, roc_auc_score, 
    confusion_matrix, classification_report
)

def evaluate_model(y_true, y_pred, y_prob, threshold=None):
    """Compute and return metrics."""
    if threshold is not None:
        y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'pr_auc': average_precision_score(y_true, y_prob),
        'roc_auc': roc_auc_score(y_true, y_prob),
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def recall_at_fpr(y_true, y_prob, fpr_target=0.01):
    """
    Compute recall at a given false positive rate.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # find threshold with fpr <= fpr_target
    idx = np.where(fpr <= fpr_target)[0]
    if len(idx) == 0:
        return 0.0
    max_tpr = tpr[idx[-1]] if len(idx) > 0 else 0.0
    return max_tpr

if __name__ == "__main__":
    pass