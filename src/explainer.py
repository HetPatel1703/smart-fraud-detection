import shap
import matplotlib.pyplot as plt

def create_explainer(model, X_background, feature_names=None):
    # TreeExplainer works directly on the XGBoost model
    explainer = shap.TreeExplainer(model, X_background, feature_names=feature_names)
    return explainer

def get_shap_values(explainer, X_sample):
    # Use the __call__ method for the modern SHAP Explanation object
    return explainer(X_sample)

def plot_waterfall(explainer, X_sample, index=0, save_path=None):
    explanation = explainer(X_sample.iloc[index:index+1])
    
    # Handle binary classification multi-dimensional outputs
    if len(explanation.values.shape) == 3:
        exp_to_plot = explanation[0, :, 1]
    else:
        exp_to_plot = explanation[0]
        
    fig = plt.figure()
    shap.plots.waterfall(exp_to_plot, show=False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return fig