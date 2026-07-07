import shap

def create_explainer(model, X_background=None, feature_names=None):
    """
    Create a SHAP TreeExplainer for the given model.
    Using tree_path_dependent bypasses the XGBoost categorical split bug.
    """
    explainer = shap.TreeExplainer(
        model, 
        feature_perturbation="tree_path_dependent"
    )
    return explainer