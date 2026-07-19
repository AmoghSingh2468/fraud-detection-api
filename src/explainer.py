"""
Thin wrapper around shap.TreeExplainer so main.py doesn't need to know
SHAP's API details. Returns the top-N contributing features for a single
prediction, which is what the resume line "surface the top contributing
features behind every prediction" refers to.
"""
import shap
import pandas as pd


class FraudExplainer:
    def __init__(self, xgb_model):
        # TreeExplainer is fast enough (~ms) for tree models on a single row,
        # unlike generic/Kernel SHAP which would blow the latency budget.
        self.explainer = shap.TreeExplainer(xgb_model)

    def top_features(self, row: pd.DataFrame, top_n: int = 5):
        shap_values = self.explainer.shap_values(row)
        contributions = list(zip(row.columns, shap_values[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return [
            {"feature": feat, "shap_value": round(float(val), 5)}
            for feat, val in contributions[:top_n]
        ]
