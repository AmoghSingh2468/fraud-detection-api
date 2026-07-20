"""
Thin wrapper around shap.TreeExplainer so main.py doesn't need to know
SHAP's API details. SHAP is imported lazily on first use rather than at
startup, which keeps ~100MB out of the idle memory footprint.
"""
import pandas as pd


class FraudExplainer:
    def __init__(self, xgb_model):
        self._model = xgb_model
        self._explainer = None

    def _get_explainer(self):
        # Deferred import: shap only loads when an explanation is first
        # requested, not when the container boots.
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self._model)
        return self._explainer

    def top_features(self, row: pd.DataFrame, top_n: int = 5):
        shap_values = self._get_explainer().shap_values(row)
        contributions = list(zip(row.columns, shap_values[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return [
            {"feature": feat, "shap_value": round(float(val), 5)}
            for feat, val in contributions[:top_n]
        ]