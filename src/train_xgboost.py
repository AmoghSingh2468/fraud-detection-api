"""
Trains the supervised XGBoost classifier on the SMOTE-balanced training set
and evaluates it on the untouched, realistically-imbalanced test set.

Run: python -m src.train_xgboost
"""
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from src import config


def main():
    X_train = pd.read_csv(config.X_TRAIN_PATH)
    y_train = pd.read_csv(config.Y_TRAIN_PATH).squeeze()
    X_test = pd.read_csv(config.X_TEST_PATH)
    y_test = pd.read_csv(config.Y_TEST_PATH).squeeze()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= config.FRAUD_THRESHOLD).astype(int)

    print("=== Classification report (test set, realistic imbalance) ===")
    print(classification_report(y_test, preds, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    print(f"PR-AUC (average precision): {average_precision_score(y_test, probs):.4f}")

    joblib.dump(model, config.XGB_MODEL_PATH)
    print(f"Saved model to {config.XGB_MODEL_PATH}")


if __name__ == "__main__":
    main()
