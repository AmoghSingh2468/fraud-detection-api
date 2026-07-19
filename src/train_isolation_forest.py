"""
Trains the unsupervised Isolation Forest on the (pre-SMOTE-composition-aware)
training features, ignoring labels entirely. Its job is to flag transactions
that are structurally anomalous even if XGBoost wouldn't flag them.

Run: python -m src.train_isolation_forest
"""
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import config


def main():
    X_train = pd.read_csv(config.X_TRAIN_PATH)

    # contamination ~ expected proportion of anomalies; we set it close to
    # the real-world fraud rate rather than the SMOTE-inflated one, since
    # Isolation Forest's job is to model what "normal" looks like.
    model = IsolationForest(
        n_estimators=200,
        contamination=0.0017,
        max_samples="auto",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train)

    joblib.dump(model, config.ISO_MODEL_PATH)
    print(f"Saved model to {config.ISO_MODEL_PATH}")


if __name__ == "__main__":
    main()
