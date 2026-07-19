"""
Loads the raw Kaggle creditcard.csv, does a stratified train/test split,
scales Time and Amount (V1-V28 are already PCA-scaled by Kaggle), applies
SMOTE to the TRAINING set only, and writes everything to data/processed/.

Run: python -m src.data_preprocessing
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src import config


def load_raw_data() -> pd.DataFrame:
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset at {config.RAW_DATA_PATH}. Download "
            "creditcard.csv from Kaggle (mlg-ulb/creditcardfraud) and "
            "place it there."
        )
    return pd.read_csv(config.RAW_DATA_PATH)


def main():
    df = load_raw_data()
    print(f"Loaded {len(df):,} rows. Fraud rate: {df['Class'].mean() * 100:.4f}%")

    X = df[config.FEATURE_COLUMNS].copy()
    y = df["Class"].copy()

    # Stratified split keeps the ~0.17% fraud ratio in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )

    # Scale Time and Amount (V1-V28 already PCA-normalized by Kaggle).
    scaler = StandardScaler()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    # SMOTE only on the training set -> no leakage into test evaluation.
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(
        f"After SMOTE: {len(X_train_res):,} rows, "
        f"fraud rate {y_train_res.mean() * 100:.2f}%"
    )

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_res.to_csv(config.X_TRAIN_PATH, index=False)
    X_test.to_csv(config.X_TEST_PATH, index=False)
    y_train_res.to_csv(config.Y_TRAIN_PATH, index=False)
    y_test.to_csv(config.Y_TEST_PATH, index=False)

    joblib.dump(scaler, config.SCALER_PATH)
    print("Saved processed splits and scaler.pkl")


if __name__ == "__main__":
    main()
