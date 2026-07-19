"""
Central configuration. Everything path/threshold-related lives here so
other modules never hardcode strings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

X_TRAIN_PATH = PROCESSED_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"

XGB_MODEL_PATH = BASE_DIR / os.getenv("XGB_MODEL_PATH", "models/xgboost_model.pkl")
ISO_MODEL_PATH = BASE_DIR / os.getenv("ISO_MODEL_PATH", "models/isolation_forest_model.pkl")
SCALER_PATH = BASE_DIR / os.getenv("SCALER_PATH", "models/scaler.pkl")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud_logs.db")

FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", 0.5))
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", -0.1))

RANDOM_STATE = 42

# The 30 input features the model expects, in order.
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
