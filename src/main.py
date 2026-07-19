"""
FastAPI application. Loads both models once at startup, exposes /predict
which combines XGBoost's supervised probability with Isolation Forest's
anomaly score into a single risk level, attaches SHAP feature attributions,
logs the result to the database, and returns everything within a
sub-100ms latency budget.

Run: python run.py   (or: uvicorn src.main:app --reload)
"""
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src import config
from src.database import init_db, get_db, PredictionLog
from src.explainer import FraudExplainer
from src.schemas import TransactionRequest, PredictionResponse

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load everything ONCE at startup -- never per-request. This is the
    # single biggest lever for hitting sub-100ms latency.
    ml_models["xgb"] = joblib.load(config.XGB_MODEL_PATH)
    ml_models["iso"] = joblib.load(config.ISO_MODEL_PATH)
    ml_models["scaler"] = joblib.load(config.SCALER_PATH)
    ml_models["explainer"] = FraudExplainer(ml_models["xgb"])
    init_db()
    yield
    ml_models.clear()


app = FastAPI(title="Real-Time Fraud Detection API", lifespan=lifespan)

FRONTEND_DIR = config.BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")


def _risk_level(fraud_prob: float, anomaly_score: float) -> str:
    is_fraud_flag = fraud_prob >= config.FRAUD_THRESHOLD
    is_anomaly_flag = anomaly_score <= config.ANOMALY_THRESHOLD

    if is_fraud_flag and is_anomaly_flag:
        return "HIGH"
    if is_fraud_flag or is_anomaly_flag:
        return "MEDIUM"
    return "LOW"


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(ml_models.keys())}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest, db: Session = Depends(get_db)):
    start = time.perf_counter()

    row = pd.DataFrame([transaction.model_dump()])[config.FEATURE_COLUMNS]

    # Scale Time/Amount with the SAME scaler fit during preprocessing.
    scaler = ml_models["scaler"]
    row[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])

    xgb_model = ml_models["xgb"]
    iso_model = ml_models["iso"]
    explainer = ml_models["explainer"]

    fraud_probability = float(xgb_model.predict_proba(row)[0, 1])
    anomaly_score = float(iso_model.decision_function(row)[0])
    risk_level = _risk_level(fraud_probability, anomaly_score)
    top_features = explainer.top_features(row, top_n=5)

    latency_ms = (time.perf_counter() - start) * 1000

    log_entry = PredictionLog(
        fraud_probability=fraud_probability,
        anomaly_score=anomaly_score,
        risk_level=risk_level,
        latency_ms=latency_ms,
    )
    db.add(log_entry)
    db.commit()

    return PredictionResponse(
        fraud_probability=round(fraud_probability, 5),
        anomaly_score=round(anomaly_score, 5),
        risk_level=risk_level,
        top_features=top_features,
        latency_ms=round(latency_ms, 2),
    )
