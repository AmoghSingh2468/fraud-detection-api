"""
Pydantic models defining the API's request and response contracts.
"""
from pydantic import BaseModel, Field
from typing import List


class TransactionRequest(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0, "V1": -2.3, "V2": 1.4, "V3": -3.1, "V4": 2.2,
                "V5": -1.1, "V6": -0.5, "V7": -2.9, "V8": 1.1, "V9": -0.8,
                "V10": -2.5, "V11": 2.0, "V12": -3.0, "V13": 0.4, "V14": -3.6,
                "V15": 0.2, "V16": -1.7, "V17": -1.9, "V18": -0.3, "V19": 0.1,
                "V20": 0.05, "V21": 0.3, "V22": 0.7, "V23": -0.02, "V24": 0.1,
                "V25": 0.2, "V26": -0.1, "V27": 0.05, "V28": 0.02, "Amount": 0.0
            }
        }


class TopFeature(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="XGBoost predicted probability of fraud")
    anomaly_score: float = Field(..., description="Isolation Forest anomaly score (lower = more anomalous)")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH")
    top_features: List[TopFeature]
    latency_ms: float
