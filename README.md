---
title: Real-Time Fraud Detection API
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
# Real-Time Fraud Detection API

FastAPI service combining a supervised XGBoost classifier with an unsupervised
Isolation Forest to flag both known fraud patterns and novel anomalies,
targeting sub-100ms inference latency. SMOTE corrects severe class imbalance
(0.17% fraud rate), SHAP TreeExplainer surfaces top contributing features per
prediction, and every request is logged (probability, risk level, latency)
to a SQLAlchemy-backed database.

## Folder structure

```
fraud-detection-api/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                  # put creditcard.csv here (see Dataset below)
â”‚   â””â”€â”€ processed/            # train/test splits get written here
â”œâ”€â”€ models/                   # trained model artifacts (.pkl) land here
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ config.py              # paths, constants, env vars
â”‚   â”œâ”€â”€ data_preprocessing.py  # load, split, scale, SMOTE
â”‚   â”œâ”€â”€ train_xgboost.py       # trains + saves the supervised model
â”‚   â”œâ”€â”€ train_isolation_forest.py  # trains + saves the anomaly model
â”‚   â”œâ”€â”€ explainer.py           # SHAP TreeExplainer wrapper
â”‚   â”œâ”€â”€ database.py            # SQLAlchemy engine/session/table
â”‚   â”œâ”€â”€ schemas.py              # Pydantic request/response models
â”‚   â””â”€â”€ main.py                # FastAPI app + /predict endpoint
â”œâ”€â”€ tests/
â”‚   â””â”€â”€ test_api.py
â”œâ”€â”€ run.py                    # uvicorn entrypoint
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ .env.example
â””â”€â”€ .gitignore
```

## Dataset

Use the Kaggle "Credit Card Fraud Detection" dataset (ULB):
https://www.kaggle.com/mlg-ulb/creditcardfraud

It has 284,807 transactions, 492 frauds â†’ 0.172% fraud rate â€” this is exactly
the number your resume line references. Columns: `Time`, `Amount`, `V1..V28`
(PCA-anonymized features), `Class` (1 = fraud).

Download `creditcard.csv` and place it at `data/raw/creditcard.csv`.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Order of execution

```bash
# 1. Preprocess (splits + scales + saves train/test to data/processed/)
python -m src.data_preprocessing

# 2. Train supervised model (XGBoost on SMOTE-balanced train set)
python -m src.train_xgboost

# 3. Train unsupervised model (Isolation Forest on the full train set)
python -m src.train_isolation_forest

# 4. Launch the API
python run.py
```

Then open http://127.0.0.1:8000/docs for interactive Swagger UI, or POST to
`/predict` with a JSON body of the 30 transaction features.

## Why two models

- **XGBoost (supervised)**: learns the actual labeled fraud patterns present
  in historical data. Strong at catching fraud that *looks like* fraud
  you've seen before.
- **Isolation Forest (unsupervised)**: trained without labels, isolates
  points that are structurally different from normal transactions. Catches
  novel fraud patterns that don't resemble anything in the labeled set.

The API combines both scores into one risk level so a transaction can be
flagged even if XGBoost alone wouldn't have caught it.

## Why SMOTE only on the training set

Fraud is 0.17% of transactions. Training directly on that imbalance means
the model can get 99.8% "accuracy" by always predicting "not fraud" â€” useless.
SMOTE synthetically oversamples the minority (fraud) class so XGBoost sees a
balanced signal during training. It's applied **after** the train/test split
so the test set stays representative of real-world imbalance (no leakage).

## Why SHAP TreeExplainer specifically

`shap.TreeExplainer` is optimized for tree-based models (XGBoost, Random
Forest, Isolation Forest) and computes exact Shapley values fast enough for
per-request use â€” unlike the generic `shap.Explainer`/KernelExplainer, which
is far too slow for a sub-100ms latency budget.

## Latency strategy

- Models are loaded **once** at FastAPI startup (`@app.on_event("startup")`),
  never per-request.
- Single-row inference avoids batching overhead.
- SHAP TreeExplainer on a single row is typically 1-5ms for a shallow-depth
  XGBoost model â€” cheap enough to run on every request.
- DB logging is done as a lightweight synchronous insert after the response
  is computed (you can switch this to a background task via
  `BackgroundTasks` if you want to shave off the DB write time from the
  measured latency).
