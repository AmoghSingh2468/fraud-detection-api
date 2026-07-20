FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

ENV DATABASE_URL=sqlite:////tmp/fraud_logs.db \
    XGB_MODEL_PATH=models/xgboost_model.pkl \
    ISO_MODEL_PATH=models/isolation_forest_model.pkl \
    SCALER_PATH=models/scaler.pkl \
    FRAUD_THRESHOLD=0.5 \
    ANOMALY_THRESHOLD=-0.1

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-7860}"]