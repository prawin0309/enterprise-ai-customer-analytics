"""Scoring service over the trained customer intelligence models.

Exposes the same three models the batch pipeline uses, so a CRM front end can
request a single account on demand instead of waiting for the nightly run.
Models and the processed feature table are loaded once at start-up; the raw CRM
extract is not needed at serving time.

    uvicorn src.api:app --reload
    curl http://127.0.0.1:8000/customers/12908/score
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

try:
    from paths import MODEL_DIR, PROCESSED_FILE
except ImportError:  # imported as ``src.api``
    from src.paths import MODEL_DIR, PROCESSED_FILE

CUSTOMER_KEY = "Customer_ID"

HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.30

logger = logging.getLogger("api")

app = FastAPI(
    title="Customer Intelligence Scoring API",
    description="Churn probability, next-quarter revenue forecast and behavioural "
    "segment for a single account.",
    version="1.0.0",
)


def load_artifacts() -> dict:
    """Load the serialised models and the training feature contract."""
    return {
        "classifier": joblib.load(MODEL_DIR / "classifier.pkl"),
        "regressor": joblib.load(MODEL_DIR / "regressor.pkl"),
        "clusterer": joblib.load(MODEL_DIR / "clusterer.pkl"),
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "pca": joblib.load(MODEL_DIR / "segmentation_pca.pkl"),
        "contract": joblib.load(MODEL_DIR / "feature_contract.pkl"),
    }


ARTIFACTS = load_artifacts()
CUSTOMERS = pd.read_csv(PROCESSED_FILE).set_index(CUSTOMER_KEY, drop=False)
logger.info("Loaded %d customers and 6 model artifacts", len(CUSTOMERS))


def classify_risk_level(probability: float) -> str:
    """Map a churn probability onto the business-facing risk band."""
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def score_row(row: pd.DataFrame) -> dict:
    """Run one customer row through all three models."""
    contract = ARTIFACTS["contract"]

    probability = float(
        ARTIFACTS["classifier"].predict_proba(row[contract["churn_features"]])[0, 1]
    )
    forecast = float(ARTIFACTS["regressor"].predict(row[contract["revenue_features"]])[0])

    segment_data = row[contract["segmentation_features"]].copy()
    for column in contract["log_scale_features"]:
        segment_data[column] = np.log1p(segment_data[column].clip(lower=0))
    reduced = ARTIFACTS["pca"].transform(ARTIFACTS["scaler"].transform(segment_data))
    cluster_id = int(ARTIFACTS["clusterer"].predict(reduced)[0])

    return {
        "customer_id": int(row[CUSTOMER_KEY].iloc[0]),
        "churn_probability": round(probability, 4),
        "risk_level": classify_risk_level(probability),
        "predicted_next_quarter_revenue_usd": round(forecast, 2),
        "revenue_at_risk_usd": round(max(forecast, 0.0) * probability, 2),
        "cluster_id": cluster_id,
    }


@app.get("/health")
def health() -> dict:
    """Readiness probe reporting what the service has loaded."""
    return {
        "status": "ok",
        "customers_loaded": len(CUSTOMERS),
        "models_loaded": sorted(ARTIFACTS),
    }


@app.get("/customers/{customer_id}/score")
def score_customer(customer_id: int) -> dict:
    """Score a single account."""
    if customer_id not in CUSTOMERS.index:
        raise HTTPException(status_code=404, detail=f"Unknown customer {customer_id}")
    return score_row(CUSTOMERS.loc[[customer_id]])
