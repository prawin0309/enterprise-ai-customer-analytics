"""Customer intelligence fusion layer.

Scores every customer with the three trained models and fuses the outputs into
a single standardised JSON profile per customer:

    churn probability + risk level      (classifier.pkl)
    next-quarter revenue forecast       (regressor.pkl)
    behavioural cluster id + label      (clusterer.pkl / scaler.pkl / pca.pkl)
    per-customer SHAP churn drivers     (TreeExplainer over the classifier)
    derived risk / opportunity scores   (fusion logic in this module)

The profiles are the hand-off contract consumed by ``llm_insights.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SOURCE_DIR = Path(r"D:\DS_F\DataSet-20260223T124234Z-1-001\DataSet")
PROCESSED_FILE = Path(r"D:\DS_FO\data\processed\processed_customer_data.csv")
MODEL_DIR = Path(r"D:\DS_FO\models")
OUTPUT_DIR = Path(r"D:\DS_FO\data\processed")
REPORT_DIR = Path(r"D:\DS_FO\reports")

PROFILES_FILE = OUTPUT_DIR / "customer_profiles.json"
PRIORITY_FILE = OUTPUT_DIR / "priority_accounts.json"
SUMMARY_FILE = REPORT_DIR / "fusion_summary.csv"

CUSTOMER_KEY = "Customer_ID"
CHURN_TARGET = "Churn"
REVENUE_TARGET = "Next_Quarter_Revenue_USD"

TOP_DRIVERS_PER_CUSTOMER = 5
PRIORITY_ACCOUNT_COUNT = 5

# Churn probability cut-offs for the business-facing risk band.
HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.30

# Descriptive attributes pulled back from the source data for readability.
CONTEXT_COLUMNS = [
    "Account_Tier",
    "Company_Size",
    "Contract_Type",
    "Tenure_Months",
    "Health_Score",
    "ACV_USD",
    "Lifetime_Revenue_USD",
    "Support_Tickets",
    "CSAT_Score",
    "NPS_Score",
    "Renewal_Probability",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fusion_layer")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_artifacts() -> dict:
    """Load the serialised models and the feature contract from training."""
    artifacts = {
        "classifier": joblib.load(MODEL_DIR / "classifier.pkl"),
        "regressor": joblib.load(MODEL_DIR / "regressor.pkl"),
        "clusterer": joblib.load(MODEL_DIR / "clusterer.pkl"),
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "pca": joblib.load(MODEL_DIR / "segmentation_pca.pkl"),
        "contract": joblib.load(MODEL_DIR / "feature_contract.pkl"),
    }
    logger.info("Loaded %d model artifacts from %s", len(artifacts), MODEL_DIR)
    return artifacts


def load_business_context() -> pd.DataFrame:
    """Read human-readable descriptive attributes from the read-only source data."""
    customers = pd.read_csv(SOURCE_DIR / "fact_customers.csv")
    customers = customers.drop_duplicates(subset=[CUSTOMER_KEY], keep="first")

    # Dimension keys are not unique in the source extract (dim_geography ships
    # duplicate Geo_ID rows), so deduplicate before joining.
    industry = pd.read_csv(SOURCE_DIR / "dim_industry.csv")[
        ["Industry_ID", "Industry", "Business_Model"]
    ].drop_duplicates(subset=["Industry_ID"], keep="first")
    geography = pd.read_csv(SOURCE_DIR / "dim_geography.csv")[
        ["Geo_ID", "Country", "Continent", "Market_Tier"]
    ].drop_duplicates(subset=["Geo_ID"], keep="first")
    product = pd.read_csv(SOURCE_DIR / "dim_product.csv")[
        ["Product_ID", "SKU", "Plan_Type", "Module", "Support_Level"]
    ].drop_duplicates(subset=["Product_ID"], keep="first")

    context = (
        customers.merge(industry, on="Industry_ID", how="left")
        .merge(geography, on="Geo_ID", how="left")
        .merge(product, on="Product_ID", how="left")
    )
    keep = [CUSTOMER_KEY] + CONTEXT_COLUMNS + [
        "Industry",
        "Business_Model",
        "Country",
        "Continent",
        "Market_Tier",
        "SKU",
        "Plan_Type",
        "Module",
        "Support_Level",
    ]
    context = context[[c for c in keep if c in context.columns]]
    logger.info("Loaded business context %s", context.shape)
    return context


# --------------------------------------------------------------------------- #
# Model scoring
# --------------------------------------------------------------------------- #


def score_all_models(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Run every customer through the classifier, regressor and clusterer."""
    contract = artifacts["contract"]
    churn_features = df[contract["churn_features"]]

    scores = pd.DataFrame({CUSTOMER_KEY: df[CUSTOMER_KEY].to_numpy()})
    scores["churn_probability"] = artifacts["classifier"].predict_proba(
        churn_features
    )[:, 1]
    scores["predicted_next_quarter_revenue_usd"] = artifacts["regressor"].predict(
        df[contract["revenue_features"]]
    )

    segment_data = df[contract["segmentation_features"]].copy()
    for column in contract["log_scale_features"]:
        segment_data[column] = np.log1p(segment_data[column].clip(lower=0))
    reduced = artifacts["pca"].transform(artifacts["scaler"].transform(segment_data))
    scores["cluster_id"] = artifacts["clusterer"].predict(reduced)

    logger.info(
        "Scored %d customers | mean churn prob %.3f | mean forecast $%.0f",
        len(scores),
        scores["churn_probability"].mean(),
        scores["predicted_next_quarter_revenue_usd"].mean(),
    )
    return scores


def compute_local_shap_drivers(
    df: pd.DataFrame, artifacts: dict
) -> list[list[dict]]:
    """Return the top per-customer SHAP churn drivers, signed by direction."""
    features = df[artifacts["contract"]["churn_features"]]
    explainer = shap.TreeExplainer(artifacts["classifier"])
    shap_values = explainer.shap_values(features)
    logger.info("Computed SHAP values %s", shap_values.shape)

    column_names = np.array(features.columns)
    ranked = np.argsort(-np.abs(shap_values), axis=1)[:, :TOP_DRIVERS_PER_CUSTOMER]

    drivers: list[list[dict]] = []
    for row_index, column_indices in enumerate(ranked):
        drivers.append(
            [
                {
                    "feature": str(column_names[column_index]),
                    "value": round(
                        float(features.iat[row_index, int(column_index)]), 4
                    ),
                    "shap_value": round(
                        float(shap_values[row_index, column_index]), 4
                    ),
                    "direction": (
                        "increases_churn_risk"
                        if shap_values[row_index, column_index] > 0
                        else "reduces_churn_risk"
                    ),
                }
                for column_index in column_indices
            ]
        )
    return drivers


# --------------------------------------------------------------------------- #
# Fusion logic
# --------------------------------------------------------------------------- #


def label_clusters(df: pd.DataFrame, scores: pd.DataFrame, artifacts: dict) -> dict:
    """Derive business-readable names for each behavioural cluster."""
    features = artifacts["contract"]["segmentation_features"]
    profile = df[features].copy()
    profile["cluster_id"] = scores["cluster_id"].to_numpy()
    profile["churn_rate"] = df[CHURN_TARGET].to_numpy()
    summary = profile.groupby("cluster_id").mean()

    median_revenue = summary["avg_monthly_revenue_usd"].median()
    median_adoption = summary["usage_mean_feature_adoption_pct"].median()
    median_churn = summary["churn_rate"].median()

    labels: dict[int, dict] = {}
    for cluster_id, row in summary.iterrows():
        high_value = row["avg_monthly_revenue_usd"] >= median_revenue
        engaged = row["usage_mean_feature_adoption_pct"] >= median_adoption
        at_risk = row["churn_rate"] >= median_churn

        if engaged and high_value and not at_risk:
            name = "Loyal High-Value Accounts"
        elif not engaged and at_risk:
            name = "Disengaged At-Risk Accounts"
        elif engaged and not high_value:
            name = "Growth-Potential Accounts"
        elif high_value and at_risk:
            name = "High-Value At-Risk Accounts"
        else:
            name = "Cost-Sensitive Steady Accounts"

        labels[int(cluster_id)] = {
            "cluster_label": name,
            "cluster_size": int((profile["cluster_id"] == cluster_id).sum()),
            "cluster_churn_rate": round(float(row["churn_rate"]), 4),
            "cluster_mean_monthly_revenue_usd": round(
                float(row["avg_monthly_revenue_usd"]), 2
            ),
            "cluster_mean_feature_adoption_pct": round(
                float(row["usage_mean_feature_adoption_pct"]), 2
            ),
        }
    logger.info("Cluster labels: %s", {k: v["cluster_label"] for k, v in labels.items()})
    return labels


def classify_risk_level(probability: float) -> str:
    """Map a churn probability onto a business-facing risk band."""
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def classify_engagement_level(adoption_percentile: float) -> str:
    """Map a feature-adoption percentile onto an engagement band."""
    if adoption_percentile >= 0.66:
        return "High"
    if adoption_percentile >= 0.33:
        return "Moderate"
    return "Low"


def fuse_scores(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    context: pd.DataFrame,
    cluster_labels: dict,
) -> pd.DataFrame:
    """Combine model outputs into derived risk and opportunity indicators."""
    fused = scores.merge(context, on=CUSTOMER_KEY, how="left")
    if len(fused) != len(scores):
        raise ValueError(
            f"Context join changed the customer grain: {len(scores)} -> {len(fused)}"
        )

    forecast = fused["predicted_next_quarter_revenue_usd"].clip(lower=0)
    fused["revenue_at_risk_usd"] = forecast * fused["churn_probability"]

    # Scores are percentile-normalised to 0-100 so they compare across accounts.
    fused["risk_score"] = (fused["churn_probability"].rank(pct=True) * 100).round(1)
    fused["revenue_potential_score"] = (forecast.rank(pct=True) * 100).round(1)
    fused["priority_score"] = (
        fused["revenue_at_risk_usd"].rank(pct=True) * 100
    ).round(1)

    adoption_percentile = df["usage_mean_feature_adoption_pct"].rank(pct=True)
    fused["engagement_level"] = [
        classify_engagement_level(p) for p in adoption_percentile
    ]
    fused["risk_level"] = [
        classify_risk_level(p) for p in fused["churn_probability"]
    ]
    fused["cluster_label"] = fused["cluster_id"].map(
        lambda c: cluster_labels[int(c)]["cluster_label"]
    )
    return fused


# --------------------------------------------------------------------------- #
# Profile assembly
# --------------------------------------------------------------------------- #


def build_profiles(
    fused: pd.DataFrame, drivers: list[list[dict]], cluster_labels: dict
) -> list[dict]:
    """Assemble one standardised JSON intelligence profile per customer."""
    profiles: list[dict] = []
    for position, row in enumerate(fused.itertuples(index=False)):
        record = row._asdict()
        cluster_id = int(record["cluster_id"])
        profiles.append(
            {
                "customer_id": int(record[CUSTOMER_KEY]),
                "account_context": {
                    "industry": record.get("Industry"),
                    "business_model": record.get("Business_Model"),
                    "country": record.get("Country"),
                    "market_tier": record.get("Market_Tier"),
                    "account_tier": record.get("Account_Tier"),
                    "company_size": record.get("Company_Size"),
                    "plan_type": record.get("Plan_Type"),
                    "module": record.get("Module"),
                    "support_level": record.get("Support_Level"),
                    "contract_type": record.get("Contract_Type"),
                    "tenure_months": _to_number(record.get("Tenure_Months")),
                },
                "churn_prediction": {
                    "probability": round(float(record["churn_probability"]), 4),
                    "risk_level": record["risk_level"],
                    "renewal_probability_crm": _to_number(
                        record.get("Renewal_Probability")
                    ),
                },
                "revenue_forecast": {
                    "predicted_next_quarter_usd": round(
                        float(record["predicted_next_quarter_revenue_usd"]), 2
                    ),
                    "current_acv_usd": _to_number(record.get("ACV_USD")),
                    "lifetime_revenue_usd": _to_number(
                        record.get("Lifetime_Revenue_USD")
                    ),
                    "revenue_at_risk_usd": round(
                        float(record["revenue_at_risk_usd"]), 2
                    ),
                },
                "segmentation": {
                    "cluster_id": cluster_id,
                    **cluster_labels[cluster_id],
                },
                "intelligence_scores": {
                    "risk_score": float(record["risk_score"]),
                    "revenue_potential_score": float(
                        record["revenue_potential_score"]
                    ),
                    "priority_score": float(record["priority_score"]),
                    "engagement_level": record["engagement_level"],
                    "health_score": _to_number(record.get("Health_Score")),
                    "csat_score": _to_number(record.get("CSAT_Score")),
                    "nps_score": _to_number(record.get("NPS_Score")),
                    "support_tickets": _to_number(record.get("Support_Tickets")),
                },
                "top_churn_drivers": drivers[position],
            }
        )
    return profiles


def _to_number(value: object) -> float | int | None:
    """Coerce a value to a JSON-safe number, mapping NaN to ``None``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    return None


def select_priority_accounts(profiles: list[dict], count: int) -> list[dict]:
    """Pick the accounts with the largest forecast revenue exposed to churn."""
    ranked = sorted(
        profiles,
        key=lambda p: p["revenue_forecast"]["revenue_at_risk_usd"],
        reverse=True,
    )
    return ranked[:count]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_fusion() -> list[dict]:
    """Score all customers, fuse the outputs, and write the profile files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_FILE)
    logger.info("Loaded %s %s", PROCESSED_FILE.name, df.shape)

    artifacts = load_artifacts()
    context = load_business_context()

    scores = score_all_models(df, artifacts)
    drivers = compute_local_shap_drivers(df, artifacts)
    cluster_labels = label_clusters(df, scores, artifacts)
    fused = fuse_scores(df, scores, context, cluster_labels)

    profiles = build_profiles(fused, drivers, cluster_labels)
    PROFILES_FILE.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    logger.info("Wrote %d profiles to %s", len(profiles), PROFILES_FILE)

    priority = select_priority_accounts(profiles, PRIORITY_ACCOUNT_COUNT)
    PRIORITY_FILE.write_text(json.dumps(priority, indent=2), encoding="utf-8")
    logger.info(
        "Wrote top %d priority accounts to %s (customer ids %s)",
        len(priority),
        PRIORITY_FILE,
        [p["customer_id"] for p in priority],
    )

    summary_columns = [
        CUSTOMER_KEY,
        "churn_probability",
        "risk_level",
        "predicted_next_quarter_revenue_usd",
        "revenue_at_risk_usd",
        "cluster_id",
        "cluster_label",
        "engagement_level",
        "risk_score",
        "revenue_potential_score",
        "priority_score",
    ]
    fused[summary_columns].to_csv(SUMMARY_FILE, index=False)
    logger.info("Wrote fusion summary to %s", SUMMARY_FILE)

    logger.info(
        "Risk distribution: %s",
        dict(fused["risk_level"].value_counts()),
    )
    logger.info(
        "Total forecast revenue at risk: $%.0f",
        fused["revenue_at_risk_usd"].sum(),
    )
    return profiles


if __name__ == "__main__":
    run_fusion()
