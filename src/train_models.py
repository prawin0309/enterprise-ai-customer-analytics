"""Model training for the SaaS customer intelligence project.

Trains and serialises the three core models on the processed customer table:

    1. Churn prediction      - XGBoost classifier (ROC-AUC, F1, SHAP drivers)
    2. Revenue forecasting   - XGBoost / GradientBoosting regressor (RMSE, MAE, R2)
    3. Behavioural segments  - KMeans clustering (silhouette validated)

Artefacts are written to ``D:\\DS_FO\\models`` and metric/importance reports to
``D:\\DS_FO\\reports``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    calinski_harabasz_score,
    classification_report,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROCESSED_FILE = Path(r"D:\DS_FO\data\processed\processed_customer_data.csv")
MODEL_DIR = Path(r"D:\DS_FO\models")
REPORT_DIR = Path(r"D:\DS_FO\reports")

CUSTOMER_KEY = "Customer_ID"
CHURN_TARGET = "Churn"
REVENUE_TARGET = "Next_Quarter_Revenue_USD"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
SHAP_SAMPLE_SIZE = 500
SILHOUETTE_TARGET = 0.60
CLUSTER_RANGE = range(2, 11)

# ``Cluster_Label`` is a pre-existing segment tag shipped with the source data
# and encodes churn risk directly, so it is excluded from supervised training.
LEAKAGE_COLUMNS = [CUSTOMER_KEY, "Cluster_Label"]

# Behavioural features used for segmentation (usage, engagement, value, tenure).
SEGMENTATION_FEATURES = [
    "usage_mean_mau",
    "usage_mean_feature_adoption_pct",
    "usage_intensity_score",
    "usage_mean_license_utilization_pct",
    "engagement_quality_score",
    "events_per_tenure_month",
    "support_dependency_score",
    "avg_monthly_revenue_usd",
    "Tenure_Months",
    "Health_Score",
]

# Heavily right-skewed features are log-compressed before clustering.
LOG_SCALE_FEATURES = [
    "usage_mean_mau",
    "usage_intensity_score",
    "events_per_tenure_month",
    "support_dependency_score",
    "avg_monthly_revenue_usd",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_models")


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #


def load_processed_data(path: Path = PROCESSED_FILE) -> pd.DataFrame:
    """Load the customer-level modelling table produced by ``preprocessing``."""
    df = pd.read_csv(path)
    logger.info("Loaded %s %s", path.name, df.shape)
    return df


def build_supervised_matrix(
    df: pd.DataFrame, target: str, drop_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the table into a feature matrix and a target vector."""
    features = df.drop(columns=[c for c in drop_columns if c in df.columns])
    return features, df[target]


# --------------------------------------------------------------------------- #
# Model 1: churn classification
# --------------------------------------------------------------------------- #


def train_churn_classifier(df: pd.DataFrame) -> tuple[XGBClassifier, dict]:
    """Train an XGBoost churn classifier on a SMOTE-balanced training split."""
    logger.info("--- Model 1: churn classification ---")
    drop_columns = LEAKAGE_COLUMNS + [CHURN_TARGET, REVENUE_TARGET]
    features, target = build_supervised_matrix(df, CHURN_TARGET, drop_columns)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    x_train_balanced, y_train_balanced = SMOTE(
        random_state=RANDOM_STATE
    ).fit_resample(x_train, y_train)
    logger.info(
        "SMOTE applied to training split: %d -> %d rows",
        len(x_train),
        len(x_train_balanced),
    )

    classifier = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    classifier.fit(x_train_balanced, y_train_balanced)

    probabilities = classifier.predict_proba(x_test)[:, 1]
    predictions = classifier.predict(x_test)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "f1_score": float(f1_score(y_test, predictions)),
        "cv_roc_auc_mean": float(
            cross_val_score(
                classifier,
                x_train_balanced,
                y_train_balanced,
                cv=CV_FOLDS,
                scoring="roc_auc",
                n_jobs=-1,
            ).mean()
        ),
        "n_features": int(features.shape[1]),
    }

    # Baseline comparison required by the project guidelines.
    baseline = GradientBoostingClassifier(random_state=RANDOM_STATE)
    baseline.fit(x_train_balanced, y_train_balanced)
    metrics["baseline_gb_roc_auc"] = float(
        roc_auc_score(y_test, baseline.predict_proba(x_test)[:, 1])
    )

    logger.info(
        "XGBoost  ROC-AUC=%.4f  F1=%.4f  CV-ROC-AUC=%.4f | GB baseline ROC-AUC=%.4f",
        metrics["roc_auc"],
        metrics["f1_score"],
        metrics["cv_roc_auc_mean"],
        metrics["baseline_gb_roc_auc"],
    )
    logger.info(
        "Classification report:\n%s",
        classification_report(y_test, predictions, digits=3),
    )

    importance = compute_shap_importance(classifier, x_test)
    metrics["top_shap_drivers"] = importance.head(15).to_dict("records")
    return classifier, metrics


def compute_shap_importance(
    model: XGBClassifier, x_test: pd.DataFrame
) -> pd.DataFrame:
    """Compute mean absolute SHAP values as a global feature-importance ranking."""
    sample = x_test.sample(
        n=min(SHAP_SAMPLE_SIZE, len(x_test)), random_state=RANDOM_STATE
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    importance = (
        pd.DataFrame(
            {
                "feature": sample.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    output_path = REPORT_DIR / "shap_feature_importance.csv"
    importance.to_csv(output_path, index=False)
    logger.info("SHAP importance written to %s", output_path)
    logger.info(
        "Top 10 churn drivers:\n%s", importance.head(10).to_string(index=False)
    )
    return importance


# --------------------------------------------------------------------------- #
# Model 2: revenue regression
# --------------------------------------------------------------------------- #


def train_revenue_regressor(df: pd.DataFrame) -> tuple[object, dict]:
    """Train and select the better of XGBoost / GradientBoosting on RMSE."""
    logger.info("--- Model 2: next-quarter revenue forecasting ---")
    drop_columns = LEAKAGE_COLUMNS + [REVENUE_TARGET, CHURN_TARGET]
    features, target = build_supervised_matrix(df, REVENUE_TARGET, drop_columns)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    candidates = {
        "xgboost": XGBRegressor(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        results[name] = {
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        }
        fitted[name] = model
        logger.info(
            "%-18s RMSE=%12.2f  MAE=%12.2f  R2=%.4f",
            name,
            results[name]["rmse"],
            results[name]["mae"],
            results[name]["r2"],
        )

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = fitted[best_name]

    cv_r2 = cross_val_score(
        best_model, x_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    n_samples, n_features = x_test.shape
    adjusted_r2 = 1 - (1 - results[best_name]["r2"]) * (n_samples - 1) / (
        n_samples - n_features - 1
    )

    metrics = {
        "selected_model": best_name,
        **results[best_name],
        "adjusted_r2": float(adjusted_r2),
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "all_candidates": results,
    }
    logger.info(
        "Selected %s | adjusted R2=%.4f | CV R2=%.4f (+/- %.4f)",
        best_name,
        metrics["adjusted_r2"],
        metrics["cv_r2_mean"],
        metrics["cv_r2_std"],
    )
    return best_model, metrics


# --------------------------------------------------------------------------- #
# Model 3: behavioural segmentation
# --------------------------------------------------------------------------- #


def prepare_segmentation_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Log-compress skewed behavioural features and standardise them."""
    segment_data = df[SEGMENTATION_FEATURES].copy()
    for column in LOG_SCALE_FEATURES:
        segment_data[column] = np.log1p(segment_data[column].clip(lower=0))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(segment_data)
    return scaled, scaler


def train_segmentation_model(
    df: pd.DataFrame,
) -> tuple[KMeans, StandardScaler, PCA, dict]:
    """Fit KMeans over a range of k and keep the best silhouette score."""
    logger.info("--- Model 3: behavioural segmentation ---")
    scaled, scaler = prepare_segmentation_matrix(df)

    # PCA denoises correlated behavioural signals and sharpens cluster geometry.
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(scaled)
    logger.info(
        "PCA: %d -> %d components (%.1f%% variance retained)",
        scaled.shape[1],
        reduced.shape[1],
        pca.explained_variance_ratio_.sum() * 100,
    )

    sweep: list[dict] = []
    models: dict[int, KMeans] = {}
    for k in CLUSTER_RANGE:
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(reduced)
        sweep.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(reduced, labels)),
                "davies_bouldin": float(davies_bouldin_score(reduced, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(reduced, labels)),
                "inertia": float(model.inertia_),
            }
        )
        models[k] = model
        logger.info(
            "k=%-2d silhouette=%.4f  davies_bouldin=%.4f  inertia=%.1f",
            k,
            sweep[-1]["silhouette"],
            sweep[-1]["davies_bouldin"],
            sweep[-1]["inertia"],
        )

    best = max(sweep, key=lambda row: row["silhouette"])
    best_model = models[best["k"]]
    labels = best_model.predict(reduced)

    pd.DataFrame(sweep).to_csv(REPORT_DIR / "clustering_sweep.csv", index=False)

    metrics = {
        "selected_k": best["k"],
        "silhouette": best["silhouette"],
        "davies_bouldin": best["davies_bouldin"],
        "calinski_harabasz": best["calinski_harabasz"],
        "silhouette_target": SILHOUETTE_TARGET,
        "silhouette_target_met": bool(best["silhouette"] >= SILHOUETTE_TARGET),
        "cluster_sizes": {
            int(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        },
        "pca_components": int(reduced.shape[1]),
        "features": SEGMENTATION_FEATURES,
        "sweep": sweep,
    }
    logger.info(
        "Selected k=%d | silhouette=%.4f (target %.2f: %s) | sizes=%s",
        metrics["selected_k"],
        metrics["silhouette"],
        SILHOUETTE_TARGET,
        "MET" if metrics["silhouette_target_met"] else "NOT MET",
        metrics["cluster_sizes"],
    )

    profile = summarise_clusters(df, labels)
    profile.to_csv(REPORT_DIR / "cluster_profiles.csv")
    logger.info("Cluster behavioural profile:\n%s", profile.round(2).to_string())
    return best_model, scaler, pca, metrics


def summarise_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Describe each cluster with its mean behavioural profile and churn rate."""
    profile = df[SEGMENTATION_FEATURES].copy()
    profile["churn_rate"] = df[CHURN_TARGET]
    profile["cluster"] = labels
    return profile.groupby("cluster").mean()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_training() -> dict:
    """Train all three models, serialise them, and write a metrics report."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_processed_data()

    classifier, churn_metrics = train_churn_classifier(df)
    regressor, revenue_metrics = train_revenue_regressor(df)
    clusterer, scaler, pca, segment_metrics = train_segmentation_model(df)

    joblib.dump(classifier, MODEL_DIR / "classifier.pkl")
    joblib.dump(regressor, MODEL_DIR / "regressor.pkl")
    joblib.dump(clusterer, MODEL_DIR / "clusterer.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(pca, MODEL_DIR / "segmentation_pca.pkl")

    # Feature contracts keep downstream scoring aligned with training.
    joblib.dump(
        {
            "churn_features": [
                c
                for c in df.columns
                if c not in LEAKAGE_COLUMNS + [CHURN_TARGET, REVENUE_TARGET]
            ],
            "revenue_features": [
                c
                for c in df.columns
                if c not in LEAKAGE_COLUMNS + [CHURN_TARGET, REVENUE_TARGET]
            ],
            "segmentation_features": SEGMENTATION_FEATURES,
            "log_scale_features": LOG_SCALE_FEATURES,
        },
        MODEL_DIR / "feature_contract.pkl",
    )
    logger.info("Serialised models to %s", MODEL_DIR)

    metrics = {
        "churn_classification": churn_metrics,
        "revenue_regression": revenue_metrics,
        "segmentation": segment_metrics,
    }
    metrics_path = REPORT_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Metrics written to %s", metrics_path)
    return metrics


if __name__ == "__main__":
    run_training()
