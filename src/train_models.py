"""Model training for the SaaS customer intelligence project.

Trains and serialises the three core models on the processed customer table:

    1. Churn prediction      - XGBoost classifier (ROC-AUC, F1, SHAP drivers)
    2. Revenue forecasting   - XGBoost / GradientBoosting regressor (RMSE, MAE, R2)
    3. Behavioural segments  - KMeans clustering (silhouette validated)

Model artefacts are written to ``models/`` and metric/importance reports to
``reports/``; both are resolved relative to the repository root by ``paths``.
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
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    calinski_harabasz_score,
    classification_report,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

try:
    from paths import MODEL_DIR, PROCESSED_FILE, REPORT_DIR
except ImportError:  # imported as ``src.train_models``
    from src.paths import MODEL_DIR, PROCESSED_FILE, REPORT_DIR

CUSTOMER_KEY = "Customer_ID"
CHURN_TARGET = "Churn"
REVENUE_TARGET = "Next_Quarter_Revenue_USD"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
SHAP_SAMPLE_SIZE = 500
SILHOUETTE_TARGET = 0.60
CLUSTER_RANGE = range(2, 11)

# Randomised search budget. Twenty candidates over three folds covers the
# depth / learning-rate trade-off that matters most for boosted trees while
# keeping a full retrain to a couple of minutes.
SEARCH_ITERATIONS = 20
SEARCH_CV_FOLDS = 3

# Prefixed with ``model__`` because the search runs over the SMOTE pipeline.
CLASSIFIER_SEARCH_SPACE = {
    "model__n_estimators": [200, 300, 400, 600, 800],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "model__min_child_weight": [1, 3, 5, 10],
}

REGRESSOR_SEARCH_SPACE = {
    "n_estimators": [300, 400, 600, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

# Neighbourhood radii probed when cross-checking KMeans against DBSCAN.
DBSCAN_EPS_RANGE = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

# ``Cluster_Label`` is a pre-existing segment tag shipped with the source data
# (values such as "At-Risk SMB") and encodes churn risk directly, so it is
# excluded from supervised training. One-hot encoding expands it into
# ``Cluster_Label_*`` dummies, so the exclusion must match on prefix - matching
# the bare column name silently excludes nothing.
LEAKAGE_COLUMNS = [CUSTOMER_KEY]
LEAKAGE_PREFIXES = ("Cluster_Label",)

# Record-count features that fall simply because a churned customer stops
# generating rows - the value encodes the outcome rather than predicting it.
# Each correlates NEGATIVELY with churn. Engagement counts are deliberately not
# in this list: they correlate POSITIVELY (churners raise more tickets), which
# is genuine distress signal, not a temporal artefact.
# Measured cost of excluding them: ROC-AUC 0.9455 -> 0.9023 (see ablation in
# reports/model_metrics.json under "leakage_ablation").
TEMPORAL_ARTIFACT_FEATURES = [
    "txn_count",
    "txn_renewal_count",
    "active_quarters",
    "usage_months_observed",
]

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


def resolve_excluded_columns(
    df: pd.DataFrame, extra: list[str], drop_temporal_artifacts: bool = False
) -> list[str]:
    """Return every column to withhold from training, expanding leakage prefixes."""
    excluded = {c for c in LEAKAGE_COLUMNS + extra if c in df.columns}
    excluded |= {c for c in df.columns if c.startswith(LEAKAGE_PREFIXES)}
    if drop_temporal_artifacts:
        excluded |= {c for c in TEMPORAL_ARTIFACT_FEATURES if c in df.columns}
    return sorted(excluded)


def build_churn_pipeline(params: dict | None = None) -> ImbPipeline:
    """SMOTE + XGBoost as one estimator, so resampling happens inside each fold.

    Cross-validating a model that was fitted on pre-balanced data is optimistic:
    synthetic minority rows generated from a validation record can appear in the
    training folds. Wrapping SMOTE in the pipeline removes that leak.
    """
    return ImbPipeline(
        [
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", build_classifier(params)),
        ]
    )


DEFAULT_CLASSIFIER_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


def build_classifier(params: dict | None = None) -> XGBClassifier:
    """Build the XGBoost churn classifier, overriding defaults with ``params``."""
    settings = dict(DEFAULT_CLASSIFIER_PARAMS)
    settings.update(params or {})
    return XGBClassifier(
        **settings,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def tune_classifier(
    x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[dict, dict]:
    """Randomised search over the SMOTE + XGBoost pipeline.

    Searching the pipeline rather than the bare estimator keeps resampling
    inside each validation fold, so the reported search score is not inflated
    by synthetic rows leaking across the split.
    """
    search = RandomizedSearchCV(
        build_churn_pipeline(),
        CLASSIFIER_SEARCH_SPACE,
        n_iter=SEARCH_ITERATIONS,
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,
    )
    search.fit(x_train, y_train)
    best = {
        key.removeprefix("model__"): value
        for key, value in search.best_params_.items()
    }
    summary = {
        "method": "RandomizedSearchCV",
        "candidates_sampled": SEARCH_ITERATIONS,
        "cv_folds": SEARCH_CV_FOLDS,
        "scoring": "roc_auc",
        "search_space": sorted(CLASSIFIER_SEARCH_SPACE),
        "best_params": best,
        "best_search_score": float(search.best_score_),
        "default_params": DEFAULT_CLASSIFIER_PARAMS,
    }
    logger.info("Classifier search: best CV ROC-AUC=%.4f with %s",
                search.best_score_, best)
    return best, summary


def tune_regressor(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict, dict]:
    """Randomised search for the XGBoost revenue regressor, scored on R²."""
    search = RandomizedSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        REGRESSOR_SEARCH_SPACE,
        n_iter=SEARCH_ITERATIONS,
        scoring="r2",
        cv=SEARCH_CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,
    )
    search.fit(x_train, y_train)
    summary = {
        "method": "RandomizedSearchCV",
        "candidates_sampled": SEARCH_ITERATIONS,
        "cv_folds": SEARCH_CV_FOLDS,
        "scoring": "r2",
        "search_space": sorted(REGRESSOR_SEARCH_SPACE),
        "best_params": dict(search.best_params_),
        "best_search_score": float(search.best_score_),
    }
    logger.info("Regressor search: best CV R2=%.4f with %s",
                search.best_score_, search.best_params_)
    return dict(search.best_params_), summary


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
    drop_columns = resolve_excluded_columns(
        df, [CHURN_TARGET, REVENUE_TARGET], drop_temporal_artifacts=True
    )
    logger.info("Withholding %d columns: %s", len(drop_columns), drop_columns)
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

    best_params, search_summary = tune_classifier(x_train, y_train)
    classifier = build_classifier(best_params)
    classifier.fit(x_train_balanced, y_train_balanced)

    probabilities = classifier.predict_proba(x_test)[:, 1]
    predictions = classifier.predict(x_test)

    # Cross-validate the SMOTE+model pipeline on the UNBALANCED training split
    # so resampling is refitted inside every fold.
    honest_cv = cross_val_score(
        build_churn_pipeline(best_params),
        x_train,
        y_train,
        cv=StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        ),
        scoring="roc_auc",
        n_jobs=-1,
    )

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "f1_score": float(f1_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "cv_roc_auc_mean": float(honest_cv.mean()),
        "cv_roc_auc_std": float(honest_cv.std()),
        "cv_method": "SMOTE inside imblearn Pipeline, refitted per fold",
        "n_features": int(features.shape[1]),
        "excluded_temporal_artifacts": TEMPORAL_ARTIFACT_FEATURES,
        "hyperparameter_search": search_summary,
    }

    # Baseline comparison required by the project guidelines.
    baseline = GradientBoostingClassifier(random_state=RANDOM_STATE)
    baseline.fit(x_train_balanced, y_train_balanced)
    metrics["baseline_gb_roc_auc"] = float(
        roc_auc_score(y_test, baseline.predict_proba(x_test)[:, 1])
    )

    logger.info(
        "XGBoost  ROC-AUC=%.4f  F1=%.4f  CV-ROC-AUC=%.4f (+/- %.4f) | "
        "GB baseline ROC-AUC=%.4f",
        metrics["roc_auc"],
        metrics["f1_score"],
        metrics["cv_roc_auc_mean"],
        metrics["cv_roc_auc_std"],
        metrics["baseline_gb_roc_auc"],
    )
    logger.info(
        "Classification report:\n%s",
        classification_report(y_test, predictions, digits=3),
    )

    metrics["leakage_ablation"] = run_leakage_ablation(df, best_params)

    importance = compute_shap_importance(classifier, x_test)
    metrics["top_shap_drivers"] = importance.head(15).to_dict("records")
    return classifier, metrics


def run_leakage_ablation(df: pd.DataFrame, params: dict | None = None) -> dict:
    """Quantify what the temporal-artefact features were contributing.

    Retrains the classifier *with* the excluded features so the report can state
    the measured cost of the exclusion rather than asserting it.
    """
    logger.info("Running leakage ablation (retraining with artefacts included)")
    drop_columns = resolve_excluded_columns(
        df, [CHURN_TARGET, REVENUE_TARGET], drop_temporal_artifacts=False
    )
    features, target = build_supervised_matrix(df, CHURN_TARGET, drop_columns)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, stratify=target,
        random_state=RANDOM_STATE,
    )
    x_balanced, y_balanced = SMOTE(random_state=RANDOM_STATE).fit_resample(
        x_train, y_train
    )
    model = build_classifier(params)
    model.fit(x_balanced, y_balanced)
    predictions = model.predict(x_test)

    honest_cv = cross_val_score(
        build_churn_pipeline(params), x_train, y_train,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=-1,
    )
    result = {
        "description": (
            "Contaminated variant retaining record-count features that fall "
            "because a churned customer stops generating rows."
        ),
        "n_features": int(features.shape[1]),
        "roc_auc": float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])),
        "f1_score": float(f1_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "cv_roc_auc_mean": float(honest_cv.mean()),
    }
    logger.info(
        "Ablation (with artefacts): ROC-AUC=%.4f F1=%.4f recall=%.4f",
        result["roc_auc"], result["f1_score"], result["recall"],
    )
    return result


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
    drop_columns = resolve_excluded_columns(df, [REVENUE_TARGET, CHURN_TARGET])
    features, target = build_supervised_matrix(df, REVENUE_TARGET, drop_columns)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    tuned_params, search_summary = tune_regressor(x_train, y_train)
    candidates = {
        "xgboost_tuned": XGBRegressor(
            **tuned_params, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost_default": XGBRegressor(
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
        "hyperparameter_search": search_summary,
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


def validate_with_alternative_algorithms(reduced: np.ndarray, k: int) -> dict:
    """Cross-check the KMeans partition against hierarchical and density methods.

    Neither is used for the production segmentation; they exist to show whether
    the weak silhouette is a KMeans artefact or a property of the data.
    """
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
        reduced
    )
    results: dict = {
        "agglomerative_ward": {
            "k": int(k),
            "silhouette": float(silhouette_score(reduced, labels)),
            "davies_bouldin": float(davies_bouldin_score(reduced, labels)),
        },
        "dbscan_sweep": [],
    }

    best_dbscan = None
    for eps in DBSCAN_EPS_RANGE:
        labels = DBSCAN(eps=eps, min_samples=10).fit_predict(reduced)
        clusters = sorted(set(labels) - {-1})
        noise_rate = float((labels == -1).mean())
        record = {
            "eps": eps,
            "clusters_found": len(clusters),
            "noise_rate": round(noise_rate, 4),
        }
        # Silhouette is undefined below two clusters and meaningless once most
        # of the data has been labelled noise.
        if len(clusters) >= 2 and noise_rate < 0.5:
            mask = labels != -1
            record["silhouette"] = float(
                silhouette_score(reduced[mask], labels[mask])
            )
            if best_dbscan is None or (
                record["silhouette"] > best_dbscan["silhouette"]
            ):
                best_dbscan = record
        results["dbscan_sweep"].append(record)

    results["dbscan_best"] = best_dbscan
    logger.info(
        "Cross-check | agglomerative silhouette=%.4f | best DBSCAN=%s",
        results["agglomerative_ward"]["silhouette"],
        best_dbscan,
    )
    return results


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
        "alternative_algorithms": validate_with_alternative_algorithms(
            reduced, best["k"]
        ),
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
    # The churn model additionally withholds the temporal-artefact features, so
    # the two contracts differ and must be resolved separately.
    churn_excluded = resolve_excluded_columns(
        df, [CHURN_TARGET, REVENUE_TARGET], drop_temporal_artifacts=True
    )
    revenue_excluded = resolve_excluded_columns(
        df, [CHURN_TARGET, REVENUE_TARGET], drop_temporal_artifacts=False
    )
    joblib.dump(
        {
            "churn_features": [c for c in df.columns if c not in churn_excluded],
            "revenue_features": [c for c in df.columns if c not in revenue_excluded],
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
