"""Render the report figures from the serialised models.

The notebook draws the same diagnostics inline; this module writes them to
``reports/figures`` as PNGs so ``generate_report`` can embed them in the PDF.
Splits are reproduced with the constants used in training, so the figures
describe the same hold-out rows the metrics were measured on.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

try:
    from paths import CLUSTER_SWEEP_FILE, FIGURE_DIR, MODEL_DIR, PROCESSED_FILE
except ImportError:  # imported as ``src.make_figures``
    from src.paths import CLUSTER_SWEEP_FILE, FIGURE_DIR, MODEL_DIR, PROCESSED_FILE

CHURN_TARGET = "Churn"
REVENUE_TARGET = "Next_Quarter_Revenue_USD"

# Must match train_models, otherwise the figures describe a different hold-out.
TEST_SIZE = 0.2
RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 500
SCATTER_SAMPLE_SIZE = 2000

DPI = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("make_figures")

plt.rcParams["figure.dpi"] = DPI
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure to the report figure directory and close it."""
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s (%.0f KB)", path.name, path.stat().st_size / 1024)


def churn_holdout(df: pd.DataFrame, contract: dict):
    """Rebuild the classifier's stratified hold-out split."""
    features = df[contract["churn_features"]]
    return train_test_split(
        features,
        df[CHURN_TARGET],
        test_size=TEST_SIZE,
        stratify=df[CHURN_TARGET],
        random_state=RANDOM_STATE,
    )


def revenue_holdout(df: pd.DataFrame, contract: dict):
    """Rebuild the regressor's hold-out split."""
    features = df[contract["revenue_features"]]
    return train_test_split(
        features, df[REVENUE_TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def plot_roc_and_confusion(classifier, x_test, y_test) -> None:
    """ROC curve and confusion matrix for the churn classifier."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_estimator(classifier, x_test, y_test, ax=axes[0], name="XGBoost")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    axes[0].set_title("Churn model — ROC curve (hold-out)")
    axes[0].legend(loc="lower right")

    ConfusionMatrixDisplay.from_estimator(
        classifier,
        x_test,
        y_test,
        ax=axes[1],
        colorbar=False,
        cmap="Blues",
        display_labels=["Retained", "Churned"],
    )
    axes[1].set_title("Confusion matrix (threshold 0.5)")
    axes[1].grid(False)

    save(fig, "churn_roc_confusion.png")


def plot_shap_summary(classifier, x_test) -> None:
    """SHAP beeswarm over a sample of the hold-out set."""
    sample = x_test.sample(
        n=min(SHAP_SAMPLE_SIZE, len(x_test)), random_state=RANDOM_STATE
    )
    shap_values = shap.TreeExplainer(classifier).shap_values(sample)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, max_display=15, show=False)
    figure = plt.gcf()
    figure.suptitle("Churn drivers — SHAP value distribution", y=1.01)
    save(figure, "churn_shap_beeswarm.png")


def plot_residuals(regressor, x_test, y_test) -> None:
    """Predicted-vs-actual, residual scatter and residual distribution."""
    predictions = regressor.predict(x_test)
    residuals = y_test.to_numpy() - predictions

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].scatter(y_test, predictions, s=8, alpha=0.35)
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    axes[0].plot(limits, limits, "k--", linewidth=0.8)
    axes[0].set_xlabel("Actual next-quarter revenue (USD)")
    axes[0].set_ylabel("Predicted (USD)")
    axes[0].set_title("Predicted vs actual")

    axes[1].scatter(predictions, residuals, s=8, alpha=0.35)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Predicted (USD)")
    axes[1].set_ylabel("Residual (USD)")
    axes[1].set_title("Residuals vs fitted")

    axes[2].hist(residuals, bins=50, edgecolor="white", linewidth=0.4)
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Residual (USD)")
    axes[2].set_title(
        f"Residual distribution (mean ${residuals.mean():,.0f})"
    )

    save(fig, "revenue_residuals.png")


def plot_cluster_scatter(df: pd.DataFrame, artifacts: dict) -> None:
    """Behavioural segments projected onto two components for visualisation."""
    contract = artifacts["contract"]
    segment_data = df[contract["segmentation_features"]].copy()
    for column in contract["log_scale_features"]:
        segment_data[column] = np.log1p(segment_data[column].clip(lower=0))

    reduced = artifacts["pca"].transform(artifacts["scaler"].transform(segment_data))
    labels = artifacts["clusterer"].predict(reduced)

    # The model keeps 6 components; a further 2-component projection is used for
    # display only and never feeds the clusterer.
    display = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(reduced)
    sample = np.random.default_rng(RANDOM_STATE).choice(
        len(display), size=min(SCATTER_SAMPLE_SIZE, len(display)), replace=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for cluster in np.unique(labels):
        mask = labels[sample] == cluster
        axes[0].scatter(
            display[sample][mask, 0],
            display[sample][mask, 1],
            s=10,
            alpha=0.5,
            label=f"Cluster {cluster}",
        )
    axes[0].set_xlabel("Principal component 1")
    axes[0].set_ylabel("Principal component 2")
    axes[0].set_title(f"Behavioural segments ({SCATTER_SAMPLE_SIZE:,} sampled accounts)")
    axes[0].legend()

    churn_by_cluster = pd.Series(df[CHURN_TARGET].to_numpy()).groupby(labels).mean()
    sizes = pd.Series(labels).value_counts().sort_index()
    bars = axes[1].bar(
        churn_by_cluster.index.astype(str), churn_by_cluster.to_numpy() * 100
    )
    for bar, size in zip(bars, sizes):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"n={size:,}",
            ha="center",
            fontsize=9,
        )
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Churn rate (%)")
    axes[1].set_title("Churn rate by segment")

    save(fig, "segment_pca_scatter.png")


def plot_cluster_sweep() -> None:
    """Elbow and silhouette curves from the k = 2..10 sweep."""
    sweep = pd.read_csv(CLUSTER_SWEEP_FILE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(sweep["k"], sweep["inertia"], marker="o")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow method")

    axes[1].plot(sweep["k"], sweep["silhouette"], marker="o", label="Silhouette")
    axes[1].axhline(
        0.60, color="crimson", linestyle="--", linewidth=0.9, label="Project target 0.60"
    )
    best = sweep.loc[sweep["silhouette"].idxmax()]
    axes[1].annotate(
        f"best {best['silhouette']:.3f} at k={int(best['k'])}",
        xy=(best["k"], best["silhouette"]),
        xytext=(best["k"] + 0.6, best["silhouette"] + 0.05),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=9,
    )
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette across k")
    axes[1].legend()

    save(fig, "clustering_sweep.png")


def generate_figures() -> list:
    """Render every figure embedded in the capstone report."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_FILE)
    artifacts = {
        "classifier": joblib.load(MODEL_DIR / "classifier.pkl"),
        "regressor": joblib.load(MODEL_DIR / "regressor.pkl"),
        "clusterer": joblib.load(MODEL_DIR / "clusterer.pkl"),
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "pca": joblib.load(MODEL_DIR / "segmentation_pca.pkl"),
        "contract": joblib.load(MODEL_DIR / "feature_contract.pkl"),
    }
    logger.info("Loaded %s %s and 6 model artifacts", PROCESSED_FILE.name, df.shape)

    _, x_test, _, y_test = churn_holdout(df, artifacts["contract"])
    plot_roc_and_confusion(artifacts["classifier"], x_test, y_test)
    plot_shap_summary(artifacts["classifier"], x_test)

    _, xr_test, _, yr_test = revenue_holdout(df, artifacts["contract"])
    plot_residuals(artifacts["regressor"], xr_test, yr_test)

    plot_cluster_scatter(df, artifacts)
    plot_cluster_sweep()

    written = sorted(FIGURE_DIR.glob("*.png"))
    logger.info("Wrote %d figures to %s", len(written), FIGURE_DIR)
    return written


if __name__ == "__main__":
    generate_figures()
