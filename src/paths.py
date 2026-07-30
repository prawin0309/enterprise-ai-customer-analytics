"""Project path resolution.

Every path used by the pipeline is derived from the repository root rather than
hard-coded, so the project runs unchanged from any checkout location.

The raw CRM extract ships with the repository under ``data/raw/`` and is
treated as read-only, so the pipeline runs from a clean clone with no further
setup. To point at an extract held elsewhere, set ``CRM_DATA_DIR``:

    setx CRM_DATA_DIR "C:\\path\\to\\DataSet"

If the variable is unset, the first directory containing ``fact_customers.csv``
from a small list of repository-relative locations is used.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
NOTEBOOK_DIR = PROJECT_DIR / "notebooks"

PROCESSED_FILE = PROCESSED_DIR / "processed_customer_data.csv"
TRAIN_BALANCED_FILE = PROCESSED_DIR / "train_balanced_smote.csv"
TEST_FILE = PROCESSED_DIR / "test_holdout.csv"
PROFILES_FILE = PROCESSED_DIR / "customer_profiles.json"
PRIORITY_FILE = PROCESSED_DIR / "priority_accounts.json"

METRICS_FILE = REPORT_DIR / "model_metrics.json"
SUMMARY_FILE = REPORT_DIR / "fusion_summary.csv"
SHAP_IMPORTANCE_FILE = REPORT_DIR / "shap_feature_importance.csv"
CLUSTER_SWEEP_FILE = REPORT_DIR / "clustering_sweep.csv"
CLUSTER_PROFILE_FILE = REPORT_DIR / "cluster_profiles.csv"
INSIGHTS_FILE = REPORT_DIR / "executive_insights.md"
PDF_REPORT_FILE = REPORT_DIR / "Capstone_Final_Report.pdf"

SOURCE_ENV_VAR = "CRM_DATA_DIR"
SENTINEL_TABLE = "fact_customers.csv"

_FALLBACK_SOURCE_DIRS = (
    DATA_DIR / "raw",
    PROJECT_DIR.parent / "DS_F" / "DataSet-20260223T124234Z-1-001" / "DataSet",
)


def resolve_source_dir() -> Path:
    """Locate the read-only CRM extract, honouring ``CRM_DATA_DIR`` first."""
    override = os.environ.get(SOURCE_ENV_VAR)
    if override:
        candidate = Path(override)
        if not (candidate / SENTINEL_TABLE).exists():
            raise FileNotFoundError(
                f"{SOURCE_ENV_VAR} points at {candidate}, which does not contain "
                f"{SENTINEL_TABLE}."
            )
        return candidate

    for candidate in _FALLBACK_SOURCE_DIRS:
        if (candidate / SENTINEL_TABLE).exists():
            return candidate

    searched = "\n  ".join(str(p) for p in _FALLBACK_SOURCE_DIRS)
    raise FileNotFoundError(
        f"CRM extract not found. Set {SOURCE_ENV_VAR} to the directory holding "
        f"{SENTINEL_TABLE}. Searched:\n  {searched}"
    )


def ensure_output_dirs() -> None:
    """Create the writable output directories if they do not already exist."""
    for directory in (PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
