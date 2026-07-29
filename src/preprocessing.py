"""Data preprocessing pipeline for the SaaS customer intelligence project.

Builds a single customer-level modelling table from the star-schema CRM
extract in the read-only source directory:

    fact_customers          (grain: customer)
    fact_transactions       (grain: transaction)   -> aggregated per customer
    fact_usage_monthly      (grain: customer-month) -> aggregated per customer
    fact_engagement_events  (grain: event)          -> aggregated per customer
    dim_geography / dim_industry / dim_product      -> descriptive attributes

The pipeline cleans, engineers features, encodes categoricals, and writes the
processed dataset. A SMOTE-balanced training split is written alongside it so
that oversampling never leaks into the hold-out set.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SOURCE_DIR = Path(r"D:\DS_F\DataSet-20260223T124234Z-1-001\DataSet")
OUTPUT_DIR = Path(r"D:\DS_FO\data\processed")

PROCESSED_FILE = OUTPUT_DIR / "processed_customer_data.csv"
TRAIN_BALANCED_FILE = OUTPUT_DIR / "train_balanced_smote.csv"
TEST_FILE = OUTPUT_DIR / "test_holdout.csv"

CUSTOMER_KEY = "Customer_ID"
CHURN_TARGET = "Churn"
REVENUE_TARGET = "Next_Quarter_Revenue_USD"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Categorical columns with more distinct values than this are label encoded
# instead of one-hot encoded, to keep the feature matrix compact.
ONE_HOT_MAX_CARDINALITY = 12

# Free-text / identifier columns that carry no predictive signal.
IDENTIFIER_COLUMNS = [
    "CSM_Name",
    "Contract_Start_Date",
    "Contract_End_Date",
    "City",
    "ISO2",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocessing")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_tables(source_dir: Path = SOURCE_DIR) -> dict[str, pd.DataFrame]:
    """Load every fact and dimension CSV from the read-only source directory."""
    table_names = [
        "fact_customers",
        "fact_transactions",
        "fact_usage_monthly",
        "fact_engagement_events",
        "dim_geography",
        "dim_industry",
        "dim_product",
    ]
    tables: dict[str, pd.DataFrame] = {}
    for name in table_names:
        path = source_dir / f"{name}.csv"
        tables[name] = pd.read_csv(path)
        logger.info("Loaded %-24s %s", name, tables[name].shape)
    return tables


# --------------------------------------------------------------------------- #
# Fact aggregation to customer grain
# --------------------------------------------------------------------------- #


def _quarter_sort_key(fiscal_quarter: pd.Series) -> pd.Series:
    """Convert a ``Q3-2023`` style label into a sortable integer (``202303``)."""
    quarter = fiscal_quarter.str.slice(1, 2).astype(int)
    year = fiscal_quarter.str.slice(3).astype(int)
    return year * 100 + quarter


def aggregate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to customer level, including quarterly revenue."""
    txn = transactions.drop_duplicates()
    txn = txn[txn[CUSTOMER_KEY].notna()].copy()

    overall = txn.groupby(CUSTOMER_KEY).agg(
        txn_count=("Transaction_ID", "count"),
        txn_total_revenue_usd=("Net_Revenue_USD", "sum"),
        txn_mean_revenue_usd=("Net_Revenue_USD", "mean"),
        txn_total_billed_usd=("Total_Billed_USD", "sum"),
        txn_addon_revenue_usd=("Add_On_Revenue_USD", "sum"),
        txn_mean_discount_pct=("Discount_Pct", "mean"),
        txn_mean_payment_delay_days=("Payment_Delay_Days", "mean"),
        txn_max_payment_delay_days=("Payment_Delay_Days", "max"),
        txn_renewal_count=(
            "Transaction_Type",
            lambda s: int((s == "Renewal").sum()),
        ),
        txn_auto_renewal_rate=(
            "Auto_Renewal",
            lambda s: float((s == "Yes").mean()),
        ),
        txn_failed_payment_rate=(
            "Payment_Status",
            lambda s: float((s != "Paid").mean()),
        ),
    )

    # Quarterly revenue profile: level, volatility and trend.
    txn["quarter_key"] = _quarter_sort_key(txn["Fiscal_Quarter"])
    quarterly = (
        txn.groupby([CUSTOMER_KEY, "quarter_key"])["Net_Revenue_USD"]
        .sum()
        .reset_index()
        .sort_values([CUSTOMER_KEY, "quarter_key"])
    )
    quarter_stats = quarterly.groupby(CUSTOMER_KEY)["Net_Revenue_USD"].agg(
        quarterly_revenue_mean="mean",
        quarterly_revenue_std="std",
        quarterly_revenue_max="max",
        quarterly_revenue_min="min",
        active_quarters="count",
    )
    last_quarter = (
        quarterly.groupby(CUSTOMER_KEY)["Net_Revenue_USD"]
        .last()
        .rename("quarterly_revenue_last")
    )
    quarter_stats = quarter_stats.join(last_quarter)
    quarter_stats["quarterly_revenue_growth_ratio"] = (
        quarter_stats["quarterly_revenue_last"]
        / quarter_stats["quarterly_revenue_mean"].replace(0, np.nan)
    )

    aggregated = overall.join(quarter_stats).reset_index()
    logger.info("Aggregated transactions -> %s", aggregated.shape)
    return aggregated


def aggregate_usage(usage: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly usage snapshots to customer level with a usage trend."""
    snapshots = usage.drop_duplicates().sort_values([CUSTOMER_KEY, "Snapshot_Month"])

    aggregated = snapshots.groupby(CUSTOMER_KEY).agg(
        usage_months_observed=("Snapshot_Month", "nunique"),
        usage_mean_dau=("Daily_Active_Users", "mean"),
        usage_mean_mau=("Monthly_Active_Users", "mean"),
        usage_mean_feature_adoption_pct=("Feature_Adoption_Pct", "mean"),
        usage_mean_api_calls=("API_Calls_Monthly", "mean"),
        usage_mean_license_utilization_pct=("License_Utilization_Pct", "mean"),
        usage_mean_sessions_total=("Sessions_Total", "mean"),
        usage_mean_session_minutes=("Avg_Session_Min", "mean"),
        usage_mean_mobile_sessions_pct=("Mobile_Sessions_Pct", "mean"),
        usage_mean_integrations_active=("Integrations_Active", "mean"),
        usage_mean_automations=("Automations_Triggered", "mean"),
        usage_mean_reports=("Reports_Generated", "mean"),
        usage_mean_storage_gb=("Storage_Used_GB", "mean"),
        usage_mean_errors=("Errors_Logged", "mean"),
        usage_mean_uptime_pct=("Uptime_Pct", "mean"),
    )

    # Usage trend: mean of the three most recent months versus earlier history.
    recent = (
        snapshots.groupby(CUSTOMER_KEY)
        .tail(3)
        .groupby(CUSTOMER_KEY)["Monthly_Active_Users"]
        .mean()
        .rename("usage_recent_mau")
    )
    aggregated = aggregated.join(recent)
    aggregated["usage_trend_ratio"] = (
        aggregated["usage_recent_mau"]
        / aggregated["usage_mean_mau"].replace(0, np.nan)
    )

    aggregated = aggregated.reset_index()
    logger.info("Aggregated usage -> %s", aggregated.shape)
    return aggregated


def aggregate_engagement(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate engagement events to customer level."""
    evt = events.drop_duplicates()

    aggregated = evt.groupby(CUSTOMER_KEY).agg(
        eng_event_count=("Engagement_Event_ID", "count"),
        eng_mean_score=("Score", "mean"),
        eng_mean_resolution_days=("Resolution_Days", "mean"),
        eng_negative_sentiment_rate=(
            "Sentiment",
            lambda s: float((s == "Negative").mean()),
        ),
        eng_positive_sentiment_rate=(
            "Sentiment",
            lambda s: float((s == "Positive").mean()),
        ),
        eng_sla_breach_rate=("SLA_Breached", lambda s: float((s == "Yes").mean())),
        eng_escalation_rate=(
            "Escalated_To_Manager",
            lambda s: float((s == "Yes").mean()),
        ),
        eng_follow_up_rate=(
            "Follow_Up_Required",
            lambda s: float((s == "Yes").mean()),
        ),
        eng_unresolved_rate=(
            "Resolution_Status",
            lambda s: float((s == "Open").mean()),
        ),
        eng_distinct_event_types=("Event_Type", "nunique"),
    )

    aggregated = aggregated.reset_index()
    logger.info("Aggregated engagement -> %s", aggregated.shape)
    return aggregated


# --------------------------------------------------------------------------- #
# Joining
# --------------------------------------------------------------------------- #


def build_customer_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join aggregated facts and dimension attributes onto the customer table."""
    customers = tables["fact_customers"].copy()

    fact_aggregates = [
        aggregate_transactions(tables["fact_transactions"]),
        aggregate_usage(tables["fact_usage_monthly"]),
        aggregate_engagement(tables["fact_engagement_events"]),
    ]
    for aggregate in fact_aggregates:
        customers = customers.merge(aggregate, on=CUSTOMER_KEY, how="left")

    dimensions = [
        (tables["dim_geography"], "Geo_ID"),
        (tables["dim_industry"], "Industry_ID"),
        (tables["dim_product"], "Product_ID"),
    ]
    for dimension, key in dimensions:
        # dim_geography ships duplicate Geo_ID rows; deduplicate so the join
        # cannot fan out the customer grain.
        deduplicated = dimension.drop_duplicates(subset=[key], keep="first")
        if len(deduplicated) < len(dimension):
            logger.warning(
                "Dropped %d duplicate %s rows before joining",
                len(dimension) - len(deduplicated),
                key,
            )
        customers = customers.merge(
            deduplicated, on=key, how="left", suffixes=("", "_dim")
        )

    # Drop duplicated dimension columns that already exist on the fact table.
    duplicate_columns = [c for c in customers.columns if c.endswith("_dim")]
    customers = customers.drop(columns=duplicate_columns)

    logger.info("Joined customer table -> %s", customers.shape)
    return customers


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate customers and impute missing values."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=[CUSTOMER_KEY], keep="first")
    logger.info(
        "Removed %d duplicate rows (%d -> %d unique customers)",
        before - len(df),
        before,
        len(df),
    )

    df = df.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in df.columns])

    numeric_columns = df.select_dtypes(include="number").columns.difference(
        [CUSTOMER_KEY, CHURN_TARGET]
    )
    categorical_columns = df.columns.difference(numeric_columns).difference(
        [CUSTOMER_KEY, CHURN_TARGET]
    )

    for column in numeric_columns:
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        if df[column].isna().any():
            mode = df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[column] = df[column].fillna(fill_value)

    remaining = int(df.isna().sum().sum())
    logger.info("Missing values after imputation: %d", remaining)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def engineer_features(df: pd.DataFrame, raw_customers: pd.DataFrame) -> pd.DataFrame:
    """Derive tenure, usage intensity, engagement and revenue ratio features."""
    df = df.copy()
    dates = raw_customers.drop_duplicates(subset=[CUSTOMER_KEY], keep="first")
    dates = dates.set_index(CUSTOMER_KEY)

    start = pd.to_datetime(
        df[CUSTOMER_KEY].map(dates["Contract_Start_Date"]), errors="coerce"
    )
    end = pd.to_datetime(
        df[CUSTOMER_KEY].map(dates["Contract_End_Date"]), errors="coerce"
    )
    reference_date = end.max()

    # --- Tenure -----------------------------------------------------------
    df["contract_duration_days"] = (end - start).dt.days
    df["tenure_days"] = (reference_date - start).dt.days
    df["tenure_years"] = df["Tenure_Months"] / 12.0
    df["days_to_contract_end"] = (end - reference_date).dt.days

    # --- Usage intensity --------------------------------------------------
    tenure_months = df["Tenure_Months"].replace(0, np.nan)
    licenses = df["Licenses_Purchased"].replace(0, np.nan)
    monthly_active = df["usage_mean_mau"].replace(0, np.nan)

    df["usage_intensity_score"] = (
        df["usage_mean_sessions_total"] / monthly_active
    ) * df["usage_mean_session_minutes"]
    df["dau_mau_ratio"] = df["usage_mean_dau"] / monthly_active
    df["api_calls_per_license"] = df["usage_mean_api_calls"] / licenses
    df["active_user_per_license"] = df["Daily_Active_Users"] / licenses
    df["automation_density"] = df["usage_mean_automations"] / monthly_active
    df["error_rate_per_session"] = df["usage_mean_errors"] / df[
        "usage_mean_sessions_total"
    ].replace(0, np.nan)

    # --- Engagement ratios ------------------------------------------------
    df["events_per_tenure_month"] = df["eng_event_count"] / tenure_months
    df["tickets_per_tenure_month"] = df["Support_Tickets"] / tenure_months
    df["escalated_ticket_ratio"] = df["Escalated_Tickets"] / df[
        "Support_Tickets"
    ].replace(0, np.nan)
    df["support_dependency_score"] = df["Support_Tickets"] / monthly_active
    df["engagement_quality_score"] = (
        df["eng_positive_sentiment_rate"] - df["eng_negative_sentiment_rate"]
    )

    # --- Revenue ----------------------------------------------------------
    df["avg_monthly_revenue_usd"] = df["Lifetime_Revenue_USD"] / tenure_months
    df["revenue_per_license_usd"] = df["ACV_USD"] / licenses
    df["addon_revenue_share"] = df["Add_On_Revenue_USD"] / df["ACV_USD"].replace(
        0, np.nan
    )
    df["payment_reliability_score"] = 1.0 / (1.0 + df["Payment_Delay_Days"].clip(lower=0))
    df["clv_approximation_usd"] = df["avg_monthly_revenue_usd"] * df["Tenure_Months"]

    engineered = [
        c
        for c in df.columns
        if c not in raw_customers.columns and not c.startswith(("usage_", "txn_", "eng_", "quarterly_"))
    ]
    logger.info("Engineered %d features: %s", len(engineered), ", ".join(engineered))

    # Ratios can produce inf/NaN when a denominator was zero; repair them.
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].median())
    return df


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def sanitise_column_name(name: str) -> str:
    """Strip characters that XGBoost rejects in feature names (``[``, ``]``, ``<``)."""
    cleaned = re.sub(r"[\[\]<>]", "", str(name))
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", cleaned)
    return cleaned.strip("_")


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode binary/high-cardinality columns and one-hot encode the rest."""
    df = df.copy()
    categorical_columns = [
        c
        for c in df.select_dtypes(include=["object", "string", "category"]).columns
        if c not in (CUSTOMER_KEY, CHURN_TARGET)
    ]

    one_hot_columns: list[str] = []
    label_columns: list[str] = []
    for column in categorical_columns:
        cardinality = df[column].nunique(dropna=False)
        if cardinality <= 2 or cardinality > ONE_HOT_MAX_CARDINALITY:
            label_columns.append(column)
        else:
            one_hot_columns.append(column)

    for column in label_columns:
        df[column] = LabelEncoder().fit_transform(df[column].astype(str))

    if one_hot_columns:
        df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True, dtype=int)

    df.columns = [sanitise_column_name(c) for c in df.columns]

    logger.info(
        "Encoded %d label columns and %d one-hot columns -> %s",
        len(label_columns),
        len(one_hot_columns),
        df.shape,
    )
    return df


# --------------------------------------------------------------------------- #
# Class balancing
# --------------------------------------------------------------------------- #


def build_balanced_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split stratified on churn, then apply SMOTE to the training split only."""
    features = df.drop(columns=[CHURN_TARGET])
    target = df[CHURN_TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    logger.info(
        "Train/test split: %d / %d rows | train churn rate %.3f",
        len(x_train),
        len(x_test),
        y_train.mean(),
    )

    # Customer_ID is an identifier, not a feature: exclude it from resampling.
    train_ids = x_train[CUSTOMER_KEY]
    resampler = SMOTE(random_state=RANDOM_STATE)
    x_resampled, y_resampled = resampler.fit_resample(
        x_train.drop(columns=[CUSTOMER_KEY]), y_train
    )

    train_balanced = x_resampled.copy()
    train_balanced[CHURN_TARGET] = y_resampled.to_numpy()
    logger.info(
        "SMOTE: %d -> %d rows | class counts %s (original ids: %d)",
        len(x_train),
        len(train_balanced),
        dict(pd.Series(y_resampled).value_counts().sort_index()),
        train_ids.nunique(),
    )

    test_set = x_test.copy()
    test_set[CHURN_TARGET] = y_test.to_numpy()
    return train_balanced, test_set


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_pipeline() -> pd.DataFrame:
    """Execute the full preprocessing pipeline and write the output files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = load_tables()
    joined = build_customer_table(tables)
    cleaned = clean_data(joined)
    engineered = engineer_features(cleaned, tables["fact_customers"])
    encoded = encode_categoricals(engineered)

    encoded.to_csv(PROCESSED_FILE, index=False)
    logger.info("Wrote %s %s", PROCESSED_FILE, encoded.shape)

    train_balanced, test_set = build_balanced_split(encoded)
    train_balanced.to_csv(TRAIN_BALANCED_FILE, index=False)
    test_set.to_csv(TEST_FILE, index=False)
    logger.info("Wrote %s %s", TRAIN_BALANCED_FILE, train_balanced.shape)
    logger.info("Wrote %s %s", TEST_FILE, test_set.shape)

    logger.info(
        "Targets available: %s (classification), %s (regression)",
        CHURN_TARGET,
        REVENUE_TARGET,
    )
    return encoded


if __name__ == "__main__":
    run_pipeline()
