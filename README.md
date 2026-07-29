# Enterprise AI-Powered Customer Analytics & Strategic Insight Framework

End-to-end customer intelligence system for a SaaS / CRM platform (Salesforce / Zoho style).
Predicts churn, forecasts next-quarter revenue, segments customers behaviourally, fuses all
three model outputs into a unified per-account intelligence profile, and uses an LLM to
generate executive-level briefings.

**Domain:** SaaS / CRM Analytics · Customer Intelligence · AI-driven Business Analytics

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Execution Instructions](#execution-instructions)
- [Model Performance](#model-performance)
- [Pipeline Details](#pipeline-details)
- [Outputs](#outputs)
- [Known Limitations](#known-limitations)

---

## Problem Statement

CRM platforms store large volumes of subscription, usage, payment and engagement data, but
business teams struggle to convert it into decisions. This project addresses four gaps:

1. **Churn risk is invisible until it happens** — no predictive early-warning signal.
2. **Revenue forecasting is manual** — no per-customer, model-driven projection.
3. **Customers are treated as one undifferentiated block** — no behavioural segmentation.
4. **Model output is unreadable to executives** — probabilities and cluster IDs are not strategy.

---

## Project Architecture

```
D:\DS_FO\
├── src/
│   ├── preprocessing.py       # Load → join → clean → engineer → encode → SMOTE
│   ├── train_models.py        # Churn classifier, revenue regressor, KMeans segmentation
│   ├── fusion_layer.py        # Fuse model outputs into per-customer JSON profiles
│   ├── llm_insights.py        # Gemini-generated executive briefings
│   └── generate_report.py     # ReportLab capstone PDF
├── data/
│   └── processed/
│       ├── processed_customer_data.csv    # 5,000 × 190 modelling table
│       ├── train_balanced_smote.csv       # SMOTE-balanced training split
│       ├── test_holdout.csv               # Untouched hold-out split
│       ├── customer_profiles.json         # 5,000 fused intelligence profiles
│       └── priority_accounts.json         # Top 5 by revenue at risk
├── models/
│   ├── classifier.pkl                     # XGBoost churn classifier
│   ├── regressor.pkl                      # GradientBoosting revenue regressor
│   ├── clusterer.pkl                      # KMeans segmentation model
│   ├── scaler.pkl                         # StandardScaler for segmentation
│   ├── segmentation_pca.pkl               # PCA transform for segmentation
│   └── feature_contract.pkl               # Feature lists for consistent scoring
├── notebooks/
│   └── EDA_and_Modeling.ipynb             # Fully executed EDA + diagnostics
├── reports/
│   ├── Capstone_Final_Report.pdf          # Submission report
│   ├── executive_insights.md              # 5 LLM-generated briefings
│   ├── model_metrics.json                 # All measured metrics
│   ├── shap_feature_importance.csv        # Global churn drivers
│   ├── clustering_sweep.csv               # k = 2..10 validation sweep
│   ├── cluster_profiles.csv               # Segment behavioural means
│   └── fusion_summary.csv                 # Flat per-account scoring table
├── requirements.txt
└── README.md
```

### Data flow

```
D:\DS_F (read-only source)
        │
        ▼
  preprocessing.py ──────► processed_customer_data.csv
        │
        ▼
  train_models.py ───────► classifier.pkl · regressor.pkl · clusterer.pkl
        │                  scaler.pkl · segmentation_pca.pkl
        ▼
  fusion_layer.py ───────► customer_profiles.json (churn + revenue + cluster + SHAP)
        │
        ├──► llm_insights.py ─────► executive_insights.md
        └──► generate_report.py ──► Capstone_Final_Report.pdf
```

---

## Dataset

Star-schema CRM extract read from `D:\DS_F` (treated as strictly read-only).

| Table | Grain | Rows | Role |
| --- | --- | --- | --- |
| `fact_customers.csv` | Customer | 5,150 | Base table, churn + revenue targets |
| `fact_transactions.csv` | Transaction | 54,858 | Revenue, payment behaviour |
| `fact_usage_monthly.csv` | Customer-month | 131,268 | Product usage snapshots |
| `fact_engagement_events.csv` | Event | 54,822 | Support, sentiment, SLA |
| `dim_geography.csv` | Geography | 183 | Country, market tier |
| `dim_industry.csv` | Industry | 22 | Sector, business model |
| `dim_product.csv` | Product | 32 | SKU, plan, support level |

**Targets:** `Churn` (binary, 11.3% positive) · `Next_Quarter_Revenue_USD` (continuous)

---

## Installation

Requires **Python 3.13** (3.14 lacks wheels for some dependencies) and Git.

```powershell
# 1. Create the virtual environment
py -3.13 -m venv D:\DS_FO\.venv

# 2. Activate it
D:\DS_FO\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r D:\DS_FO\requirements.txt
```

### API key (LLM stage only)

`llm_insights.py` reads the Gemini key from the environment — it is never hard-coded.

```powershell
setx GEMINI_API_KEY "your-key-here"
```

Restart the shell afterwards so the variable is visible. Get a key at
<https://aistudio.google.com/apikey>. Optionally pin a model with `setx GEMINI_MODEL "gemini-2.5-pro"`.

---

## Execution Instructions

Run the stages in order — each consumes the previous stage's output.

```powershell
D:\DS_FO\.venv\Scripts\Activate.ps1

python src\preprocessing.py      # ~15 s  → processed_customer_data.csv
python src\train_models.py       # ~90 s  → 6 model artefacts + metrics
python src\fusion_layer.py       # ~5 s   → 5,000 JSON intelligence profiles
python src\llm_insights.py       # ~90 s  → executive_insights.md   (needs API key)
python src\generate_report.py    # ~2 s   → Capstone_Final_Report.pdf
```

Re-execute the notebook:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=900 notebooks\EDA_and_Modeling.ipynb
```

Or explore interactively:

```powershell
python -m jupyter lab notebooks\EDA_and_Modeling.ipynb
```

---

## Model Performance

All figures measured on a stratified 20% hold-out set (`random_state=42`), 5,000 customers,
184 modelling features. The pre-existing `Cluster_Label_*` segment tags shipped with the
source data are withheld from both supervised models as leakage.

### Model 1 — Churn Prediction (XGBoost Classifier)

| Metric | Value | Notes |
| --- | --- | --- |
| **ROC-AUC** | **0.9455** | Hold-out; the figure to quote |
| **F1-score (churn class)** | **0.6881** | Minority-class balance |
| Precision (churn) | 0.714 | Of flagged accounts, 71.4% do churn |
| Recall (churn) | 0.664 | Catches 66.4% of actual churners |
| Accuracy | 0.932 | Overall correctness |
| GradientBoosting baseline ROC-AUC | 0.9384 | XGBoost selected |

**Top SHAP churn drivers:** `txn_count` · `Renewal_Risk_Flag_Low` ·
`txn_mean_payment_delay_days` · `Relative_Churn_Risk_Medium` · `txn_renewal_count`

### Model 2 — Revenue Forecasting (GradientBoosting Regressor)

| Metric | Value |
| --- | --- |
| **RMSE** | **$1,363.28** |
| **MAE** | **$692.37** |
| **R²** | **0.9674** |
| Adjusted R² | 0.9600 |
| Cross-validated R² | 0.9557 ± 0.0105 |
| XGBoost alternative RMSE | $1,390.61 (R² 0.9661) |

### Model 3 — Behavioural Segmentation (KMeans + PCA)

| Metric | Value | Target | Status |
| --- | --- | --- | --- |
| **Silhouette score** | **0.3760** | > 0.60 | ❌ Not met |
| Davies-Bouldin index | 1.1886 | lower better | — |
| Calinski-Harabasz | 2,723.4 | higher better | — |
| Clusters selected (k) | 2 | — | Best across k = 2..10 |
| PCA components | 6 | — | 90.7% variance retained |

| Cluster | Label | Accounts | Mean MAU | Adoption | Avg Monthly Rev | Churn Rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Disengaged At-Risk Accounts | 1,401 | 46 | 30.2% | $576 | **30.0%** |
| 1 | Loyal High-Value Accounts | 3,599 | 202 | 73.3% | $2,127 | **4.0%** |

The silhouette target was **not** met — see [Known Limitations](#known-limitations). The
segments remain commercially sharp: a 7.5× churn-rate gap between them.

### Fused Portfolio View

| Indicator | Value |
| --- | --- |
| Accounts profiled | 5,000 |
| High-risk accounts | 549 |
| Medium-risk accounts | 27 |
| Low-risk accounts | 4,424 |
| **Total forecast revenue at risk** | **$711,300** |

---

## Pipeline Details

### 1. Preprocessing (`src/preprocessing.py`)

- Aggregates three fact tables from their native grain to customer grain.
- Joins dimensions on `Geo_ID` / `Industry_ID` / `Product_ID`, **deduplicating dimension keys
  first** — `dim_geography` ships 5 duplicate `Geo_ID` rows that otherwise fan out the grain.
- Removes 150 duplicate customer records → 5,000 unique accounts.
- Median imputation (numerical) and mode imputation (categorical) → 0 missing cells.
- **45 engineered features** across four families:
  - *Tenure:* `tenure_days`, `contract_duration_days`, `days_to_contract_end`
  - *Usage intensity:* `usage_intensity_score`, `dau_mau_ratio`, `api_calls_per_license`,
    `error_rate_per_session`
  - *Engagement:* `events_per_tenure_month`, `escalated_ticket_ratio`,
    `support_dependency_score`, `engagement_quality_score`
  - *Revenue:* `quarterly_revenue_mean/std/last/growth_ratio`, `avg_monthly_revenue_usd`,
    `clv_approximation_usd`, `payment_reliability_score`
- Label encoding (binary + high-cardinality) and one-hot encoding (low-cardinality) → 190 columns.
- **SMOTE applied to the training split only**, after the stratified split, so no synthetic
  record can reach the hold-out set (11.3% → 50/50, 4,000 → 7,096 rows).

### 2. Training (`src/train_models.py`)

Compares two algorithms per supervised task, cross-validates, and sweeps k = 2..10 for
clustering with silhouette / Davies-Bouldin / Calinski-Harabasz validation. Computes SHAP
values via `TreeExplainer`.

### 3. Fusion (`src/fusion_layer.py`)

Emits one standardised JSON profile per customer:

```json
{
  "customer_id": 12908,
  "account_context":     { "industry": "Healthcare", "country": "Singapore", "...": "..." },
  "churn_prediction":    { "probability": 0.8724, "risk_level": "High" },
  "revenue_forecast":    { "predicted_next_quarter_usd": 15868.45,
                           "revenue_at_risk_usd": 13844.37 },
  "segmentation":        { "cluster_id": 1, "cluster_label": "Loyal High-Value Accounts" },
  "intelligence_scores": { "risk_score": 90.3, "revenue_potential_score": 88.3,
                           "priority_score": 100.0, "engagement_level": "High" },
  "top_churn_drivers":   [ { "feature": "txn_count", "shap_value": 2.655,
                             "direction": "increases_churn_risk" } ]
}
```

Derived indicators: `revenue_at_risk_usd = forecast × churn_probability`, plus
percentile-normalised risk, potential and priority scores.

### 4. LLM Insights (`src/llm_insights.py`)

Passes profiles to Gemini under a system instruction that forbids invented metrics. Each
briefing covers risk level, revenue impact, segment interpretation, prioritised
recommendations with owner and timeframe, and renewal + upsell plays.

> **Model note:** the originally specified `gemini-pro` has been retired from the API. The
> script probes candidate models at runtime and uses the first the credential can serve.
> `google-generativeai` is end-of-life upstream; `google-genai` is its replacement.

---

## Outputs

| Deliverable | Path |
| --- | --- |
| Processed dataset | `data/processed/processed_customer_data.csv` |
| Intelligence profiles | `data/processed/customer_profiles.json` |
| Serialised models | `models/*.pkl` |
| Executed notebook (15 charts) | `notebooks/EDA_and_Modeling.ipynb` |
| Executive briefings | `reports/executive_insights.md` |
| Capstone PDF report | `reports/Capstone_Final_Report.pdf` |
| Metrics | `reports/model_metrics.json` |

---

## Known Limitations

1. **Potential target leakage — `txn_count`.** The strongest churn driver (mean |SHAP| 2.89,
   ~2.5× the next feature) is transaction count. Customers who churn stop transacting, so
   this partly encodes the outcome rather than predicting it. A production rebuild should
   ablate it and re-measure honest performance.

2. **Optimistic cross-validated ROC-AUC.** The `cv_roc_auc_mean` of 0.9954 in
   `model_metrics.json` is computed over SMOTE-balanced folds, so synthetic minority samples
   leak across fold boundaries. **Quote the hold-out ROC-AUC of 0.9455 instead.** The fix is
   to resample inside an `imblearn.pipeline.Pipeline` evaluated per fold.

3. **Silhouette target not met.** 0.3760 versus the 0.60 target. SaaS behavioural features
   are continuous and overlapping, so accounts form a gradient rather than separable spheres;
   no k in 2..10 approached the target. Reducing to 2 PCA components would inflate the score
   geometrically at the cost of business meaning — not done here.

4. **Negative quarterly revenue.** `quarterly_revenue_mean` reaches −$3,079 for some accounts
   (refunds and credits in the transaction table). Legitimate data, but it rules out naive log
   transformation of revenue.

5. **Static snapshot.** The system scores a point-in-time extract. Production use requires
   scheduled re-scoring, drift monitoring and retraining triggers.

6. **Unvalidated LLM output.** Briefings are constrained by prompt design but not
   automatically fact-checked. A validator comparing generated figures against the source
   profile is the natural next step.

---

## Tech Stack

Python 3.13 · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP ·
matplotlib · seaborn · joblib · google-generativeai (Gemini) · ReportLab · Jupyter
