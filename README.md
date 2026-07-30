# Enterprise AI-Powered Customer Analytics & Strategic Insight Framework

End-to-end customer intelligence system for a SaaS / CRM platform (Salesforce / Zoho style).
Predicts churn, forecasts next-quarter revenue, segments customers behaviourally, fuses all
three model outputs into a unified per-account intelligence profile, and uses an LLM to
generate executive-level briefings.

**Repository:** <https://github.com/prawin0309/enterprise-ai-customer-analytics>

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
customer-intelligence/
├── src/
│   ├── paths.py               # Repo-relative path resolution (no absolute paths)
│   ├── preprocessing.py       # Load → join → clean → engineer → encode → SMOTE
│   ├── train_models.py        # Churn classifier, revenue regressor, KMeans segmentation
│   ├── fusion_layer.py        # Fuse model outputs into per-customer JSON profiles
│   ├── llm_insights.py        # Gemini-generated executive briefings
│   ├── make_figures.py        # Report figures (ROC, SHAP, residuals, clusters)
│   ├── generate_report.py     # ReportLab capstone PDF
│   └── api.py                 # FastAPI scoring service
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
│   ├── fusion_quality.json                # Profiling-layer validation
│   ├── insight_audit.json                 # LLM currency-figure audit
│   ├── figures/                           # PNGs embedded in the PDF
│   └── fusion_summary.csv                 # Flat per-account scoring table
├── requirements.txt
└── README.md
```

### Data flow

```
CRM_DATA_DIR (read-only source extract)
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
        ├──► llm_insights.py ─────► executive_insights.md + insight_audit.json
        ├──► make_figures.py ─────► reports/figures/*.png
        ├──► generate_report.py ──► Capstone_Final_Report.pdf
        └──► api.py ──────────────► on-demand scoring endpoint
```

---

## Dataset

Star-schema CRM extract, treated as strictly read-only. Its location is resolved by
`src/paths.py` from the `CRM_DATA_DIR` environment variable — nothing in the codebase
hard-codes an absolute path.

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

### Source data location

`src/paths.py` resolves the read-only CRM extract from `CRM_DATA_DIR`, falling back to
`data/raw/` and a sibling `DS_F/` directory. Point it at the folder holding
`fact_customers.csv`:

```powershell
setx CRM_DATA_DIR "D:\DS_F\DataSet-20260223T124234Z-1-001\DataSet"
```

Every writable path (`data/`, `models/`, `reports/`) is derived from the repository root,
so the project runs unchanged from any checkout location.

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
python src\train_models.py       # ~3 min → 6 model artefacts + metrics (incl. search)
python src\fusion_layer.py       # ~5 s   → 5,000 JSON intelligence profiles
python src\llm_insights.py       # ~90 s  → executive_insights.md + insight_audit.json
python src\make_figures.py       # ~5 s   → 5 PNGs in reports/figures/
python src\generate_report.py    # ~3 s   → Capstone_Final_Report.pdf (9 pages, 5 figures)
```

`llm_insights.py` needs `GEMINI_API_KEY`. `generate_report.py` degrades to a tables-only
document if `make_figures.py` has not been run, rather than failing.

### Scoring API

`src/api.py` serves the same three models for a single account on demand. It loads the
`.pkl` artefacts and the processed feature table at start-up and does not need the raw CRM
extract.

```powershell
uvicorn src.api:app --reload
curl http://127.0.0.1:8000/customers/12908/score
```

```json
{
  "customer_id": 12908,
  "churn_probability": 0.9303,
  "risk_level": "High",
  "predicted_next_quarter_revenue_usd": 15926.51,
  "revenue_at_risk_usd": 14816.35,
  "cluster_id": 1
}
```

Interactive docs at `/docs`; readiness probe at `/health`.

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
180 modelling features. Two families of columns are withheld from supervised training as
leakage: the pre-baked `Cluster_Label_*` segment tags shipped with the source data, and four
record-count features that encode the churn outcome (see
[Target-leakage handling](#target-leakage-handling)).

### Model 1 — Churn Prediction (XGBoost Classifier)

| Metric | Value | Notes |
| --- | --- | --- |
| **ROC-AUC** | **0.9113** | Hold-out; the figure to quote |
| **F1-score (churn class)** | **0.6146** | Minority-class balance |
| Precision (churn) | 0.747 | Of flagged accounts, 74.7% do churn |
| Recall (churn) | 0.522 | Catches 52.2% of actual churners |
| Accuracy | 0.926 | Overall correctness |
| **Cross-validated ROC-AUC** | **0.9080 ± 0.0103** | SMOTE refitted inside each fold |
| GradientBoosting baseline ROC-AUC | 0.8689 | XGBoost selected |

Hyperparameters come from `RandomizedSearchCV` (20 sampled configurations, 3-fold
stratified CV, ROC-AUC scoring) run over the entire SMOTE + XGBoost pipeline, so
resampling is refitted inside every search fold. Selected: `n_estimators=800`,
`max_depth=6`, `learning_rate=0.08`, `subsample=0.9`, `colsample_bytree=1.0`,
`min_child_weight=3`, `reg_lambda=1.0` (search score 0.9085).

**Top SHAP churn drivers:** `Renewal_Risk_Flag_Low` · `txn_total_revenue_usd` ·
`txn_total_billed_usd` · `txn_mean_revenue_usd` · `escalated_ticket_ratio`

#### Target-leakage handling

Four record-count features fall simply because a churned customer **stops generating rows**,
so their value encodes the outcome rather than predicting it. All four correlate *negatively*
with churn. They are withheld from the shipped model:

`txn_count` · `txn_renewal_count` · `active_quarters` · `usage_months_observed`

Engagement counts (`eng_event_count`, `eng_distinct_event_types`) are deliberately **kept** —
they correlate *positively* with churn (+0.27), because distressed customers raise *more*
tickets. That is genuine leading signal, not a temporal artefact.

The cost of this exclusion was measured, not assumed (`leakage_ablation` in
`reports/model_metrics.json`):

| Variant | Features | ROC-AUC | F1 | Recall |
| --- | --- | --- | --- | --- |
| **Shipped model (leak-free)** | 180 | **0.9113** | **0.6146** | **0.5221** |
| Contaminated variant (ablation only) | 184 | 0.9425 | 0.7032 | 0.6814 |

The contaminated variant scores ~0.031 higher ROC-AUC partly by reading the outcome it is
meant to forecast. **Quote the leak-free numbers.**

A second caveat sits alongside this one. `Renewal_Risk_Flag_Low` is the strongest SHAP
driver and is itself a CRM-assigned risk rating, so the classifier is partly reproducing
an existing human or rules-based judgement. It is populated before renewal and is
therefore legitimate at prediction time, but the headline ROC-AUC should not be read as
wholly independent evidence of new signal.

### Model 2 — Revenue Forecasting (GradientBoosting Regressor)

| Metric | Value |
| --- | --- |
| **RMSE** | **$1,363.28** |
| **MAE** | **$692.37** |
| **R²** | **0.9674** |
| Adjusted R² | 0.9600 |
| Cross-validated R² | 0.9557 ± 0.0105 |
| Tuned XGBoost alternative RMSE | $1,378.95 (R² 0.9666) |
| Untuned XGBoost alternative RMSE | $1,390.61 (R² 0.9661) |

The XGBoost candidate was tuned by `RandomizedSearchCV` (20 configurations, 3-fold CV,
R² scoring; best: `n_estimators=400`, `max_depth=4`, `learning_rate=0.02`,
`subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=1.0`). GradientBoosting still wins
on hold-out RMSE, so it is the shipped regressor.

### Model 3 — Behavioural Segmentation (KMeans + PCA)

| Metric | Value | Target | Status |
| --- | --- | --- | --- |
| **Silhouette score** | **0.3760** | > 0.60 | ❌ Not met |
| Davies-Bouldin index | 1.1886 | lower better | — |
| Calinski-Harabasz | 2,723.4 | higher better | — |
| Clusters selected (k) | 2 | — | Best across k = 2..10 |
| PCA components | 6 | — | 90.7% variance retained |
| Ward hierarchical silhouette (k=2) | 0.3832 | — | Cross-check; same partition |
| DBSCAN silhouette (eps=1.0) | 0.6110 | — | Clears target, but drops 9.1% as noise |

| Cluster | Label | Accounts | Mean MAU | Adoption | Avg Monthly Rev | Churn Rate |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Disengaged At-Risk Accounts | 1,401 | 46 | 30.2% | $576 | **30.0%** |
| 1 | Loyal High-Value Accounts | 3,599 | 202 | 73.3% | $2,127 | **4.0%** |

The silhouette target was **not** met — see [Known Limitations](#known-limitations). Ward
hierarchical clustering reproduces the same partition, so the weak separation is a
property of the data rather than a KMeans artefact. DBSCAN clears the threshold only by
labelling 9.1% of accounts as noise — it declines to classify exactly the accounts sitting
between the groups, which is the population Customer Success most needs an answer for. The
shipped segments remain commercially sharp: a 7.5× churn-rate gap between them.

### Fusion Layer Validation

Measured by `assess_fusion_quality()` and written to `reports/fusion_quality.json`.

| Criterion | Measure | Result |
| --- | --- | --- |
| Data integrity | Profile completeness | **100%** of 5,000 profiles carry all six blocks |
| Integration accuracy | Model output consistency | **100%** — revenue at risk reconciles to forecast × churn probability |
| Logical consistency | Risk vs CRM health | Spearman **−0.522** — churn risk falls as health rises |
| Grain preservation | One profile per customer | Yes |
| Scalability | Generation rate | ~2,400 profiles/second, single process |
| Business utility | Account prioritisation | **10.4%** of accounts carry **88.0%** of total revenue at risk |

### LLM Factuality Audit

`audit_briefing()` checks every currency figure in every briefing against the source
profile, accepting monthly / quarterly / annual restatements, `k` and `m` shorthand, and
rounding within 1%. Written to `reports/insight_audit.json`.

| Measure | Result |
| --- | --- |
| Briefings fully traceable | **5 of 5** |
| Currency figures quoted | 53 |
| Figures flagged for review | **0** |

A flagged figure would be a review candidate, not proof of fabrication — the model may
legitimately combine two profile values. The check covers currency only; percentages,
dates and reasoning are not verified.

### Fused Portfolio View

| Indicator | Value |
| --- | --- |
| Accounts profiled | 5,000 |
| High-risk accounts | 521 |
| Medium-risk accounts | 34 |
| Low-risk accounts | 4,445 |
| **Total forecast revenue at risk** | **$624,460** |

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

Tunes both supervised models with `RandomizedSearchCV`, compares each against at least one
untuned baseline, and sweeps k = 2..10 for clustering with silhouette / Davies-Bouldin /
Calinski-Harabasz validation plus Ward-hierarchical and DBSCAN cross-checks. Computes SHAP
values via `TreeExplainer`.

Cross-validation uses an `imblearn.pipeline.Pipeline` so **SMOTE is refitted inside every
fold** on that fold's training portion only. Cross-validating a model already fitted on
pre-balanced data is optimistic — synthetic rows derived from a validation record leak into
the training folds. That earlier approach reported 0.9954 against the honest 0.9032.

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
| Executed notebook (16 charts) | `notebooks/EDA_and_Modeling.ipynb` |
| Executive briefings | `reports/executive_insights.md` |
| Capstone PDF report (9 pages, 5 figures) | `reports/Capstone_Final_Report.pdf` |
| Report figures | `reports/figures/*.png` |
| Metrics | `reports/model_metrics.json` |
| Fusion layer validation | `reports/fusion_quality.json` |
| LLM factuality audit | `reports/insight_audit.json` |
| Scoring API | `src/api.py` |

---

## Known Limitations

1. **Residual leakage risk.** The four clearest temporal artefacts are withheld and the cost
   measured, but other aggregates — `txn_total_revenue_usd`, `Lifetime_Revenue_USD` — also
   scale with how long an account survived. A production rebuild should compute every feature
   from a fixed observation window that closes before the prediction date.

2. **Recall ceiling.** The leak-free model catches 52.2% of churners at 74.7% precision.
   Raising recall means lowering the decision threshold and accepting more false positives;
   the right operating point depends on the cost of a retention outreach versus a lost
   account. No threshold tuning was performed — the default 0.5 cut-off is used.

3. **Silhouette target not met.** 0.3760 versus the 0.60 target. SaaS behavioural features
   are continuous and overlapping, so accounts form a gradient rather than separable spheres;
   no k in 2..10 approached the target, and Ward hierarchical clustering reproduces the same
   partition. DBSCAN reaches 0.6110 only by discarding 9.1% of accounts as noise before the
   score is computed. Soft or model-based clustering, which assigns membership probabilities
   instead of hard labels, is the more promising direction. Reducing to 2 PCA components
   would inflate the score geometrically at the cost of business meaning — not done here.

4. **CRM risk flag as a predictor.** `Renewal_Risk_Flag_Low` dominates the SHAP ranking and
   is a CRM-assigned rating rather than raw behaviour. It is available before renewal, so it
   is not leakage, but part of the model's accuracy reflects an existing judgement rather
   than newly discovered signal.

5. **Negative quarterly revenue.** `quarterly_revenue_mean` reaches −$3,079 for some accounts
   (refunds and credits in the transaction table). Legitimate data, but it rules out naive log
   transformation of revenue.

6. **Static snapshot.** The system scores a point-in-time extract. Production use requires
   scheduled re-scoring, drift monitoring and retraining triggers.

7. **Partial LLM validation.** The factuality audit checks currency figures only.
   Percentages, dates, named roles and the reasoning connecting them are not verified, and
   no automated check can judge whether a recommendation is commercially sound. Briefings
   are a drafting aid for a human owner, not an unreviewed output.

---

## Tech Stack

Python 3.13 · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP ·
matplotlib · seaborn · joblib · google-generativeai (Gemini) · ReportLab · FastAPI ·
uvicorn · Jupyter
