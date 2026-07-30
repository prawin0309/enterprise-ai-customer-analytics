"""Generate the submission-ready capstone PDF report with ReportLab.

Maps each machine-learning component to the business KPI it serves, and
summarises measured model performance, segmentation results, the fused
intelligence layer and the LLM insight stage.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

try:
    from paths import (
        CLUSTER_PROFILE_FILE as CLUSTER_FILE,
        FIGURE_DIR,
        METRICS_FILE,
        PDF_REPORT_FILE as OUTPUT_FILE,
        PROCESSED_DIR,
        PROJECT_DIR,
        REPORT_DIR,
        SHAP_IMPORTANCE_FILE as SHAP_FILE,
        SUMMARY_FILE as FUSION_FILE,
    )
except ImportError:  # imported as ``src.generate_report``
    from src.paths import (
        CLUSTER_PROFILE_FILE as CLUSTER_FILE,
        FIGURE_DIR,
        METRICS_FILE,
        PDF_REPORT_FILE as OUTPUT_FILE,
        PROCESSED_DIR,
        PROJECT_DIR,
        REPORT_DIR,
        SHAP_IMPORTANCE_FILE as SHAP_FILE,
        SUMMARY_FILE as FUSION_FILE,
    )

FUSION_QUALITY_FILE = REPORT_DIR / "fusion_quality.json"
INSIGHT_AUDIT_FILE = REPORT_DIR / "insight_audit.json"

BRAND_COLOR = colors.HexColor("#1F4E79")
ACCENT_COLOR = colors.HexColor("#2E75B6")
LIGHT_ROW = colors.HexColor("#EAF1F8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_report")


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #


def build_styles() -> dict:
    """Return the paragraph styles used throughout the report."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=21,
            leading=26,
            textColor=BRAND_COLOR,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=18,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading1"],
            fontSize=14,
            leading=18,
            textColor=BRAND_COLOR,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "subheading": ParagraphStyle(
            "SubHeading",
            parent=base["Heading2"],
            fontSize=11.5,
            leading=15,
            textColor=ACCENT_COLOR,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyJustified",
            parent=base["Normal"],
            fontSize=9.5,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "BodyBullet",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13.5,
            leftIndent=14,
            bulletIndent=4,
            spaceAfter=3,
        ),
        "cell": ParagraphStyle(
            "TableCell", parent=base["Normal"], fontSize=8, leading=10.5
        ),
        "cell_header": ParagraphStyle(
            "TableCellHeader",
            parent=base["Normal"],
            fontSize=8,
            leading=10.5,
            textColor=colors.white,
            fontName="Helvetica-Bold",
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#666666"),
            spaceAfter=10,
        ),
    }


FIGURE_WIDTH = 16.0 * cm


def figure(name: str, caption: str, styles: dict, aspect: float) -> list:
    """Return an image flowable plus its caption, or nothing if not rendered.

    ``make_figures.py`` writes the PNGs. The report degrades to a tables-only
    document rather than failing if that stage has not been run.
    """
    path = FIGURE_DIR / name
    if not path.exists():
        logger.warning("Figure %s missing - run src/make_figures.py", name)
        return []
    return [
        Spacer(1, 0.25 * cm),
        Image(str(path), width=FIGURE_WIDTH, height=FIGURE_WIDTH * aspect),
        Paragraph(caption, styles["caption"]),
        Spacer(1, 0.25 * cm),
    ]


def format_params(params: dict) -> str:
    """Render a hyperparameter dict as a compact ``key=value`` list."""
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def make_table(rows: list[list], styles: dict, widths: list[float]) -> Table:
    """Build a branded table whose first row is the header."""
    data = [
        [
            Paragraph(str(cell), styles["cell_header"] if index == 0 else styles["cell"])
            for cell in row
        ]
        for index, row in enumerate(rows)
    ]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C9DC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_ROW]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #


def draw_page_furniture(canvas, document) -> None:
    """Draw the header rule and page footer on every page."""
    canvas.saveState()
    width, height = A4

    canvas.setStrokeColor(BRAND_COLOR)
    canvas.setLineWidth(1.4)
    canvas.line(2 * cm, height - 1.5 * cm, width - 2 * cm, height - 1.5 * cm)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(
        2 * cm,
        height - 1.25 * cm,
        "Enterprise AI-Powered Customer Analytics & Strategic Insight Framework",
    )
    canvas.line(2 * cm, 1.5 * cm, width - 2 * cm, 1.5 * cm)
    canvas.drawString(2 * cm, 1.1 * cm, f"Generated {date.today().isoformat()}")
    canvas.drawRightString(width - 2 * cm, 1.1 * cm, f"Page {document.page}")
    canvas.restoreState()


# --------------------------------------------------------------------------- #
# Content sections
# --------------------------------------------------------------------------- #


def build_story(styles: dict, data: dict) -> list:
    """Assemble the full flowable story for the report."""
    metrics = data["metrics"]
    churn = metrics["churn_classification"]
    revenue = metrics["revenue_regression"]
    segmentation = metrics["segmentation"]
    fusion_quality = data.get("fusion_quality") or {}
    insight_audit = data.get("insight_audit") or {}
    fusion = data["fusion"]

    story: list = []
    full_width = [11.5 * cm, 5.5 * cm]

    # --- Title page ------------------------------------------------------
    story.append(Spacer(1, 2.5 * cm))
    story.append(
        Paragraph(
            "Enterprise AI-Powered Customer Analytics "
            "&amp; Strategic Insight Framework",
            styles["title"],
        )
    )
    story.append(
        Paragraph(
            "Churn Prediction, Revenue Forecasting, Behavioural Segmentation "
            "and LLM-Generated Executive Intelligence<br/>"
            "for a SaaS / CRM Platform (Salesforce / Zoho style)",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 0.6 * cm))
    story.append(
        make_table(
            [
                ["Attribute", "Detail"],
                ["Domain", "SaaS / CRM Analytics / Customer Intelligence"],
                ["Dataset", "Star schema: 4 fact tables, 3 dimension tables"],
                ["Modelling grain", f"{len(fusion):,} customers (one row each)"],
                ["Features engineered", f"{churn['n_features']} modelling features"],
                ["Classification target", "Churn (binary)"],
                ["Regression target", "Next_Quarter_Revenue_USD (continuous)"],
                ["Report date", date.today().strftime("%d %B %Y")],
            ],
            styles,
            full_width,
        )
    )
    story.append(PageBreak())

    # --- 1. Executive summary -------------------------------------------
    story.append(Paragraph("1. Executive Summary", styles["heading"]))
    story.append(
        Paragraph(
            "This project delivers an end-to-end customer intelligence system for a "
            "subscription SaaS CRM platform. Four relational fact tables and three "
            f"dimension tables were consolidated into a single customer-level table of "
            f"{len(fusion):,} accounts and {churn['n_features']} engineered features. "
            "Three machine-learning models were trained on that table, their outputs "
            "fused into a standardised per-account intelligence profile, and a large "
            "language model used to translate those profiles into executive briefings.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            f"The churn classifier achieves a hold-out ROC-AUC of "
            f"<b>{churn['roc_auc']:.4f}</b> and an F1 of <b>{churn['f1_score']:.4f}</b> "
            f"on the minority churn class, after withholding four record-count "
            f"features identified as target leakage. The revenue model explains "
            f"<b>{revenue['r2']:.1%}</b> of next-quarter revenue variance with an RMSE "
            f"of <b>${revenue['rmse']:,.0f}</b>. Segmentation resolves the portfolio "
            f"into <b>{segmentation['selected_k']}</b> behavioural clusters whose churn "
            "rates differ by a factor of roughly seven. Fusing these outputs identifies "
            f"<b>${fusion['revenue_at_risk_usd'].sum():,.0f}</b> of next-quarter revenue "
            "sitting in accounts with elevated churn probability.",
            styles["body"],
        )
    )

    # --- 2. Business problem to ML mapping -------------------------------
    story.append(Paragraph("2. Business Problem to ML Task Mapping", styles["heading"]))
    story.append(
        Paragraph(
            "Every technical component is anchored to a measurable business "
            "objective, so model performance translates directly into commercial "
            "outcomes.",
            styles["body"],
        )
    )
    story.append(
        make_table(
            [
                ["Business Problem", "ML Task", "Output", "Business KPI Served"],
                [
                    "Customers cancelling subscriptions",
                    "Binary classification (XGBoost)",
                    "Churn probability 0-1",
                    "Retention rate, customer lifetime value",
                ],
                [
                    "Unreliable revenue planning",
                    "Regression (GradientBoosting)",
                    "Next-quarter revenue (USD)",
                    "Forecast accuracy, quota setting",
                ],
                [
                    "Undifferentiated marketing",
                    "Clustering (KMeans + PCA)",
                    "Behavioural cluster ID",
                    "Campaign conversion, personalisation",
                ],
                [
                    "Unclear account prioritisation",
                    "Feature fusion / scoring",
                    "Risk, potential, priority scores",
                    "Revenue at risk, CSM coverage",
                ],
                [
                    "Technical output not board-ready",
                    "LLM narrative generation",
                    "Executive briefing text",
                    "Decision latency, reporting effort",
                ],
            ],
            styles,
            [4.2 * cm, 4.0 * cm, 3.8 * cm, 5.0 * cm],
        )
    )

    # --- 3. Methodology ---------------------------------------------------
    story.append(Paragraph("3. Methodology", styles["heading"]))
    story.append(Paragraph("3.1 Data engineering", styles["subheading"]))
    for text in [
        "Fact tables aggregated from their native grain to customer grain: "
        "transactions (54,858 rows), monthly usage snapshots (131,268 rows) and "
        "engagement events (54,822 rows).",
        "Dimension attributes joined on Geo_ID, Industry_ID and Product_ID. "
        "Duplicate dimension keys were removed first — dim_geography ships five "
        "duplicate Geo_ID rows that would otherwise fan out the customer grain.",
        "150 duplicate customer records removed, leaving 5,000 unique accounts.",
        "Missing values imputed with the median (numerical) and mode (categorical); "
        "zero missing cells remain.",
    ]:
        story.append(Paragraph(text, styles["bullet"], bulletText="\u2022"))

    story.append(Paragraph("3.2 Feature engineering", styles["subheading"]))
    for text in [
        "<b>Tenure:</b> contract duration, tenure in days and years, days to "
        "contract end.",
        "<b>Usage intensity:</b> sessions per active user weighted by session "
        "length, DAU/MAU ratio, API calls per licence, error rate per session.",
        "<b>Engagement ratios:</b> events per tenure month, escalated ticket ratio, "
        "support dependency, sentiment-based engagement quality score.",
        "<b>Revenue:</b> quarterly revenue mean, volatility, latest value and growth "
        "ratio, average monthly revenue, CLV approximation, payment reliability.",
    ]:
        story.append(Paragraph(text, styles["bullet"], bulletText="\u2022"))

    story.append(Paragraph("3.3 Encoding and class balance", styles["subheading"]))
    story.append(
        Paragraph(
            "Binary and high-cardinality categoricals were label encoded; "
            "low-cardinality nominals were one-hot encoded, producing 190 columns. "
            "The churn target is imbalanced at roughly 11.3% positives. SMOTE was "
            "applied <b>only to the training split after the stratified train/test "
            "split</b>, so no synthetic record can reach the hold-out set.",
            styles["body"],
        )
    )

    story.append(PageBreak())

    # --- 4. Model performance --------------------------------------------
    story.append(Paragraph("4. Model Performance", styles["heading"]))
    story.append(Paragraph("4.1 Churn classification", styles["subheading"]))
    story.append(
        make_table(
            [
                ["Metric", "Value", "Business Interpretation"],
                [
                    "ROC-AUC (hold-out)",
                    f"{churn['roc_auc']:.4f}",
                    "Separates churners from retained accounts reliably",
                ],
                [
                    "F1-score (churn class)",
                    f"{churn['f1_score']:.4f}",
                    "Balanced precision and recall on the minority class",
                ],
                [
                    "Precision (churn class)",
                    f"{churn['precision']:.4f}",
                    "Share of flagged accounts that genuinely churn",
                ],
                [
                    "Recall (churn class)",
                    f"{churn['recall']:.4f}",
                    "Share of actual churners the model catches",
                ],
                [
                    "Cross-validated ROC-AUC",
                    f"{churn['cv_roc_auc_mean']:.4f} ± "
                    f"{churn['cv_roc_auc_std']:.4f}",
                    "SMOTE refitted inside each fold - no resampling leak",
                ],
                [
                    "Baseline GradientBoosting ROC-AUC",
                    f"{churn['baseline_gb_roc_auc']:.4f}",
                    "XGBoost selected as the stronger model",
                ],
                [
                    "Features used",
                    f"{churn['n_features']}",
                    "After withholding leakage-prone columns",
                ],
            ],
            styles,
            [4.6 * cm, 3.4 * cm, 9.0 * cm],
        )
    )
    search = churn["hyperparameter_search"]
    story.append(
        Paragraph(
            f"Hyperparameters were selected by {search['method']} over "
            f"{search['candidates_sampled']} sampled configurations with "
            f"{search['cv_folds']}-fold stratified cross-validation, scoring "
            "ROC-AUC. The search ran over the whole SMOTE + XGBoost pipeline, so "
            "resampling was refitted inside every fold rather than applied once "
            "beforehand. Best configuration: "
            f"{format_params(search['best_params'])} (search score "
            f"{search['best_search_score']:.4f}).",
            styles["caption"],
        )
    )
    story.append(
        Paragraph(
            "The cross-validated figure is produced by an imbalanced-learn pipeline "
            "in which SMOTE is refitted on the training portion of every fold. "
            "Cross-validating a model already fitted on pre-balanced data would be "
            "optimistic, because synthetic rows derived from a validation record can "
            "reach the training folds; that earlier approach reported 0.9954 against "
            f"the honest {churn['cv_roc_auc_mean']:.4f}.",
            styles["caption"],
        )
    )

    ablation = churn.get("leakage_ablation")
    if ablation:
        story.append(
            Paragraph("4.1.1 Target-leakage ablation", styles["subheading"])
        )
        story.append(
            Paragraph(
                "Four record-count features (transaction count, renewal count, "
                "active quarters, months of usage observed) fall simply because a "
                "churned customer stops generating rows, so their value encodes the "
                "outcome rather than predicting it. They are withheld from the "
                "shipped model. Engagement counts are deliberately retained: they "
                "correlate <i>positively</i> with churn (distressed customers raise "
                "more tickets), which is genuine leading signal. The table below "
                "states the measured cost of that decision.",
                styles["body"],
            )
        )
        story.append(
            make_table(
                [
                    ["Variant", "Features", "ROC-AUC", "F1", "Recall"],
                    [
                        "Shipped model (leak-free)",
                        f"{churn['n_features']}",
                        f"{churn['roc_auc']:.4f}",
                        f"{churn['f1_score']:.4f}",
                        f"{churn['recall']:.4f}",
                    ],
                    [
                        "Contaminated variant (ablation only)",
                        f"{ablation['n_features']}",
                        f"{ablation['roc_auc']:.4f}",
                        f"{ablation['f1_score']:.4f}",
                        f"{ablation['recall']:.4f}",
                    ],
                ],
                styles,
                [6.4 * cm, 2.4 * cm, 2.8 * cm, 2.6 * cm, 2.8 * cm],
            )
        )
        story.append(
            Paragraph(
                f"Excluding the artefacts costs "
                f"{ablation['roc_auc'] - churn['roc_auc']:.4f} ROC-AUC and "
                f"{ablation['recall'] - churn['recall']:.4f} recall. The lower "
                "figure is the honest one: the contaminated variant scores well "
                "partly by reading the outcome it is meant to forecast.",
                styles["caption"],
            )
        )

    story.extend(
        figure(
            "churn_roc_confusion.png",
            "Figure 1 — Churn model ROC curve and confusion matrix on the 1,000-row "
            "hold-out set. The 0.5 threshold trades recall for precision; 54 churners "
            "are missed and 20 retained accounts are flagged.",
            styles,
            0.42,
        )
    )

    story.append(Paragraph("4.2 Explainable AI — SHAP churn drivers", styles["subheading"]))
    shap_rows = [["Rank", "Feature", "Mean |SHAP|", "Business Meaning"]]
    meanings = {
        "txn_count": "Transaction frequency — low volume signals disengagement",
        "Renewal_Risk_Flag_Low": "CRM renewal-risk flag set to Low",
        "txn_mean_payment_delay_days": "Average invoice payment delay",
        "escalated_ticket_ratio": "Share of support tickets escalated",
        "Relative_Churn_Risk_Medium": "Industry-level baseline churn risk",
        "txn_renewal_count": "Number of renewal transactions completed",
        "usage_mean_license_utilization_pct": "Licences actually used vs purchased",
        "Relative_Churn_Risk_Low": "Industry-level baseline churn risk",
        "Renewal_Risk_Flag_Medium": "CRM renewal-risk flag set to Medium",
        "tickets_per_tenure_month": "Support ticket rate per month of tenure",
    }
    for rank, row in enumerate(data["shap"].head(10).itertuples(index=False), start=1):
        shap_rows.append(
            [
                rank,
                row.feature,
                f"{row.mean_abs_shap:.4f}",
                meanings.get(row.feature, "Model-derived behavioural signal"),
            ]
        )
    story.append(make_table(shap_rows, styles, [1.2 * cm, 5.6 * cm, 2.4 * cm, 7.8 * cm]))

    story.extend(
        figure(
            "churn_shap_beeswarm.png",
            "Figure 2 — SHAP value distribution across 500 hold-out accounts. Red "
            "points are high feature values, blue are low; points to the right push "
            "the prediction toward churn.",
            styles,
            0.68,
        )
    )

    story.append(Paragraph("4.3 Revenue forecasting", styles["subheading"]))
    story.append(
        make_table(
            [
                ["Metric", "Value", "Business Interpretation"],
                [
                    "Selected model",
                    revenue["selected_model"],
                    "Chosen over the alternative on hold-out RMSE",
                ],
                [
                    "RMSE",
                    f"${revenue['rmse']:,.2f}",
                    "Typical forecast error, penalising large misses",
                ],
                [
                    "MAE",
                    f"${revenue['mae']:,.2f}",
                    "Average absolute error per account per quarter",
                ],
                ["R\u00b2", f"{revenue['r2']:.4f}", "Share of revenue variance explained"],
                [
                    "Adjusted R\u00b2",
                    f"{revenue['adjusted_r2']:.4f}",
                    "Corrected for the number of features used",
                ],
                [
                    "Cross-validated R\u00b2",
                    f"{revenue['cv_r2_mean']:.4f} \u00b1 {revenue['cv_r2_std']:.4f}",
                    "Stability across five folds",
                ],
            ],
            styles,
            [4.0 * cm, 3.6 * cm, 9.4 * cm],
        )
    )

    story.extend(
        figure(
            "revenue_residuals.png",
            "Figure 3 — Revenue model diagnostics: predicted versus actual, residuals "
            "against fitted values, and the residual distribution. Residuals are "
            "centred and show no systematic pattern across the fitted range.",
            styles,
            0.30,
        )
    )

    revenue_search = revenue["hyperparameter_search"]
    story.append(
        Paragraph(
            f"The XGBoost candidate was tuned by {revenue_search['method']} over "
            f"{revenue_search['candidates_sampled']} sampled configurations "
            f"({revenue_search['cv_folds']}-fold cross-validation, R\u00b2 "
            "scoring) and compared against an untuned XGBoost and a "
            "GradientBoosting baseline. Best configuration: "
            f"{format_params(revenue_search['best_params'])}.",
            styles["caption"],
        )
    )

    story.append(Paragraph("4.4 Behavioural segmentation", styles["subheading"]))
    story.append(
        make_table(
            [
                ["Metric", "Value", "Assessment"],
                [
                    "Clusters selected (k)",
                    str(segmentation["selected_k"]),
                    "Chosen by best silhouette across k = 2 to 10",
                ],
                [
                    "Silhouette score",
                    f"{segmentation['silhouette']:.4f}",
                    f"Below the {segmentation['silhouette_target']:.2f} project target",
                ],
                [
                    "Davies-Bouldin index",
                    f"{segmentation['davies_bouldin']:.4f}",
                    "Lower is better; indicates moderate compactness",
                ],
                [
                    "Calinski-Harabasz",
                    f"{segmentation['calinski_harabasz']:,.1f}",
                    "Higher is better; confirms real separation",
                ],
                [
                    "PCA components",
                    str(segmentation["pca_components"]),
                    "90% of behavioural variance retained",
                ],
            ],
            styles,
            [4.0 * cm, 3.6 * cm, 9.4 * cm],
        )
    )
    alternatives = segmentation.get("alternative_algorithms")
    if alternatives:
        agglomerative = alternatives["agglomerative_ward"]
        dbscan_best = alternatives.get("dbscan_best")
        if dbscan_best:
            dbscan_text = (
                f"DBSCAN peaked at silhouette {dbscan_best['silhouette']:.4f} "
                f"(eps={dbscan_best['eps']}, "
                f"{dbscan_best['clusters_found']} clusters, "
                f"{dbscan_best['noise_rate']:.1%} of accounts labelled noise)"
            )
        else:
            dbscan_text = (
                "DBSCAN found no radius that produced two or more clusters while "
                "keeping noise below half the dataset"
            )
        story.append(
            Paragraph(
                "<b>Cross-check against other algorithms.</b> Ward hierarchical "
                "clustering at the same k scores silhouette "
                f"{agglomerative['silhouette']:.4f} - effectively the same partition "
                "as KMeans, so the weak separation is not a KMeans artefact. "
                f"{dbscan_text}. DBSCAN scores higher only because silhouette is "
                "computed after its noise points are discarded: it clears the "
                "threshold by refusing to classify the accounts that sit between "
                "the groups, which is exactly the population Customer Success needs "
                "an answer for. A hard partition over every account remains the "
                "right choice for this use case.",
                styles["body"],
            )
        )

    story.append(
        Paragraph(
            "<b>Honest assessment:</b> the 0.60 silhouette target was not met. The best "
            f"achievable score was {segmentation['silhouette']:.4f} at k="
            f"{segmentation['selected_k']}. SaaS behavioural features are continuous "
            "and overlapping, so accounts form a gradient rather than well-separated "
            "spheres, and no k between 2 and 10 approached the target. The clusters "
            "remain commercially useful: churn rates differ roughly sevenfold between "
            "them, which is the property the business actually needs.",
            styles["body"],
        )
    )

    story.append(PageBreak())

    # --- 5. Segment profile ----------------------------------------------
    story.extend(
        figure(
            "clustering_sweep.png",
            "Figure 4 — Elbow and silhouette curves across k = 2 to 10. No k reaches "
            "the 0.60 project target.",
            styles,
            0.37,
        )
    )
    story.extend(
        figure(
            "segment_pca_scatter.png",
            "Figure 5 — Behavioural segments projected onto two principal components "
            "(display only; the model clusters on six), with churn rate by segment. "
            "The groups overlap geometrically but separate sharply on churn.",
            styles,
            0.40,
        )
    )

    story.append(Paragraph("5. Customer Segment Profiles", styles["heading"]))
    cluster_rows = [
        [
            "Cluster",
            "Accounts",
            "Mean MAU",
            "Adoption %",
            "Avg Monthly Rev",
            "Churn Rate",
        ]
    ]
    for cluster_id, row in data["clusters"].iterrows():
        size = int((fusion["cluster_id"] == cluster_id).sum())
        cluster_rows.append(
            [
                f"{cluster_id}",
                f"{size:,}",
                f"{row['usage_mean_mau']:,.0f}",
                f"{row['usage_mean_feature_adoption_pct']:.1f}%",
                f"${row['avg_monthly_revenue_usd']:,.0f}",
                f"{row['churn_rate']:.1%}",
            ]
        )
    story.append(
        make_table(
            cluster_rows,
            styles,
            [2.0 * cm, 2.4 * cm, 2.4 * cm, 2.6 * cm, 3.6 * cm, 2.6 * cm],
        )
    )
    story.append(
        Paragraph(
            "Cluster 0 (<i>Disengaged At-Risk Accounts</i>) combines low usage, low "
            "adoption and heavy support dependency with a materially higher churn "
            "rate. Cluster 1 (<i>Loyal High-Value Accounts</i>) shows roughly four "
            "times the monthly active users, more than double the feature adoption, "
            "and near-negligible churn. Retention spend belongs in cluster 0; expansion "
            "and upsell motions belong in cluster 1.",
            styles["body"],
        )
    )

    # --- 6. Fusion layer --------------------------------------------------
    story.append(Paragraph("6. Unified Customer Intelligence Layer", styles["heading"]))
    story.append(
        Paragraph(
            "The fusion layer scores every account with all three models and emits a "
            "standardised JSON profile containing churn probability and risk band, "
            "next-quarter revenue forecast, revenue at risk, cluster assignment and "
            "label, percentile-normalised risk / potential / priority scores, and the "
            "top five per-account SHAP drivers with direction of effect.",
            styles["body"],
        )
    )
    risk_counts = fusion["risk_level"].value_counts()
    story.append(
        make_table(
            [
                ["Portfolio Indicator", "Value"],
                ["Accounts profiled", f"{len(fusion):,}"],
                ["High-risk accounts", f"{risk_counts.get('High', 0):,}"],
                ["Medium-risk accounts", f"{risk_counts.get('Medium', 0):,}"],
                ["Low-risk accounts", f"{risk_counts.get('Low', 0):,}"],
                [
                    "Total forecast revenue at risk",
                    f"${fusion['revenue_at_risk_usd'].sum():,.0f}",
                ],
                [
                    "Mean churn probability",
                    f"{fusion['churn_probability'].mean():.1%}",
                ],
                [
                    "Top 5 accounts, revenue at risk",
                    f"${fusion.nlargest(5, 'revenue_at_risk_usd')['revenue_at_risk_usd'].sum():,.0f}",
                ],
            ],
            styles,
            full_width,
        )
    )

    # --- 7. LLM layer -----------------------------------------------------
    if fusion_quality:
        story.append(Paragraph("6.1 Fusion layer validation", styles["subheading"]))
        story.append(
            make_table(
                [
                    ["Evaluation criterion", "Measure", "Result"],
                    [
                        "Data integrity",
                        "Profile completeness",
                        f"{fusion_quality['profile_completeness']:.1%} of "
                        f"{fusion_quality['profiles_generated']:,} profiles carry "
                        "all six blocks",
                    ],
                    [
                        "Integration accuracy",
                        "Model output consistency",
                        f"{fusion_quality['integration_accuracy']:.1%} - revenue at "
                        "risk reconciles to forecast x churn probability",
                    ],
                    [
                        "Logical consistency",
                        "Risk vs CRM health",
                        f"Spearman {fusion_quality['risk_health_alignment_spearman']:.3f} "
                        "- churn risk falls as health rises, as expected",
                    ],
                    [
                        "Grain preservation",
                        "One profile per customer",
                        "Yes" if fusion_quality["grain_preserved"] else "No",
                    ],
                    [
                        "Scalability",
                        "Profile generation rate",
                        f"{fusion_quality['profiles_per_second']:,.0f} profiles per "
                        "second, single process",
                    ],
                    [
                        "Business utility",
                        "Account prioritisation",
                        f"{fusion_quality['high_risk_share_of_accounts']:.1%} of "
                        "accounts carry "
                        f"{fusion_quality['high_risk_share_of_revenue_at_risk']:.1%} "
                        "of total revenue at risk",
                    ],
                ],
                styles,
                [4.2 * cm, 4.2 * cm, 8.6 * cm],
            )
        )
        story.append(
            Paragraph(
                "The prioritisation figure is the one that matters commercially: "
                "concentrating retention effort on the high-risk band reaches the "
                "large majority of exposed revenue while contacting roughly a tenth "
                "of the book. Full detail in reports/fusion_quality.json.",
                styles["caption"],
            )
        )

    story.append(Paragraph("7. LLM-Based Executive Insight Generation", styles["heading"]))
    story.append(
        Paragraph(
            "Structured intelligence profiles are passed to a Gemini model under a "
            "system instruction that forbids invented metrics and requires every claim "
            "to trace back to a supplied value. Each briefing covers account risk "
            "level, revenue impact, behavioural segment interpretation, prioritised "
            "strategic recommendations with owner and timeframe, and a targeted "
            "renewal plus upsell play. Five briefings were generated for the accounts "
            "carrying the highest forecast revenue at risk and exported to "
            "reports/executive_insights.md.",
            styles["body"],
        )
    )
    if insight_audit:
        story.append(
            Paragraph(
                "<b>Factuality audit.</b> Every currency figure in every briefing is checked "
                "against the source profile, accepting monthly, quarterly and annual "
                "restatements and rounding within one percent. Result: "
                f"{insight_audit['briefings_fully_traceable']} of "
                f"{insight_audit['briefings_audited']} briefings are fully "
                f"traceable, with {insight_audit['figures_requiring_review']} of "
                f"{insight_audit['currency_figures_quoted']} quoted figures flagged "
                "for review. A flagged figure is a review candidate rather than proof "
                "of fabrication - the model may legitimately combine two profile "
                "values. Detail in reports/insight_audit.json.",
                styles["body"],
            )
        )

    story.append(
        Paragraph(
            "Implementation note: the originally specified gemini-pro model has been "
            "retired from the API. The pipeline probes current models at runtime and "
            "uses the first one the credential can serve.",
            styles["caption"],
        )
    )

    # --- 8. Limitations ---------------------------------------------------
    story.append(Paragraph("8. Limitations and Future Work", styles["heading"]))
    for text in [
        "<b>Residual leakage risk.</b> The four clearest temporal artefacts are now "
        "withheld and the cost measured (section 4.1.1), but other aggregates such as "
        "lifetime revenue also scale with how long an account survived. A production "
        "rebuild should compute every feature from a fixed observation window that "
        "closes before the prediction date.",
        "<b>Recall ceiling.</b> The leak-free model catches "
        f"{churn['recall']:.1%} of churners at "
        f"{churn['precision']:.1%} precision. Raising recall means lowering the "
        "decision threshold and accepting more false positives; the right operating "
        "point depends on the cost of a retention outreach versus a lost account.",
        "<b>Segmentation separation.</b> Silhouette of "
        f"{segmentation['silhouette']:.3f} falls short of the "
        f"{segmentation['silhouette_target']:.2f} target. Ward hierarchical "
        "clustering reproduces it almost exactly, and DBSCAN only clears the "
        "threshold by setting roughly a tenth of accounts aside as noise. Soft or "
        "model-based clustering, which assigns membership probabilities instead of "
        "hard labels, is the more promising direction for gradient-shaped "
        "behavioural data.",
        "<b>CRM risk flag as a predictor.</b> Renewal_Risk_Flag is the strongest "
        "SHAP driver and is itself a CRM-assigned risk rating, so the classifier is "
        "partly learning to reproduce an existing human or rules-based judgement. It "
        "is populated before renewal and is therefore legitimate at prediction time, "
        "but the headline ROC-AUC should not be read as wholly independent evidence "
        "of new signal.",
        "<b>Static snapshot.</b> The system scores a point-in-time extract. Production "
        "deployment needs scheduled re-scoring, drift monitoring and retraining "
        "triggers.",
        "<b>Insight validation.</b> The factuality audit checks currency figures only. "
        "Percentages, dates, named roles and the reasoning connecting them are not "
        "verified, and no automated check can assess whether a recommendation is "
        "commercially sound. Briefings remain a drafting aid for a human owner, not "
        "an unreviewed output.",
    ]:
        story.append(Paragraph(text, styles["bullet"], bulletText="\u2022"))

    # --- 9. Deliverables --------------------------------------------------
    story.append(Paragraph("9. Deliverables", styles["heading"]))
    story.append(
        make_table(
            [
                ["Artefact", "Location"],
                ["Preprocessing pipeline", "src/preprocessing.py"],
                ["Model training", "src/train_models.py"],
                ["Intelligence fusion layer", "src/fusion_layer.py"],
                ["LLM insight generation", "src/llm_insights.py"],
                ["PDF report generator", "src/generate_report.py"],
                ["Executed analysis notebook", "notebooks/EDA_and_Modeling.ipynb"],
                ["Report figures", "reports/figures/*.png"],
                ["Figure generation", "src/make_figures.py"],
                ["Scoring API (FastAPI)", "src/api.py"],
                ["Fusion layer quality report", "reports/fusion_quality.json"],
                ["LLM factuality audit", "reports/insight_audit.json"],
                ["Serialised models", "models/*.pkl"],
                ["Processed dataset", "data/processed/processed_customer_data.csv"],
                ["Customer intelligence profiles", "data/processed/customer_profiles.json"],
                ["Executive briefings", "reports/executive_insights.md"],
                ["Model metrics", "reports/model_metrics.json"],
                ["This report", "reports/Capstone_Final_Report.pdf"],
            ],
            styles,
            full_width,
        )
    )
    return story


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def load_optional_json(path) -> dict:
    """Load a JSON artefact, or return an empty dict if its stage has not run."""
    if not path.exists():
        logger.warning("%s missing - its section will be omitted", path.name)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs() -> dict:
    """Load every artefact the report renders."""
    return {
        "metrics": json.loads(METRICS_FILE.read_text(encoding="utf-8")),
        "fusion_quality": load_optional_json(FUSION_QUALITY_FILE),
        "insight_audit": load_optional_json(INSIGHT_AUDIT_FILE),
        "shap": pd.read_csv(SHAP_FILE),
        "fusion": pd.read_csv(FUSION_FILE),
        "clusters": pd.read_csv(CLUSTER_FILE, index_col=0),
    }


def generate_report() -> Path:
    """Render the capstone PDF report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    styles = build_styles()

    document = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Enterprise AI-Powered Customer Analytics Capstone Report",
        author="Customer Intelligence Team",
    )
    document.build(
        build_story(styles, data),
        onFirstPage=draw_page_furniture,
        onLaterPages=draw_page_furniture,
    )
    logger.info(
        "Wrote %s (%.1f KB)", OUTPUT_FILE, OUTPUT_FILE.stat().st_size / 1024
    )
    return OUTPUT_FILE


if __name__ == "__main__":
    generate_report()
