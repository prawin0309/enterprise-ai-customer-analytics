"""LLM-based executive insight generation.

Reads the unified customer intelligence profiles produced by ``fusion_layer``
and asks a Gemini model to translate each structured profile into an
executive-level briefing covering account risk, revenue impact, strategic
recommendations and targeted renewal / upsell offers.

The API key is read from the environment (never hard-coded):

    setx GEMINI_API_KEY "your-key-here"
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from pathlib import Path

import google.generativeai as genai

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

try:
    from paths import INSIGHTS_FILE as OUTPUT_FILE, PRIORITY_FILE, PROFILES_FILE
except ImportError:  # imported as ``src.llm_insights``
    from src.paths import (
        INSIGHTS_FILE as OUTPUT_FILE,
        PRIORITY_FILE,
        PROFILES_FILE,
    )

API_KEY_VARIABLES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")

# ``gemini-pro`` was retired from the v1beta API. Model listings also advertise
# models the key cannot actually call, so each candidate is probed with a live
# request and the first one that answers is used.
#
# Free-tier quota is tracked per model, so the list deliberately mixes tiers:
# when the flagship models are exhausted, the lite variants normally still have
# headroom and keep the pipeline running.
MODEL_CANDIDATES = (
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemma-4-31b-it",
)

# Errors that mean "this model is unusable right now, try the next one".
QUOTA_ERROR_NAMES = ("ResourceExhausted", "NotFound", "PermissionDenied")

INSIGHT_COUNT = 5
# Current Gemini models spend part of the output budget on internal reasoning
# tokens, so the ceiling is set well above the ~450-word target briefing.
GENERATION_CONFIG = {
    "temperature": 0.4,
    "top_p": 0.9,
    "max_output_tokens": 8192,
}

SYSTEM_INSTRUCTION = textwrap.dedent(
    """
    You are a senior Customer Success strategist for an enterprise SaaS CRM
    platform. You turn machine-learning outputs into concise, decision-ready
    briefings for executives.

    Rules:
    - Ground every statement in the numbers supplied in the profile.
    - Never invent metrics, dates, contacts or events that are not in the data.
    - Refer to the SHAP drivers by their business meaning, not the raw
      feature name.
    - Be specific and quantitative. No filler, no hedging.
    """
).strip()

PROMPT_TEMPLATE = textwrap.dedent(
    """
    Produce an executive briefing for the account below.

    Customer intelligence profile (JSON):
    ```json
    {profile_json}
    ```

    Structure the response in exactly these five markdown sections:

    ### 1. Account Risk Level
    State the risk band and churn probability, and explain what drives it using
    the top SHAP drivers.

    ### 2. Revenue Impact
    Quantify next-quarter forecast revenue, revenue at risk, and how this
    account compares to the portfolio using its percentile scores.

    ### 3. Behavioural Segment Interpretation
    Explain what the assigned cluster means commercially and how this account
    behaves relative to that segment.

    ### 4. Strategic Recommendations
    Give 3-4 prioritised, concrete actions with an owner role and a timeframe.

    ### 5. Targeted Renewal / Upsell Offer
    Recommend one renewal play and one upsell or cross-sell play, each with a
    rationale tied to the account's data.

    Keep the whole briefing under 450 words.
    """
).strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llm_insights")


# --------------------------------------------------------------------------- #
# Client setup
# --------------------------------------------------------------------------- #


def resolve_api_key() -> str:
    """Return the Gemini API key from the environment, or raise a clear error."""
    for variable in API_KEY_VARIABLES:
        key = os.environ.get(variable)
        if key:
            logger.info("Using API key from %s", variable)
            return key
    raise RuntimeError(
        "No Gemini API key found. Set one of "
        f"{', '.join(API_KEY_VARIABLES)} and re-run. "
        'PowerShell: setx GEMINI_API_KEY "your-key-here" (then restart the shell).'
    )


class ModelPool:
    """Serve generations from the first usable model, failing over on quota errors.

    Free-tier quota is per model and can run out mid-job, so the pool advances to
    the next candidate rather than aborting a partially generated report.
    """

    def __init__(self, candidates: tuple[str, ...]) -> None:
        self._candidates = list(candidates)
        self._index = 0
        self._model: genai.GenerativeModel | None = None
        self.failures: dict[str, str] = {}

    @property
    def model_name(self) -> str:
        """Name of the model currently serving requests."""
        return self._candidates[self._index]

    def _build(self, name: str) -> genai.GenerativeModel:
        return genai.GenerativeModel(
            model_name=name,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config=GENERATION_CONFIG,
        )

    def _advance(self) -> bool:
        """Move to the next candidate. Returns False when none are left."""
        self._index += 1
        self._model = None
        return self._index < len(self._candidates)

    def generate(self, prompt: str):
        """Generate a response, failing over while candidates remain."""
        while self._index < len(self._candidates):
            name = self._candidates[self._index]
            if self._model is None:
                self._model = self._build(name)
            try:
                return self._model.generate_content(prompt)
            except Exception as error:  # noqa: BLE001 - drives model failover
                error_name = type(error).__name__
                self.failures[name] = error_name
                if error_name not in QUOTA_ERROR_NAMES:
                    raise
                logger.warning("Model %s unusable (%s); failing over", name, error_name)
                if not self._advance():
                    break
        raise RuntimeError(
            "Every candidate model is unavailable or out of quota. "
            f"Failures: {self.failures}"
        )


def resolve_model_pool(api_key: str) -> ModelPool:
    """Configure the client and return a pool primed on a working model."""
    genai.configure(api_key=api_key)

    override = os.environ.get("GEMINI_MODEL")
    candidates = (override,) + MODEL_CANDIDATES if override else MODEL_CANDIDATES

    pool = ModelPool(candidates)
    pool.generate("Reply with the single word: ready")
    logger.info("Selected model: %s", pool.model_name)
    return pool


# --------------------------------------------------------------------------- #
# Profile selection
# --------------------------------------------------------------------------- #


def load_profiles(count: int = INSIGHT_COUNT) -> list[dict]:
    """Load the accounts to brief on, preferring the pre-ranked priority file."""
    if PRIORITY_FILE.exists():
        profiles = json.loads(PRIORITY_FILE.read_text(encoding="utf-8"))
        logger.info("Loaded %d priority accounts from %s", len(profiles), PRIORITY_FILE)
    else:
        profiles = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
        profiles = sorted(
            profiles,
            key=lambda p: p["revenue_forecast"]["revenue_at_risk_usd"],
            reverse=True,
        )
        logger.info("Loaded %d profiles from %s", len(profiles), PROFILES_FILE)
    return profiles[:count]


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate_insight(pool: ModelPool, profile: dict) -> str:
    """Generate one executive briefing for a single customer profile."""
    prompt = PROMPT_TEMPLATE.format(profile_json=json.dumps(profile, indent=2))
    response = pool.generate(prompt)

    # A MAX_TOKENS finish reason means the briefing was cut off mid-sentence.
    # The enum renders as an int, so compare by name rather than by str().
    finish_reason = response.candidates[0].finish_reason
    if getattr(finish_reason, "name", str(finish_reason)) != "STOP":
        logger.warning(
            "Customer %d: briefing finished with reason %s (possible truncation)",
            profile["customer_id"],
            finish_reason,
        )
    return response.text.strip()


def render_report(profiles: list[dict], insights: list[str], model_name: str) -> str:
    """Assemble the generated briefings into a single markdown report."""
    total_at_risk = sum(
        p["revenue_forecast"]["revenue_at_risk_usd"] for p in profiles
    )
    lines = [
        "# Executive Customer Intelligence Briefings",
        "",
        "Generated by the AI-powered customer intelligence system: churn "
        "classification, revenue regression and behavioural clustering fused "
        "into per-account profiles, then narrated by an LLM.",
        "",
        f"- **Model:** `{model_name}`",
        f"- **Accounts briefed:** {len(profiles)} (highest forecast revenue at risk)",
        f"- **Combined revenue at risk:** ${total_at_risk:,.0f}",
        "",
        "---",
        "",
    ]

    for index, (profile, insight) in enumerate(zip(profiles, insights), start=1):
        churn = profile["churn_prediction"]
        revenue = profile["revenue_forecast"]
        segment = profile["segmentation"]
        context = profile["account_context"]

        lines.extend(
            [
                f"## Briefing {index} — Customer {profile['customer_id']}",
                "",
                f"| Field | Value |",
                f"| --- | --- |",
                f"| Industry | {context.get('industry')} |",
                f"| Country | {context.get('country')} |",
                f"| Plan / Tier | {context.get('plan_type')} / "
                f"{context.get('account_tier')} |",
                f"| Churn probability | {churn['probability']:.1%} "
                f"({churn['risk_level']} risk) |",
                f"| Forecast next quarter | ${revenue['predicted_next_quarter_usd']:,.0f} |",
                f"| Revenue at risk | ${revenue['revenue_at_risk_usd']:,.0f} |",
                f"| Segment | {segment['cluster_label']} (cluster "
                f"{segment['cluster_id']}) |",
                "",
                insight,
                "",
                "---",
                "",
            ]
        )

    lines.append(
        "> Prompt design: each briefing is generated from the standardised JSON "
        "intelligence profile only. The system instruction constrains the model "
        "to the supplied metrics and forbids invented facts, so every claim is "
        "traceable to a model output or a CRM field."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_insight_generation() -> Path:
    """Generate executive briefings for the priority accounts and write them out."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    api_key = resolve_api_key()
    pool = resolve_model_pool(api_key)
    profiles = load_profiles()

    insights: list[str] = []
    for index, profile in enumerate(profiles, start=1):
        logger.info(
            "Generating briefing %d/%d for customer %d "
            "(churn %.1f%%, at risk $%.0f)",
            index,
            len(profiles),
            profile["customer_id"],
            profile["churn_prediction"]["probability"] * 100,
            profile["revenue_forecast"]["revenue_at_risk_usd"],
        )
        insights.append(generate_insight(pool, profile))

    report = render_report(profiles, insights, pool.model_name)
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    logger.info("Wrote %d briefings to %s", len(insights), OUTPUT_FILE)
    return OUTPUT_FILE


if __name__ == "__main__":
    run_insight_generation()
