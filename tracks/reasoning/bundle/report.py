"""
Markdown report generation for the analysis bundle pipeline.

Generates LLM_CONTEXT_SHIFT_REPORT.md covering all models evaluated.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from shared.endpoints import PROTOCOL_VERSION
from tracks.reasoning.endpoints import (
    CLASS_CORRESPONDENCE_T3,
    CONTEXT_SENSITIVITY_T3,
    CONTEXT_SPECIFICITY_T3,
    DIRECTIONAL_ALIGNMENT_T3,
)

_AGGREGATE_LABEL = "aggregate"


def _fmt(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(f) or math.isinf(f):
        return "N/A"
    return f"{f:.{decimals}f}"


def _fmt_p(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(f) or math.isinf(f):
        return "N/A"
    if f < 0.0001:
        return "< 0.0001"
    return f"{f:.4f}"


def _fmt_ci(lo: Any, hi: Any, decimals: int = 4) -> str:
    return f"[{_fmt(lo, decimals)}, {_fmt(hi, decimals)}]"


def _categories(category_type: str) -> str:
    """``"privacy categories"``, or ``"categories"`` with no category type."""
    return f"{category_type} categories" if category_type else "categories"


def _category_adjective(category_type: str) -> str:
    """``"privacy-category"``, or ``"category"`` with no category type."""
    return f"{category_type}-category" if category_type else "category"


def _interpretation_paragraph(
    model: str, flag: str, row: pd.Series, category_type: str = "privacy"
) -> str:
    h1 = _fmt(row.get("context_sensitivity_mean_abs_delta_correct"), 4)
    h1s = _fmt(row.get("context_sensitivity_mean_abs_delta_shuffled"), 4)
    h4 = _fmt(row.get("context_specificity_mean_alignment_difference"), 4)
    h4_ci = _fmt_ci(row.get("context_specificity_ci_low"), row.get("context_specificity_ci_high"))

    if flag == "no_context_sensitivity":
        return (
            f"**{model}**: The model shows no detectable context sensitivity "
            f"(context sensitivity mean |Δ| = {h1} ≤ ε). Adding patient context — whether correct "
            f"or shuffled — does not change the model's {_category_adjective(category_type)} scores. "
            "No conclusions about directional alignment, class-level correspondence or "
            "context specificity can be drawn."
        )
    if flag == "nonspecific_context_inflation":
        return (
            f"**{model}**: The model shows context sensitivity "
            f"(context sensitivity correct = {h1}, shuffled = {h1s}), but the context-specificity confidence "
            f"interval for the alignment difference includes zero ({h4_ci}). "
            "This means correct context and shuffled context produce comparable "
            "directional movement. The dominant pattern is **nonspecific context "
            "inflation**: the model reacts to any extra text, not specifically to "
            "the correct patient's information."
        )
    if flag == "modest_patient_specific_alignment":
        return (
            f"**{model}**: The model shows context sensitivity and correct context "
            f"aligns better than shuffled (context specificity mean diff = {h4}, 95% CI = {h4_ci}). "
            "The CI excludes zero, but the effect is modest (< 0.05). This supports "
            "a **modest patient-specific context-alignment signal**."
        )
    if flag == "strong_patient_specific_alignment":
        return (
            f"**{model}**: The model shows substantial context sensitivity and correct "
            f"context produces meaningfully stronger directional alignment than shuffled "
            f"context (context specificity mean diff = {h4}, 95% CI = {h4_ci}). This supports a "
            "**strong patient-specific context-alignment signal**."
        )
    return (
        f"**{model}**: Results could not be fully computed. Check the validation "
        "summary for details."
    )


def generate_markdown_report(
    hypothesis_summary: pd.DataFrame,
    h1_df: pd.DataFrame,
    h2_df: pd.DataFrame,
    h3_effects_df: pd.DataFrame,
    h3_corr_df: pd.DataFrame,
    h4_df: pd.DataFrame,
    model_comparison: pd.DataFrame,
    validation_summary: pd.DataFrame,
    output_path: str,
    run_params: Optional[Dict[str, Any]] = None,
    categories: Optional[List[str]] = None,
    category_type: str = "privacy",
) -> None:
    """Write LLM_CONTEXT_SHIFT_REPORT.md to *output_path*.

    Parameters
    ----------
    hypothesis_summary:
        From ``make_hypothesis_summary``.
    h1_df, h2_df, h3_effects_df, h3_corr_df, h4_df:
        Per-model analysis DataFrames (concatenated across models).
    model_comparison:
        From ``make_model_comparison_summary``.
    validation_summary:
        From ``validate_inputs``.
    output_path:
        Destination .md file path.
    run_params:
        Dict of CLI / run parameters (seed, epsilon, n_bootstrap, etc.).
    categories:
        Canonical category names for per-category tables.
    category_type:
        Word placed before "categories" in the report text (the reasoning
        config's ``prompt.category_type``; ``""`` for none).
    """
    category_type = category_type.strip()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cats = categories or []
    params = run_params or {}
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    models = list(hypothesis_summary["model"].unique()) if not hypothesis_summary.empty else []

    sections = [
        _section_title(timestamp),
        _section_objective(category_type),
        _section_data_pairing(hypothesis_summary, cats, category_type),
        _section_models(models),
        _section_validation(validation_summary),
        _section_h1(h1_df, cats),
        _section_h2(h2_df, cats),
        _section_h3(h3_effects_df, h3_corr_df, cats, category_type),
        _section_h4(h4_df, cats),
        _section_model_comparison(model_comparison),
        _section_interpretation(hypothesis_summary, category_type),
        _section_limitations(),
        _section_reproducibility(params),
    ]

    content = "\n\n".join(s for s in sections if s) + "\n"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _section_title(timestamp: str) -> str:
    return (
        f"# LLM Context-Shift Analysis Report\n\n**Generated:** {timestamp}  \n"
        f"**Protocol version:** {PROTOCOL_VERSION}"
    )


def _section_objective(category_type: str = "privacy") -> str:
    return (
        "## Objective\n\n"
        f"This report evaluates whether LLMs change their {_category_adjective(category_type)} judgments when "
        "patient context is added, and whether those changes align with physician "
        "survey-to-interview judgment shifts more than shuffled (mismatched) context does.\n\n"
        f"**Primary endpoint: {CONTEXT_SPECIFICITY_T3.heading}** — the difference in directional alignment between correct "
        "and shuffled context, restricted to items where physicians shifted their judgment. "
        "A positive context-specificity result with a CI excluding zero provides evidence of patient-specific "
        "context alignment.\n\n"
        "This analysis does **not** test prediction accuracy. Brier score, F1, AUROC, and "
        "final-label accuracy are not reported. The physician survey-to-interview shift is the "
        "human reference delta; correct-context LLM effects must always be interpreted relative "
        "to shuffled-context effects. High context sensitivity with weak context specificity "
        "indicates nonspecific context "
        "inflation, not patient-specific alignment."
    )


def _section_data_pairing(
    hypothesis_summary: pd.DataFrame, cats: list[str], category_type: str = "privacy"
) -> str:
    n_patients = int(hypothesis_summary["n_patients"].max()) if not hypothesis_summary.empty and "n_patients" in hypothesis_summary.columns else "—"
    n_cats = len(cats) if cats else 10
    return (
        "## Data and Pairing\n\n"
        f"- **Patients:** {n_patients}\n"
        f"- **{_categories(category_type).capitalize()}:** {n_cats}\n"
        "- **Physician conditions:** Survey (context-free) and Interview (context-aware)\n"
        "- **LLM conditions:** context-free, correct-context, shuffled-context\n"
        "- **Delta physician:** interview_consensus − survey_consensus per patient × item × category\n"
        "- **Delta LLM correct:** correct_context_score − context_free_score\n"
        "- **Delta LLM shuffled:** shuffled_context_score − context_free_score\n\n"
        "Cells are patient × item × category tuples. Alignment is defined as "
        "sign(Δ_physician) × Δ_LLM and is only computed where Δ_physician ≠ 0."
    )


def _section_models(models: list[str]) -> str:
    if not models:
        return "## Models Evaluated\n\nNo models found."
    items = "\n".join(f"- `{m}`" for m in models)
    return f"## Models Evaluated\n\n{items}"


def _section_validation(validation_summary: pd.DataFrame) -> str:
    if validation_summary is None or validation_summary.empty:
        return "## Validation Summary\n\nNo validation results available."
    header = "| Check | Description | Status | Count | Notes |"
    sep = "|---|---|---|---|---|"
    rows_md = []
    for _, row in validation_summary.iterrows():
        rows_md.append(
            f"| {row.get('check_id', '')} | {row.get('description', '')} | "
            f"{row.get('status', '')} | {row.get('count', '')} | {row.get('notes', '')} |"
        )
    return "## Validation Summary\n\n" + "\n".join([header, sep] + rows_md)


def _section_h1(h1_df: pd.DataFrame, cats: list[str]) -> str:
    lines = [
        f"## {CONTEXT_SENSITIVITY_T3.heading}\n",
        "**Question:** Does the correct patient's context change the LLM's judgments more than another patient's context does?\n",
        "Computed for both correct and shuffled contexts. Almost any added text moves the scores, "
        "so the test is whether the correct patient's context moves them more than the shuffled one: "
        "mean |Δ_correct| − |Δ_shuffled|, with a patient-cluster bootstrap CI and a one-sided "
        "patient-cluster sign-flip p-value.\n",
    ]

    for model, grp in (h1_df.groupby("model") if h1_df is not None and not h1_df.empty else []):
        agg = grp[grp["category"] == _AGGREGATE_LABEL]
        if agg.empty:
            continue
        r = agg.iloc[0]
        lines.append(f"### Model: `{model}`\n")
        lines.append(f"- **Mean |Δ_LLM| correct:** {_fmt(r.get('mean_abs_delta_correct'))} "
                     f"95% CI {_fmt_ci(r.get('ci_low_correct'), r.get('ci_high_correct'))}")
        lines.append(f"- **Mean |Δ_LLM| shuffled:** {_fmt(r.get('mean_abs_delta_shuffled'))} "
                     f"95% CI {_fmt_ci(r.get('ci_low_shuffled'), r.get('ci_high_shuffled'))}")
        lines.append(f"- **Difference (correct − shuffled):** {_fmt(r.get('mean_abs_delta_difference'))} "
                     f"95% CI {_fmt_ci(r.get('ci_low_difference'), r.get('ci_high_difference'))}, "
                     f"sign-flip p = {_fmt_p(r.get('paired_permutation_p'))}")
        lines.append(f"- **Proportion changed (correct):** {_fmt(r.get('proportion_changed_correct'))}")
        lines.append(f"- **Proportion changed (shuffled):** {_fmt(r.get('proportion_changed_shuffled'))}\n")

        # Per-category table
        lines.append("| Category | Mean \\|Δ\\| Correct | CI Correct | Mean \\|Δ\\| Shuffled | CI Shuffled |")
        lines.append("|---|---|---|---|---|")
        for cat in cats:
            cat_row = grp[grp["category"] == cat]
            if cat_row.empty:
                continue
            cr = cat_row.iloc[0]
            lines.append(
                f"| {cat} | {_fmt(cr.get('mean_abs_delta_correct'))} | "
                f"{_fmt_ci(cr.get('ci_low_correct'), cr.get('ci_high_correct'))} | "
                f"{_fmt(cr.get('mean_abs_delta_shuffled'))} | "
                f"{_fmt_ci(cr.get('ci_low_shuffled'), cr.get('ci_high_shuffled'))} |"
            )
        lines.append("")

    return "\n".join(lines)


def _section_h2(h2_df: pd.DataFrame, cats: list[str]) -> str:
    lines = [
        f"## {DIRECTIONAL_ALIGNMENT_T3.heading}\n",
        "**Question:** When physicians shift after context, does the LLM move in the same direction?\n",
        "Restricted to cells where Δ_physician ≠ 0. LLM deltas below epsilon count as no movement; "
        "sign agreement counts only cells where the LLM moved, and the share that did not is "
        "reported separately. Chance agreement is not 0.5. The reference is a permutation "
        "null that shuffles each patient's model deltas among that patient's items.\n",
    ]

    for model, grp in (h2_df.groupby("model") if h2_df is not None and not h2_df.empty else []):
        agg = grp[grp["category"] == _AGGREGATE_LABEL]
        if agg.empty:
            continue
        r = agg.iloc[0]
        lines.append(f"### Model: `{model}`\n")
        lines.append(f"- **Mean alignment (correct):** {_fmt(r.get('mean_alignment_correct'))} "
                     f"95% CI {_fmt_ci(r.get('ci_low_correct'), r.get('ci_high_correct'))}")
        lines.append(f"- **Mean alignment (shuffled):** {_fmt(r.get('mean_alignment_shuffled'))} "
                     f"95% CI {_fmt_ci(r.get('ci_low_shuffled'), r.get('ci_high_shuffled'))}")
        lines.append(f"- **Sign agreement (correct):** {_fmt(r.get('sign_agreement_correct'))}")
        lines.append(f"- **Sign agreement (shuffled):** {_fmt(r.get('sign_agreement_shuffled'))}")
        lines.append(f"- **Chance agreement (permutation null):** {_fmt(r.get('null_sign_agreement_correct'))}, "
                     f"p = {_fmt_p(r.get('permutation_p_correct'))}")
        lines.append(f"- **Shift cells with no LLM movement (left out of sign agreement):** "
                     f"correct {_fmt(r.get('unchanged_rate_correct'))}, "
                     f"shuffled {_fmt(r.get('unchanged_rate_shuffled'))}\n")

        lines.append("| Category | Sign Agree Correct | Sign Agree Shuffled | n Shift Cells |")
        lines.append("|---|---|---|---|")
        for cat in cats:
            cat_row = grp[grp["category"] == cat]
            if cat_row.empty:
                continue
            cr = cat_row.iloc[0]
            lines.append(
                f"| {cat} | {_fmt(cr.get('sign_agreement_correct'))} | "
                f"{_fmt(cr.get('sign_agreement_shuffled'))} | "
                f"{cr.get('n_shift_cells', 'N/A')} |"
            )
        lines.append("")

    return "\n".join(lines)


def _section_h3(
    h3_effects: pd.DataFrame,
    h3_corr: pd.DataFrame,
    cats: list[str],
    category_type: str = "privacy",
) -> str:
    lines = [
        f"## {CLASS_CORRESPONDENCE_T3.heading}\n",
        f"**Question:** Are the same {_categories(category_type)} context-sensitive for physicians and the LLM?\n",
        "Pearson and Spearman correlations between class-level mean physician deltas and LLM deltas. "
        "P-values from permutation tests (permute category labels).\n",
    ]

    if h3_corr is not None and not h3_corr.empty:
        lines.append("### Correlation Summary\n")
        lines.append("| Model | Pearson Correct | Spearman Correct | Pearson Shuffled | "
                     "Spearman Shuffled | Δ Pearson | Δ Spearman | p Pearson | p Spearman |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, row in h3_corr.iterrows():
            lines.append(
                f"| {row.get('model','')} | {_fmt(row.get('pearson_correct'))} | "
                f"{_fmt(row.get('spearman_correct'))} | {_fmt(row.get('pearson_shuffled'))} | "
                f"{_fmt(row.get('spearman_shuffled'))} | "
                f"{_fmt(row.get('pearson_correct_minus_shuffled'))} | "
                f"{_fmt(row.get('spearman_correct_minus_shuffled'))} | "
                f"{_fmt_p(row.get('permutation_p_pearson_correct'))} | "
                f"{_fmt_p(row.get('permutation_p_spearman_correct'))} |"
            )
        lines.append("")

    if h3_effects is not None and not h3_effects.empty:
        lines.append("### Class-Level Mean Deltas\n")
        for model, grp in h3_effects.groupby("model"):
            lines.append(f"#### Model: `{model}`\n")
            lines.append("| Category | Mean Δ Physician | Mean \\|Δ\\| Physician | "
                         "Mean Δ LLM Correct | Mean Δ LLM Shuffled |")
            lines.append("|---|---|---|---|---|")
            for cat in cats:
                cat_row = grp[grp["category"] == cat]
                if cat_row.empty:
                    continue
                cr = cat_row.iloc[0]
                lines.append(
                    f"| {cat} | {_fmt(cr.get('mean_delta_physician'))} | "
                    f"{_fmt(cr.get('mean_abs_delta_physician'))} | "
                    f"{_fmt(cr.get('mean_delta_llm_correct'))} | "
                    f"{_fmt(cr.get('mean_delta_llm_shuffled'))} |"
                )
            lines.append("")

    return "\n".join(lines)


def _section_h4(h4_df: pd.DataFrame, cats: list[str]) -> str:
    lines = [
        f"## {CONTEXT_SPECIFICITY_T3.heading}\n",
        "**Question:** Is the LLM responding to the correct patient context, "
        "or merely to the presence of extra context text?\n",
        "**Primary endpoint.** Restricted to cells where Δ_physician ≠ 0. "
        "Mean alignment difference = alignment_correct − alignment_shuffled. "
        "Positive value with CI > 0 indicates patient-specific alignment.\n",
    ]

    for model, grp in (h4_df.groupby("model") if h4_df is not None and not h4_df.empty else []):
        agg = grp[grp["category"] == _AGGREGATE_LABEL]
        if agg.empty:
            continue
        r = agg.iloc[0]
        lines.append(f"### Model: `{model}`\n")
        lines.append(f"- **Mean alignment difference (correct − shuffled):** "
                     f"{_fmt(r.get('mean_alignment_difference'))}")
        lines.append(f"- **95% CI:** {_fmt_ci(r.get('ci_low_difference'), r.get('ci_high_difference'))}")
        lines.append(f"- **Patient-cluster sign-flip p-value:** {_fmt_p(r.get('paired_permutation_p'))}\n")

        lines.append("| Category | Mean Diff | CI | n Shift Cells | p-value |")
        lines.append("|---|---|---|---|---|")
        for cat in cats:
            cat_row = grp[grp["category"] == cat]
            if cat_row.empty:
                continue
            cr = cat_row.iloc[0]
            lines.append(
                f"| {cat} | {_fmt(cr.get('mean_alignment_difference'))} | "
                f"{_fmt_ci(cr.get('ci_low_difference'), cr.get('ci_high_difference'))} | "
                f"{cr.get('n_shift_cells', 'N/A')} | "
                f"{_fmt_p(cr.get('paired_permutation_p'))} |"
            )
        lines.append("")

    return "\n".join(lines)


def _section_model_comparison(model_comparison: pd.DataFrame) -> str:
    if model_comparison is None or model_comparison.empty:
        return "## Model Comparison\n\nNo models to compare."
    lines = [
        "## Model Comparison\n",
        "Models ranked by context-specificity mean alignment difference (descending). "
        "Context specificity is the primary patient-specific context endpoint.\n",
    ]
    cols = ["rank", "model", "context_specificity_mean_alignment_difference", "context_specificity_ci_low", "context_specificity_ci_high",
            "context_specificity_p_value", "context_sensitivity_mean_abs_delta_correct", "context_sensitivity_mean_abs_delta_shuffled",
            "interpretation_flag"]
    avail = [c for c in cols if c in model_comparison.columns]
    header = "| " + " | ".join(avail) + " |"
    sep = "|" + "|".join(["---"] * len(avail)) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in model_comparison.iterrows():
        vals = []
        for c in avail:
            v = row[c]
            if c in ("rank",):
                vals.append(str(v))
            elif c in ("model", "interpretation_flag"):
                vals.append(str(v))
            elif "p_value" in c or "p_" in c:
                vals.append(_fmt_p(v))
            else:
                vals.append(_fmt(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _section_interpretation(
    hypothesis_summary: pd.DataFrame, category_type: str = "privacy"
) -> str:
    if hypothesis_summary is None or hypothesis_summary.empty:
        return "## Interpretation\n\nNo results to interpret."
    lines = ["## Interpretation\n"]
    for _, row in hypothesis_summary.iterrows():
        model = str(row.get("model", "Unknown"))
        flag = str(row.get("interpretation_flag", "failed_validation"))
        lines.append(_interpretation_paragraph(model, flag, row, category_type))
        lines.append("")
    return "\n".join(lines)


def _section_limitations() -> str:
    return (
        "## Limitations\n\n"
        "1. **Small cohort.** Results are scoped to the patients and physicians in this dataset. "
        "Generalisability to broader populations is unknown.\n"
        "2. **No prediction accuracy claims.** Track 3 measures context sensitivity and "
        "directional alignment only. It does not evaluate classification performance.\n"
        "3. **Bootstrap CI validity.** Patient-cluster bootstrap CIs assume exchangeability of "
        "patient clusters. With small patient counts, bootstrap distributions may be coarse.\n"
        "4. **Shuffled context control.** The shuffled-context condition uses a single random "
        "draw per patient-item pair. Results may differ with alternative shuffling strategies.\n"
        "5. **LLM temperature.** Results may differ with stochastic sampling (temperature > 0).\n"
        "6. **Context specificity.** High context sensitivity with weak context specificity "
        "indicates nonspecific context inflation. "
        "Context specificity is the primary endpoint; context sensitivity alone cannot "
        "distinguish patient-specific from generic responses to context."
    )


def _section_reproducibility(params: dict) -> str:
    lines = [
        "## Reproducibility\n",
        "All analyses are fully deterministic given the parameters below.\n",
    ]
    for k, v in params.items():
        lines.append(f"- **{k}:** {v}")
    if not params:
        lines.append("- Run parameters not recorded.")
    return "\n".join(lines)
