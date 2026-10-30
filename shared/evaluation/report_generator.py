"""
Report Generator for Track 3 (reasoning).

Produces a single markdown file per model run using Python string formatting
only (no external templating library).


Canonical location: shared/evaluation/report_generator.py
"""

# Terminology mapping (canonical terms):
#   physician / human rater → reference_observer
#   Patient                 → context_entity_id
#   Item / input_text       → task_instance
#   class / category        → output_dimension
#   model_id / llm_*        → candidate_id
#   physician_survey/survey → reference_context_free
#   physician_interview/interview → reference_correct_context
#   delta_physician/human_shift   → delta_reference

from __future__ import annotations

import math
import os
from typing import Any

from shared.endpoints import (
    CLASS_CORRESPONDENCE,
    CONTEXT_SENSITIVITY,
    CONTEXT_SPECIFICITY,
    DIRECTIONAL_ALIGNMENT,
    PROTOCOL_VERSION,
    Endpoint,
)
from shared.label_space import DEFAULT_LABEL_SPACE, LabelSpace

# ---------------------------------------------------------------------------
# Default category names (canonical order)
# ---------------------------------------------------------------------------

DEFAULT_CATEGORY_NAMES: list[str] = DEFAULT_LABEL_SPACE.keys()


def _fmt(value: Any, decimals: int = 4) -> str:
    """Format a numeric value, returning 'N/A' for NaN/None."""
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(f) or math.isinf(f):
        return "N/A"
    return f"{f:.{decimals}f}"


def _fmt_ci(lo: Any, hi: Any, decimals: int = 4) -> str:
    """Format a confidence interval as '[lo, hi]'."""
    return f"[{_fmt(lo, decimals)}, {_fmt(hi, decimals)}]"


def _fmt_p(value: Any) -> str:
    """Format a p-value with 4 decimal places, or 'N/A'."""
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


class ReportGenerator:
    """Generates a markdown analysis report for one model run.

    Parameters
    ----------
    h1_result:
        Dict returned by ``HypothesisAnalyzer.compute_h1``.
        Expected keys: ``mean_abs_delta``, ``ci_lower``, ``ci_upper``,
        ``per_category`` (dict of per output_dimension dicts with
        ``mean_abs_delta``, ``ci_lower``, ``ci_upper``), and optionally the
        shuffled-context reference: ``mean_abs_delta_shuffled``,
        ``mean_abs_delta_difference`` with its CI, ``p_value``,
        ``zero_delta_rate`` and ``zero_delta_rate_shuffled``.
    h2_result:
        Dict returned by ``HypothesisAnalyzer.compute_h2``.
        Expected keys: ``mean_alignment``, ``ci_lower_alignment``,
        ``ci_upper_alignment``, ``sign_agreement_rate``,
        ``ci_lower_sign_agree``, ``ci_upper_sign_agree``, and optionally
        ``null_sign_agreement_rate``, ``p_value``, ``zero_delta_rate`` and
        the ``*_shuffled`` rates.
    h3_result:
        Dict returned by ``HypothesisAnalyzer.compute_h3``.
        Expected keys: ``pearson_r``, ``p_value``,
        ``mean_delta_physician``, ``mean_delta_llm``.
    h4_result:
        Dict returned by ``HypothesisAnalyzer.compute_h4``.
        Expected keys: ``mean_diff``, ``ci_lower``, ``ci_upper``,
        ``p_value``.
    model_id:
        candidate_id — Hugging Face model identifier (e.g.
        ``"meta-llama/Llama-3.1-8B-Instruct"``).
    run_timestamp:
        ISO 8601 timestamp string for the run.
    category_names:
        output_dimension keys, matching the keys of the per-category
        results. Ignored when ``label_space`` is given. Defaults to the keys
        of the ten SHARES categories.
    h2_per_category:
        Optional dict mapping output_dimension name → dict with
        ``sign_agreement_rate``, ``ci_lower``, ``ci_upper``.
        If ``None``, the per-category directional-alignment section notes that the
        breakdown is not available.
    label_space:
        Optional :class:`~shared.label_space.LabelSpace`. Results are looked
        up by its keys and tables show its display names.
    category_type:
        Word placed before "categories" in the report text (the reasoning
        config's ``prompt.category_type``; ``""`` for none).
    legacy_aliases:
        Optional endpoint name → old hypothesis number (e.g.
        ``{"context_sensitivity": "H1"}``), shown in headings as
        "(formerly H1)".
    """

    def __init__(
        self,
        h1_result: dict[str, Any],
        h2_result: dict[str, Any],
        h3_result: dict[str, Any],
        h4_result: dict[str, Any],
        model_id: str,
        run_timestamp: str,
        category_names: list[str] | None = None,
        h2_per_category: dict[str, dict[str, Any]] | None = None,
        is_dry_run: bool = False,
        n_context_entities: int | None = None,
        n_task_pairs: int | None = None,
        label_space: LabelSpace | None = None,
        category_type: str = "privacy",
        legacy_aliases: dict[str, str] | None = None,
    ) -> None:
        self._h1 = h1_result
        self._h2 = h2_result
        self._h3 = h3_result
        self._h4 = h4_result
        self._model_id = model_id  # candidate_id
        self._run_timestamp = run_timestamp
        if label_space is not None:
            self._category_names: list[str] = label_space.keys()
            self._category_labels: list[str] = label_space.display_names()
        else:
            self._category_names = (
                list(category_names) if category_names is not None else DEFAULT_CATEGORY_NAMES
            )
            self._category_labels = list(self._category_names)
        self._h2_per_category = h2_per_category
        self._category_type = category_type.strip()
        self._legacy_aliases = dict(legacy_aliases or {})
        self._is_dry_run = is_dry_run
        self._n_context_entities = n_context_entities
        self._n_task_pairs = n_task_pairs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, output_path: str) -> None:
        """Write the markdown report to *output_path*.

        Creates parent directories if they do not exist.

        Parameters
        ----------
        output_path:
            Filesystem path for the output ``.md`` file.
        """
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        content = self._build_report()
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_report(self) -> str:
        sections = [
            self._section_header(),
            self._section_methods(),
            self._section_results(),
            self._section_summary_table(),
            self._section_limitations(),
        ]
        return "\n\n".join(sections) + "\n"

    # --- Cohort description -----------------------------------------

    def _entity_phrase(self) -> str:
        if self._n_context_entities is None:
            return "patients"
        return f"**{self._n_context_entities} patients**"

    def _heading(self, endpoint: Endpoint) -> str:
        """Endpoint title, with its legacy hypothesis number if one is known."""
        alias = self._legacy_aliases.get(endpoint.name)
        return f"{endpoint.title} (formerly {alias})" if alias else endpoint.title

    def _categories(self) -> str:
        """``"privacy categories"``, or ``"categories"`` with no category type."""
        return f"{self._category_type} categories" if self._category_type else "categories"

    def _category_adjective(self) -> str:
        """``"privacy-category"``, or ``"category"`` with no category type."""
        return f"{self._category_type}-category" if self._category_type else "category"

    def _cohort_sentence(self) -> str:
        if self._n_context_entities is None:
            return ""
        sentence = f"The dataset comprises {self._entity_phrase()} (context_entity_id)"
        if self._n_task_pairs is not None:
            sentence += f" and **{self._n_task_pairs} patient-item pairs**"
        return sentence + ". "

    # --- Header -------------------------------------------------------

    def _section_header(self) -> str:
        lines = []
        if self._is_dry_run:
            lines.append(
                "> **DRY RUN — Mock data. Results are not scientifically "
                "interpretable. This report was generated with deterministic "
                "mock LLM responses for pipeline validation only.**\n"
            )
        lines.append(
            "# Track 3 (Reasoning) Context-Shift Report\n\n"
            f"**Model:** {self._model_id}  \n"
            f"**Run timestamp:** {self._run_timestamp}  \n"
            f"**Protocol version:** {PROTOCOL_VERSION}"
        )
        return "\n".join(lines)

    # --- Methods ------------------------------------------------------

    def _section_methods(self) -> str:
        return (
            "## Methods\n\n"
            "### Dataset\n\n"
            "Track 3 uses a paired dataset of EHR items evaluated by physicians "
            "in two conditions: a context-free survey phase (reference_context_free) "
            "and a context-aware interview phase (reference_correct_context). "
            f"{self._cohort_sentence()}"
            "Reference observers assessed items (task_instance) across "
            f"{len(self._category_names)} {self._categories()} (output_dimension), yielding "
            "paired consensus labels (Physician_Survey_Consensus and "
            "Physician_Interview_Consensus). The physician judgment delta "
            "(delta_reference) is defined as the interview consensus minus the survey "
            "consensus.\n\n"
            "### LLM Evaluation\n\n"
            f"The LLM under evaluation is **{self._model_id}** (candidate_id). "
            "Each EHR item (task_instance) is presented to the LLM under three "
            "conditions:\n\n"
            "1. **Context-free:** item text only, no patient clinical context.\n"
            "2. **Correct-context:** item text with the correct patient's clinical "
            "snapshot (summary, medical history, allergies, medication history, "
            "social history, labs, radiology, and procedures).\n"
            "3. **Shuffled-context:** item text with a randomly selected *different* "
            "patient's clinical snapshot (control condition).\n\n"
            "The LLM outputs a JSON object with probability scores (0–1) for each of "
            f"the {self._categories()} (output_dimension). Delta_LLM_Correct is "
            "defined as the correct-context score minus the context-free score. "
            "Delta_LLM_Shuffled is defined as the shuffled-context score minus the "
            "context-free score.\n\n"
            "### Statistical Procedures\n\n"
            "All confidence intervals are 95% patient-cluster bootstrap CIs "
            "(1 000 resamples) that resample entire patient clusters "
            "(context_entity_id) with replacement "
            "to respect within-patient correlation. Permutation tests use 10 000 "
            "permutations. P-values are one-sided (fraction of permuted statistics "
            "≥ observed statistic). Context sensitivity is tested against the "
            "shuffled-context condition (does the correct patient's context move "
            "the scores more than another patient's?), and context specificity "
            "compares alignment under the two contexts. Both use a patient-cluster "
            "sign-flip test: each patient's cells are flipped together, and with "
            "few patients all sign patterns are enumerated, so the smallest "
            "attainable p-value is 2^−(number of patients). Directional "
            "alignment is tested against a permutation null that shuffles each "
            "patient's model deltas among that patient's items. Sign agreement "
            "counts only cells where the model moved; how often it did not move "
            "is reported separately. Chance agreement is this null's mean, not "
            "0.5, because the model and the observers may each tend to move one "
            "way.\n\n"
            "This report does **not** include Brier score, F1, AUROC, or any measure "
            "of prediction accuracy. The endpoints are context sensitivity, "
            "directional alignment, class-level correspondence, and correct-versus-"
            "shuffled control."
        )

    # --- Results ------------------------------------------------------

    def _section_results(self) -> str:
        parts = [
            "## Results",
            self._subsection_h1(),
            self._subsection_h2(),
            self._subsection_h3(),
            self._subsection_h4(),
        ]
        return "\n\n".join(parts)

    def _subsection_h1(self) -> str:
        h1 = self._h1
        point = _fmt(h1.get("mean_abs_delta"))
        ci = _fmt_ci(h1.get("ci_lower"), h1.get("ci_upper"))

        lines = [
            f"### {self._heading(CONTEXT_SENSITIVITY)}\n",
            f"**Hypothesis:** The LLM (candidate_id) changes its {self._category_adjective()} "
            "scores (output_dimension) more when the correct patient's context "
            "(context_entity_id) is added than when another patient's (shuffled) "
            "context is (mean |Delta_LLM_Correct| − mean |Delta_LLM_Shuffled| > 0). "
            "Almost any added text moves the scores, so the shuffled context, "
            "not zero, is the reference.\n",
            f"- **Mean absolute delta, correct context:** {point} (95% CI {ci})  ",
            f"- **Mean absolute delta, shuffled context:** "
            f"{_fmt(h1.get('mean_abs_delta_shuffled'))} "
            f"(95% CI {_fmt_ci(h1.get('ci_lower_shuffled'), h1.get('ci_upper_shuffled'))})  ",
            f"- **Difference (correct − shuffled):** "
            f"{_fmt(h1.get('mean_abs_delta_difference'))} "
            f"(95% CI {_fmt_ci(h1.get('ci_lower_difference'), h1.get('ci_upper_difference'))})  ",
            f"- **Patient-cluster sign-flip p-value:** {_fmt_p(h1.get('p_value'))}  ",
            f"- **Cells with zero delta:** correct {_fmt(h1.get('zero_delta_rate'))}, "
            f"shuffled {_fmt(h1.get('zero_delta_rate_shuffled'))}\n",
            "#### Per-Category Breakdown\n",
            "| Category | Mean Absolute Delta | 95% CI | Shuffled | Correct − Shuffled |",
            "|---|---|---|---|---|",
        ]

        per_cat: dict[str, Any] = h1.get("per_category", {})
        for cat, label in zip(self._category_names, self._category_labels):
            cat_data = per_cat.get(cat, {})
            cat_point = _fmt(cat_data.get("mean_abs_delta"))
            cat_ci = _fmt_ci(cat_data.get("ci_lower"), cat_data.get("ci_upper"))
            cat_s = _fmt(cat_data.get("mean_abs_delta_shuffled"))
            cat_d = _fmt(cat_data.get("mean_abs_delta_difference"))
            lines.append(f"| {label} | {cat_point} | {cat_ci} | {cat_s} | {cat_d} |")

        return "\n".join(lines)

    def _subsection_h2(self) -> str:
        h2 = self._h2
        mean_align = _fmt(h2.get("mean_alignment"))
        ci_align = _fmt_ci(h2.get("ci_lower_alignment"), h2.get("ci_upper_alignment"))
        sign_rate = _fmt(h2.get("sign_agreement_rate"))
        ci_sign = _fmt_ci(h2.get("ci_lower_sign_agree"), h2.get("ci_upper_sign_agree"))

        lines = [
            f"### {self._heading(DIRECTIONAL_ALIGNMENT)}\n",
            "**Hypothesis:** LLM context-induced deltas are directionally aligned "
            "with reference_observer judgment deltas (delta_reference; restricted "
            "to cells where delta_reference ≠ 0).\n",
            f"- **Mean alignment score:** {mean_align}  ",
            f"- **95% CI (alignment):** {ci_align}  ",
            f"- **Sign agreement rate:** {sign_rate}  ",
            f"- **95% CI (sign agreement):** {ci_sign}  ",
            f"- **Chance agreement (within-patient permutation null):** "
            f"{_fmt(h2.get('null_sign_agreement_rate'))}  ",
            f"- **Permutation p-value:** {_fmt_p(h2.get('p_value'))}  ",
            f"- **Sign agreement rate, shuffled context:** "
            f"{_fmt(h2.get('sign_agreement_rate_shuffled'))}  ",
            f"- **Mean alignment, shuffled context:** "
            f"{_fmt(h2.get('mean_alignment_shuffled'))}  ",
            f"- **Shift cells with zero model delta (left out of sign agreement):** "
            f"correct {_fmt(h2.get('zero_delta_rate'))}, "
            f"shuffled {_fmt(h2.get('zero_delta_rate_shuffled'))}\n",
            "#### Per-Category Sign Agreement Rate\n",
        ]

        if self._h2_per_category is not None:
            lines += [
                "| Category | Sign Agreement Rate | 95% CI |",
                "|---|---|---|",
            ]
            for cat, label in zip(self._category_names, self._category_labels):
                cat_data = self._h2_per_category.get(cat, {})
                rate = _fmt(cat_data.get("sign_agreement_rate"))
                cat_ci = _fmt_ci(cat_data.get("ci_lower"), cat_data.get("ci_upper"))
                lines.append(f"| {label} | {rate} | {cat_ci} |")
        else:
            lines.append(
                "*Per-category directional-alignment breakdown not available in the current "
                "implementation. It is computed on the aggregate set of non-zero "
                "delta_reference cells.*"
            )

        return "\n".join(lines)

    def _subsection_h3(self) -> str:
        h3 = self._h3
        r_val = _fmt(h3.get("pearson_r"))
        p_val = _fmt_p(h3.get("p_value"))

        # Class-level delta table
        mean_dp = h3.get("mean_delta_physician")
        mean_dl = h3.get("mean_delta_llm")

        lines = [
            f"### {self._heading(CLASS_CORRESPONDENCE)}\n",
            "**Hypothesis:** The pattern of LLM context effects across "
            f"{self._categories()} (output_dimension) correlates with the pattern of "
            "reference_observer judgment shifts (delta_reference) "
            "(Pearson r of class-level mean deltas).\n",
            f"- **Pearson r:** {r_val}  ",
            f"- **Permutation p-value:** {p_val}\n",
            "#### Class-Level Mean Deltas\n",
            "| Category | Mean Delta Physician | Mean Delta LLM |",
            "|---|---|---|",
        ]

        for i, cat in enumerate(self._category_labels):
            dp_val = "N/A"
            dl_val = "N/A"
            try:
                if mean_dp is not None:
                    dp_val = _fmt(float(mean_dp[i]))
            except (IndexError, TypeError):
                pass
            try:
                if mean_dl is not None:
                    dl_val = _fmt(float(mean_dl[i]))
            except (IndexError, TypeError):
                pass
            lines.append(f"| {cat} | {dp_val} | {dl_val} |")

        return "\n".join(lines)

    def _subsection_h4(self) -> str:
        h4 = self._h4

        # Handle skipped context specificity (shuffled-context absent)
        if h4.get("status") == "skipped":
            reason = h4.get("reason", "unknown reason")
            lines = [
                f"### {self._heading(CONTEXT_SPECIFICITY)}\n",
                "**Hypothesis:** Correct patient context (reference_correct_context) "
                "produces greater directional alignment with reference_observer deltas "
                "(delta_reference) than shuffled (mismatched) context "
                "(restricted to cells where delta_reference ≠ 0).\n",
                f"**Status:** Skipped — {reason}",
            ]
            return "\n".join(lines)

        mean_diff = _fmt(h4.get("mean_diff"))
        ci = _fmt_ci(h4.get("ci_lower"), h4.get("ci_upper"))
        p_val = _fmt_p(h4.get("p_value"))

        lines = [
            f"### {self._heading(CONTEXT_SPECIFICITY)}\n",
            "**Hypothesis:** Correct patient context (reference_correct_context) "
            "produces greater directional alignment with reference_observer deltas "
            "(delta_reference) than shuffled (mismatched) context "
            "(restricted to cells where delta_reference ≠ 0).\n",
            f"- **Mean alignment difference (correct − shuffled):** {mean_diff}  ",
            f"- **95% CI:** {ci}  ",
            f"- **Patient-cluster sign-flip p-value:** {p_val}",
        ]

        return "\n".join(lines)

    # --- Summary Table ------------------------------------------------

    def _section_summary_table(self) -> str:
        h1 = self._h1
        h2 = self._h2
        h3 = self._h3
        h4 = self._h4

        rows = [
            (
                f"{CONTEXT_SENSITIVITY.title} (mean absolute delta)",
                _fmt(h1.get("mean_abs_delta")),
                _fmt_ci(h1.get("ci_lower"), h1.get("ci_upper")),
                "—",
            ),
            (
                f"{CONTEXT_SENSITIVITY.title} (correct − shuffled mean absolute delta)",
                _fmt(h1.get("mean_abs_delta_difference")),
                _fmt_ci(h1.get("ci_lower_difference"), h1.get("ci_upper_difference")),
                _fmt_p(h1.get("p_value")),
            ),
            (
                f"{DIRECTIONAL_ALIGNMENT.title} (sign agreement rate; chance "
                f"{_fmt(h2.get('null_sign_agreement_rate'))})",
                _fmt(h2.get("sign_agreement_rate")),
                _fmt_ci(h2.get("ci_lower_sign_agree"), h2.get("ci_upper_sign_agree")),
                _fmt_p(h2.get("p_value")),
            ),
            (
                f"{DIRECTIONAL_ALIGNMENT.title} (mean alignment)",
                _fmt(h2.get("mean_alignment")),
                _fmt_ci(h2.get("ci_lower_alignment"), h2.get("ci_upper_alignment")),
                "—",
            ),
            (
                f"{CLASS_CORRESPONDENCE.title} (r)",
                _fmt(h3.get("pearson_r")),
                "—",
                _fmt_p(h3.get("p_value")),
            ),
            (
                f"{CONTEXT_SPECIFICITY.title} (correct − shuffled mean diff)",
                _fmt(h4.get("mean_diff")) if h4.get("status") != "skipped" else "Skipped",
                _fmt_ci(h4.get("ci_lower"), h4.get("ci_upper")) if h4.get("status") != "skipped" else "—",
                _fmt_p(h4.get("p_value")) if h4.get("status") != "skipped" else "—",
            ),
        ]

        header = "| Endpoint | Estimate | 95% CI | p-value |"
        separator = "|---|---|---|---|"
        table_rows = [f"| {e} | {est} | {ci} | {pv} |" for e, est, ci, pv in rows]

        return (
            "## Summary Table\n\n"
            + "\n".join([header, separator] + table_rows)
        )

    # --- Limitations --------------------------------------------------

    def _section_limitations(self) -> str:
        return (
            "## Limitations\n\n"
            "1. **Cohort scope:** Results are scoped to the "
            f"{self._entity_phrase()} (context_entity_id) and reference "
            "observers (reference_observer) in the evaluated dataset. "
            "Generalisability to broader "
            "populations is unknown.\n"
            "2. **Single model run:** Each report reflects a single model "
            f"({self._model_id}) evaluated at a single point in time "
            f"({self._run_timestamp}). Results may vary across model versions "
            "or inference configurations.\n"
            "3. **No prediction accuracy claims:** Track 3 measures context "
            "sensitivity and directional alignment only. It does not evaluate "
            "classification performance and does not report Brier score, F1, "
            "AUROC, or any measure of prediction accuracy.\n"
            "4. **Bootstrap CI validity:** Patient-cluster bootstrap CIs assume "
            "exchangeability of patient clusters (context_entity_id). With few "
            "patient clusters, bootstrap distributions may be coarse.\n"
            "5. **Shuffled context control:** The shuffled-context condition "
            "uses a single random draw per patient-item pair. Results may "
            "differ with alternative shuffling strategies.\n"
            "6. **LLM temperature:** All LLM calls use temperature 0.0 "
            "(deterministic decoding where supported). Results may differ "
            "with stochastic sampling."
        )
