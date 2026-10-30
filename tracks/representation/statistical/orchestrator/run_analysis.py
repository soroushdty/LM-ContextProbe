"""Top-level statistical analysis orchestrator for Track 1.

See the Track 1 endpoints in docs/methodology.md.

Calls all hypothesis tests, ICC analysis, entropy, stratum, and figure
functions in a fixed order and saves outputs to a structured directory.

Entry point: run_statistical_analysis()
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OUTPUTS = {
    "tau_robustness.csv":      "τ=0.5 vs F1-optimal robustness check",
    "directional_alignment_per_class.csv": "Directional alignment (formerly H1) per class",
    "directional_alignment_pooled.json":   "Directional alignment pooled over classes (permutation test)",
    "directional_alignment_cmh.json":      "Directional alignment across classes (CMH, permutation p-value)",
    "context_specificity_pooled.json":     "Context specificity pooled over classes (correct vs shuffled context)",
    "context_specificity_per_class.csv":   "Context specificity per class",
    "brier_improvement_per_class.csv":     "Brier improvement (formerly H2) per class",
    "brier_improvement_summary.json":      "Brier improvement macro-average summary",
    "wasserstein_distance.csv":            "Per-class Wasserstein distance to the label distribution",
    "icc_results.csv":         "Rater-level ICC(2,1) model-vs-physician",
    "icc_summary.json":        "Rater-level ICC summary statistics",
    "calibration_ece.csv":     "Per-class ECE with >0.10 flag and bootstrap CIs",
    "entropy_change.csv":      "Context-induced entropy change",
    "entropy_pearson.json":    "Pearson r between physician and model entropy change (descriptive only)",
    "stratum_analysis.csv":    "Stratum repeated vs novel",
    "ablation_scores.csv":     "Sub-field ablation attribution A(k,c)",
    "arch_comparison.csv":     "Architectural comparison",
    "figures/figure1_delta_histogram.png":       "Figure 1",
    "figures/figure2_interphysician_agreement.png": "Figure 2",
    "figures/figure3_icc_horizontal_plot.png":   "Figure 3",
    "figures/figure4_entropy_scatter.png":       "Figure 4",
}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=lambda x: None if x != x else x)


def _save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Sub-field ablation
# ---------------------------------------------------------------------------

def _run_ablation(
    context_json: dict,
    ensemble_bundle_path: Path,
    y_hat_ca: np.ndarray,
    patient_ids: np.ndarray,
    item_texts: np.ndarray,
    class_list: list[str],
    llm: str,
    cfg: dict,
    X_items_test: np.ndarray | None = None,
) -> pd.DataFrame | None:
    """Compute per-sub-field, per-class ablation scores A(k,c).

    A(k,c) = (1/n) Σ_i |ŷ_ca(i,c) − ŷ_ca^(−k)(i,c)|

    Loads the trained EnsemblePredictor from bundle_path, re-encodes each
    patient's context with one sub-field blanked out, rebuilds fusion features
    for the affected items, applies the Stage 2 model(s), and measures the
    mean absolute change in prediction.

    Returns None if ablation cannot be run (missing bundle, context, or Stage 2).
    """
    if not ensemble_bundle_path.exists():
        logger.warning(
            "Ablation skipped: ensemble bundle not found at %s", ensemble_bundle_path
        )
        return None

    if not context_json:
        logger.warning("Ablation skipped: context_json is empty.")
        return None

    if X_items_test is None:
        logger.warning(
            "Ablation skipped: X_items_test not provided — cannot rebuild fusion features."
        )
        return None

    try:
        import joblib
        from tracks.representation.training.stage2.stage2_context import (
            _CONTEXT_FIELDS,
            build_context_string,
            build_fusion_features,
        )
        from shared.embeddings.compute_embeddings import compute_embeddings
        from tracks.representation.models.EnsemblePredictor import EnsemblePredictor
        from tracks.representation.training.stage2.fit_stage2_fusion import apply_stage2_fusion, project_stage2_embeddings

        payload = joblib.load(ensemble_bundle_path)
        if not isinstance(payload, dict) or payload.get("bundle_format") != "pdm_ensemble_v1":
            logger.warning("Ablation skipped: unrecognised bundle format.")
            return None
        ens = EnsemblePredictor.from_bundle_dict(payload)

        if not ens.stage2 or not any(m is not None for m in ens.stage2):
            logger.warning("Ablation skipped: no Stage 2 models in bundle.")
            return None

        stage2_context_vectors = ens.stage2_context_vectors or {}
        stage2_models = [m for m in ens.stage2 if m is not None]

        n = len(item_texts)
        n_classes = len(class_list)
        n_fields = len(_CONTEXT_FIELDS)
        ablation_delta = np.zeros((n_fields, n_classes), dtype=np.float64)
        ca = np.asarray(y_hat_ca, dtype=np.float64)

        # Compute a consistent Stage-2-only baseline before entering the field
        # loop.  y_hat_ca comes from predict_proba, which averages ALL fold
        # members (substituting Stage-1 predictions for folds whose Stage-2
        # slot is None).  The ablation loop averages only the non-None Stage-2
        # models, so using y_hat_ca directly as the baseline introduces a
        # systematic bias when any fold lacks a Stage-2 model.  By re-applying
        # the same stage2_models to the original context vectors we guarantee
        # that both ca_baseline and ablated_ca use an identical aggregation,
        # so attribution scores reflect only the ablation effect.
        # For passthrough models, use zeros as a consistent y_cf placeholder so
        # baseline and ablated predictions are comparable (delta is context-only).
        _y_cf_zeros = np.zeros((1, len(class_list)), dtype=np.float32)
        ca_baseline = ca.copy()
        for _i in range(n):
            _pid = int(float(patient_ids[_i]))
            _c_p = stage2_context_vectors.get(_pid)
            if _c_p is None:
                continue  # patient has no stored context vector; keep ca[_i]
            _e_i = X_items_test[_i]
            _per_model_preds = []
            for m in stage2_models:
                _e_proj, _ctx_proj = project_stage2_embeddings(
                    m, _e_i.reshape(1, -1), {_pid: _c_p}
                )
                _z = build_fusion_features(
                    _e_proj[0], _ctx_proj[_pid],
                    fusion_strategy=m["fusion_strategy"], r=m["r"],
                )
                _per_model_preds.append(
                    apply_stage2_fusion(m, _z.reshape(1, -1), y_cf=_y_cf_zeros)[0]
                )
            ca_baseline[_i] = np.mean(_per_model_preds, axis=0).astype(np.float64)

        # Pre-encode all unique ablated context strings per field to avoid
        # re-encoding the same patient multiple times within a field.
        for k_idx, field in enumerate(_CONTEXT_FIELDS):
            logger.info(
                "Ablation: processing sub-field '%s' (%d/%d).", field, k_idx + 1, n_fields
            )

            # Build ablated context string per unique patient.
            unique_pids = {int(float(p)) for p in patient_ids
                           if str(int(float(p))) in context_json}
            pid_to_ablated_str: dict[int, str] = {}
            for pid in unique_pids:
                ctx = context_json[str(pid)]
                ablated_ctx = {k: ("" if k == field else v) for k, v in ctx.items()}
                pid_to_ablated_str[pid] = build_context_string(ablated_ctx)

            if not pid_to_ablated_str:
                continue

            abl_series = pd.Series(list(pid_to_ablated_str.values()))
            try:
                abl_emb_dict = compute_embeddings(llm, abl_series, schema="pandas", cfg=cfg)
            except Exception as enc_exc:
                logger.warning("Ablation field '%s' embedding failed: %s", field, enc_exc)
                continue

            # Build ablated fusion features row-by-row.
            abl_preds_rows = []
            for i in range(n):
                pid = int(float(patient_ids[i]))
                abl_ctx_str = pid_to_ablated_str.get(pid)
                if abl_ctx_str is None or abl_ctx_str not in abl_emb_dict:
                    # Embedding unavailable: fall back to the Stage-2-only
                    # baseline so the contribution of this row to the
                    # attribution delta is exactly zero.
                    abl_preds_rows.append(ca_baseline[i])
                    continue
                abl_c_p = abl_emb_dict[abl_ctx_str]
                e_i = X_items_test[i]
                # Average predictions across all Stage 2 fold models, projecting
                # through each model's PCA (if any) before building fusion features.
                fold_preds = []
                for m in stage2_models:
                    _e_proj, _ctx_proj = project_stage2_embeddings(
                        m, e_i.reshape(1, -1), {pid: abl_c_p}
                    )
                    z_abl = build_fusion_features(
                        _e_proj[0], _ctx_proj[pid],
                        fusion_strategy=m["fusion_strategy"], r=m["r"],
                    )
                    fold_preds.append(
                        apply_stage2_fusion(m, z_abl.reshape(1, -1), y_cf=_y_cf_zeros)[0]
                    )
                abl_preds_rows.append(np.mean(fold_preds, axis=0).astype(np.float64))

            ablated_ca = np.array(abl_preds_rows, dtype=np.float64)
            # Compare ablated predictions against the consistent Stage-2-only
            # baseline (not the raw y_hat_ca) so that differences are purely
            # attributable to the field ablation and not to predict_proba's
            # aggregation mixing Stage-1 fallbacks for None-fold slots.
            ablation_delta[k_idx] = np.mean(np.abs(ca_baseline - ablated_ca), axis=0)

        df = pd.DataFrame(
            ablation_delta,
            index=_CONTEXT_FIELDS,
            columns=class_list,
        )
        df.index.name = "sub_field"
        return df.reset_index()

    except Exception as exc:
        logger.warning("Ablation failed: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_statistical_analysis(
    y_survey: np.ndarray,
    y_interview: np.ndarray,
    y_hat_cf: np.ndarray,
    y_hat_ca: np.ndarray,
    patient_ids: np.ndarray,
    item_texts: np.ndarray,
    class_list: list[str],
    avg_thresh_f1opt: np.ndarray,
    tau_fixed: float,
    dataset_path: Path,
    sheet_names: dict,
    context_json: dict | None,
    llm: str,
    cfg: dict,
    output_dir: Path,
    n_resamples: int = 1000,
    n_permutations: int = 10_000,
    strata: np.ndarray | None = None,
    arch_predictions: dict | None = None,
    ensemble_bundle_path: Path | None = None,
    item_texts_train: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    X_items_test: np.ndarray | None = None,
    y_hat_ca_leave_pair_out: np.ndarray | None = None,
    y_hat_shuffled: np.ndarray | None = None,
) -> None:
    """Run all statistical analyses and save outputs to output_dir.

    Args:
        y_survey:           (n_pairs, n_classes) pair-aggregated survey labels.
        y_interview:        (n_pairs, n_classes) pair-aggregated interview labels.
        y_hat_cf:           (n_pairs, n_classes) context-free predictions (Stage 1 only).
        y_hat_ca:           (n_pairs, n_classes) context-aware predictions (Stage 2).
        y_hat_ca_leave_pair_out, y_hat_shuffled:
                            (n_pairs, n_classes) predictions of the leave-pair-out
                            Stage 2 heads with the row's own context and with its
                            swap partner's. Context specificity compares the two;
                            it is skipped if either is None.
        patient_ids:        (n_pairs,) patient IDs.
        item_texts:         (n_pairs,) item text strings.
        class_list:         Class names, length n_classes.
        avg_thresh_f1opt:   (n_classes,) mean inner-fold F1-optimal thresholds.
        tau_fixed:          Fixed threshold (cfg['tau'], default 0.5).
        dataset_path:       Path to dataset.xlsx (for individual physician labels).
        sheet_names:        dict with keys 'train', 'test', 'interview'.
        context_json:       Raw context JSON dict (patient ID → sub-field dict).
        llm:                Encoder model ID (for ablation re-encoding).
        cfg:                Full pipeline config dict.
        output_dir:         Root directory for all statistical outputs.
        n_resamples:        Bootstrap CI resamples (default 1000).
        n_permutations:     Permutation / sign-flip replicates (default 10,000).
        strata:             (n_pairs,) 'repeated'/'novel' labels; computed if None.
        arch_predictions:   {arch_name: (n_pairs, n_classes)} for arch comparison; skipped if None.
        ensemble_bundle_path: Path to ensemble_bundle.joblib for ablation.
        item_texts_train:   Training item strings for strata assignment.
        rng:                Optional numpy Generator for reproducibility.
    """
    import time
    t0 = time.perf_counter()
    logger.info("Starting statistical analysis → %s", output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if rng is None:
        seed = cfg.get("global_seed", 42)
        rng = np.random.default_rng(seed)

    stat_cfg = cfg.get("statistical_analysis", {}) or {}
    min_nonzero = int(stat_cfg.get("confirmatory_min_nonzero", 15))
    # Separate generator for the patient-cluster sign-flip tests, so they do
    # not consume draws from ``rng`` (whose sequence drives every bootstrap CI).
    perm_rng = np.random.default_rng(int(cfg.get("global_seed", 42)) + 1)
    # And one for the within-patient permutation nulls of directional
    # alignment, so they leave the sign-flip draws (used by Brier
    # improvement) unchanged.
    null_rng = np.random.default_rng(int(cfg.get("global_seed", 42)) + 2)
    # And one for context specificity, so adding it leaves every other
    # output unchanged.
    spec_rng = np.random.default_rng(int(cfg.get("global_seed", 42)) + 3)
    fig_dpi = int(stat_cfg.get("figure_dpi", 150))

    patient_col = cfg.get("patient_col", "Patient")
    physician_col = cfg.get("physician_col", "Physician")
    item_col = cfg.get("item_col", "Item")

    # ------------------------------------------------------------------
    # Step 1 — Core deltas and confirmatory filter
    # ------------------------------------------------------------------
    from shared.statistical.delta import (
        compute_physician_deltas,
        compute_model_deltas,
        confirmatory_eligible_classes,
    )
    delta_p = compute_physician_deltas(y_survey, y_interview)
    delta_m = compute_model_deltas(y_hat_cf, y_hat_ca)
    eligible_classes, descriptive_classes = confirmatory_eligible_classes(
        delta_p, class_list, min_nonzero=min_nonzero
    )
    logger.info(
        "Confirmatory-eligible: %s | Descriptive-only: %s",
        eligible_classes, descriptive_classes,
    )

    # ------------------------------------------------------------------
    # Step 2 — τ robustness check
    # ------------------------------------------------------------------
    from shared.statistical.tau_check import tau_robustness_check, tau_robustness_summary
    tau_df = tau_robustness_check(tau_fixed, avg_thresh_f1opt, class_list)
    _save_csv(output_dir / "tau_robustness.csv", tau_df)
    logger.info("τ check: %s", tau_robustness_summary(tau_df))

    # ------------------------------------------------------------------
    # Step 3 — Load individual physician labels (for ICC + Figure 2)
    # ------------------------------------------------------------------
    from shared.statistical.icc import load_individual_physician_labels

    # Build a raw-item → canonical-item standardizer that matches the JSON-
    # based item standardization applied during preprocessing (remove_spaces=
    # False, casefold-only normalization, then reverse-lookup from mapping).
    # This ensures individual_labels keys align with item_texts_test keys.
    _item_std = None
    try:
        _map_path_val = cfg.get("DIR_JSON_MAP")
        if _map_path_val:
            from pathlib import Path as _Path
            from shared.utils.mapping_utils import _build_reverse_lookup, _load_grouped_mapping
            from shared.utils.text_utils import normalize_for_matching as _nfm
            from shared.preprocessing.input_loading import _resolve_cfg_path
            _map_path = _Path(_resolve_cfg_path(cfg, "DIR_JSON_MAP"))
            if _map_path.exists() and _map_path.is_file():
                _grouped = _load_grouped_mapping(_map_path, remove_spaces=False)
                _reverse = _build_reverse_lookup(_grouped)
                def _item_std(raw: str, _r=_reverse, _nfm=_nfm) -> str:
                    return _r.get(_nfm(raw, strip=False), raw)
                logger.info(
                    "ICC item standardizer built from %s (%d canonical entries).",
                    _map_path, len(_grouped),
                )
    except Exception as _std_exc:
        logger.warning(
            "Could not build ICC item standardizer from DIR_JSON_MAP — "
            "falling back to raw item strings (ICC matching may be degraded): %s",
            _std_exc,
        )

    individual_labels: dict = {}
    try:
        individual_labels = load_individual_physician_labels(
            Path(dataset_path), sheet_names,
            patient_col, physician_col, item_col, class_list,
            item_standardizer=_item_std,
        )
    except Exception as exc:
        logger.warning("Could not load individual physician labels: %s", exc)

    # ------------------------------------------------------------------
    # Step 4 — Directional alignment (H1) per class
    # ------------------------------------------------------------------
    from ..hypotheses.h1 import h1_binomial_per_class, h1_permutation_test, h1_cmh_test
    h1_df = h1_binomial_per_class(
        delta_p, delta_m, class_list, eligible_classes,
        patient_ids, n_resamples, rng,
        n_permutations=n_permutations, perm_rng=perm_rng, null_rng=null_rng,
    )
    _save_csv(output_dir / "directional_alignment_per_class.csv", h1_df)
    logger.info(
        "Directional alignment per class done: %d confirmatory classes.",
        len(eligible_classes),
    )

    # ------------------------------------------------------------------
    # Step 5 — Directional alignment pooled (permutation test)
    # ------------------------------------------------------------------
    h1_agg = h1_permutation_test(
        delta_p, delta_m, patient_ids,
        n_permutations=n_permutations,
        eligible_classes=eligible_classes,
        class_list=class_list,
        rng=rng,
        n_resamples=n_resamples,
    )
    _save_json(output_dir / "directional_alignment_pooled.json", h1_agg)
    logger.info(
        "Directional alignment pooled: rate=%.3f, p=%.4f",
        h1_agg["aggregate_rate"], h1_agg["p_value"],
    )

    # ------------------------------------------------------------------
    # Step 6 — Directional alignment across classes (CMH)
    # ------------------------------------------------------------------
    h1_cmh = h1_cmh_test(
        delta_p, delta_m, eligible_classes, class_list,
        patient_ids=patient_ids, n_permutations=n_permutations, perm_rng=perm_rng,
        null_rng=null_rng,
    )
    _save_json(output_dir / "directional_alignment_cmh.json", h1_cmh)
    logger.info(
        "Directional alignment CMH: OR=%.3f, p=%.4f, permutation p=%.4f",
        h1_cmh["common_odds_ratio"], h1_cmh["p_cmh"], h1_cmh["p_permutation"],
    )

    # ------------------------------------------------------------------
    # Step 6.5 — Context specificity (correct vs shuffled context)
    # ------------------------------------------------------------------
    if y_hat_ca_leave_pair_out is not None and y_hat_shuffled is not None:
        from ..hypotheses.context_specificity import (
            context_specificity_per_class,
            context_specificity_pooled,
        )
        # Both conditions come from the same leave-pair-out head, not from the
        # fold-pure head behind delta_m (see docs/methodology.md).
        delta_m_pair = compute_model_deltas(y_hat_cf, y_hat_ca_leave_pair_out)
        delta_s = compute_model_deltas(y_hat_cf, y_hat_shuffled)
        spec_pooled = context_specificity_pooled(
            delta_p, delta_m_pair, delta_s, patient_ids, eligible_classes, class_list,
            n_resamples=n_resamples, n_permutations=n_permutations, rng=spec_rng,
        )
        _save_json(output_dir / "context_specificity_pooled.json", spec_pooled)
        spec_df = context_specificity_per_class(
            delta_p, delta_m_pair, delta_s, patient_ids, class_list, eligible_classes,
            n_resamples=n_resamples, n_permutations=n_permutations, rng=spec_rng,
        )
        _save_csv(output_dir / "context_specificity_per_class.csv", spec_df)
        logger.info(
            "Context specificity pooled: alignment difference=%.4f, p=%.4f",
            spec_pooled["mean_alignment_difference"], spec_pooled["p_value"],
        )
    else:
        logger.info("Context specificity skipped: no shuffled-context predictions.")

    # ------------------------------------------------------------------
    # Step 7 — Brier improvement (H2)
    # ------------------------------------------------------------------
    from ..hypotheses.h2 import h2_wilcoxon_per_class, h2_wasserstein_per_class, h2_macro_summary
    h2_df = h2_wilcoxon_per_class(
        y_hat_cf, y_hat_ca, y_interview, class_list, patient_ids, n_resamples, rng,
        n_permutations=n_permutations, perm_rng=perm_rng,
    )
    _save_csv(output_dir / "brier_improvement_per_class.csv", h2_df)

    h2_summ = h2_macro_summary(h2_df, y_hat_cf, y_hat_ca, y_interview, patient_ids, n_resamples, rng)
    _save_json(output_dir / "brier_improvement_summary.json", h2_summ)
    logger.info(
        "Brier improvement (macro): CF=%.4f CA=%.4f improve=%.4f, sig_classes=%d/%d",
        h2_summ["macro_BS_cf"], h2_summ["macro_BS_ca"], h2_summ["macro_improvement"],
        h2_summ["n_classes_significant"], h2_summ["n_classes_total"],
    )

    # ------------------------------------------------------------------
    # Step 8 — Wasserstein distance (descriptive)
    # ------------------------------------------------------------------
    was_df = h2_wasserstein_per_class(
        y_hat_cf, y_hat_ca, y_interview, class_list, patient_ids, n_resamples, rng,
    )
    _save_csv(output_dir / "wasserstein_distance.csv", was_df)

    # ------------------------------------------------------------------
    # Step 9 — ICC rater-level analysis
    # ------------------------------------------------------------------
    from shared.statistical.icc import rater_icc_analysis
    icc_results: dict = {}
    if individual_labels:
        try:
            icc_results = rater_icc_analysis(
                y_hat_ca, individual_labels, patient_ids, item_texts,
                class_list, n_resamples, rng,
            )
            # Save per-pair ICCs to CSV
            mp_df = pd.DataFrame(icc_results.get("model_physician_iccs", []))
            hh_df = pd.DataFrame(icc_results.get("human_human_iccs", []))
            icc_df = pd.concat(
                [mp_df.assign(rater_type="model_physician"),
                 hh_df.assign(rater_type="human_human")],
                ignore_index=True,
            )
            _save_csv(output_dir / "icc_results.csv", icc_df)
            icc_summary = {k: v for k, v in icc_results.items()
                           if k not in ("model_physician_iccs", "human_human_iccs")}
            _save_json(output_dir / "icc_summary.json", icc_summary)
            logger.info(
                "ICC: model-physician mean=%.3f, human-human mean=%.3f",
                icc_results.get("model_physician_icc_mean", float("nan")),
                icc_results.get("human_human_icc_mean", float("nan")),
            )
        except Exception as exc:
            logger.warning("ICC analysis failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Step 9.5 — Calibration ECE per class
    # ------------------------------------------------------------------
    from shared.evaluation.calibration import calibration_ece_per_class
    ece_df = calibration_ece_per_class(
        y_hat_ca, y_interview, class_list, patient_ids, n_resamples, rng=rng,
    )
    _save_csv(output_dir / "calibration_ece.csv", ece_df)
    n_flagged = int(ece_df["flagged"].sum())
    if n_flagged:
        logger.warning(
            "Calibration: %d/%d classes have ECE > 0.10: %s",
            n_flagged, len(class_list),
            ece_df.loc[ece_df["flagged"], "Class"].tolist(),
        )
    else:
        logger.info("Calibration: all %d classes have ECE ≤ 0.10.", len(class_list))

    # ------------------------------------------------------------------
    # Step 10 — Context-induced entropy change
    # ------------------------------------------------------------------
    from shared.statistical.entropy import (
        physician_entropy_change,
        model_entropy_change,
        entropy_pearson_r,
    )
    phys_ent_delta = physician_entropy_change(y_survey, y_interview)
    model_ent_delta = model_entropy_change(y_hat_cf, y_hat_ca)

    ent_df_rows = []
    for c_idx, cls in enumerate(class_list):
        for i in range(len(item_texts)):
            ent_df_rows.append({
                "item": item_texts[i],
                "patient": patient_ids[i],
                "class": cls,
                "phys_entropy_change": float(phys_ent_delta[i, c_idx]),
                "model_entropy_change": float(model_ent_delta[i, c_idx]),
            })
    ent_df = pd.DataFrame(ent_df_rows)
    _save_csv(output_dir / "entropy_change.csv", ent_df)

    pearson_result = entropy_pearson_r(phys_ent_delta, model_ent_delta, eligible_classes, class_list)
    _save_json(output_dir / "entropy_pearson.json", pearson_result)
    logger.info(
        "Entropy Pearson r=%.3f (p=%.4f, descriptive only)",
        pearson_result["pearson_r"],
        pearson_result["p_value_descriptive_only"],
    )

    # ------------------------------------------------------------------
    # Step 11 — Stratum analysis
    # ------------------------------------------------------------------
    from ..reporting.stratum import assign_test_strata, stratum_comparison
    if strata is None:
        if item_texts_train is not None:
            strata = assign_test_strata(item_texts, item_texts_train)
        else:
            logger.warning(
                "Stratum analysis skipped: strata not provided and item_texts_train is None."
            )
            strata = None

    if strata is not None:
        strat_df = stratum_comparison(
            delta_p, delta_m, y_hat_cf, y_hat_ca, y_interview,
            strata, patient_ids, eligible_classes, class_list, n_resamples, rng,
        )
        _save_csv(output_dir / "stratum_analysis.csv", strat_df)
        logger.info("Stratum analysis done.")

    # ------------------------------------------------------------------
    # Step 12 — Sub-field ablation
    # ------------------------------------------------------------------
    if ensemble_bundle_path is not None and context_json:
        abl_df = _run_ablation(
            context_json, ensemble_bundle_path, y_hat_ca,
            patient_ids, item_texts, class_list, llm, cfg,
            X_items_test=X_items_test,
        )
        if abl_df is not None:
            _save_csv(output_dir / "ablation_scores.csv", abl_df)
            logger.info("Ablation scores saved.")
    else:
        logger.info("Ablation skipped (no bundle path or context_json).")

    # ------------------------------------------------------------------
    # Step 13 — Architectural comparison
    # ------------------------------------------------------------------
    if arch_predictions:
        from ..reporting.arch_compare import arch_comparison_table
        # Width of the item/context vectors entering Stage 2.
        _stage2_dim = cfg.get("stage2_pca_n_components") or (
            X_items_test.shape[1] if X_items_test is not None else None
        )
        arch_df = arch_comparison_table(
            arch_predictions, y_interview, class_list,
            patient_ids, avg_thresh_f1opt, n_resamples, rng,
            embedding_dim=int(_stage2_dim) if _stage2_dim is not None else None,
            lowrank_r=int(cfg.get("lowrank_bilinear_r", 8)),
        )
        _save_csv(output_dir / "arch_comparison.csv", arch_df)
        logger.info("Arch comparison done: %d architectures.", len(arch_predictions))

    # ------------------------------------------------------------------
    # Step 14 — Figures
    # ------------------------------------------------------------------
    from ..reporting.figures import (
        figure1_delta_histogram,
        figure2_interphysician_agreement,
        figure3_icc_horizontal_plot,
        figure4_entropy_scatter,
    )
    try:
        figure1_delta_histogram(delta_p, class_list,
                                fig_dir / "figure1_delta_histogram.png", dpi=fig_dpi)
    except Exception as exc:
        logger.warning("Figure 1 failed: %s", exc)

    if individual_labels:
        try:
            figure2_interphysician_agreement(individual_labels, class_list,
                                             fig_dir / "figure2_interphysician_agreement.png",
                                             dpi=fig_dpi)
        except Exception as exc:
            logger.warning("Figure 2 failed: %s", exc)

    if icc_results:
        try:
            figure3_icc_horizontal_plot(icc_results,
                                        fig_dir / "figure3_icc_horizontal_plot.png",
                                        dpi=fig_dpi)
        except Exception as exc:
            logger.warning("Figure 3 failed: %s", exc)

    try:
        figure4_entropy_scatter(
            phys_ent_delta, model_ent_delta,
            eligible_classes, class_list,
            pearson_result["pearson_r"],
            fig_dir / "figure4_entropy_scatter.png",
            dpi=fig_dpi,
        )
    except Exception as exc:
        logger.warning("Figure 4 failed: %s", exc)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Statistical analysis complete (%.1fs). Outputs in %s.", elapsed, output_dir
    )
