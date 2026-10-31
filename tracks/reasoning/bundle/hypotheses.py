"""
Endpoint computation for the analysis bundle pipeline (shared/endpoints.py):
context sensitivity, directional alignment, class-level correspondence and
context specificity (formerly Track 3's H1–H4).

Each function accepts a ``cells_df`` (paired cell deltas for a single model),
the model name, and statistical parameters.  All CIs use patient-cluster
bootstrap; p-values use permutation tests.

Context sensitivity and directional alignment use the shuffled-context
condition as their reference: a paired sign-flip test of |Δ_correct| −
|Δ_shuffled|, and a permutation null that shuffles each patient's model
deltas among that patient's items.

Reuses:
- ``shared.statistical.bootstrap.patient_block_bootstrap``
- ``shared.statistical.cluster_tests.within_cluster_permutation_test``
- ``shared.evaluation.hypothesis_analyzer.permutation_test_h3``
- ``shared.evaluation.hypothesis_analyzer.permutation_test_h4``
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats

from shared.statistical.bootstrap import patient_block_bootstrap
from shared.statistical.delta import sign_agreement_rate
from shared.statistical.cluster_tests import (
    cluster_sign_flip_test,
    within_cluster_permutation_test,
)
from shared.evaluation.hypothesis_analyzer import permutation_test_h3, permutation_test_h4

logger = logging.getLogger(__name__)

_AGGREGATE_LABEL = "aggregate"


# ---------------------------------------------------------------------------
# Internal bootstrap helpers
# ---------------------------------------------------------------------------


def _bootstrap_scalar(
    values: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstrap: int,
    seed: int,
    agg_fn=np.mean,
) -> Tuple[float, float, float]:
    """Return (point, ci_low, ci_high) using patient-cluster bootstrap."""
    values = np.asarray(values, dtype=float)
    patient_ids = np.asarray(patient_ids)
    point = float(agg_fn(values))

    def _stat(idx: np.ndarray) -> float:
        return float(agg_fn(values[idx]))

    rng = np.random.default_rng(seed)
    lo, hi = patient_block_bootstrap(
        metric_fn=_stat,
        patient_ids=patient_ids,
        n_resamples=n_bootstrap,
        rng=rng,
        ci_level=0.95,
    )
    return point, lo, hi


def _permutation_test_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Permutation test for Spearman correlation (permute y labels)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed_r, _ = scipy.stats.spearmanr(x, y)
    permuted = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(y)
        r_perm, _ = scipy.stats.spearmanr(x, shuffled)
        permuted[i] = r_perm
    return float(np.mean(permuted >= observed_r))


# ---------------------------------------------------------------------------
# Context sensitivity (formerly H1)
# ---------------------------------------------------------------------------


def compute_h1(
    cells_df: pd.DataFrame,
    model: str,
    categories: List[str],
    epsilon: float = 0.01,
    n_bootstrap: int = 1000,
    seed: int = 2026,
    n_permutations: int = 10000,
) -> pd.DataFrame:
    """Compute context sensitivity (formerly H1) for correct and shuffled conditions.

    The test is whether the correct patient's context moves the scores more
    than another patient's: the paired difference |Δ_correct| − |Δ_shuffled|
    gets a patient-cluster bootstrap CI and a one-sided patient-cluster
    sign-flip p-value.

    Parameters
    ----------
    cells_df:
        Paired cell deltas for a single model (output of
        ``build_paired_cell_deltas`` filtered to one model).
    model:
        Model identifier string (written to the output DataFrame).
    categories:
        Canonical category names present in ``cells_df``.
    epsilon:
        Threshold for determining whether the LLM "changed" its score.
    n_bootstrap:
        Number of patient-cluster bootstrap resamples.
    seed:
        RNG seed.
    n_permutations:
        Monte Carlo replicates for the sign-flip test when exact enumeration
        is too large.

    Returns
    -------
    pd.DataFrame
        One aggregate row + one per-category row.
        Columns: model, category, n_patients, n_patient_items, n_cells,
        mean_abs_delta_correct, ci_low_correct, ci_high_correct,
        mean_abs_delta_shuffled, ci_low_shuffled, ci_high_shuffled,
        mean_abs_delta_difference, ci_low_difference, ci_high_difference,
        paired_permutation_p, proportion_changed_correct,
        proportion_changed_shuffled, epsilon.
    """
    df = cells_df.copy()
    patient_col = "patient" if "patient" in df.columns else "patient_id"
    patient_ids = df[patient_col].astype(str).to_numpy()

    rows: list[dict] = []

    def _compute_row(mask: np.ndarray, category: str) -> dict:
        sub = df[mask]
        if len(sub) == 0:
            return _nan_h1_row(model, category, epsilon)
        pids = sub[patient_col].astype(str).to_numpy()
        n_patients = int(sub[patient_col].nunique())
        n_patient_items = int(
            sub.groupby([patient_col, "item_text"]).ngroups
            if "item_text" in sub.columns
            else len(sub)
        )
        n_cells = len(sub)

        dc = sub["delta_llm_correct"].to_numpy(dtype=float)
        ds = sub["delta_llm_shuffled"].to_numpy(dtype=float)
        abs_dc = np.abs(dc)
        abs_ds = np.abs(ds)

        pt_c, lo_c, hi_c = _bootstrap_scalar(abs_dc, pids, n_bootstrap, seed)
        pt_s, lo_s, hi_s = _bootstrap_scalar(abs_ds, pids, n_bootstrap, seed + 1)
        pt_d, lo_d, hi_d = _bootstrap_scalar(abs_dc - abs_ds, pids, n_bootstrap, seed + 2)
        p_diff = cluster_sign_flip_test(
            abs_dc - abs_ds, pids, n_permutations, np.random.default_rng(seed + 3)
        )

        prop_c = float(np.mean(abs_dc > epsilon))
        prop_s = float(np.mean(abs_ds > epsilon))

        return {
            "model": model,
            "category": category,
            "n_patients": n_patients,
            "n_patient_items": n_patient_items,
            "n_cells": n_cells,
            "mean_abs_delta_correct": pt_c,
            "ci_low_correct": lo_c,
            "ci_high_correct": hi_c,
            "mean_abs_delta_shuffled": pt_s,
            "ci_low_shuffled": lo_s,
            "ci_high_shuffled": hi_s,
            "mean_abs_delta_difference": pt_d,
            "ci_low_difference": lo_d,
            "ci_high_difference": hi_d,
            "paired_permutation_p": p_diff,
            "proportion_changed_correct": prop_c,
            "proportion_changed_shuffled": prop_s,
            "epsilon": epsilon,
        }

    # Aggregate row
    rows.append(_compute_row(np.ones(len(df), dtype=bool), _AGGREGATE_LABEL))

    # Per-category rows
    for cat in categories:
        if "category" in df.columns:
            mask = df["category"].to_numpy() == cat
        else:
            mask = np.ones(len(df), dtype=bool)
        rows.append(_compute_row(mask, cat))

    return pd.DataFrame(rows)


def _nan_h1_row(model: str, category: str, epsilon: float) -> dict:
    nan = float("nan")
    return {
        "model": model, "category": category,
        "n_patients": 0, "n_patient_items": 0, "n_cells": 0,
        "mean_abs_delta_correct": nan, "ci_low_correct": nan, "ci_high_correct": nan,
        "mean_abs_delta_shuffled": nan, "ci_low_shuffled": nan, "ci_high_shuffled": nan,
        "mean_abs_delta_difference": nan, "ci_low_difference": nan, "ci_high_difference": nan,
        "paired_permutation_p": nan,
        "proportion_changed_correct": nan, "proportion_changed_shuffled": nan,
        "epsilon": epsilon,
    }


# ---------------------------------------------------------------------------
# Directional alignment (formerly H2)
# ---------------------------------------------------------------------------


def compute_h2(
    cells_df: pd.DataFrame,
    model: str,
    categories: List[str],
    epsilon: float = 0.01,
    n_bootstrap: int = 1000,
    seed: int = 2026,
    n_permutations: int = 10000,
) -> pd.DataFrame:
    """Compute directional alignment (formerly H2) (correct and shuffled) where Δ_physician ≠ 0.

    The sign-agreement rates leave out cells where the LLM did not move
    (|Δ_LLM| < epsilon; reported as ``unchanged_rate_*``): counting them as
    disagreement made an LLM that moves where the physicians move, in random
    directions, look aligned (#36). Chance agreement is still not 0.5. The
    aggregate row adds a permutation null: each patient's model delta rows
    (all categories of an item together) are shuffled among that patient's
    items, and the sign-agreement rate is recomputed.

    Parameters
    ----------
    cells_df:
        Paired cell deltas for a single model.
    model:
        Model identifier.
    categories:
        Canonical category names.
    epsilon:
        Threshold for near-zero LLM deltas (treated as no movement).
    n_bootstrap:
        Bootstrap resamples.
    seed:
        RNG seed.
    n_permutations:
        Permutations for the null of the aggregate sign-agreement rate.

    Returns
    -------
    pd.DataFrame
        One aggregate row + one per-category row.
        Columns: model, category, n_patients, n_shift_cells,
        mean_alignment_correct, ci_low_correct, ci_high_correct,
        mean_alignment_shuffled, ci_low_shuffled, ci_high_shuffled,
        sign_agreement_correct, sign_agreement_shuffled (on shift cells
        where the LLM moved), unchanged_rate_correct, unchanged_rate_shuffled (share of shift
        cells with |Δ_LLM| < epsilon), null_sign_agreement_correct and
        permutation_p_correct (aggregate row only; NaN per category),
        epsilon.
    """
    df = cells_df.copy()
    patient_col = "patient" if "patient" in df.columns else "patient_id"

    rows: list[dict] = []

    def _compute_row(mask: np.ndarray, category: str) -> dict:
        sub = df[mask & (df["delta_physician"] != 0.0)]
        if len(sub) == 0:
            return _nan_h2_row(model, category, epsilon)

        pids = sub[patient_col].astype(str).to_numpy()
        n_patients = int(sub[patient_col].nunique())
        n_shift_cells = len(sub)

        sign_dp = np.sign(sub["delta_physician"].to_numpy(dtype=float))
        dc = sub["delta_llm_correct"].to_numpy(dtype=float)
        ds = sub["delta_llm_shuffled"].to_numpy(dtype=float)

        # Apply epsilon threshold for near-zero LLM movement
        dc_adj = np.where(np.abs(dc) < epsilon, 0.0, dc)
        ds_adj = np.where(np.abs(ds) < epsilon, 0.0, ds)

        align_c = sign_dp * dc_adj
        align_s = sign_dp * ds_adj

        pt_c, lo_c, hi_c = _bootstrap_scalar(align_c, pids, n_bootstrap, seed)
        pt_s, lo_s, hi_s = _bootstrap_scalar(align_s, pids, n_bootstrap, seed + 1)

        sa_c = sign_agreement_rate(sign_dp, dc_adj)
        sa_s = sign_agreement_rate(sign_dp, ds_adj)

        null_c = p_c = float("nan")
        if category == _AGGREGATE_LABEL:
            null = _sign_agreement_null(df, patient_col, epsilon, n_permutations, seed + 2)
            null_c, p_c = null["null_mean"], null["p_value"]

        return {
            "model": model,
            "category": category,
            "n_patients": n_patients,
            "n_shift_cells": n_shift_cells,
            "mean_alignment_correct": pt_c,
            "ci_low_correct": lo_c,
            "ci_high_correct": hi_c,
            "mean_alignment_shuffled": pt_s,
            "ci_low_shuffled": lo_s,
            "ci_high_shuffled": hi_s,
            "sign_agreement_correct": sa_c,
            "sign_agreement_shuffled": sa_s,
            "unchanged_rate_correct": float(np.mean(dc_adj == 0.0)),
            "unchanged_rate_shuffled": float(np.mean(ds_adj == 0.0)),
            "null_sign_agreement_correct": null_c,
            "permutation_p_correct": p_c,
            "epsilon": epsilon,
        }

    rows.append(_compute_row(np.ones(len(df), dtype=bool), _AGGREGATE_LABEL))
    for cat in categories:
        if "category" in df.columns:
            mask = df["category"].to_numpy() == cat
        else:
            mask = np.ones(len(df), dtype=bool)
        rows.append(_compute_row(mask, cat))

    return pd.DataFrame(rows)


def _nan_h2_row(model: str, category: str, epsilon: float) -> dict:
    nan = float("nan")
    return {
        "model": model, "category": category, "n_patients": 0, "n_shift_cells": 0,
        "mean_alignment_correct": nan, "ci_low_correct": nan, "ci_high_correct": nan,
        "mean_alignment_shuffled": nan, "ci_low_shuffled": nan, "ci_high_shuffled": nan,
        "sign_agreement_correct": nan, "sign_agreement_shuffled": nan,
        "unchanged_rate_correct": nan, "unchanged_rate_shuffled": nan,
        "null_sign_agreement_correct": nan, "permutation_p_correct": nan,
        "epsilon": epsilon,
    }


def _sign_agreement_null(
    df: pd.DataFrame,
    patient_col: str,
    epsilon: float,
    n_permutations: int,
    seed: int,
) -> dict[str, float]:
    """Within-patient permutation null for the aggregate sign-agreement rate.

    Cells are pivoted to one row per (patient, item) with one column per
    category, so an item's categories move together. Missing cells are NaN
    and do not count.
    """
    keys = [patient_col, "item_text"] if "item_text" in df.columns else [patient_col]
    wide = df.pivot_table(
        index=keys, columns="category",
        values=["delta_physician", "delta_llm_correct"], aggfunc="mean",
    )
    dp = wide["delta_physician"].to_numpy(dtype=float)
    dm = wide["delta_llm_correct"].to_numpy(dtype=float)
    dm = np.where(np.abs(dm) < epsilon, 0.0, dm)
    pids = wide.index.get_level_values(patient_col).astype(str).to_numpy()
    sign_dp = np.sign(dp)
    shift = (dp != 0.0) & ~np.isnan(dp)

    def _rate(dm_: np.ndarray) -> float:
        # Cells where the LLM did not move (or is missing) do not count (#36).
        valid = shift & ~np.isnan(dm_) & (dm_ != 0.0)
        if not np.any(valid):
            return float("nan")
        return float(np.mean((np.sign(dm_) == sign_dp)[valid]))

    return within_cluster_permutation_test(
        _rate, dm, pids, n_permutations=n_permutations,
        rng=np.random.default_rng(seed),
    )


# ---------------------------------------------------------------------------
# Class-level correspondence (formerly H3)
# ---------------------------------------------------------------------------


def compute_h3(
    cells_df: pd.DataFrame,
    model: str,
    categories: List[str],
    n_permutations: int = 10000,
    seed: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute class-level correspondence (formerly H3) between physician and LLM deltas.

    Parameters
    ----------
    cells_df:
        Paired cell deltas for a single model.
    model:
        Model identifier.
    categories:
        Canonical category names.
    n_permutations:
        Number of permutation iterations.
    seed:
        RNG seed.

    Returns
    -------
    (effects_df, correlations_df)
        effects_df: one row per category + aggregate.
        correlations_df: one row for the model with Pearson/Spearman for
        both correct and shuffled contexts.
    """
    df = cells_df.copy()

    # ------------------------------------------------------------------
    # Per-category means
    # ------------------------------------------------------------------
    effect_rows: list[dict] = []
    cat_mean_dp: list[float] = []
    cat_mean_llm_c: list[float] = []
    cat_mean_llm_s: list[float] = []

    for cat in categories:
        if "category" in df.columns:
            sub = df[df["category"] == cat]
        else:
            sub = df
        if len(sub) == 0:
            cat_mean_dp.append(np.nan)
            cat_mean_llm_c.append(np.nan)
            cat_mean_llm_s.append(np.nan)
            effect_rows.append(_nan_h3_effect_row(model, cat))
            continue
        mdp = float(sub["delta_physician"].mean())
        madp = float(sub["delta_physician"].abs().mean())
        mdc = float(sub["delta_llm_correct"].mean())
        madc = float(sub["delta_llm_correct"].abs().mean())
        mds = float(sub["delta_llm_shuffled"].mean())
        mads = float(sub["delta_llm_shuffled"].abs().mean())
        cat_mean_dp.append(mdp)
        cat_mean_llm_c.append(mdc)
        cat_mean_llm_s.append(mds)
        effect_rows.append(
            {
                "model": model,
                "category": cat,
                "mean_delta_physician": mdp,
                "mean_abs_delta_physician": madp,
                "mean_delta_llm_correct": mdc,
                "mean_abs_delta_llm_correct": madc,
                "mean_delta_llm_shuffled": mds,
                "mean_abs_delta_llm_shuffled": mads,
            }
        )

    # Aggregate row
    agg_row = {
        "model": model,
        "category": _AGGREGATE_LABEL,
        "mean_delta_physician": float(df["delta_physician"].mean()),
        "mean_abs_delta_physician": float(df["delta_physician"].abs().mean()),
        "mean_delta_llm_correct": float(df["delta_llm_correct"].mean()),
        "mean_abs_delta_llm_correct": float(df["delta_llm_correct"].abs().mean()),
        "mean_delta_llm_shuffled": float(df["delta_llm_shuffled"].mean()),
        "mean_abs_delta_llm_shuffled": float(df["delta_llm_shuffled"].abs().mean()),
    }
    effects_df = pd.DataFrame([agg_row] + effect_rows)

    # ------------------------------------------------------------------
    # Correlations across category-level means
    # ------------------------------------------------------------------
    x = np.array(cat_mean_dp, dtype=float)
    yc = np.array(cat_mean_llm_c, dtype=float)
    ys = np.array(cat_mean_llm_s, dtype=float)

    valid = ~(np.isnan(x) | np.isnan(yc) | np.isnan(ys))
    n_cats = int(valid.sum())

    if n_cats < 3:
        nan = float("nan")
        corr_row = {
            "model": model,
            "pearson_correct": nan, "spearman_correct": nan,
            "pearson_shuffled": nan, "spearman_shuffled": nan,
            "pearson_correct_minus_shuffled": nan,
            "spearman_correct_minus_shuffled": nan,
            "permutation_p_pearson_correct": nan,
            "permutation_p_spearman_correct": nan,
            "n_categories": n_cats,
        }
    else:
        xv, ycv, ysv = x[valid], yc[valid], ys[valid]
        pearson_c, _ = scipy.stats.pearsonr(xv, ycv)
        spearman_c, _ = scipy.stats.spearmanr(xv, ycv)
        pearson_s, _ = scipy.stats.pearsonr(xv, ysv)
        spearman_s, _ = scipy.stats.spearmanr(xv, ysv)

        rng_p = np.random.default_rng(seed)
        rng_s = np.random.default_rng(seed + 1)
        p_pearson = permutation_test_h3(xv, ycv, n_permutations, rng_p)
        p_spearman = _permutation_test_spearman(xv, ycv, n_permutations, rng_s)

        corr_row = {
            "model": model,
            "pearson_correct": float(pearson_c),
            "spearman_correct": float(spearman_c),
            "pearson_shuffled": float(pearson_s),
            "spearman_shuffled": float(spearman_s),
            "pearson_correct_minus_shuffled": float(pearson_c - pearson_s),
            "spearman_correct_minus_shuffled": float(spearman_c - spearman_s),
            "permutation_p_pearson_correct": p_pearson,
            "permutation_p_spearman_correct": p_spearman,
            "n_categories": n_cats,
        }

    correlations_df = pd.DataFrame([corr_row])
    return effects_df, correlations_df


def _nan_h3_effect_row(model: str, category: str) -> dict:
    nan = float("nan")
    return {
        "model": model, "category": category,
        "mean_delta_physician": nan, "mean_abs_delta_physician": nan,
        "mean_delta_llm_correct": nan, "mean_abs_delta_llm_correct": nan,
        "mean_delta_llm_shuffled": nan, "mean_abs_delta_llm_shuffled": nan,
    }


# ---------------------------------------------------------------------------
# Context specificity: correct vs shuffled context (formerly H4)
# ---------------------------------------------------------------------------


def compute_h4(
    cells_df: pd.DataFrame,
    model: str,
    categories: List[str],
    n_bootstrap: int = 1000,
    n_permutations: int = 10000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Compute context specificity (formerly H4): correct-context alignment vs shuffled-context (Δ_physician ≠ 0).

    Parameters
    ----------
    cells_df:
        Paired cell deltas for a single model.
    model:
        Model identifier.
    categories:
        Canonical category names.
    n_bootstrap:
        Bootstrap resamples for CI.
    n_permutations:
        Permutation iterations for p-value.
    seed:
        RNG seed.

    Returns
    -------
    pd.DataFrame
        One aggregate row + one per-category row.
        Columns: model, category, n_patients, n_shift_cells,
        mean_alignment_correct, mean_alignment_shuffled,
        mean_alignment_difference, ci_low_difference, ci_high_difference,
        paired_permutation_p.
    """
    df = cells_df.copy()
    patient_col = "patient" if "patient" in df.columns else "patient_id"

    rows: list[dict] = []

    def _compute_row(mask: np.ndarray, category: str) -> dict:
        sub = df[mask & (df["delta_physician"] != 0.0)]
        if len(sub) == 0:
            return _nan_h4_row(model, category)

        pids = sub[patient_col].astype(str).to_numpy()
        n_patients = int(sub[patient_col].nunique())
        n_shift_cells = len(sub)

        ac = sub["alignment_correct"].to_numpy(dtype=float)
        as_ = sub["alignment_shuffled"].to_numpy(dtype=float)
        diff = sub["alignment_difference"].to_numpy(dtype=float)

        mean_ac = float(np.mean(ac))
        mean_as = float(np.mean(as_))
        pt_diff, lo, hi = _bootstrap_scalar(diff, pids, n_bootstrap, seed)

        rng_perm = np.random.default_rng(seed + 2)
        p = permutation_test_h4(ac, as_, pids, n_permutations, rng_perm)

        return {
            "model": model,
            "category": category,
            "n_patients": n_patients,
            "n_shift_cells": n_shift_cells,
            "mean_alignment_correct": mean_ac,
            "mean_alignment_shuffled": mean_as,
            "mean_alignment_difference": pt_diff,
            "ci_low_difference": lo,
            "ci_high_difference": hi,
            "paired_permutation_p": p,
        }

    rows.append(_compute_row(np.ones(len(df), dtype=bool), _AGGREGATE_LABEL))
    for cat in categories:
        if "category" in df.columns:
            mask = df["category"].to_numpy() == cat
        else:
            mask = np.ones(len(df), dtype=bool)
        rows.append(_compute_row(mask, cat))

    return pd.DataFrame(rows)


def _nan_h4_row(model: str, category: str) -> dict:
    nan = float("nan")
    return {
        "model": model, "category": category,
        "n_patients": 0, "n_shift_cells": 0,
        "mean_alignment_correct": nan, "mean_alignment_shuffled": nan,
        "mean_alignment_difference": nan,
        "ci_low_difference": nan, "ci_high_difference": nan,
        "paired_permutation_p": nan,
    }
