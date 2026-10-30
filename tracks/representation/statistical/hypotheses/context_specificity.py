"""Context specificity for Track 1.

Same endpoint and statistic as Track 3's context specificity (formerly
Track 3's H4); see shared/endpoints.py.  Both tracks call
``shared.evaluation.hypothesis_analyzer.context_specificity_test``.

On cells with a non-zero physician delta, the alignment of a model delta is
``sign(Δ_p) · Δ_m``.  The statistic is the mean of
``alignment(Δ_m) − alignment(Δ_shuffled)``, where Δ_shuffled is the change
the fold-pure Stage 2 head makes when it is fed another patient's context
vector.  A positive value means the model moves toward the physicians more
with the correct patient's context than with another patient's, rather than
shifting the same way whatever context it is given.

Two outputs:
  1. Pooled over confirmatory-eligible classes, as Track 1's pooled
     directional alignment is: patient-cluster bootstrap CI and one-sided
     patient-cluster sign-flip test.  This is the confirmatory result.
  2. Per class, with the same statistic and test and BH correction across
     eligible classes.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from shared.evaluation.hypothesis_analyzer import context_specificity_test

from .h1 import _bh_correct

logger = logging.getLogger(__name__)


def _complete_rows(*arrays: np.ndarray) -> np.ndarray:
    """Rows with no NaN in any of the (n, n_classes) arrays."""
    return ~np.any(np.stack([np.isnan(a) for a in arrays]), axis=(0, 2))


def _result_fields(res: dict) -> dict:
    return {
        "n_cells": res["n_cells"],
        "n_patients": res["n_patients"],
        "mean_alignment_correct": res["mean_alignment_correct"],
        "mean_alignment_shuffled": res["mean_alignment_shuffled"],
        "mean_alignment_difference": res["mean_diff"],
        "ci_lower": res["ci_lower"],
        "ci_upper": res["ci_upper"],
        "p_value": res["p_value"],
    }


def context_specificity_pooled(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    delta_shuffled: np.ndarray,
    patient_ids: np.ndarray,
    eligible_classes: list[str],
    class_list: list[str],
    n_resamples: int = 1000,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Context specificity pooled over confirmatory-eligible classes.

    Rows with NaN in any delta (for example a test row whose fold has no
    Stage 2 head) are left out.

    Returns a dict with keys:
        classes, n_rows, n_cells, n_patients, mean_alignment_correct,
        mean_alignment_shuffled, mean_alignment_difference, ci_lower,
        ci_upper, p_value
    """
    dp = np.asarray(delta_p, dtype=np.float64)
    dm = np.asarray(delta_m, dtype=np.float64)
    ds = np.asarray(delta_shuffled, dtype=np.float64)
    pid = np.asarray(patient_ids)

    eligible_set = set(eligible_classes)
    col_mask = np.array([cls in eligible_set for cls in class_list], dtype=bool)
    keep = _complete_rows(dp, dm, ds)
    if not keep.all():
        logger.warning(
            "Context specificity: %d of %d rows have missing predictions and are left out.",
            int((~keep).sum()), keep.size,
        )

    res = context_specificity_test(
        delta_model=dm[keep][:, col_mask],
        delta_shuffled=ds[keep][:, col_mask],
        delta_reference=dp[keep][:, col_mask],
        patient_ids=pid[keep],
        n_bootstrap=n_resamples,
        n_permutations=n_permutations,
        bootstrap_rng=rng,
        permutation_rng=rng,
    )
    return {
        "classes": [cls for cls in class_list if cls in eligible_set],
        "n_rows": int(keep.sum()),
        **_result_fields(res),
    }


def context_specificity_per_class(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    delta_shuffled: np.ndarray,
    patient_ids: np.ndarray,
    class_list: list[str],
    eligible_classes: list[str],
    n_resamples: int = 1000,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Context specificity for each class.

    BH correction (q=0.05) is applied to ``p_value`` across
    confirmatory-eligible classes only.

    Returns a DataFrame with columns:
      Class, n_cells, n_patients, mean_alignment_correct,
      mean_alignment_shuffled, mean_alignment_difference, ci_lower,
      ci_upper, p_value, bh_adj_p, confirmatory
    """
    dp = np.asarray(delta_p, dtype=np.float64)
    dm = np.asarray(delta_m, dtype=np.float64)
    ds = np.asarray(delta_shuffled, dtype=np.float64)
    pid = np.asarray(patient_ids)
    keep = _complete_rows(dp, dm, ds)
    eligible_set = set(eligible_classes)

    rows = []
    confirmatory_p_values: list[float] = []
    confirmatory_indices: list[int] = []
    for c_idx, cls in enumerate(class_list):
        res = context_specificity_test(
            delta_model=dm[keep][:, c_idx : c_idx + 1],
            delta_shuffled=ds[keep][:, c_idx : c_idx + 1],
            delta_reference=dp[keep][:, c_idx : c_idx + 1],
            patient_ids=pid[keep],
            n_bootstrap=n_resamples,
            n_permutations=n_permutations,
            bootstrap_rng=rng,
            permutation_rng=rng,
        )
        rows.append({
            "Class": cls,
            **_result_fields(res),
            "bh_adj_p": float("nan"),  # filled below
            "confirmatory": cls in eligible_set,
        })
        if cls in eligible_set and res["n_cells"] > 0:
            # Untestable (fewer than two patients) counts as p = 1, as in H1.
            p_val = res["p_value"]
            confirmatory_p_values.append(1.0 if np.isnan(p_val) else p_val)
            confirmatory_indices.append(c_idx)

    df = pd.DataFrame(rows)
    if confirmatory_p_values:
        adj = _bh_correct(np.array(confirmatory_p_values), q=0.05)
        for adj_val, row_idx in zip(adj, confirmatory_indices):
            df.at[row_idx, "bh_adj_p"] = float(adj_val)
    return df
