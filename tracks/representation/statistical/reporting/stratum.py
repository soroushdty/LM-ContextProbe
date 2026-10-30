"""Stratum repeated vs. novel sub-analysis.

Each paired (item, patient) observation is assigned to one of
two evaluation strata:
  repeated — item text appeared in Stage 1 training under a different patient
  novel    — item with no Stage 1 training appearance

The Mann-Whitney U test compares H1 sign agreement rate and H2 Brier
improvement between the two strata.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from shared.statistical.bootstrap import bootstrap_scalar
from shared.statistical.delta import sign_agreement_cells, sign_agreement_mask


def assign_test_strata(
    item_texts_test: np.ndarray,
    item_texts_train: np.ndarray,
) -> np.ndarray:
    """Assign each test item to 'repeated' or 'novel'.

    Uses the same normalize_for_matching() function as the fold loop to
    ensure consistent text normalisation.

    Args:
        item_texts_test:  (n_test,) item strings for the paired eval items.
        item_texts_train: (n_train,) item strings from the training set (all patients).

    Returns:
        (n_test,) str array: each element is 'repeated' or 'novel'.
    """
    from shared.utils.text_utils import normalize_for_matching

    train_norms = set(
        normalize_for_matching(s) for s in item_texts_train
    )
    strata = np.where(
        np.array([normalize_for_matching(s) in train_norms for s in item_texts_test]),
        "repeated",
        "novel",
    )
    return strata


def stratum_comparison(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    y_hat_cf: np.ndarray,
    y_hat_ca: np.ndarray,
    y_interview: np.ndarray,
    strata: np.ndarray,
    patient_ids: np.ndarray,
    eligible_classes: list[str],
    class_list: list[str],
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per-stratum sign agreement rate and Brier improvement with Mann-Whitney U.

    For each stratum (repeated / novel):
      - sign_agree_rate: pooled over confirmatory-eligible classes, on cells with Δ_p ≠ 0 and Δ_m ≠ 0
      - brier_improvement: mean(BS_cf - BS_ca) pooled over all classes
      - 95% patient-level bootstrap CI on each

    Returns DataFrame with columns:
        stratum, n_items, sign_agree_rate, ci_lower_sar, ci_upper_sar,
        brier_improvement, ci_lower_bi, ci_upper_bi,
        mannwhitney_sar_U, mannwhitney_sar_p,
        mannwhitney_bi_U,  mannwhitney_bi_p
    """
    dp = np.asarray(delta_p, dtype=np.float32)
    dm = np.asarray(delta_m, dtype=np.float32)
    cf = np.asarray(y_hat_cf, dtype=np.float64)
    ca = np.asarray(y_hat_ca, dtype=np.float64)
    yi = np.asarray(y_interview, dtype=np.float64)
    pid = np.asarray(patient_ids)
    st = np.asarray(strata)

    eligible_set = set(eligible_classes)
    eligible_cols = np.array([cls in eligible_set for cls in class_list])

    strata_vals = ["repeated", "novel"]
    stratum_data: dict[str, dict] = {}

    for sv in strata_vals:
        mask = st == sv
        if mask.sum() == 0:
            stratum_data[sv] = {
                "n_items": 0,
                "sar_vals": np.array([], dtype=float),
                "bi_vals": np.array([], dtype=float),
                "pid_vals": np.array([], dtype=object),
            }
            continue

        dp_s = dp[mask]
        dm_s = dm[mask]
        cf_s = cf[mask]
        ca_s = ca[mask]
        yi_s = yi[mask]
        pid_s = pid[mask]

        # Sign agreement — pool eligible classes
        dp_elig = dp_s[:, eligible_cols]
        dm_elig = dm_s[:, eligible_cols]
        agree = sign_agreement_mask(dp_elig, dm_elig)
        nz = sign_agreement_cells(dp_elig, dm_elig)
        # Per-item: fraction of eligible-class positions that agree (among cells
        # where both Δ_p and Δ_m are non-zero; #36)
        nz_count = nz.sum(axis=1)
        agree_count = agree.sum(axis=1)
        # Only include items that have at least one such cell
        has_nz = nz_count > 0
        sar_vals = np.where(has_nz, agree_count / np.maximum(nz_count, 1), float("nan"))
        sar_vals = sar_vals[has_nz]
        pid_sar = pid_s[has_nz]

        # Brier improvement per item (pooled over all classes)
        bi_vals = ((cf_s - yi_s) ** 2 - (ca_s - yi_s) ** 2).mean(axis=1)

        stratum_data[sv] = {
            "n_items": int(mask.sum()),
            "sar_vals": sar_vals,
            "bi_vals": bi_vals,
            "pid_vals": pid_s,
            "pid_sar": pid_sar,
        }

    rows = []
    for sv in strata_vals:
        d = stratum_data[sv]
        if d["n_items"] == 0:
            rows.append({
                "stratum": sv, "n_items": 0,
                "sign_agree_rate": float("nan"), "ci_lower_sar": float("nan"), "ci_upper_sar": float("nan"),
                "brier_improvement": float("nan"), "ci_lower_bi": float("nan"), "ci_upper_bi": float("nan"),
            })
            continue

        sar_vals = d["sar_vals"]
        bi_vals = d["bi_vals"]
        pid_vals = d["pid_vals"]
        pid_sar = d.get("pid_sar", pid_vals)

        sar_pt, sar_lo, sar_hi = bootstrap_scalar(sar_vals, pid_sar, np.nanmean, n_resamples, rng) if len(sar_vals) else (float("nan"), float("nan"), float("nan"))
        bi_pt, bi_lo, bi_hi = bootstrap_scalar(bi_vals, pid_vals, np.mean, n_resamples, rng)

        rows.append({
            "stratum": sv,
            "n_items": d["n_items"],
            "sign_agree_rate": float(np.nanmean(sar_vals)) if len(sar_vals) else float("nan"),
            "ci_lower_sar": sar_lo,
            "ci_upper_sar": sar_hi,
            "brier_improvement": float(bi_vals.mean()),
            "ci_lower_bi": bi_lo,
            "ci_upper_bi": bi_hi,
        })

    df = pd.DataFrame(rows)

    # Mann-Whitney U between strata
    rep = stratum_data["repeated"]
    nov = stratum_data["novel"]

    for metric, key in [("sar", "sar_vals"), ("bi", "bi_vals")]:
        r_vals = rep.get(key, np.array([]))
        n_vals = nov.get(key, np.array([]))
        r_valid = r_vals[~np.isnan(r_vals)] if len(r_vals) else r_vals
        n_valid = n_vals[~np.isnan(n_vals)] if len(n_vals) else n_vals

        if len(r_valid) >= 1 and len(n_valid) >= 1:
            try:
                res = mannwhitneyu(r_valid, n_valid, alternative="two-sided")
                df[f"mannwhitney_{metric}_U"] = float(res.statistic)
                df[f"mannwhitney_{metric}_p"] = float(res.pvalue)
            except Exception:
                df[f"mannwhitney_{metric}_U"] = float("nan")
                df[f"mannwhitney_{metric}_p"] = float("nan")
        else:
            df[f"mannwhitney_{metric}_U"] = float("nan")
            df[f"mannwhitney_{metric}_p"] = float("nan")

    return df
