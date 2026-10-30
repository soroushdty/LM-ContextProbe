"""Directional alignment (Track 1's H1) hypothesis tests.

Same endpoint as Track 3's directional alignment (formerly Track 3's H2);
see shared/endpoints.py.

Three prespecified inferences (see the Track 1 endpoints in docs/methodology.md):
  1. Per-class within-patient permutation test of sign agreement against
     its chance level (BH FDR q=0.05); the sign-flip test against 0.5 and
     the exact binomial p-value are kept as references
  2. Patient-clustered permutation test on aggregate sign agreement
  3. Cochran-Mantel-Haenszel (CMH) pooled cross-class inference, with a
     within-patient permutation p-value for the common odds ratio

The confirmatory tests compare with a within-patient permutation null
rather than with 0.5: a model whose deltas mostly go the way most reference
deltas go agrees more than half the time without matching any particular
item.  The null shuffles the model's deltas among the items of each patient
and keeps the reference deltas in place, so it keeps how often and in which
direction the model moves and breaks only the item-level match.

Every test counts only cells where both the reference and the model moved
(Δ_p ≠ 0 and Δ_m ≠ 0; ``shared.statistical.delta.sign_agreement_cells``).
Counting a zero model delta as disagreement let a model that moves where
the reference moves, in random directions, pass these tests (#36).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from shared.statistical.bootstrap import bootstrap_rate
from shared.statistical.cluster_tests import (
    cluster_sign_flip_test,
    within_cluster_permutation_test,
)
from shared.statistical.delta import (
    sign_agreement_cells,
    sign_agreement_mask,
    sign_agreement_rate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BH correction helper (scipy ≥1.11 has false_discovery_control; fall back)
# ---------------------------------------------------------------------------

def _bh_correct(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    try:
        from scipy.stats import false_discovery_control
        return false_discovery_control(p_values, method="bh")
    except ImportError:
        pass
    # Manual BH
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    ranks = np.argsort(order) + 1  # 1-based
    adj = np.minimum(1.0, p_values * n / ranks)
    # Enforce monotonicity (take cumulative minimum from largest rank)
    adj_sorted = adj[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = adj_sorted
    return adj


# ---------------------------------------------------------------------------
# 1. Per-class binomial test
# ---------------------------------------------------------------------------

def h1_binomial_per_class(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    class_list: list[str],
    eligible_classes: list[str],
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
    n_permutations: int = 10_000,
    perm_rng: np.random.Generator | None = None,
    null_rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per-class test of sign agreement against its chance level (H1 per-class).

    For each class c, restricted to items with Δ_p(i,c) ≠ 0 and Δ_m(i,c) ≠ 0
    (``n_moved``; ``n_nonzero`` counts Δ_p ≠ 0 alone):
      - Compute sign agreement rate (proportion where sign(Δ_m)==sign(Δ_p))
      - Within-patient permutation test: the class's model deltas are
        shuffled among each patient's items, the reference deltas stay in
        place, and the rate is recomputed.  ``null_rate`` is the null's mean
        (the chance agreement rate) and ``perm_p`` the one-sided p-value.
        This is the confirmatory p-value.
      - Patient-cluster sign-flip test on (agree − 0.5) (``cluster_p``) and
        the exact binomial test of rate > 0.5 (``binom_p``), for reference
        only: 0.5 is not the chance level when the model and the reference
        shift mostly in one direction.
      - 95% patient-level bootstrap CI on the agreement rate
    BH FDR correction (q=0.05) on perm_p, across confirmatory-eligible
    classes only.

    ``perm_rng`` drives the sign-flip test's Monte Carlo path (unused when the
    test is exact) and ``null_rng`` the permutation test; both are separate
    from ``rng`` so that the tests do not change the bootstrap draws.

    Returns a DataFrame with columns:
      Class, n_nonzero, n_moved, sign_agree_rate, ci_lower, ci_upper,
      null_rate, perm_p, cluster_p, binom_p, bh_adj_p, confirmatory
    """
    dp = np.asarray(delta_p, dtype=np.float32)
    dm = np.asarray(delta_m, dtype=np.float32)
    pid = np.asarray(patient_ids)
    eligible_set = set(eligible_classes)

    rows = []
    confirmatory_p_values: list[float] = []
    confirmatory_indices: list[int] = []

    for c_idx, cls in enumerate(class_list):
        n_nz = int((dp[:, c_idx] != 0.0).sum())
        moved = sign_agreement_cells(dp[:, c_idx], dm[:, c_idx])
        n_moved = int(moved.sum())

        if n_moved == 0:
            rows.append({
                "Class": cls,
                "n_nonzero": n_nz,
                "n_moved": 0,
                "sign_agree_rate": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "null_rate": float("nan"),
                "perm_p": float("nan"),
                "cluster_p": float("nan"),
                "binom_p": float("nan"),
                "bh_adj_p": float("nan"),
                "confirmatory": cls in eligible_set,
            })
            continue

        agree_1d = sign_agreement_mask(dp[:, c_idx], dm[:, c_idx])[moved]
        pid_moved = pid[moved]

        k = int(agree_1d.sum())
        rate = float(k / n_moved)

        # Patient-level bootstrap CI
        point, ci_lo, ci_hi = bootstrap_rate(agree_1d, pid_moved, n_resamples, rng)

        # Exact binomial test (unclustered; descriptive reference)
        binom_p = float(binomtest(k, n_moved, p=0.5, alternative="greater").pvalue)

        # Patient-cluster sign-flip test against 0.5 (reference)
        cluster_p = cluster_sign_flip_test(
            agree_1d.astype(float) - 0.5, pid_moved, n_permutations, perm_rng,
        )

        # Within-patient permutation test against chance (confirmatory)
        def _rate(dm_col: np.ndarray, _dp=dp[:, c_idx]) -> float:
            return sign_agreement_rate(_dp, dm_col)

        null = within_cluster_permutation_test(
            _rate, dm[:, c_idx], pid, n_permutations, null_rng,
        )
        p_val = null["p_value"]

        row_idx = len(rows)
        rows.append({
            "Class": cls,
            "n_nonzero": n_nz,
            "n_moved": n_moved,
            "sign_agree_rate": rate,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "null_rate": null["null_mean"],
            "perm_p": p_val,
            "cluster_p": cluster_p,
            "binom_p": binom_p,
            "bh_adj_p": float("nan"),  # filled below
            "confirmatory": cls in eligible_set,
        })

        if cls in eligible_set:
            # Untestable (fewer than two patients) counts as p = 1, as in H2.
            confirmatory_p_values.append(1.0 if np.isnan(p_val) else p_val)
            confirmatory_indices.append(row_idx)

    df = pd.DataFrame(rows)

    if confirmatory_p_values:
        adj = _bh_correct(np.array(confirmatory_p_values), q=0.05)
        for adj_val, row_idx in zip(adj, confirmatory_indices):
            df.at[row_idx, "bh_adj_p"] = float(adj_val)

    return df


# ---------------------------------------------------------------------------
# 2. Patient-clustered permutation test
# ---------------------------------------------------------------------------

def h1_permutation_test(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    patient_ids: np.ndarray,
    n_permutations: int = 10_000,
    eligible_classes: list[str] | None = None,
    class_list: list[str] | None = None,
    rng: np.random.Generator | None = None,
    n_resamples: int = 1000,
) -> dict:
    """Patient-clustered permutation test on aggregate sign agreement (H1 aggregate).

    Pooled over (item, class) pairs where both the physician and the model
    delta are non-zero, across all confirmatory-eligible classes (or all
    classes if eligible_classes is None).

    Permutation: model deltas are permuted within patient (entire patient row
    vector is permuted together within each patient block), preserving
    within-patient correlation.

    Returns dict with keys:
        aggregate_rate, ci_lower, ci_upper, null_mean, null_std, p_value
    """
    dp = np.asarray(delta_p, dtype=np.float32)
    dm = np.asarray(delta_m, dtype=np.float32)
    pid = np.asarray(patient_ids)

    if rng is None:
        rng = np.random.default_rng()

    # Restrict to eligible classes if specified
    if eligible_classes is not None and class_list is not None:
        eligible_set = set(eligible_classes)
        col_mask = np.array([cls in eligible_set for cls in class_list])
        dp = dp[:, col_mask]
        dm = dm[:, col_mask]

    # Observed aggregate rate
    def _aggregate_rate(dp_: np.ndarray, dm_: np.ndarray) -> float:
        return sign_agreement_rate(dp_, dm_)

    observed = _aggregate_rate(dp, dm)

    # Short-circuit: if no cell has both deltas non-zero (e.g. all dp == 0),
    # the statistic is undefined everywhere — return all NaN without running the
    # permutation loop (which would otherwise trigger np.nanmean("Mean of empty
    # slice") warnings on an all-NaN null_vals array).
    if np.isnan(observed):
        return {
            "aggregate_rate": observed,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "p_value": float("nan"),
            "n_permutations": n_permutations,
        }

    # Bootstrap CI on observed rate
    unique_pts = np.unique(pid)
    pt_to_ix = {pt: np.where(pid == pt)[0] for pt in unique_pts}

    def _boot_rate(idx: np.ndarray) -> float:
        return _aggregate_rate(dp[idx], dm[idx])

    from shared.statistical.bootstrap import patient_block_bootstrap
    ci_lo, ci_hi = patient_block_bootstrap(_boot_rate, pid, n_resamples, rng)

    # Null distribution: permute dm within each patient block
    null_vals = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        dm_perm = dm.copy()
        for pt in unique_pts:
            ix = pt_to_ix[pt]
            # Permute the row ordering within this patient's items
            dm_perm[ix] = dm_perm[rng.permutation(ix)]
        null_vals[i] = _aggregate_rate(dp, dm_perm)

    null_mean = float(np.nanmean(null_vals))
    null_std = float(np.nanstd(null_vals))
    p_value = float(np.mean(null_vals >= observed))

    return {
        "aggregate_rate": observed,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# 3. Cochran-Mantel-Haenszel test
# ---------------------------------------------------------------------------

def h1_cmh_test(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
    eligible_classes: list[str],
    class_list: list[str],
    patient_ids: np.ndarray | None = None,
    n_permutations: int = 10_000,
    perm_rng: np.random.Generator | None = None,
    null_rng: np.random.Generator | None = None,
) -> dict:
    """Class-stratified CMH test for pooled sign-agreement evidence (H1 pooled).

    For each confirmatory-eligible class, builds a valid 2×2 contingency table
    stratified by physician-delta direction:

        Rows    — physician direction:  Δ_p > 0 (row 0) vs Δ_p < 0 (row 1)
        Columns — model direction:      Δ_m > 0 (col 0) vs Δ_m < 0 (col 1)

        Cell (0,0): Δ_p>0, Δ_m>0  → physician-model both positive (agree)
        Cell (0,1): Δ_p>0, Δ_m<0  → physician positive, model negative (disagree)
        Cell (1,0): Δ_p<0, Δ_m>0  → physician negative, model positive (disagree)
        Cell (1,1): Δ_p<0, Δ_m<0  → physician-model both negative (agree)

        Cells where the model did not move (Δ_m = 0) are left out (#36).

        Columns are CONSISTENT across rows (both represent model direction),
        so the common odds ratio is interpretable: OR > 1 means the model
        moves up more often when the reference moved up than when it moved
        down.  Unlike the raw agreement rate, OR = 1 for a model that moves
        up at the same rate whatever the reference did.

    Strata where either physician direction is absent (all Δ_p in one direction)
    are skipped to avoid degenerate tables with a zero row margin.

    Combines tables via statsmodels StratifiedTable (common-odds-ratio test).
    The CMH test counts cells as independent.  When ``patient_ids`` is given,
    ``p_permutation`` is the confirmatory, patient-aware counterpart of
    ``p_cmh``: a one-sided within-patient permutation test of the
    Mantel-Haenszel common odds ratio > 1, in which each patient's model
    delta rows (all included classes together) are shuffled among that
    patient's items.  ``p_cluster`` (patient-cluster sign-flip test of pooled
    sign agreement > 0.5 on the same cells) is kept for reference only.

    Returns dict with keys:
        common_odds_ratio, ci_lower, ci_upper, chi2_cmh, p_cmh,
        p_permutation, p_cluster, n_strata
    Falls back to a manual Mantel-Haenszel calculation if statsmodels is
    unavailable.
    """
    dp = np.asarray(delta_p, dtype=np.float32)
    dm = np.asarray(delta_m, dtype=np.float32)
    eligible_set = set(eligible_classes)

    tables: list[np.ndarray] = []
    included: list[int] = []
    pooled_agree: list[np.ndarray] = []
    pooled_pids: list[np.ndarray] = []
    pid = np.asarray(patient_ids) if patient_ids is not None else None
    for c_idx, cls in enumerate(class_list):
        if cls not in eligible_set:
            continue
        dp_col = dp[:, c_idx]
        dm_col = dm[:, c_idx]

        # Physician-positive and physician-negative item masks
        pos = dp_col > 0.0
        neg = dp_col < 0.0

        # Skip degenerate strata: need at least one item in each physician direction
        if pos.sum() == 0 or neg.sum() == 0:
            continue

        # 2×2: rows = physician direction, cols = model direction (dm>0 vs dm<0)
        # Columns have the SAME meaning across both rows — this is required for a
        # valid CMH common-odds-ratio test.
        up, down = dm_col > 0.0, dm_col < 0.0
        a = int((pos & up).sum())    # Δ_p>0, Δ_m>0  (agree for pos physician)
        b = int((pos & down).sum())  # Δ_p>0, Δ_m<0  (disagree for pos physician)
        c = int((neg & up).sum())    # Δ_p<0, Δ_m>0  (disagree for neg physician)
        d = int((neg & down).sum())  # Δ_p<0, Δ_m<0  (agree for neg physician)

        # Skip if any margin is zero (degenerate for CMH)
        if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue

        tables.append(np.array([[a, b], [c, d]], dtype=float))
        included.append(c_idx)
        if pid is not None:
            moved = sign_agreement_cells(dp_col, dm_col)
            agree = (pos & up) | (neg & down)
            pooled_agree.append(agree[moved].astype(float))
            pooled_pids.append(pid[moved])

    if not tables:
        return {
            "common_odds_ratio": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "chi2_cmh": float("nan"),
            "p_cmh": float("nan"),
            "p_permutation": float("nan"),
            "p_cluster": float("nan"),
            "n_strata": 0,
        }

    p_cluster = p_permutation = float("nan")
    if pid is not None:
        p_cluster = cluster_sign_flip_test(
            np.concatenate(pooled_agree) - 0.5,
            np.concatenate(pooled_pids),
            n_permutations,
            perm_rng,
        )
        dp_inc = dp[:, included]
        pos_inc, neg_inc = dp_inc > 0.0, dp_inc < 0.0

        def _mh_odds_ratio(dm_inc: np.ndarray) -> float:
            up, down = dm_inc > 0.0, dm_inc < 0.0
            n = ((pos_inc | neg_inc) & (up | down)).sum(axis=0)
            n = np.where(n == 0, 1, n)  # a stratum with no moved cells adds 0
            num = ((pos_inc & up).sum(axis=0) * (neg_inc & down).sum(axis=0) / n).sum()
            den = ((pos_inc & down).sum(axis=0) * (neg_inc & up).sum(axis=0) / n).sum()
            if den == 0.0:
                return float("inf") if num > 0.0 else float("nan")
            return float(num / den)

        p_permutation = within_cluster_permutation_test(
            _mh_odds_ratio, dm[:, included], pid, n_permutations, null_rng,
        )["p_value"]

    try:
        from statsmodels.stats.contingency_tables import StratifiedTable
        st = StratifiedTable(np.stack(tables, axis=2))
        summary = st.test_null_odds()
        est = st.oddsratio_pooled
        ci = st.oddsratio_pooled_confint()
        return {
            "common_odds_ratio": float(est),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "chi2_cmh": float(summary.statistic),
            "p_cmh": float(summary.pvalue),
            "p_permutation": p_permutation,
            "p_cluster": p_cluster,
            "n_strata": len(tables),
        }
    except Exception as exc:
        logger.warning("statsmodels StratifiedTable failed (%s); using manual MH.", exc)
        return {**_manual_mh(tables), "p_permutation": p_permutation, "p_cluster": p_cluster}


def _manual_mh(tables: list[np.ndarray]) -> dict:
    """Manual Mantel-Haenszel common odds ratio (without CI)."""
    from scipy.stats import chi2

    num = denom = 0.0
    chi2_num = 0.0

    for t in tables:
        a, b = t[0, 0], t[0, 1]
        c, d = t[1, 0], t[1, 1]
        n = a + b + c + d
        if n == 0:
            continue
        num += a * d / n
        denom += b * c / n
        E_a = (a + b) * (a + c) / n
        V_a = (a + b) * (c + d) * (a + c) * (b + d) / (n ** 2 * (n - 1)) if n > 1 else 0.0
        chi2_num += (a - E_a)

    or_mh = num / denom if denom != 0 else float("nan")
    stat = chi2_num ** 2 / max(sum(
        (t[0, 0] + t[0, 1]) * (t[1, 0] + t[1, 1]) * (t[0, 0] + t[1, 0]) *
        (t[0, 1] + t[1, 1]) / ((sum(t.flat)) ** 2 * (sum(t.flat) - 1))
        for t in tables
        if sum(t.flat) > 1
    ), 1e-12)

    p = float(1.0 - chi2.cdf(stat, df=1))
    return {
        "common_odds_ratio": float(or_mh),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "chi2_cmh": float(stat),
        "p_cmh": p,
        "n_strata": len(tables),
    }
