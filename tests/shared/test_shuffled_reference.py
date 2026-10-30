"""Shuffled-context reference and chance level for Track 3's context
sensitivity and directional alignment (#17).

Context sensitivity is tested against the shuffled-context condition, and
directional alignment against a permutation null that shuffles each
patient's model deltas among that patient's items.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from shared.evaluation.hypothesis_analyzer import HypothesisAnalyzer
from shared.statistical.cluster_tests import within_cluster_permutation_test
from tracks.reasoning.bundle.hypotheses import compute_h1, compute_h2

N_PATIENTS = 8
ITEMS_PER_PATIENT = 6
D = 3

_CFG = {
    "bootstrap_seed": 0,
    "permutation_seed": 0,
    "n_bootstrap_resamples": 200,
    "n_permutations": 2000,
}


class _Dataset:
    def __init__(self) -> None:
        self.patient_ids = np.repeat(
            [f"P{i}" for i in range(N_PATIENTS)], ITEMS_PER_PATIENT
        )
        self.category_names = [f"c{j}" for j in range(D)]


def _analyzer() -> HypothesisAnalyzer:
    return HypothesisAnalyzer(_Dataset(), _CFG)


N = N_PATIENTS * ITEMS_PER_PATIENT
CF = np.full((N, D), 0.5)


def _item_signs(seed: int = 1) -> np.ndarray:
    """A ±1 reference shift per cell that varies between the items of a patient."""
    return np.random.default_rng(seed).choice([-1.0, 1.0], size=(N, D))


# ---------------------------------------------------------------------------
# within_cluster_permutation_test
# ---------------------------------------------------------------------------


class TestWithinClusterPermutationTest:
    def test_rows_identical_within_cluster_give_p_one(self):
        """Permuting identical rows changes nothing, so the null equals the observed value."""
        rows = np.repeat(np.arange(4.0), 3)[:, None]
        ids = np.repeat(["a", "b", "c", "d"], 3)
        target = rows[:, 0] * 2.0
        res = within_cluster_permutation_test(
            lambda r: float(np.mean(r[:, 0] * target)), rows, ids,
            n_permutations=200, rng=np.random.default_rng(0),
        )
        assert res["null_mean"] == pytest.approx(res["observed"])
        assert res["p_value"] == pytest.approx(1.0)

    def test_item_level_match_is_detected(self):
        target = _item_signs()
        pids = _Dataset().patient_ids
        res = within_cluster_permutation_test(
            lambda r: float(np.mean(np.sign(r) == target)), target.copy(), pids,
            n_permutations=500, rng=np.random.default_rng(0),
        )
        assert res["observed"] == pytest.approx(1.0)
        assert 0.35 < res["null_mean"] < 0.65
        assert res["p_value"] == pytest.approx(1 / 501)

    def test_nan_statistic_returns_nan(self):
        res = within_cluster_permutation_test(
            lambda r: float("nan"), np.zeros((4, 1)), np.array(["a", "a", "b", "b"]),
            n_permutations=10,
        )
        assert all(math.isnan(v) for v in res.values())

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            within_cluster_permutation_test(
                lambda r: 0.0, np.zeros((3, 1)), np.array(["a", "b"]),
            )


# ---------------------------------------------------------------------------
# HypothesisAnalyzer — context sensitivity
# ---------------------------------------------------------------------------


class TestContextSensitivityReference:
    def test_any_context_moves_scores_equally(self):
        """Correct and shuffled context move the scores by the same amount: no evidence."""
        cc = CF + 0.2 * _item_signs(1)
        sc = CF - 0.2 * _item_signs(2)
        res = _analyzer().compute_h1(CF, cc, sc)
        assert res["mean_abs_delta"] == pytest.approx(0.2)
        assert res["ci_lower"] > 0  # the old check would have called this a finding
        assert res["mean_abs_delta_shuffled"] == pytest.approx(0.2)
        assert res["mean_abs_delta_difference"] == pytest.approx(0.0, abs=1e-12)
        assert res["p_value"] == pytest.approx(1.0)

    def test_correct_context_moves_scores_more(self):
        cc = CF + 0.3 * _item_signs(1)
        sc = CF + 0.1 * _item_signs(2)
        res = _analyzer().compute_h1(CF, cc, sc)
        assert res["mean_abs_delta_difference"] == pytest.approx(0.2)
        assert res["ci_lower_difference"] > 0
        # Exact sign-flip test: smallest attainable p-value is 2^-n_patients.
        assert res["p_value"] == pytest.approx(2.0 ** -N_PATIENTS)
        for cat in res["per_category"].values():
            assert cat["mean_abs_delta_difference"] == pytest.approx(0.2)

    def test_zero_delta_rates(self):
        cc = CF.copy()
        cc[: N // 2] += 0.1
        sc = CF + 0.1
        res = _analyzer().compute_h1(CF, cc, sc)
        assert res["zero_delta_rate"] == pytest.approx(0.5)
        assert res["zero_delta_rate_shuffled"] == pytest.approx(0.0)

    def test_rows_with_invalid_shuffled_scores_are_left_out(self):
        cc = CF + 0.3
        sc = CF + 0.1
        sc[0, 1] = np.nan
        res = _analyzer().compute_h1(CF, cc, sc)
        assert res["mean_abs_delta"] == pytest.approx(0.3)  # uses every row
        assert res["mean_abs_delta_shuffled"] == pytest.approx(0.1)
        assert res["mean_abs_delta_difference"] == pytest.approx(0.2)

    def test_without_shuffled_scores_reference_is_nan(self):
        res = _analyzer().compute_h1(CF, CF + 0.1)
        assert res["mean_abs_delta"] == pytest.approx(0.1)
        for key in ("mean_abs_delta_shuffled", "mean_abs_delta_difference", "p_value"):
            assert math.isnan(res[key])


# ---------------------------------------------------------------------------
# HypothesisAnalyzer — directional alignment
# ---------------------------------------------------------------------------


class TestDirectionalAlignmentReference:
    def test_uniform_upward_drift_is_chance_agreement(self):
        """Physicians mostly shift up and the model always nudges up.

        Agreement is high, but it says nothing about the items: the
        permutation null reaches the same rate, so the test finds nothing.
        """
        dp = np.full((N, D), 1.0)
        dp[::5] = -1.0
        cc = CF + 0.1
        res = _analyzer().compute_h2(CF, cc, dp, CF + 0.1)
        assert res["sign_agreement_rate"] > 0.75
        assert res["null_sign_agreement_rate"] == pytest.approx(res["sign_agreement_rate"])
        assert res["p_value"] > 0.5
        assert res["sign_agreement_rate_shuffled"] == pytest.approx(res["sign_agreement_rate"])

    def test_item_specific_agreement_beats_chance(self):
        dp = _item_signs(1)
        cc = CF + 0.1 * dp
        res = _analyzer().compute_h2(CF, cc, dp)
        assert res["sign_agreement_rate"] == pytest.approx(1.0)
        assert 0.35 < res["null_sign_agreement_rate"] < 0.65
        assert res["p_value"] < 0.01
        assert math.isnan(res["sign_agreement_rate_shuffled"])

    def test_zero_model_deltas_are_left_out(self):
        """Cells where the model did not move count neither way (#36): with the
        model unmoved on half the shift cells and random elsewhere, chance
        agreement stays near 0.5, and the unmoved share is reported apart."""
        dp = _item_signs(1)
        noise = _item_signs(3)
        cc = CF + 0.1 * noise
        cc[: N // 2] = CF[: N // 2]
        res = _analyzer().compute_h2(CF, cc, dp)
        assert res["zero_delta_rate"] == pytest.approx(0.5)
        assert res["null_sign_agreement_rate"] == pytest.approx(0.5, abs=0.08)
        moved = cc != CF
        expected = np.mean((np.sign(cc - CF) == dp)[moved])
        assert res["sign_agreement_rate"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Post-hoc bundle
# ---------------------------------------------------------------------------


def _cells(dp: np.ndarray, dc: np.ndarray, ds: np.ndarray) -> pd.DataFrame:
    pids = _Dataset().patient_ids
    rows = []
    for i in range(N):
        for j in range(D):
            rows.append({
                "model": "m", "patient": pids[i], "item_text": f"item{i}",
                "category": f"c{j}", "delta_physician": dp[i, j],
                "delta_llm_correct": dc[i, j], "delta_llm_shuffled": ds[i, j],
            })
    return pd.DataFrame(rows)


CATS = [f"c{j}" for j in range(D)]


class TestBundleReference:
    def test_h1_difference_and_sign_flip(self):
        cells = _cells(_item_signs(1), 0.3 * _item_signs(1), 0.1 * _item_signs(2))
        h1 = compute_h1(cells, "m", CATS, n_bootstrap=100, seed=0, n_permutations=2000)
        agg = h1[h1["category"] == "aggregate"].iloc[0]
        assert agg["mean_abs_delta_difference"] == pytest.approx(0.2)
        assert agg["ci_low_difference"] > 0
        assert agg["paired_permutation_p"] == pytest.approx(2.0 ** -N_PATIENTS)

    def test_h2_permutation_null_on_aggregate_row(self):
        dp = _item_signs(1)
        cells = _cells(dp, 0.1 * dp, np.zeros((N, D)))
        h2 = compute_h2(cells, "m", CATS, n_bootstrap=100, seed=0, n_permutations=500)
        agg = h2[h2["category"] == "aggregate"].iloc[0]
        assert agg["sign_agreement_correct"] == pytest.approx(1.0)
        assert 0.35 < agg["null_sign_agreement_correct"] < 0.65
        assert agg["permutation_p_correct"] == pytest.approx(1 / 501)
        assert agg["unchanged_rate_shuffled"] == pytest.approx(1.0)
        per_cat = h2[h2["category"] != "aggregate"]
        assert per_cat["permutation_p_correct"].isna().all()

    def test_h2_uniform_drift_matches_null(self):
        dp = np.full((N, D), 1.0)
        dp[::5] = -1.0
        cells = _cells(dp, np.full((N, D), 0.1), np.full((N, D), 0.1))
        h2 = compute_h2(cells, "m", CATS, n_bootstrap=100, seed=0, n_permutations=500)
        agg = h2[h2["category"] == "aggregate"].iloc[0]
        assert agg["null_sign_agreement_correct"] == pytest.approx(agg["sign_agreement_correct"])
        assert agg["permutation_p_correct"] > 0.5
