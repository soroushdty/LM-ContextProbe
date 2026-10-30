"""Sign agreement leaves out cells where the model did not move (#36).

Models and reference observers both tend to move on the same (ambiguous)
items. When a zero model delta counted as disagreement, a model that moves
exactly where the reference moves, in random directions, beat the
within-patient permutation null almost every time. Each test below plants
that model and checks that no directional-alignment test calls it aligned
more often than chance allows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.evaluation.hypothesis_analyzer import HypothesisAnalyzer
from shared.statistical.delta import sign_agreement_cells, sign_agreement_rate
from tracks.reasoning.bundle.hypotheses import compute_h2 as bundle_h2
from tracks.representation.statistical.hypotheses.h1 import (
    h1_binomial_per_class,
    h1_cmh_test,
    h1_permutation_test,
)

N_PATIENTS, ITEMS, D = 8, 12, 3
N_SEEDS = 30
CLASSES = [f"c{j}" for j in range(D)]
PIDS = np.repeat(np.arange(N_PATIENTS), ITEMS)


def _colocated(seed: int):
    """Reference shifts on ~40% of cells, mostly up; the model moves on exactly
    those cells, in a random direction, and nowhere else."""
    rng = np.random.default_rng(seed)
    shifts = rng.random((PIDS.size, D)) < 0.4
    dp = np.where(shifts, rng.choice([1.0, -0.5, 0.5], size=shifts.shape, p=[0.5, 0.2, 0.3]), 0.0)
    dm = np.where(shifts, rng.choice([-0.1, 0.1], size=shifts.shape), 0.0)
    return dp, dm


def _rejections(p_values) -> int:
    return int(np.sum(np.asarray(p_values) < 0.05))


# With 30 seeds at alpha = 0.05, more than 6 rejections has probability < 0.01.
MAX_REJECTIONS = 6


def test_rate_counts_only_cells_where_both_moved():
    dp = np.array([[1.0, -1.0, 0.0, 0.5]])
    dm = np.array([[0.2, 0.0, 0.3, -0.1]])
    np.testing.assert_array_equal(sign_agreement_cells(dp, dm), [[True, False, False, True]])
    assert sign_agreement_rate(dp, dm) == pytest.approx(0.5)
    assert np.isnan(sign_agreement_rate(dp, np.zeros_like(dm)))


def test_track3_analyzer():
    class _Paired:
        patient_ids = PIDS
        category_names = CLASSES

    p = []
    for seed in range(N_SEEDS):
        dp, dm = _colocated(seed)
        cf = np.full(dp.shape, 0.5)
        an = HypothesisAnalyzer(_Paired(), {"n_bootstrap_resamples": 20, "n_permutations": 300,
                                            "permutation_seed": seed})
        p.append(an.compute_h2(cf, cf + dm, dp)["p_value"])
    assert _rejections(p) <= MAX_REJECTIONS


def test_bundle_aggregate_row():
    p = []
    for seed in range(N_SEEDS):
        dp, dm = _colocated(seed)
        rows = [
            {"model": "m", "patient": f"P{PIDS[i]}", "item_text": f"i{i}", "category": CLASSES[j],
             "delta_physician": dp[i, j], "delta_llm_correct": dm[i, j], "delta_llm_shuffled": 0.0}
            for i in range(PIDS.size) for j in range(D)
        ]
        res = bundle_h2(pd.DataFrame(rows), "m", CLASSES, n_bootstrap=20, n_permutations=300, seed=seed)
        p.append(res[res["category"] == "aggregate"].iloc[0]["permutation_p_correct"])
    assert _rejections(p) <= MAX_REJECTIONS


@pytest.mark.parametrize("test", ["per_class", "pooled", "cmh"])
def test_track1(test):
    p = []
    for seed in range(N_SEEDS):
        dp, dm = _colocated(seed)
        rng = np.random.default_rng(seed)
        if test == "per_class":
            df = h1_binomial_per_class(dp, dm, CLASSES, CLASSES, PIDS, 20, rng,
                                       n_permutations=300, perm_rng=rng, null_rng=rng)
            p.extend(df["perm_p"].tolist())
        elif test == "pooled":
            p.append(h1_permutation_test(dp, dm, PIDS, n_permutations=300, rng=rng, n_resamples=20)["p_value"])
        else:
            p.append(h1_cmh_test(dp, dm, CLASSES, CLASSES, patient_ids=PIDS, n_permutations=300,
                                 perm_rng=rng, null_rng=rng)["p_permutation"])
    limit = MAX_REJECTIONS * (D if test == "per_class" else 1)
    assert _rejections(p) <= limit


def test_item_specific_agreement_is_still_detected():
    """The fix must not cost the test its power: a model that matches the
    reference's direction on the cells where it moves is still found."""
    class _Paired:
        patient_ids = PIDS
        category_names = CLASSES

    dp, _ = _colocated(0)
    cf = np.full(dp.shape, 0.5)
    res = HypothesisAnalyzer(_Paired(), {"n_bootstrap_resamples": 20, "n_permutations": 300}).compute_h2(
        cf, cf + 0.1 * np.sign(dp), dp,
    )
    assert res["sign_agreement_rate"] == pytest.approx(1.0)
    assert res["p_value"] < 0.01
