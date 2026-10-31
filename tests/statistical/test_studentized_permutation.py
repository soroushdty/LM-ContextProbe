"""Within-patient permutation test of sign agreement, studentized by the number
of moved cells (#39).

The sign-agreement rate counts only cells where both the reference and the
model moved.  Models move most on the items where the reference moves, so
permuted model rows land on fewer counted cells, and the raw permutation null
was too wide: with scores rounded to 0.05 the test rejected in about 2% of
null replicates at alpha = 0.05.  Comparing ``(rate - null_mean) * sqrt(n)``
instead brings the level back to about 4-5%; the simulation study
(docs/simulation.md) has the numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from shared.statistical.cluster_tests import within_cluster_permutation_test
from shared.statistical.delta import sign_agreement_cells, sign_agreement_rate


def _rate_and_size(dp):
    return (lambda dm: sign_agreement_rate(dp, dm),
            lambda dm: float(sign_agreement_cells(dp, dm).sum()))


class TestSizeFn:
    def test_constant_size_gives_the_raw_p_value(self):
        rng = np.random.default_rng(0)
        dp = rng.choice([-1.0, 1.0], size=(40, 3))
        dm = rng.normal(size=(40, 3))  # continuous: every cell moved
        pids = np.repeat(np.arange(8), 5)
        rate, size = _rate_and_size(dp)
        raw = within_cluster_permutation_test(rate, dm, pids, 300, np.random.default_rng(1))
        stud = within_cluster_permutation_test(rate, dm, pids, 300, np.random.default_rng(1), size_fn=size)
        assert stud == raw

    def test_p_value_is_computed_on_the_studentized_scale(self):
        rng = np.random.default_rng(2)
        dp = rng.choice([-1.0, 0.0, 1.0], size=(40, 3))
        dm = np.where(rng.random((40, 3)) < 0.5, 0.0, rng.normal(size=(40, 3)))
        pids = np.repeat(np.arange(8), 5)
        rate, size = _rate_and_size(dp)
        res = within_cluster_permutation_test(rate, dm, pids, 300, np.random.default_rng(3), size_fn=size)

        # Same permutations by hand.
        perm_rng = np.random.default_rng(3)
        groups = [np.flatnonzero(pids == p) for p in np.unique(pids)]
        null, sizes = [], []
        for _ in range(300):
            perm = np.arange(40)
            for g in groups:
                perm[g] = perm_rng.permutation(g)
            null.append(rate(dm[perm]))
            sizes.append(size(dm[perm]))
        null, sizes = np.array(null), np.array(sizes)
        z_null = (null - null.mean()) * np.sqrt(sizes)
        z_obs = (rate(dm) - null.mean()) * np.sqrt(size(dm))
        assert res["null_mean"] == pytest.approx(null.mean())
        assert res["null_std"] == pytest.approx(null.std())
        assert res["p_value"] == pytest.approx((1 + np.sum(z_null >= z_obs - 1e-12)) / 301)

