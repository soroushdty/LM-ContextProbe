"""Patient-cluster randomization tests.

Cells (for example (patient, item, class) observations) from the same
patient share a context snapshot and a reference observer group, so they
are correlated.  Tests that treat cells as independent understate the
p-value.  The sign-flip test here keeps each patient's cells together: under
the null, each patient's summed contribution is symmetric about zero, so its
sign can be flipped as a block.

With few patients the test is exact: all ``2 ** n_clusters`` sign patterns
are enumerated when that number does not exceed ``n_permutations``.  This
also means the smallest attainable one-sided p-value is ``2 ** -n_clusters``
(1/64 with six patients), which is the honest limit of a design with that
many patients.

The within-cluster permutation test asks a different question: whether a
model's deltas line up with the reference deltas of *the same item*, rather
than merely with the patient's items in general.  It shuffles the model's
delta rows among the items of each patient and leaves the reference deltas
in place.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np


def cluster_sign_flip_test(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """One-sided cluster sign-flip test of ``mean(values) > 0``.

    The statistic is the mean over all cells.  Each null replicate multiplies
    every cell of a cluster by the same random sign, so within-cluster
    correlation is preserved.

    Args:
        values: (n,) cell-level values whose mean is zero under the null
            (for example ``agree - 0.5`` or a paired difference).
        cluster_ids: (n,) cluster (patient) ID for each cell.
        n_permutations: Monte Carlo replicates when exact enumeration is too
            large.  Exact enumeration is used when
            ``2 ** n_clusters <= n_permutations``.
        rng: Generator for the Monte Carlo path.  Not used when the test is
            exact.

    Returns:
        One-sided p-value.  NaN if there are no cells or fewer than two
        clusters; 1.0 if every cluster sums to zero.
    """
    values = np.asarray(values, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    if values.shape != cluster_ids.shape:
        raise ValueError(
            f"values and cluster_ids must have the same shape, got "
            f"{values.shape} and {cluster_ids.shape}"
        )
    if values.size == 0:
        return float("nan")

    _, inverse = np.unique(cluster_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=values)  # (n_clusters,)
    n_clusters = sums.size
    if n_clusters < 2:
        return float("nan")
    if np.all(sums == 0.0):
        return 1.0

    observed = float(sums.sum())
    tol = 1e-12 * max(1.0, float(np.abs(sums).sum()))

    if 2 ** n_clusters <= n_permutations:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n_clusters)))
        null = signs @ sums
        return float(np.mean(null >= observed - tol))

    if rng is None:
        rng = np.random.default_rng()
    signs = rng.choice(np.array([1.0, -1.0]), size=(n_permutations, n_clusters))
    null = signs @ sums
    return float((1 + np.sum(null >= observed - tol)) / (n_permutations + 1))


def within_cluster_permutation_test(
    statistic_fn: Callable[[np.ndarray], float],
    rows: np.ndarray,
    cluster_ids: np.ndarray,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
    size_fn: Callable[[np.ndarray], float] | None = None,
) -> dict[str, float]:
    """One-sided test that permutes ``rows`` within each cluster.

    Each null replicate reorders the rows of every cluster at random and
    recomputes ``statistic_fn``.  A row moves as a whole, so dependence
    between the columns of one row (for example the classes of one item) is
    kept.  Clusters with a single row contribute nothing to the null.

    ``size_fn`` studentizes a statistic that is a mean over a number of
    cells that itself depends on the permutation.  A sign-agreement rate
    counts only cells where the model moved, and models move most on the
    items where the reference moves, so permuted rows land on fewer counted
    cells and the raw null is too wide: the test rejected in about 2% of
    null replicates at alpha = 0.05 (#39).  With ``size_fn``, the observed
    and every null replicate are compared as
    ``(statistic - null_mean) * sqrt(size_fn(rows))``.

    Args:
        statistic_fn: Maps a permuted copy of ``rows`` (same shape) to a
            scalar.  It closes over whatever stays fixed, such as the
            reference deltas.
        rows: (n, ...) array whose rows are exchangeable within a cluster
            under the null (for example model deltas, one row per item).
        cluster_ids: (n,) cluster (patient) ID for each row.
        n_permutations: Monte Carlo replicates.
        rng: Generator for the permutations.
        size_fn: Optional; maps rows (observed or permuted) to the number of
            cells the statistic averages over.

    Returns:
        Dict with ``observed``, ``null_mean`` (the statistic's value expected
        by chance), ``null_std`` and ``p_value`` = (1 + #{null >= observed}) /
        (n_permutations + 1), on the studentized scale when ``size_fn`` is
        given.  All NaN if the observed statistic is NaN.
    """
    rows = np.asarray(rows)
    cluster_ids = np.asarray(cluster_ids)
    if rows.shape[0] != cluster_ids.shape[0]:
        raise ValueError(
            f"rows and cluster_ids must have the same length, got "
            f"{rows.shape[0]} and {cluster_ids.shape[0]}"
        )

    nan = float("nan")
    observed = float(statistic_fn(rows))
    if np.isnan(observed):
        return {"observed": nan, "null_mean": nan, "null_std": nan, "p_value": nan}

    if rng is None:
        rng = np.random.default_rng()

    groups = [np.flatnonzero(cluster_ids == c) for c in np.unique(cluster_ids)]
    groups = [g for g in groups if g.size > 1]
    order = np.arange(rows.shape[0])

    null = np.empty(n_permutations, dtype=np.float64)
    sizes = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = order.copy()
        for g in groups:
            perm[g] = rng.permutation(g)
        null[i] = statistic_fn(rows[perm])
        if size_fn is not None:
            sizes[i] = size_fn(rows[perm])

    null_mean = float(np.nanmean(null))
    finite = null[np.isfinite(null)]  # an odds ratio can be infinite
    null_std = float(np.std(finite)) if finite.size else nan
    test_null, test_observed = null, observed
    if size_fn is not None:
        test_null = (null - null_mean) * np.sqrt(sizes)
        test_observed = (observed - null_mean) * np.sqrt(float(size_fn(rows)))

    tol = 1e-12 * max(1.0, abs(test_observed))
    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": float((1 + np.sum(test_null >= test_observed - tol)) / (n_permutations + 1)),
    }
