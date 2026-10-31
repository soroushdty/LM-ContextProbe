"""Delta computation utilities for H1 / H2 analysis.

Δ_p(i,c) = y_interview(i,c) − y_survey(i,c)   physician delta
Δ_m(i,c) = ŷ_ca(i,c) − ŷ_cf(i,c)             model delta

All inputs are pair-aggregated labels in {0.0, 0.5, 1.0} for physician
deltas, and continuous predictions in [0, 1] for model deltas.
"""
from __future__ import annotations

import numpy as np


def compute_physician_deltas(
    y_survey: np.ndarray,
    y_interview: np.ndarray,
) -> np.ndarray:
    """Δ_p(i,c) = y_interview − y_survey.

    Args:
        y_survey:   (n, n_classes) pair-aggregated survey labels in {0, 0.5, 1}.
        y_interview:(n, n_classes) pair-aggregated interview labels in {0, 0.5, 1}.

    Returns:
        (n, n_classes) float32 array with values in {-1, -0.5, 0, +0.5, +1}.
    """
    y_s = np.asarray(y_survey, dtype=np.float32)
    y_i = np.asarray(y_interview, dtype=np.float32)
    return (y_i - y_s).astype(np.float32)


def compute_model_deltas(
    y_hat_cf: np.ndarray,
    y_hat_ca: np.ndarray,
) -> np.ndarray:
    """Δ_m(i,c) = ŷ_ca − ŷ_cf.

    Args:
        y_hat_cf: (n, n_classes) context-free predictions in [0, 1].
        y_hat_ca: (n, n_classes) context-aware predictions in [0, 1].

    Returns:
        (n, n_classes) float32 array with values in [-1, +1].
    """
    cf = np.asarray(y_hat_cf, dtype=np.float32)
    ca = np.asarray(y_hat_ca, dtype=np.float32)
    return (ca - cf).astype(np.float32)


def sign_agreement_mask(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
) -> np.ndarray:
    """Boolean mask: sign(Δ_m) == sign(Δ_p) where Δ_p != 0.

    Returns a boolean array of shape (n, n_classes).  Entries where
    Δ_p == 0 are set to False (not counted in sign agreement).

    A positive/negative sign is defined by np.sign(), so zero Δ_m does
    not agree with any non-zero Δ_p direction.  Sign-agreement rates
    divide by :func:`sign_agreement_cells`, which leaves zero Δ_m out.
    """
    dp = np.asarray(delta_p, dtype=np.float32)
    dm = np.asarray(delta_m, dtype=np.float32)
    nonzero_mask = dp != 0.0
    same_sign = np.sign(dp) == np.sign(dm)
    return nonzero_mask & same_sign


def sign_agreement_cells(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
) -> np.ndarray:
    """Boolean mask of the cells a sign-agreement rate is computed on.

    A cell counts when both the reference and the model moved
    (Δ_p != 0 and Δ_m != 0).  A zero Δ_m is left out rather than counted
    as disagreement: models and reference observers both tend to move on
    the same (ambiguous) items, so counting the model's zeros as
    disagreement lets "moves where the reference moves" pass for "moves
    in the same direction" (#36).  How often the model does not move is
    reported separately, as the zero-delta rate.
    """
    dp = np.asarray(delta_p, dtype=np.float32)
    dm = np.asarray(delta_m, dtype=np.float32)
    return (dp != 0.0) & (dm != 0.0)


def sign_agreement_rate(
    delta_p: np.ndarray,
    delta_m: np.ndarray,
) -> float:
    """Share of :func:`sign_agreement_cells` where sign(Δ_m) == sign(Δ_p).

    NaN when no cell counts.
    """
    counted = sign_agreement_cells(delta_p, delta_m)
    n = int(counted.sum())
    if n == 0:
        return float("nan")
    return float(sign_agreement_mask(delta_p, delta_m).sum() / n)


def confirmatory_eligible_classes(
    delta_p: np.ndarray,
    class_list: list[str],
    min_nonzero: int = 15,
) -> tuple[list[str], list[str]]:
    """Partition classes into confirmatory-eligible and descriptive-only.

    Classes with fewer than `min_nonzero` non-zero pair-aggregated
    physician deltas are prespecified as descriptive-only (see
    the Track 1 endpoints in docs/methodology.md).

    Args:
        delta_p:    (n, n_classes) physician delta matrix.
        class_list: List of class names, length n_classes.
        min_nonzero:Threshold; default 15.

    Returns:
        (eligible_classes, descriptive_only_classes) — both lists of str.
    """
    dp = np.asarray(delta_p, dtype=np.float32)
    eligible: list[str] = []
    descriptive: list[str] = []
    for c_idx, cls in enumerate(class_list):
        n_nz = int(np.sum(dp[:, c_idx] != 0.0))
        if n_nz >= min_nonzero:
            eligible.append(cls)
        else:
            descriptive.append(cls)
    return eligible, descriptive


def nonzero_counts_per_class(
    delta_p: np.ndarray,
    class_list: list[str],
) -> dict[str, int]:
    """Count non-zero physician deltas per class.

    Useful for annotating figures and deciding confirmatory eligibility.
    """
    dp = np.asarray(delta_p, dtype=np.float32)
    return {
        cls: int(np.sum(dp[:, c_idx] != 0.0))
        for c_idx, cls in enumerate(class_list)
    }
