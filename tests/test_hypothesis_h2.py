"""Tests for tracks.reasoning.bundle.hypotheses.compute_h2."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tracks.reasoning.bundle.hypotheses import compute_h2

CATS = ["behavioral_health", "diagnoses"]


def _make_cells(
    delta_physician: float = 0.3,
    delta_correct: float = 0.2,
    delta_shuffled: float = -0.1,
    n_patients: int = 3,
    model: str = "m",
) -> pd.DataFrame:
    rows = []
    for pid in [f"P{i}" for i in range(n_patients)]:
        for item in ["Item0", "Item1"]:
            for cat in CATS:
                sign_dp = np.sign(delta_physician) if delta_physician != 0 else 0.0
                rows.append(
                    {
                        "model": model,
                        "patient": pid,
                        "item_text": item,
                        "category": cat,
                        "delta_physician": delta_physician,
                        "delta_llm_correct": delta_correct,
                        "delta_llm_shuffled": delta_shuffled,
                        "alignment_correct": sign_dp * delta_correct,
                        "alignment_shuffled": sign_dp * delta_shuffled,
                        "alignment_difference": sign_dp * delta_correct - sign_dp * delta_shuffled,
                        "physician_shift_indicator": delta_physician != 0,
                    }
                )
    return pd.DataFrame(rows)


class TestComputeH2:
    def test_perfect_sign_alignment_correct(self):
        """When LLM and physician both shift up, sign_agreement_correct = 1."""
        cells = _make_cells(delta_physician=0.3, delta_correct=0.2)
        h2 = compute_h2(cells, "m", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h2[h2["category"] == "aggregate"].iloc[0]
        assert agg["sign_agreement_correct"] == pytest.approx(1.0)

    def test_opposite_sign_alignment_correct(self):
        """When LLM shifts opposite to physician, sign_agreement_correct = 0."""
        cells = _make_cells(delta_physician=0.3, delta_correct=-0.2)
        h2 = compute_h2(cells, "m", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h2[h2["category"] == "aggregate"].iloc[0]
        assert agg["sign_agreement_correct"] == pytest.approx(0.0)

    def test_no_physician_shifts_returns_nan(self):
        """When delta_physician=0 for all rows, H2 values are NaN."""
        cells = _make_cells(delta_physician=0.0)
        h2 = compute_h2(cells, "m", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h2[h2["category"] == "aggregate"].iloc[0]
        assert math.isnan(agg["mean_alignment_correct"])
        assert math.isnan(agg["sign_agreement_correct"])

    def test_mean_alignment_correct_value(self):
        """Mean alignment = sign(dp) × delta_llm_correct."""
        # delta_physician=0.3 (sign=1), delta_correct=0.2 → alignment=0.2
        cells = _make_cells(delta_physician=0.3, delta_correct=0.2)
        h2 = compute_h2(cells, "m", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h2[h2["category"] == "aggregate"].iloc[0]
        assert agg["mean_alignment_correct"] == pytest.approx(0.2, abs=1e-9)

    def test_shuffled_alignment_differs_from_correct(self):
        cells = _make_cells(delta_physician=0.3, delta_correct=0.2, delta_shuffled=-0.1)
        h2 = compute_h2(cells, "m", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h2[h2["category"] == "aggregate"].iloc[0]
        assert agg["mean_alignment_correct"] > agg["mean_alignment_shuffled"]

    def test_epsilon_threshold_zeroes_small_delta(self):
        """LLM delta below epsilon counts as unchanged and is left out of sign agreement (#36)."""
        # delta_correct = 0.005 < epsilon=0.01 → unchanged; no cell left to agree or disagree
        cells = _make_cells(delta_physician=0.3, delta_correct=0.005)
        h2 = compute_h2(cells, "m", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        agg = h2[h2["category"] == "aggregate"].iloc[0]
        assert agg["unchanged_rate_correct"] == pytest.approx(1.0)
        assert math.isnan(agg["sign_agreement_correct"])

    def test_ci_keys_present(self):
        cells = _make_cells()
        h2 = compute_h2(cells, "m", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        for col in ("ci_low_correct", "ci_high_correct", "ci_low_shuffled", "ci_high_shuffled"):
            assert col in h2.columns

    def test_per_category_rows_present(self):
        cells = _make_cells()
        h2 = compute_h2(cells, "m", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        for cat in CATS:
            assert cat in h2["category"].values

    def test_required_columns_present(self):
        cells = _make_cells()
        h2 = compute_h2(cells, "m", CATS, epsilon=0.01, n_bootstrap=50, seed=0)
        required = [
            "model", "category", "n_patients", "n_shift_cells",
            "mean_alignment_correct", "ci_low_correct", "ci_high_correct",
            "mean_alignment_shuffled", "ci_low_shuffled", "ci_high_shuffled",
            "sign_agreement_correct", "sign_agreement_shuffled", "epsilon",
        ]
        for col in required:
            assert col in h2.columns, f"Missing: {col}"
