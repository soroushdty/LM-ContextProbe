"""Simulated studies with planted effects (#18)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared.simulation import SCENARIOS, SimulationDesign, simulate_study

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "simulate_endpoints.py"


def _script():
    spec = importlib.util.spec_from_file_location("simulate_endpoints", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _big(scenario: str, effect: float = 1.0, **kw):
    design = SimulationDesign(scenario=scenario, effect=effect, n_patients=300, items_per_patient=20, **kw)
    return simulate_study(design, np.random.default_rng(0))


def _deltas(study):
    dp = study.delta_reference
    dm = study.correct_context_scores - study.context_free_scores
    ds = study.shuffled_context_scores - study.context_free_scores
    return dp, dm, ds


def _specificity(study) -> float:
    dp, dm, ds = _deltas(study)
    nz = dp != 0
    return float(np.mean((np.sign(dp) * (dm - ds))[nz]))


def _agreement(dp, dm) -> float:
    moved = (dp != 0) & (dm != 0)
    return float(np.mean((np.sign(dp) == np.sign(dm))[moved]))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class TestGenerator:
    def test_same_seed_same_study(self):
        d = SimulationDesign(scenario="aligned", effect=0.5, n_patients=5, items_per_patient=4)
        a = simulate_study(d, np.random.default_rng(3))
        b = simulate_study(d, np.random.default_rng(3))
        np.testing.assert_array_equal(a.correct_context_scores, b.correct_context_scores)
        np.testing.assert_array_equal(a.reference_correct_context, b.reference_correct_context)

    def test_shapes_and_label_values(self):
        d = SimulationDesign(n_patients=5, items_per_patient=4, n_categories=3, n_observers=2)
        st = simulate_study(d, np.random.default_rng(0))
        assert st.context_free_scores.shape == (20, 3)
        np.testing.assert_array_equal(st.patient_ids, np.repeat(np.arange(5), 4))
        assert set(np.unique(st.reference_context_free)) <= {0.0, 0.5, 1.0}
        assert set(np.unique(st.delta_reference)) <= {-1.0, -0.5, 0.0, 0.5, 1.0}

    def test_scores_rounded_and_in_range(self):
        st = simulate_study(SimulationDesign(score_round=0.05), np.random.default_rng(0))
        for scores in (st.context_free_scores, st.correct_context_scores, st.shuffled_context_scores):
            assert scores.min() >= 0 and scores.max() <= 1
            np.testing.assert_allclose(scores / 0.05, np.round(scores / 0.05), atol=1e-9)
        cont = simulate_study(SimulationDesign(score_round=None), np.random.default_rng(0))
        dm = cont.correct_context_scores - cont.context_free_scores
        assert np.all(dm != 0)

    def test_shuffled_context_is_another_patient(self):
        st = simulate_study(SimulationDesign(n_patients=4, items_per_patient=50), np.random.default_rng(0))
        assert not np.any(st.shuffled_patient_ids == st.patient_ids)
        assert set(st.shuffled_patient_ids) == {0, 1, 2, 3}

    def test_rejects_unknown_scenario_and_single_patient(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            SimulationDesign(scenario="null")
        with pytest.raises(ValueError, match="two patients"):
            SimulationDesign(n_patients=1)
        with pytest.raises(ValueError, match=r"effect in \[0, 1\]"):
            SimulationDesign(scenario="attenuated", effect=1.5)

    def test_reference_is_mostly_unchanged_and_mostly_up(self):
        dp = _big("unrelated").delta_reference
        assert 0.2 < np.mean(dp != 0) < 0.5
        assert np.mean(dp[dp != 0] > 0) > 0.55


class TestPlantedEffects:
    """The scenarios plant what their docstrings say, in a large sample."""

    def test_unrelated_correct_and_shuffled_are_exchangeable(self):
        st = _big("unrelated")
        dp, dm, ds = _deltas(st)
        assert abs(_specificity(st)) < 0.003
        assert abs(np.mean(np.abs(dm)) - np.mean(np.abs(ds))) < 0.003
        assert _agreement(dp, dm) == pytest.approx(0.5, abs=0.02)

    def test_aligned_is_specific(self):
        st = _big("aligned")
        dp, dm, ds = _deltas(st)
        assert _specificity(st) > 0.02
        assert _agreement(dp, dm) > _agreement(dp, ds) + 0.05

    def test_nonspecific_agrees_but_is_not_specific(self):
        st = _big("nonspecific")
        dp, dm, ds = _deltas(st)
        assert _agreement(dp, dm) > 0.55
        assert _agreement(dp, ds) == pytest.approx(_agreement(dp, dm), abs=0.02)
        assert abs(_specificity(st)) < 0.003

    def test_partial_reverses_half_the_categories(self):
        st = _big("partial")
        dp, dm, _ = _deltas(st)
        half = dp.shape[1] // 2
        assert _agreement(dp[:, :half], dm[:, :half]) > 0.55
        assert _agreement(dp[:, half:], dm[:, half:]) < 0.45

    def test_attenuated_moves_less_with_another_patients_context(self):
        st = _big("attenuated", effect=0.5)
        dp, dm, ds = _deltas(st)
        assert np.mean(np.abs(dm)) > 1.3 * np.mean(np.abs(ds))
        # Directions stay unrelated to the reference.
        assert _agreement(dp, dm) == pytest.approx(0.5, abs=0.02)
        assert abs(_specificity(st)) < 0.003

    def test_attenuated_with_no_effect_is_unrelated(self):
        a, b = _big("attenuated", effect=0.0), _big("unrelated")
        np.testing.assert_array_equal(a.shuffled_context_scores, b.shuffled_context_scores)

    def test_scenario_names(self):
        assert SCENARIOS == ("unrelated", "aligned", "nonspecific", "partial", "attenuated")


# ---------------------------------------------------------------------------
# Study runner (scripts/simulate_endpoints.py)
# ---------------------------------------------------------------------------


class TestRunner:
    def test_every_test_reports_a_p_value(self):
        se = _script()
        st = simulate_study(SimulationDesign(scenario="aligned", effect=0.5, n_patients=6, items_per_patient=10),
                            np.random.default_rng(0))
        rows, ci_rows = se.evaluate_study(st, n_bootstrap=50, n_permutations=100, seed=1)
        got = {(r["track"], r["endpoint"], r["test"]) for r in rows}
        assert got == {(t, e, n) for t, e, n, _ in se.TESTS}
        assert all(0.0 <= r["p_value"] <= 1.0 for r in rows)
        assert {(r["quantity"], r["method"]) for r in ci_rows} == {
            (q, m) for q in se.CI_QUANTITIES for m in ("patient_cluster", "item_level")
        }

    def test_population_value_is_zero_when_exchangeable(self):
        se = _script()
        for scenario in ("unrelated", "nonspecific"):
            truth = se.population_value(SimulationDesign(scenario=scenario, effect=1.0))
            assert truth == dict.fromkeys(se.CI_QUANTITIES, 0.0)
        aligned = se.population_value(SimulationDesign(scenario="aligned", effect=1.0), n_patients=300)
        assert aligned["context_specificity_difference"] > 0

    def test_end_to_end_writes_outputs(self, tmp_path):
        se = _script()
        out = tmp_path / "sim"
        code = se.main([
            "--out", str(out), "--patients", "4", "--items", "6", "--effects", "1.0",
            "--clustering-sds", "2.0", "--attenuations", "0.5", "--reps-null", "2", "--reps", "1",
            "--n-bootstrap", "20", "--n-permutations", "40", "--workers", "1",
        ])
        assert code == 0
        rej = pd.read_csv(out / "rejection_rates.csv", keep_default_na=False)
        assert set(rej["scenario"]) == {"unrelated", "aligned", "nonspecific", "partial", "attenuated"}
        assert set(rej["model_patient_sd"]) == {0.3, 2.0}
        assert set(rej.loc[rej["scenario"] == "attenuated", "effect"]) == {0.5}
        cov = pd.read_csv(out / "ci_coverage.csv", keep_default_na=False)
        assert set(cov["method"]) == {"patient_cluster", "item_level"}
        design = json.loads((out / "design.json").read_text())
        assert design["seed"] == 2026
        assert {p.name for p in (out / "figures").iterdir()} == {
            "false_positive_rate.png", "power_aligned.png", "power_nonspecific.png",
            "power_partial.png", "power_attenuated.png", "ci_coverage.png", "clustering.png",
        }

    def test_same_seed_same_replicates(self, tmp_path):
        se = _script()
        args = ["--patients", "4", "--items", "6", "--effects", "1.0", "--clustering-sds", "",
                "--reps-null", "2", "--reps", "1", "--n-bootstrap", "20", "--n-permutations", "40",
                "--workers", "1"]
        se.main(["--out", str(tmp_path / "a"), *args])
        se.main(["--out", str(tmp_path / "b"), *args])
        a = pd.read_csv(tmp_path / "a" / "replicates.csv", keep_default_na=False)
        b = pd.read_csv(tmp_path / "b" / "replicates.csv", keep_default_na=False)
        pd.testing.assert_frame_equal(a, b)
