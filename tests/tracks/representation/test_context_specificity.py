"""Track 1 shuffled-context condition and context specificity (#11)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from shared.evaluation.hypothesis_analyzer import (
    HypothesisAnalyzer,
    context_specificity_test,
)
from tracks.representation.statistical.hypotheses.context_specificity import (
    context_specificity_per_class,
    context_specificity_pooled,
)
from tracks.representation.training.orchestrator import train_ensemble_pipeline as tep
from tracks.representation.training.stage2.stage2_context import (
    pair_patients_for_context_swap,
)

# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


class TestPairPatientsForContextSwap:
    @pytest.mark.parametrize("n", [2, 3, 6, 7])
    def test_every_patient_in_exactly_one_group(self, n):
        groups = pair_patients_for_context_swap(range(1, n + 1), seed=0)
        members = [p for g in groups for p in g]
        assert sorted(members) == list(range(1, n + 1))
        sizes = sorted(len(g) for g in groups)
        assert sizes == [2] * (n // 2 - 1) + ([3] if n % 2 else [2])

    def test_same_seed_same_groups(self):
        a = pair_patients_for_context_swap(range(10), seed=7)
        assert a == pair_patients_for_context_swap(range(10), seed=7)
        assert a != pair_patients_for_context_swap(range(10), seed=8)

    def test_input_order_and_duplicates_do_not_matter(self):
        a = pair_patients_for_context_swap([1, 2, 3, 4, 5], seed=3)
        b = pair_patients_for_context_swap([5, 3, 3, 1, 4, 2, 2], seed=3)
        assert a == b

    def test_fewer_than_two_patients(self):
        assert pair_patients_for_context_swap([4, 4], seed=0) == []


# ---------------------------------------------------------------------------
# Leave-pair-out scoring
# ---------------------------------------------------------------------------


def _swap_inputs(n_patients=6, items=4, d=6, k=2, same_context=False):
    rng = np.random.default_rng(0)
    pids = np.repeat(np.arange(1, n_patients + 1), items)
    n = pids.size
    shared = rng.standard_normal(d).astype(np.float32)
    ctx = {
        p: shared.copy() if same_context else rng.standard_normal(d).astype(np.float32)
        for p in range(1, n_patients + 1)
    }
    cfg = {
        "stage2_alpha_options": [0.1], "stage2_lr": 0.05, "stage2_num_epochs": 20,
        "stage2_early_stopping_patience": 5, "fusion_strategy": "2d",
        "shuffled_context_seed": 5, "calibration_min_samples": 3,
    }
    return {
        "X_test": rng.standard_normal((n, d)).astype(np.float32),
        "Y_test_interview": (rng.random((n, k)) > 0.4).astype(np.float32),
        "Y_test_survey": (rng.random((n, k)) > 0.6).astype(np.float32),
        "patient_ids_test": pids,
        "context_vectors": ctx,
        "y_cf": rng.uniform(0.1, 0.9, (n, k)).astype(np.float32),
        "class_list": [f"c{i}" for i in range(k)],
        "cfg": cfg,
    }


class TestLeavePairOutContextSwap:
    def test_heads_never_see_the_patients_they_score(self, monkeypatch):
        fitted_on = []
        real_fit = tep._fit_stage2_head

        def spy(X_rows, patient_ids_rows, *args, **kwargs):
            fitted_on.append(set(np.asarray(patient_ids_rows).tolist()))
            return real_fit(X_rows, patient_ids_rows, *args, **kwargs)

        monkeypatch.setattr(tep, "_fit_stage2_head", spy)
        inputs = _swap_inputs()
        out = tep._leave_pair_out_context_swap(**inputs)

        assert len(fitted_on) == len(out["groups"]) == 3
        for group, seen in zip(out["groups"], fitted_on):
            assert seen.isdisjoint(group)
            assert seen == set(range(1, 7)) - set(group)

    def test_donor_is_the_swap_partner(self):
        inputs = _swap_inputs(n_patients=7)
        out = tep._leave_pair_out_context_swap(**inputs)
        pids = inputs["patient_ids_test"]
        partner = {}
        for g in out["groups"]:
            for i, p in enumerate(g):
                partner[p] = g[(i + 1) % len(g)]
        np.testing.assert_array_equal(out["donors"], [partner[p] for p in pids])
        assert not np.any(out["donors"] == pids)

    def test_only_the_context_differs(self):
        """With every patient given the same context vector, the swap changes
        nothing: correct and shuffled come from the same head, item and ŷ_cf."""
        out = tep._leave_pair_out_context_swap(**_swap_inputs(same_context=True))
        np.testing.assert_allclose(out["correct"], out["shuffled"], rtol=0, atol=1e-6)

    def test_distinct_contexts_move_the_scores(self):
        out = tep._leave_pair_out_context_swap(**_swap_inputs())
        assert not np.isnan(out["correct"]).any()
        assert np.nanmax(np.abs(out["correct"] - out["shuffled"])) > 1e-3

    def test_rows_without_stage1_prediction_are_not_scored(self):
        inputs = _swap_inputs()
        inputs["y_cf"][:2] = np.nan
        out = tep._leave_pair_out_context_swap(**inputs)
        assert np.isnan(out["correct"][:2]).all() and np.isnan(out["shuffled"][:2]).all()
        assert (out["donors"][:2] == -1).all()
        assert not np.isnan(out["correct"][2:]).any()

    def test_skipped_with_one_patient(self):
        inputs = _swap_inputs()
        inputs["context_vectors"] = {1: inputs["context_vectors"][1]}
        assert tep._leave_pair_out_context_swap(**inputs) is None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _run_pipeline(tmp_path, context_vectors=None, n_features=8, n_classes=2):
    rng = np.random.default_rng(0)
    n_patients, items = 4, 3
    n_train = n_patients * items
    patient_ids_train = np.repeat(np.arange(1, n_patients + 1), items)
    patient_ids_test = np.repeat(np.arange(1, n_patients + 1), items)
    X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    Y_train = (rng.random((n_train, n_classes)) > 0.5).astype(np.float32)
    X_test = rng.standard_normal((n_train, n_features)).astype(np.float32)
    Y_test = (rng.random((n_train, n_classes)) > 0.5).astype(np.float32)
    if context_vectors is None:
        context_vectors = {
            p: rng.standard_normal(n_features).astype(np.float32)
            for p in range(1, n_patients + 1)
        }
    cfg = {
        "global_seed": 0, "lr": 0.001, "weight_decay": 0.0,
        "batch_size": n_train, "weight_cap": 10.0,
        "pca_n_components": n_features, "pca_whiten": False,
        "class_weight_mode": "cui", "class_weight_beta": 0.999,
        "num_epochs": 1, "early_stopping_patience": 1,
        "hidden_dims": [], "dropout": 0.0, "activation": "gelu",
        "primary_threshold": "tau", "tau": 0.5, "objective": "F1",
        "default_classes": [f"c{i}" for i in range(n_classes)],
        "calibration_method": "isotonic", "calibration_min_samples": 3,
        "DIR_MODEL": str(tmp_path / "model"),
        "GPU_MONITOR": {"enabled": False}, "log_hp_search_grid": False,
        "stage2_alpha_options": [0.1], "stage2_lr": 0.05,
        "stage2_num_epochs": 5, "stage2_early_stopping_patience": 3,
        "fusion_strategy": "2d", "shuffled_context_seed": 11,
    }
    result = tep.train_ensemble_pipeline(
        X_train, Y_train, patient_ids_train, X_test, Y_test, cfg,
        patient_ids_test=patient_ids_test,
        Y_test_interview=(rng.random((n_train, n_classes)) > 0.5).astype(np.float32),
        Y_test_survey=(rng.random((n_train, n_classes)) > 0.5).astype(np.float32),
        context_vectors=context_vectors,
    )
    return result, patient_ids_test, tmp_path / "model"


class TestPipelineShuffledContext:
    def test_predictions_and_audit_map_are_returned(self, tmp_path):
        result, pids, model_dir = _run_pipeline(tmp_path)
        correct = result["test_probs_ca_leave_pair_out"]
        shuffled = result["test_probs_shuffled_leave_pair_out"]
        assert correct.shape == shuffled.shape == result["test_probs_ca_fold_pure"].shape
        assert not np.isnan(correct).any() and not np.isnan(shuffled).any()

        donors = result["shuffled_context_donors"]
        assert not np.any(donors == pids)
        groups = pair_patients_for_context_swap(range(1, 5), seed=11)
        partner = {g[0]: g[1] for g in groups} | {g[1]: g[0] for g in groups}
        np.testing.assert_array_equal(donors, [partner[p] for p in pids])

        audit = pd.read_csv(model_dir / "test_shuffled_context_map.csv")
        np.testing.assert_array_equal(audit["patient_id"], pids)
        np.testing.assert_array_equal(audit["shuffled_patient_id"], donors)

    def test_skipped_when_only_one_patient_has_context(self, tmp_path):
        ctx = {1: np.random.default_rng(1).standard_normal(8).astype(np.float32)}
        result, _, model_dir = _run_pipeline(tmp_path, context_vectors=ctx)
        assert result["test_probs_ca_fold_pure"] is not None
        assert result["test_probs_shuffled_leave_pair_out"] is None
        assert result["test_probs_ca_leave_pair_out"] is None
        assert result["shuffled_context_donors"] is None
        assert not (model_dir / "test_shuffled_context_map.csv").exists()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

CLASSES = ["a", "b"]


def _deltas(n_patients=8, items=4, seed=0):
    rng = np.random.default_rng(seed)
    n = n_patients * items
    pids = np.repeat(np.arange(n_patients), items)
    dp = rng.choice([-0.5, 0.5], size=(n, len(CLASSES)))
    return dp, pids, rng


class TestContextSpecificityStatistics:
    def test_context_blind_shift_is_not_specific(self):
        """A model that shifts the same way whatever context it is given
        aligns with the physicians, but no better than with shuffled context."""
        dp, pids, _ = _deltas()
        dm = np.full_like(dp, 0.1)
        res = context_specificity_pooled(dp, dm, dm.copy(), pids, CLASSES, CLASSES, 200, 1000,
                                         np.random.default_rng(0))
        assert res["mean_alignment_difference"] == 0.0
        assert res["p_value"] == 1.0

    def test_item_specific_shift_beats_shuffled(self):
        dp, pids, rng = _deltas()
        dm = 0.2 * np.sign(dp)                       # follows the physicians
        ds = 0.2 * rng.choice([-1.0, 1.0], size=dp.shape)  # unrelated
        res = context_specificity_pooled(dp, dm, ds, pids, CLASSES, CLASSES, 200, 10_000,
                                         np.random.default_rng(0))
        assert res["mean_alignment_difference"] > 0
        assert res["ci_lower"] > 0
        assert res["p_value"] < 0.05
        assert res["n_patients"] == 8
        assert res["n_cells"] == dp.size

    def test_same_statistic_as_track3(self):
        """Track 1 and Track 3 report the same number for the same deltas."""
        dp, pids, rng = _deltas()
        cf = rng.random(dp.shape)
        ca = np.clip(cf + 0.1 * np.sign(dp) + 0.05 * rng.standard_normal(dp.shape), 0, 1)
        sh = np.clip(cf + 0.05 * rng.standard_normal(dp.shape), 0, 1)

        class _Paired:
            patient_ids = pids
            category_names = CLASSES

        analyzer = HypothesisAnalyzer(_Paired(), {"n_bootstrap_resamples": 200, "n_permutations": 1000})
        t3 = analyzer.compute_h4(cf, ca, sh, dp)
        t1 = context_specificity_pooled(dp, ca - cf, sh - cf, pids, CLASSES, CLASSES, 200, 1000,
                                        np.random.default_rng(0))
        assert t1["mean_alignment_difference"] == pytest.approx(t3["mean_diff"])
        assert t1["p_value"] == pytest.approx(t3["p_value"])

    def test_pooled_uses_eligible_classes_only(self):
        dp, pids, _ = _deltas()
        dm = np.zeros_like(dp)
        dm[:, 0] = 0.2 * np.sign(dp[:, 0])
        ds = np.zeros_like(dp)
        res = context_specificity_pooled(dp, dm, ds, pids, ["a"], CLASSES, 100, 1000,
                                         np.random.default_rng(0))
        assert res["classes"] == ["a"]
        assert res["n_cells"] == dp.shape[0]
        assert res["mean_alignment_difference"] == pytest.approx(0.2)

    def test_rows_with_missing_predictions_are_left_out(self):
        dp, pids, _ = _deltas()
        dm = 0.2 * np.sign(dp)
        ds = np.zeros_like(dp)
        dm[0, 0] = np.nan
        res = context_specificity_pooled(dp, dm, ds, pids, CLASSES, CLASSES, 100, 1000,
                                         np.random.default_rng(0))
        assert res["n_rows"] == dp.shape[0] - 1
        assert res["mean_alignment_difference"] == pytest.approx(0.2)

    def test_per_class_bh_over_eligible_classes(self):
        dp, pids, _ = _deltas()
        dm = 0.2 * np.sign(dp)
        ds = np.zeros_like(dp)
        df = context_specificity_per_class(dp, dm, ds, pids, CLASSES, ["a"], 100, 1000,
                                           np.random.default_rng(0))
        assert list(df["Class"]) == CLASSES
        a, b = df.iloc[0], df.iloc[1]
        assert bool(a["confirmatory"]) and not bool(b["confirmatory"])
        assert a["bh_adj_p"] == pytest.approx(a["p_value"])  # one test: BH is a no-op
        assert np.isnan(b["bh_adj_p"])
        assert b["mean_alignment_difference"] == pytest.approx(0.2)

    def test_no_reference_shift_gives_nan(self):
        res = context_specificity_test(
            np.ones((4, 1)), np.zeros((4, 1)), np.zeros((4, 1)), np.arange(4),
        )
        assert res["n_cells"] == 0
        assert np.isnan(res["mean_diff"]) and np.isnan(res["p_value"])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestRunAnalysisWritesContextSpecificity:
    def _call(self, tmp_path, y_hat_shuffled):
        from tracks.representation.statistical.orchestrator.run_analysis import (
            run_statistical_analysis,
        )

        rng = np.random.default_rng(0)
        n_patients, items = 6, 5
        n = n_patients * items
        pids = np.repeat(np.arange(n_patients), items)
        y_survey = rng.choice([0.0, 1.0], size=(n, 2))
        y_interview = rng.choice([0.0, 1.0], size=(n, 2))
        y_cf = rng.random((n, 2))
        y_ca = np.clip(y_cf + 0.1 * (y_interview - y_survey), 0, 1)
        out = tmp_path / "stats"
        run_statistical_analysis(
            y_survey=y_survey, y_interview=y_interview,
            y_hat_cf=y_cf, y_hat_ca=y_ca,
            patient_ids=pids, item_texts=np.array([f"i{k}" for k in range(n)]),
            class_list=CLASSES, avg_thresh_f1opt=np.full(2, 0.5), tau_fixed=0.5,
            dataset_path=tmp_path / "missing.xlsx", sheet_names={},
            context_json=None, llm="none",
            cfg={"global_seed": 0, "statistical_analysis": {"confirmatory_min_nonzero": 1}},
            output_dir=out, n_resamples=50, n_permutations=200,
            y_hat_ca_leave_pair_out=None if y_hat_shuffled is None else y_ca,
            y_hat_shuffled=None if y_hat_shuffled is None else y_hat_shuffled(y_cf),
        )
        return out

    def test_outputs_written_with_shuffled_predictions(self, tmp_path):
        out = self._call(tmp_path, lambda cf: cf.copy())
        pooled = json.loads((out / "context_specificity_pooled.json").read_text())
        assert pooled["classes"] == CLASSES
        assert pooled["mean_alignment_difference"] > 0
        df = pd.read_csv(out / "context_specificity_per_class.csv")
        assert list(df["Class"]) == CLASSES

    def test_skipped_without_shuffled_predictions(self, tmp_path):
        out = self._call(tmp_path, None)
        assert not (out / "context_specificity_pooled.json").exists()
        assert (out / "directional_alignment_pooled.json").exists()
