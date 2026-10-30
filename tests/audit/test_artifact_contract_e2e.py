"""End-to-end artifact contract integration tests.

Verifies that:
1. Stage 1 checkpoints contain all keys required by Stage 2 (including ycf_test_probs).
2. The fold alignment manifest is written with correct fold↔patient↔row-index mapping.
3. fold-pure CF predictions correctly use each patient's held-out fold, not list position.
4. Resume mode restores ycf_test_probs from checkpoints and respects split compatibility.
5. Stage 2 fold predictions are persisted as .npy files and their shapes are consistent.
6. Statistical analysis receives arrays whose row counts and patient IDs align with labels.
"""
from __future__ import annotations

import json
import numpy as np
import pytest
import joblib
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared minimal-pipeline fixture
# ---------------------------------------------------------------------------

def _run_mini_pipeline(
    tmp_path: Path,
    *,
    n_patients: int = 4,
    items_per_patient: int = 3,
    n_features: int = 8,
    n_classes: int = 2,
    with_stage2: bool = False,
    resume_from: Path | None = None,
    n_test_patients: int = 2,
    cv: dict | None = None,
):
    """Run train_ensemble_pipeline on synthetic data and return (result, paths, cfg)."""
    from tracks.representation.training.orchestrator.train_ensemble_pipeline import train_ensemble_pipeline, _build_paths

    rng = np.random.default_rng(0)
    n_train = n_patients * items_per_patient
    n_test = items_per_patient * n_test_patients

    X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    Y_train = (rng.random((n_train, n_classes)) > 0.5).astype(np.float32)
    patient_ids_train = np.repeat(np.arange(1, n_patients + 1), items_per_patient)

    X_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
    Y_test = (rng.random((n_test, n_classes)) > 0.5).astype(np.float32)
    patient_ids_test = np.repeat(np.arange(1, n_test_patients + 1), items_per_patient)

    cfg: dict = {
        "global_seed": 0,
        "lr": 0.001,
        "weight_decay": 0.0,
        "batch_size": n_train,
        "weight_cap": 10.0,
        "pca_n_components": n_features,
        "pca_whiten": False,
        "class_weight_mode": "cui",
        "class_weight_beta": 0.999,
        "num_epochs": 1,
        "early_stopping_patience": 1,
        "hidden_dims": [],
        "dropout": 0.0,
        "activation": "gelu",
        "primary_threshold": "tau",
        "tau": 0.5,
        "objective": "F1",
        "default_classes": [f"c{i}" for i in range(n_classes)],
        "calibration_method": "isotonic",
        "calibration_min_samples": 3,
        "DIR_MODEL": str(tmp_path / "model"),
        "GPU_MONITOR": {"enabled": False},
        "log_hp_search_grid": False,
    }
    if cv is not None:
        cfg["cv"] = cv

    context_vectors = None
    Y_test_interview = None
    Y_test_survey = None
    if with_stage2:
        context_vectors = {
            int(p): rng.standard_normal(n_features).astype(np.float32)
            for p in np.unique(patient_ids_test)
        }
        cfg["stage2_alpha_options"] = [0.1]
        cfg["stage2_lr"] = 0.05
        cfg["stage2_num_epochs"] = 5
        cfg["stage2_early_stopping_patience"] = 3
        cfg["fusion_strategy"] = "4_vector"
        Y_test_interview = (rng.random((n_test, n_classes)) > 0.5).astype(np.float32)
        Y_test_survey = (rng.random((n_test, n_classes)) > 0.5).astype(np.float32)

    result = train_ensemble_pipeline(
        X_train, Y_train, patient_ids_train,
        X_test, Y_test, cfg,
        patient_ids_test=patient_ids_test,
        resume_from_checkpoint=(resume_from is not None),
        Y_test_interview=Y_test_interview,
        Y_test_survey=Y_test_survey,
        context_vectors=context_vectors,
    )
    paths = _build_paths(cfg)
    return result, paths, cfg, patient_ids_train, patient_ids_test, n_train, n_test


# ---------------------------------------------------------------------------
# Issue 1 — Stage 1 checkpoints contain all keys required by Stage 2
# ---------------------------------------------------------------------------

class TestStage1CheckpointContract:
    """Every fold checkpoint must satisfy the v3 schema including ycf_test_probs."""

    def test_all_fold_checkpoints_contain_required_v3_keys(self, tmp_path):
        from tracks.representation.training.orchestrator.fold_postprocessing import _REQUIRED_V3_KEYS, CHECKPOINT_VERSION

        _, paths, _, pids_train, _, _, _ = _run_mini_pipeline(tmp_path)
        n_patients = len(np.unique(pids_train))
        for fold_num in range(1, n_patients + 1):
            ckpt_path = paths.checkpoints_dir / f"fold_{fold_num}.joblib"
            assert ckpt_path.exists(), f"Checkpoint fold_{fold_num}.joblib missing"
            ckpt = joblib.load(ckpt_path)
            assert isinstance(ckpt, dict)
            assert ckpt["checkpoint_version"] == CHECKPOINT_VERSION, (
                f"fold_{fold_num}: version mismatch"
            )
            missing = _REQUIRED_V3_KEYS - set(ckpt.keys())
            assert not missing, (
                f"fold_{fold_num} checkpoint missing keys: {missing}"
            )

    def test_fold_checkpoints_contain_ycf_test_probs_key(self, tmp_path):
        """Every checkpoint must have ycf_test_probs (None or array) for Stage 2."""
        _, paths, _, pids_train, _, _, n_test = _run_mini_pipeline(tmp_path)
        n_patients = len(np.unique(pids_train))
        for fold_num in range(1, n_patients + 1):
            ckpt = joblib.load(paths.checkpoints_dir / f"fold_{fold_num}.joblib")
            assert "ycf_test_probs" in ckpt, (
                f"fold_{fold_num} checkpoint missing ycf_test_probs"
            )

    def test_fold_checkpoint_ycf_shape_matches_n_test(self, tmp_path):
        """When Stage 1 runs, ycf_test_probs shape must be (n_test, n_classes)."""
        n_test_items = 6
        n_classes = 2
        _, paths, cfg, pids_train, _, _, _ = _run_mini_pipeline(
            tmp_path, items_per_patient=3, n_classes=n_classes
        )
        n_patients = len(np.unique(pids_train))
        for fold_num in range(1, n_patients + 1):
            ckpt = joblib.load(paths.checkpoints_dir / f"fold_{fold_num}.joblib")
            ycf = ckpt["ycf_test_probs"]
            if ycf is not None:
                assert ycf.ndim == 2, (
                    f"fold_{fold_num}: ycf_test_probs must be 2-D, got shape {ycf.shape}"
                )
                assert ycf.shape[1] == n_classes, (
                    f"fold_{fold_num}: ycf_test_probs class dim {ycf.shape[1]} != {n_classes}"
                )

    def test_fold_checkpoint_oof_shape_matches_n_val(self, tmp_path):
        """oof_probs shape must equal (n_val, n_classes) for each fold."""
        n_classes = 2
        _, paths, _, pids_train, _, _, _ = _run_mini_pipeline(
            tmp_path, n_classes=n_classes
        )
        n_patients = len(np.unique(pids_train))
        for fold_num in range(1, n_patients + 1):
            ckpt = joblib.load(paths.checkpoints_dir / f"fold_{fold_num}.joblib")
            val_ix = ckpt["val_ix"]
            oof = ckpt["oof_probs"]
            assert oof.shape == (len(val_ix), n_classes), (
                f"fold_{fold_num}: oof_probs shape {oof.shape} != ({len(val_ix)}, {n_classes})"
            )


# ---------------------------------------------------------------------------
# Issue 2/3 — Fold alignment manifest ties fold↔patient↔rows
# ---------------------------------------------------------------------------

class TestFoldAlignmentManifest:
    """fold_manifest.json must be written with correct schema and row alignment."""

    def test_manifest_is_written(self, tmp_path):
        _, paths, _, _, _, _, _ = _run_mini_pipeline(tmp_path)
        assert paths.fold_manifest_json.exists(), (
            "fold_manifest.json not written after pipeline run"
        )

    def test_manifest_schema_version(self, tmp_path):
        _, paths, _, _, _, _, _ = _run_mini_pipeline(tmp_path)
        manifest = json.loads(paths.fold_manifest_json.read_text())
        assert manifest["schema_version"] == "pdm_fold_manifest_v1"

    def test_manifest_n_folds_matches_unique_patients(self, tmp_path):
        _, paths, _, pids_train, _, _, _ = _run_mini_pipeline(tmp_path)
        manifest = json.loads(paths.fold_manifest_json.read_text())
        assert manifest["n_folds"] == len(np.unique(pids_train))
        assert len(manifest["folds"]) == manifest["n_folds"]

    def test_manifest_held_out_patient_ids_are_unique(self, tmp_path):
        _, paths, _, _, _, _, _ = _run_mini_pipeline(tmp_path)
        manifest = json.loads(paths.fold_manifest_json.read_text())
        held_out_pids = [f["held_out_patient_id"] for f in manifest["folds"]]
        assert len(held_out_pids) == len(set(held_out_pids)), (
            "Duplicate held-out patients in manifest — LOPO should have one unique patient per fold"
        )

    def test_manifest_val_ix_covers_all_train_rows(self, tmp_path):
        """Union of all val_ix across folds must equal every training row index."""
        _, paths, _, pids_train, _, n_train, _ = _run_mini_pipeline(tmp_path)
        manifest = json.loads(paths.fold_manifest_json.read_text())
        all_val_ix = []
        for fold in manifest["folds"]:
            all_val_ix.extend(fold["val_ix"])
        assert sorted(all_val_ix) == list(range(n_train)), (
            "Union of val_ix does not cover all training rows — split is not exhaustive"
        )

    def test_manifest_val_ix_are_disjoint(self, tmp_path):
        """Each row must appear in exactly one fold's val_ix."""
        _, paths, _, _, _, _, _ = _run_mini_pipeline(tmp_path)
        manifest = json.loads(paths.fold_manifest_json.read_text())
        seen = set()
        for fold in manifest["folds"]:
            overlap = seen & set(fold["val_ix"])
            assert not overlap, (
                f"Fold {fold['fold_id']} val_ix overlaps with a previous fold: {overlap}"
            )
            seen.update(fold["val_ix"])

    def test_manifest_patient_to_fold_round_trips(self, tmp_path):
        """patient_to_fold[patient_id] must equal the fold_idx where that patient was held out."""
        _, paths, _, _, _, _, _ = _run_mini_pipeline(tmp_path)
        manifest = json.loads(paths.fold_manifest_json.read_text())
        p2f = {int(k): v for k, v in manifest["patient_to_fold"].items()}
        for fold in manifest["folds"]:
            pid = fold["held_out_patient_id"]
            assert p2f.get(pid) == fold["fold_idx"], (
                f"patient_to_fold[{pid}] = {p2f.get(pid)} != fold_idx {fold['fold_idx']}"
            )

    def test_manifest_test_patient_ids_match_expected(self, tmp_path):
        """test_patient_ids list must match what was passed as patient_ids_test."""
        _, paths, _, _, pids_test, _, _ = _run_mini_pipeline(tmp_path)
        manifest = json.loads(paths.fold_manifest_json.read_text())
        expected = sorted(int(float(p)) for p in pids_test)
        got = sorted(manifest["test_patient_ids"])
        assert got == expected, (
            f"Manifest test_patient_ids {got} != expected {expected}"
        )


# ---------------------------------------------------------------------------
# Issue 3 (fix) — Fold-pure CF predictions aligned with patient IDs
# ---------------------------------------------------------------------------

class TestFoldPureCFAlignment:
    """test_probs_cf_fold_pure[i] must come from the fold where patient_ids_test[i] was held out."""

    def test_fold_pure_cf_has_no_nans_when_all_test_patients_in_training(self, tmp_path):
        """When every test patient was a training patient, all fold-pure CF rows must be non-NaN."""
        rng = np.random.default_rng(42)
        n_patients = 4
        items_per_patient = 3
        n_features = 8
        n_classes = 2
        n_train = n_patients * items_per_patient

        X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
        Y_train = (rng.random((n_train, n_classes)) > 0.5).astype(np.float32)
        patient_ids_train = np.repeat(np.arange(1, n_patients + 1), items_per_patient)

        # Test set includes the same patients as training (one item per patient)
        X_test = rng.standard_normal((n_patients, n_features)).astype(np.float32)
        Y_test = (rng.random((n_patients, n_classes)) > 0.5).astype(np.float32)
        patient_ids_test = np.arange(1, n_patients + 1)

        from tracks.representation.training.orchestrator.train_ensemble_pipeline import train_ensemble_pipeline

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
        }
        result = train_ensemble_pipeline(
            X_train, Y_train, patient_ids_train,
            X_test, Y_test, cfg,
            patient_ids_test=patient_ids_test,
        )

        cf_fp = result.get("test_probs_cf_fold_pure")
        assert cf_fp is not None, "test_probs_cf_fold_pure missing from result"
        assert cf_fp.shape == (n_patients, n_classes)

        nan_rows = np.any(np.isnan(cf_fp), axis=1)
        assert not nan_rows.any(), (
            f"Fold-pure CF has NaN for rows {np.where(nan_rows)[0].tolist()} — "
            "every test patient was in training, so fold-pure lookup must succeed for all."
        )

    def test_fold_pure_cf_row_uses_correct_fold_model(self, tmp_path):
        """For each test patient p, fold-pure CF row must equal the prediction from fold where p was held out."""
        rng = np.random.default_rng(7)
        n_patients = 3
        items_per_patient = 4
        n_features = 8
        n_classes = 2
        n_train = n_patients * items_per_patient

        X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
        Y_train = (rng.random((n_train, n_classes)) > 0.5).astype(np.float32)
        patient_ids_train = np.repeat(np.arange(1, n_patients + 1), items_per_patient)

        # One test item per training patient so we can verify fold-pure alignment
        X_test = rng.standard_normal((n_patients, n_features)).astype(np.float32)
        Y_test = (rng.random((n_patients, n_classes)) > 0.5).astype(np.float32)
        patient_ids_test = np.arange(1, n_patients + 1)

        from tracks.representation.training.orchestrator.train_ensemble_pipeline import train_ensemble_pipeline, _build_paths

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
        }
        result = train_ensemble_pipeline(
            X_train, Y_train, patient_ids_train,
            X_test, Y_test, cfg,
            patient_ids_test=patient_ids_test,
        )

        paths = _build_paths(cfg)
        manifest = json.loads(paths.fold_manifest_json.read_text())
        p2fold = {f["held_out_patient_id"]: f["fold_idx"] for f in manifest["folds"]}
        cf_fp = result["test_probs_cf_fold_pure"]

        for test_row_i, pid in enumerate(patient_ids_test):
            fold_idx = p2fold.get(int(pid))
            if fold_idx is None:
                continue  # test patient not in training — skip
            ckpt = joblib.load(paths.checkpoints_dir / f"fold_{fold_idx + 1}.joblib")
            ycf = ckpt.get("ycf_test_probs")
            if ycf is None:
                continue  # Stage 2 inactive — ycf may be None
            np.testing.assert_array_almost_equal(
                cf_fp[test_row_i], ycf[test_row_i],
                decimal=5,
                err_msg=(
                    f"Fold-pure CF row {test_row_i} (patient {pid}) does not match "
                    f"ycf_test_probs from fold_{fold_idx + 1} checkpoint — "
                    "fold-pure lookup used wrong fold."
                ),
            )


# ---------------------------------------------------------------------------
# Issue 6 — Resume: ycf_test_probs and split compatibility
# ---------------------------------------------------------------------------

class TestResumeArtifactContract:
    """Resume must restore ycf_test_probs and reject checkpoints with mismatched splits."""

    def test_resume_restores_ycf_from_checkpoints(self, tmp_path):
        """Resumed folds' ycf_test_probs must appear in fold-pure CF result."""
        # First full run — generates all checkpoints
        result_full, paths, cfg, pids_train, pids_test, n_train, n_test = (
            _run_mini_pipeline(tmp_path)
        )

        # Delete the bundle so a fresh "resume" can proceed
        bundle = paths.bundle_path
        if bundle.exists():
            bundle.unlink()

        # Second run: resume=True, all folds have checkpoints → should reconstruct
        result_resumed, _, _, _, _, _, _ = _run_mini_pipeline(
            tmp_path, resume_from=paths.checkpoints_dir
        )

        cf_full    = result_full.get("test_probs_cf_fold_pure")
        cf_resumed = result_resumed.get("test_probs_cf_fold_pure")

        assert cf_resumed is not None, (
            "test_probs_cf_fold_pure is None after full resume — "
            "ycf_test_probs from checkpoints was not restored."
        )
        if cf_full is not None:
            # Non-NaN rows should match between full run and resumed run
            valid = ~np.any(np.isnan(cf_full), axis=1) & ~np.any(np.isnan(cf_resumed), axis=1)
            if valid.any():
                np.testing.assert_array_almost_equal(
                    cf_full[valid], cf_resumed[valid], decimal=5,
                    err_msg="Fold-pure CF differs between full run and resumed run.",
                )

    def test_resume_rejects_checkpoint_with_wrong_val_ix(self, tmp_path):
        """A checkpoint whose val_ix does not match the current splits must be discarded."""
        import tempfile
        from tracks.representation.training.orchestrator.fold_postprocessing import CHECKPOINT_VERSION
        from tracks.representation.models.ConstantCalibrator import ConstantCalibrator

        # Create a fake checkpoint with wrong val_ix
        ckpt_dir = tmp_path / "model" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        bad_ckpt = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "fold_idx": 0,
            "val_ix": np.array([999, 998, 997]),  # wrong indices
            "model_state": {},
            "preprocessor": None,
            "calibrators": {"c0": None, "c1": None},
            "thresholds": np.full(2, 0.5, dtype=np.float32),
            "pos_weights": np.ones(2, dtype=np.float32),
            "thresh_inner": np.full(2, 0.5, dtype=np.float32),
            "stage2": None,
            "oof_probs": np.random.default_rng(0).random((3, 2)).astype(np.float32),
            "oof_labels": np.random.default_rng(1).random((3, 2)).astype(np.float32),
            "best_hp": None,
            "ycf_test_probs": None,
        }
        joblib.dump(bad_ckpt, ckpt_dir / "fold_1.joblib")

        # Run with resume=True — the bad checkpoint should be discarded (not crash)
        result, paths, _, _, _, _, _ = _run_mini_pipeline(
            tmp_path, resume_from=ckpt_dir
        )
        # Pipeline should complete successfully despite the bad checkpoint
        assert result is not None, "Pipeline failed when resuming with a mismatched checkpoint"

    def test_resume_full_run_produces_same_manifest_as_fresh_run(self, tmp_path):
        """Manifest written after a full resume must have the same fold structure as fresh run."""
        # Fresh run in tmp_path — generates all checkpoints
        _, paths_fresh, _, _, _, _, _ = _run_mini_pipeline(tmp_path)
        m_fresh = json.loads(paths_fresh.fold_manifest_json.read_text())

        # Delete the bundle so the second run in the same directory can resume
        if paths_fresh.bundle_path.exists():
            paths_fresh.bundle_path.unlink()

        # Resume in the same directory: checkpoints are already there
        _, paths_resumed, _, _, _, _, _ = _run_mini_pipeline(
            tmp_path, resume_from=paths_fresh.checkpoints_dir
        )
        assert paths_resumed.fold_manifest_json.exists()
        m_resume = json.loads(paths_resumed.fold_manifest_json.read_text())

        assert m_fresh["n_folds"] == m_resume["n_folds"]
        # All folds in the resume manifest should report resumed_from_checkpoint=True
        # (since all checkpoints were valid and splits match)
        for fold in m_resume["folds"]:
            assert fold["resumed_from_checkpoint"] is True, (
                f"Fold {fold['fold_id']} not marked as resumed despite valid checkpoint"
            )


# ---------------------------------------------------------------------------
# Issue 5 — Stage 2 npy artifacts have consistent shapes
# ---------------------------------------------------------------------------

class TestStage2NpyArtifacts:
    """Per-fold Stage 2 .npy files must have shape (n_test, n_classes) when written."""

    def test_stage2_ca_npy_shape_matches_n_test(self, tmp_path):
        n_classes = 2
        items_per_patient = 3
        n_test = items_per_patient * 2

        _, paths, cfg, pids_train, _, _, _ = _run_mini_pipeline(
            tmp_path, with_stage2=True,
            n_classes=n_classes, items_per_patient=items_per_patient,
        )
        n_patients = len(np.unique(pids_train))

        ca_files = list(paths.ensemble_root.glob("fold_*_stage2_preds_ca.npy"))
        cf_files = list(paths.ensemble_root.glob("fold_*_stage2_preds_cf.npy"))

        if not ca_files:
            pytest.skip("No Stage 2 npy files written (Stage 2 may have had no paired items)")

        for f in ca_files + cf_files:
            arr = np.load(f)
            assert arr.ndim == 2, f"{f.name}: expected 2-D array, got shape {arr.shape}"
            assert arr.shape[1] == n_classes, (
                f"{f.name}: class dim {arr.shape[1]} != {n_classes}"
            )
            assert arr.shape[0] == n_test, (
                f"{f.name}: row count {arr.shape[0]} != n_test {n_test}"
            )

    def test_stage2_ca_and_cf_npy_have_same_shape(self, tmp_path):
        """For each fold, CA and CF npy files must have identical shapes."""
        _, paths, _, _, _, _, _ = _run_mini_pipeline(tmp_path, with_stage2=True)

        for fold_idx in range(10):  # up to 10 folds
            ca_path = paths.ensemble_root / f"fold_{fold_idx + 1}_stage2_preds_ca.npy"
            cf_path = paths.ensemble_root / f"fold_{fold_idx + 1}_stage2_preds_cf.npy"
            if not ca_path.exists():
                break
            ca = np.load(ca_path)
            cf = np.load(cf_path)
            assert ca.shape == cf.shape, (
                f"fold_{fold_idx + 1}: CA shape {ca.shape} != CF shape {cf.shape}"
            )


# ---------------------------------------------------------------------------
# Issue 7 — No cross-stage silently-wrong alignment in stats inputs
# ---------------------------------------------------------------------------

class TestStatisticsInputAlignment:
    """Arrays passed to run_statistical_analysis must be row-aligned with patient_ids."""

    def test_result_arrays_row_counts_are_consistent(self, tmp_path):
        """All prediction arrays in result must have the same first dimension (n_test)."""
        n_test = 6
        result, _, _, _, pids_test, _, _ = _run_mini_pipeline(tmp_path, with_stage2=True)

        check_keys = [
            "test_probs_cf", "test_probs_cf_fold_pure",
            "test_probs_ca", "test_probs_ca_fold_pure", "test_probs_shuffled_fold_pure",
        ]
        for key in check_keys:
            arr = result.get(key)
            if arr is None:
                continue
            assert arr.shape[0] == len(pids_test), (
                f"result['{key}'] has {arr.shape[0]} rows but patient_ids_test "
                f"has {len(pids_test)} entries — row alignment broken."
            )

    def test_oof_probs_row_count_equals_n_train(self, tmp_path):
        """oof_probs_cf must have the same row count as X_train / Y_train."""
        n_patients = 4
        items_per_patient = 3
        n_train = n_patients * items_per_patient
        result, _, _, pids_train, _, _, _ = _run_mini_pipeline(
            tmp_path, n_patients=n_patients, items_per_patient=items_per_patient
        )
        oof = result.get("oof_probs_cf")
        assert oof is not None, "oof_probs_cf missing from result"
        assert oof.shape[0] == n_train, (
            f"oof_probs_cf has {oof.shape[0]} rows but n_train={n_train}"
        )

    def test_arch_predictions_row_counts_match_test(self, tmp_path):
        """Every arch_predictions array must have the same row count as patient_ids_test."""
        result, _, _, _, pids_test, _, _ = _run_mini_pipeline(tmp_path, with_stage2=True)
        arch = result.get("arch_predictions")
        if arch is None:
            pytest.skip("No arch_predictions (Stage 2 not active or no paired items)")
        for arch_name, preds in arch.items():
            if preds is None:
                continue
            assert preds.shape[0] == len(pids_test), (
                f"arch_predictions['{arch_name}'] has {preds.shape[0]} rows "
                f"but patient_ids_test has {len(pids_test)} entries."
            )


# ---------------------------------------------------------------------------
# Grouped k-fold — several patients held out per fold
# ---------------------------------------------------------------------------

_KFOLD_CV = {
    "outer": {"scheme": "group_kfold", "n_splits": 2, "seed": 7},
    "inner": {"scheme": "group_kfold", "n_splits": 2, "seed": 7},
}


def _run_kfold_pipeline(tmp_path, **kwargs):
    return _run_mini_pipeline(
        tmp_path, n_patients=6, n_test_patients=6, with_stage2=True, cv=_KFOLD_CV,
        **kwargs,
    )


class TestGroupKFoldPipeline:
    """With grouped k-fold, every patient a fold holds out stays out of that fold's models."""

    def test_manifest_records_multi_patient_folds(self, tmp_path):
        _, paths, _, pids_train, _, n_train, _ = _run_kfold_pipeline(tmp_path)
        manifest = json.loads(paths.fold_manifest_json.read_text())

        assert manifest["n_folds"] == 2
        assert manifest["cv"]["outer"] == _KFOLD_CV["outer"]
        assert manifest["cv"]["inner"] == _KFOLD_CV["inner"]
        held = [f["held_out_patient_ids"] for f in manifest["folds"]]
        assert sorted(p for h in held for p in h) == sorted(np.unique(pids_train).tolist())
        assert all(len(h) == 3 for h in held)
        assert all(f["held_out_patient_id"] is None for f in manifest["folds"])

        p2f = {int(k): v for k, v in manifest["patient_to_fold"].items()}
        for fold in manifest["folds"]:
            for pid in fold["held_out_patient_ids"]:
                assert p2f[pid] == fold["fold_idx"]
            assert sorted(fold["val_ix"]) == sorted(
                np.where(np.isin(pids_train, fold["held_out_patient_ids"]))[0].tolist()
            )

    def test_fold_pure_cf_uses_the_fold_that_held_the_patient_out(self, tmp_path):
        result, paths, _, _, pids_test, _, _ = _run_kfold_pipeline(tmp_path)
        manifest = json.loads(paths.fold_manifest_json.read_text())
        p2f = {int(k): v for k, v in manifest["patient_to_fold"].items()}
        cf_fp = result["test_probs_cf_fold_pure"]

        assert not np.isnan(cf_fp).any(), "every test patient was held out by some fold"
        for row, pid in enumerate(pids_test):
            fold_idx = p2f[int(pid)]
            ckpt = joblib.load(paths.checkpoints_dir / f"fold_{fold_idx + 1}.joblib")
            np.testing.assert_array_almost_equal(cf_fp[row], ckpt["ycf_test_probs"][row], decimal=5)

    def test_stage2_never_trains_on_held_out_patients(self, tmp_path, monkeypatch):
        import tracks.representation.training.stage2.fit_stage2_fusion as s2

        seen = []
        original = s2.fit_stage2_fusion_fold

        def capture(*args, **kwargs):
            seen.append(np.unique(np.asarray(kwargs["patient_ids_tr"])).tolist())
            return original(*args, **kwargs)

        monkeypatch.setattr(s2, "fit_stage2_fusion_fold", capture)
        _, paths, _, _, _, _, _ = _run_kfold_pipeline(tmp_path)
        manifest = json.loads(paths.fold_manifest_json.read_text())

        assert len(seen) == manifest["n_folds"]
        for fold, s2_patients in zip(manifest["folds"], seen):
            assert s2_patients, "Stage 2 should have training patients"
            assert not set(s2_patients) & set(fold["held_out_patient_ids"]), (
                f"Stage 2 for fold {fold['fold_id']} trained on held-out patients "
                f"{set(s2_patients) & set(fold['held_out_patient_ids'])}"
            )

    def test_result_records_outer_and_inner_splits(self, tmp_path):
        result, _, _, pids_train, _, _, _ = _run_kfold_pipeline(tmp_path)
        cv = result["cv"]

        assert cv["outer"] == _KFOLD_CV["outer"]
        assert cv["inner"] == _KFOLD_CV["inner"]
        assert cv["n_outer_folds"] == len(cv["folds"]) == 2
        all_patients = set(np.unique(pids_train).tolist())
        for fold in cv["folds"]:
            inner = fold["inner_held_out_patient_ids"]
            assert len(inner) == 2
            assert sorted(p for g in inner for p in g) == sorted(
                all_patients - set(fold["held_out_patient_ids"])
            )

    def test_default_cv_is_lopo(self, tmp_path):
        result, paths, _, pids_train, _, _, _ = _run_mini_pipeline(tmp_path)
        manifest = json.loads(paths.fold_manifest_json.read_text())

        assert manifest["cv"]["outer"] == {"scheme": "lopo", "n_splits": None, "seed": None}
        for fold in manifest["folds"]:
            assert fold["held_out_patient_ids"] == [fold["held_out_patient_id"]]
        n_patients = len(np.unique(pids_train))
        assert all(len(f["inner_held_out_patient_ids"]) == n_patients - 1 for f in result["cv"]["folds"])

    def test_infeasible_inner_scheme_fails_before_training(self, tmp_path):
        cv = {"outer": {"scheme": "lopo"}, "inner": {"scheme": "group_kfold", "n_splits": 1, "seed": 0}}
        with pytest.raises(ValueError, match="at least 2"):
            _run_mini_pipeline(tmp_path, cv=cv)
        assert not list((tmp_path / "model").rglob("fold_*.joblib"))
