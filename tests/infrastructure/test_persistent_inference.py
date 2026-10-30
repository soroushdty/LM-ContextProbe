"""Tests for persistent inference behavior in EnsemblePredictor and ModelRegistry."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest
import torch

from tracks.representation.models.EnsemblePredictor import EnsemblePredictor, load_ensemble_predictor
from tracks.representation.models.ModelRegistry import ModelRegistry
from tracks.representation.models.MultiLabelModel import MultiLabelModel


# ── helpers ───────────────────────────────────────────────────────────────────

def _lr_state_dict(n_in: int, n_out: int) -> dict:
    """Logistic-regression state dict — all-zero weights, valid for load_state_dict."""
    return {
        "linear.weight": torch.zeros(n_out, n_in),
        "linear.bias": torch.zeros(n_out),
    }


def _passthrough_prep():
    """Preprocessor stub whose transform returns its input as float32 numpy."""
    prep = MagicMock()
    prep.transform.side_effect = lambda X: np.asarray(X, dtype=np.float32)
    return prep


def make_predictor(
    n_features: int = 4,
    n_classes: int = 2,
    n_folds: int = 2,
) -> EnsemblePredictor:
    """Minimal CPU-only EnsemblePredictor with fake logistic-regression folds."""
    class_names = [f"c{i}" for i in range(n_classes)]
    return EnsemblePredictor(
        models=[_lr_state_dict(n_features, n_classes) for _ in range(n_folds)],
        preprocessors=[_passthrough_prep() for _ in range(n_folds)],
        calibrators=[{cls: None for cls in class_names} for _ in range(n_folds)],
        thresholds=[0.5] * n_classes,
        device="cpu",
        class_list=class_names,
        model_arch={"hidden_dims": [], "dropout": 0.3, "activation": "gelu"},
        stage2=None,
    )


def _touch_bundle(tmp_path: Path, name: str = "ensemble.joblib") -> Path:
    p = tmp_path / name
    p.touch()
    return p


# ── materialize / release ─────────────────────────────────────────────────────

class TestMaterialize:
    def test_materialize_idempotent(self):
        pred = make_predictor()
        pred.materialize()
        first_list = pred._live_models
        assert first_list is not None
        assert len(first_list) == 2  # one live model per fold

        pred.materialize()  # must be a no-op
        assert pred._live_models is first_list  # same list object — no reallocation

    def test_predict_proba_uses_live_models(self, monkeypatch):
        pred = make_predictor()
        X = np.zeros((1, 4), dtype=np.float32)

        call_count = 0
        original_lsd = MultiLabelModel.load_state_dict

        def counting_lsd(self_m, state_dict, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_lsd(self_m, state_dict, **kwargs)

        monkeypatch.setattr(MultiLabelModel, "load_state_dict", counting_lsd)

        pred.predict_proba(X)
        after_first = call_count
        assert after_first > 0  # materialization occurred during first call

        pred.predict_proba(X)
        assert call_count == after_first  # live models reused — no second load

    def test_release_clears_live_models(self, monkeypatch):
        # Patch so the test does not require a CUDA device
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

        pred = make_predictor()
        pred.materialize()
        assert pred._live_models is not None

        pred.release()
        assert pred._live_models is None


# ── registry ──────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_registry_cache_hit(self, monkeypatch, tmp_path):
        bundle = _touch_bundle(tmp_path)
        payload = make_predictor().to_bundle_dict()
        load_calls = 0

        def fake_load(p):
            nonlocal load_calls
            load_calls += 1
            return payload

        monkeypatch.setattr(joblib, "load", fake_load)

        r1 = load_ensemble_predictor(bundle, use_registry=True)
        r2 = load_ensemble_predictor(bundle, use_registry=True)

        assert load_calls == 1
        assert r1 is r2

    def test_registry_invalidate_on_mtime_change(self, monkeypatch, tmp_path):
        bundle = _touch_bundle(tmp_path)
        payload = make_predictor().to_bundle_dict()
        load_calls = 0

        def fake_load(p):
            nonlocal load_calls
            load_calls += 1
            return payload

        monkeypatch.setattr(joblib, "load", fake_load)

        load_ensemble_predictor(bundle, use_registry=True)
        assert load_calls == 1

        # Advance mtime by 2 s — exceeds any filesystem timestamp resolution
        stat = bundle.stat()
        new_time = stat.st_mtime + 2.0
        os.utime(bundle, (new_time, new_time))

        load_ensemble_predictor(bundle, use_registry=True)
        assert load_calls == 2  # stale mtime → cache miss → re-deserialized

    def test_registry_lru_eviction(self, tmp_path):
        registry = ModelRegistry(max_size=1)
        path_a = _touch_bundle(tmp_path, "a.joblib")
        path_b = _touch_bundle(tmp_path, "b.joblib")

        # MagicMock(spec=…) gives us a spy on release() for free
        pred_a = MagicMock(spec=EnsemblePredictor)
        pred_b = MagicMock(spec=EnsemblePredictor)

        registry.put(path_a, "cpu", pred_a)
        assert len(registry) == 1

        registry.put(path_b, "cpu", pred_b)  # pred_a evicted
        assert len(registry) == 1

        pred_a.release.assert_called_once()
        pred_b.release.assert_not_called()

    def test_registry_thread_safety(self, monkeypatch, tmp_path):
        """Concurrent reads against a warm cache never re-deserialize."""
        bundle = _touch_bundle(tmp_path)
        payload = make_predictor().to_bundle_dict()
        load_calls: list[int] = []

        def fake_load(p):
            load_calls.append(1)
            return payload

        monkeypatch.setattr(joblib, "load", fake_load)

        # Populate the registry with a single sequential load
        first = load_ensemble_predictor(bundle, use_registry=True)
        assert len(load_calls) == 1

        # 8 threads race simultaneously against the warm cache
        results: list[EnsemblePredictor] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            pred = load_ensemble_predictor(bundle, use_registry=True)
            with results_lock:
                results.append(pred)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(load_calls) == 1  # warm cache: no additional deserialization
        assert len(results) == 8
        assert all(r is first for r in results)


# ── config flags ──────────────────────────────────────────────────────────────

class TestConfigFlags:
    def test_eager_materialize_flag(self, monkeypatch, tmp_path):
        bundle = _touch_bundle(tmp_path)
        monkeypatch.setattr(joblib, "load", lambda p: make_predictor().to_bundle_dict())

        cfg = {"inference": {"persistent_models": True, "eager_materialize": True}}
        pred = load_ensemble_predictor(bundle, cfg=cfg)

        # Live models must be populated before any predict_proba call
        assert pred._live_models is not None

    def test_persistent_models_disabled(self, monkeypatch, tmp_path):
        """With persistent_models=False the registry is bypassed:
        each call deserializes fresh and returns a distinct instance."""
        bundle = _touch_bundle(tmp_path)
        load_calls = 0

        def fake_load(p):
            nonlocal load_calls
            load_calls += 1
            return make_predictor().to_bundle_dict()

        monkeypatch.setattr(joblib, "load", fake_load)

        cfg = {"inference": {"persistent_models": False}}
        r1 = load_ensemble_predictor(bundle, cfg=cfg)
        r2 = load_ensemble_predictor(bundle, cfg=cfg)

        assert load_calls == 2  # registry bypassed: both calls hit disk
        assert r1 is not r2     # no cross-call caching


# ── latency regression ────────────────────────────────────────────────────────

@pytest.mark.slow
class TestLatency:
    def test_latency_regression(self):
        """Second predict_proba() reuses live models and must be ≥2× faster than first."""
        # 20 folds amplifies materialization overhead over forward-pass cost;
        # a small batch keeps the forward pass from hiding it (with 500 rows
        # the expected speedup is only ~2.2x, too close to the threshold).
        pred = make_predictor(n_features=50, n_classes=10, n_folds=20)
        X = np.random.default_rng(0).random((50, 50)).astype(np.float32)

        # Burn-in: absorb one-time Python / torch import overhead
        _warm = make_predictor(n_features=2, n_classes=1, n_folds=1)
        _warm.predict_proba(np.zeros((1, 2), dtype=np.float32))

        N = 15
        times_first: list[float] = []
        times_second: list[float] = []
        for _ in range(N):
            pred.release()
            t0 = time.perf_counter()
            pred.predict_proba(X)   # materializes on this call
            t1 = time.perf_counter()
            pred.predict_proba(X)   # reuses live models
            t2 = time.perf_counter()
            times_first.append(t1 - t0)
            times_second.append(t2 - t1)

        # Fastest of N: interference from other processes (for example other
        # pytest-xdist workers) only ever adds time, so the minimum is the
        # least noisy estimate of each call's own cost.
        best_first = min(times_first)
        best_second = min(times_second)

        assert best_second * 2 <= best_first, (
            f"Expected ≥2× speedup on reuse: "
            f"first={best_first * 1e3:.2f}ms  second={best_second * 1e3:.2f}ms"
        )
