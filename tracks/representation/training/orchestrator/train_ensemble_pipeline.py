import csv as _csv
import os
import logging
import time
from pathlib import Path
import numpy as np
from ..shared.resolve_thresholds import resolve_thresholds
import pandas as pd

# Use module-level logger for improved logging
logger = logging.getLogger(__name__)
import torch
from copy import deepcopy
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
)

from ..shared.set_seeds import set_seeds
from ..stage1.train_single_model import train_single_model
from .compute_metrics_and_save import compute_metrics_df, write_metrics_csv
from .plot_ensemble_figures import compute_ensemble_curves, write_ensemble_figures
from shared.cv import held_out_patients, patient_splits, patient_to_fold, resolve_cv
from ..shared.soft_label_utils import (
    _soft_scores,
    _soft_average_precision,
    _soft_roc_auc,
    _soft_pr_curve,
)
from .gpu_monitor import GpuMonitor
from ..stage1.hp_search import _run_hp_candidate
from .fold_postprocessing import _postprocess_fold, _apply_fold_result, detect_completed_folds
from ..shared.fit_calibrators import fit_calibrators
from ..shared.threshold_tuning import threshold_tuning

from tracks.representation.models.Preprocessor import Preprocessor
from tracks.representation.models.EnsemblePredictor import EnsemblePredictor, write_ensemble_manifest
from tracks.representation.models.MultiLabelModel import MultiLabelModel
from shared.utils.text_utils import normalize_for_matching
from shared.utils.array_utils import as_float32_array as _as_float32_array
from concurrent.futures import ThreadPoolExecutor, Future

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRIC_ORDER = ["Precision", "Recall", "MCC", "F1", "AUC ROC", "AUC PR", "AP"]


def _resolve_class_list(cfg):
    for key in ("default_classes", "classes", "class_list"):
        values = cfg.get(key)
        if isinstance(values, (list, tuple)) and len(values) > 0:
            return list(values)
    raise KeyError(
        "Missing class names in config. Provide one of: 'default_classes', 'classes', or 'class_list'."
    )


def _resolve_effective_class_list(cfg, target_dim: int):
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")

    base = _resolve_class_list(cfg)
    if len(base) == target_dim:
        return base

    active = cfg.get("active_classes")
    if isinstance(active, (list, tuple)) and len(active) == target_dim:
        return list(active)

    logging.warning(
        "Class list width mismatch (class_list=%d, y_dim=%d). Falling back to generic class names.",
        len(base),
        target_dim,
    )
    return [f"class_{i}" for i in range(target_dim)]


def _make_epoch_writer(path: str):
    """Return a callback that appends one CSV row per epoch to *path*.

    The header is written immediately when the closure is created so the file
    exists even if the training run is interrupted on epoch 0.
    """
    with open(path, "w", newline="", encoding="utf-8") as _f:
        _csv.writer(_f).writerow(["epoch", "loss", "val_score", "patience", "is_best"])

    def _cb(epoch, loss, val_score, patience, is_best):
        with open(path, "a", newline="", encoding="utf-8") as _f:
            _csv.writer(_f).writerow([epoch, loss, val_score, patience, int(is_best)])

    return _cb


def _resolve_ensemble_root(cfg, root: str = "./") -> str:
    model_dir = cfg.get("DIR_MODEL")
    if isinstance(model_dir, str) and model_dir.strip() != "":
        return os.path.join(root, model_dir)

    return os.path.join(root, "output", "model")


import json as _json_mod

from dataclasses import dataclass


def _write_metric_metadata(meta_path: str, **kwargs) -> None:
    """Write a JSON sidecar file describing a metrics CSV."""
    with open(meta_path, "w", encoding="utf-8") as _f:
        _json_mod.dump(kwargs, _f, indent=2)


def _cv_record(
    outer_cv, inner_cv, outer_splits, outer_held_out, patient_ids_train, inner_splits_by_fold,
) -> dict:
    """Describe the outer and inner splits by held-out patient IDs, for the run manifest.

    The Stage 1 inner splits are listed per outer fold. Stage 2's inner loops
    use the same inner scheme over the patients in Stage 2's training rows.
    """
    folds = []
    for fold_idx, ((train_ix, val_ix), inner_splits) in enumerate(
        zip(outer_splits, inner_splits_by_fold)
    ):
        patient_ids_tr = patient_ids_train[train_ix]
        folds.append({
            "fold_idx": fold_idx,
            "held_out_patient_ids": outer_held_out[fold_idx],
            "n_train": int(len(train_ix)),
            "n_val": int(len(val_ix)),
            "inner_held_out_patient_ids": [
                [int(float(p)) for p in patients]
                for patients in held_out_patients(patient_ids_tr, inner_splits)
            ],
        })
    return {
        "outer": outer_cv.to_dict(),
        "inner": inner_cv.to_dict(),
        "n_outer_folds": len(outer_splits),
        "folds": folds,
    }


def _log_metric_summary(
    log, filename: str, prediction_source: str,
    target_label_source: str, thresholding_mode: str,
    df, *, is_primary: bool,
) -> None:
    """Log a one-line summary for a generated metrics CSV."""
    macro_row = df[df["Class"] == "Macro Average"] if hasattr(df, "__len__") else None
    if macro_row is not None and not macro_row.empty:
        _f1 = macro_row["F1"].iloc[0]
        _auc_roc = macro_row["AUC ROC"].iloc[0]
        _auc_pr = macro_row["AUC PR"].iloc[0]
    else:
        _f1 = _auc_roc = _auc_pr = float("nan")
    _tag = "PRIMARY" if is_primary else "DIAGNOSTIC"
    log.info(
        "[METRIC][%s] %s | pred=%s | label=%s | thresh=%s | "
        "macro_F1=%.4f | macro_AUROC=%.4f | macro_AUPRC=%.4f",
        _tag, filename, prediction_source, target_label_source,
        thresholding_mode, _f1, _auc_roc, _auc_pr,
    )


@dataclass(frozen=True)
class TrainingPathBundle:
    """All output paths for one training run, constructed once before the fold loop."""
    ensemble_root:        Path
    dir_figures_train:    Path
    dir_figures_test:     Path
    checkpoints_dir:      Path
    bundle_path:          Path
    reproducibility_dir:  Path
    cv_folds_csv:         Path
    threshold_report_csv: Path
    gpu_summary_json:     Path
    gpu_chart_png:        Path

    def fold_hp_search_csv(self, fold_idx: int) -> Path:
        return self.ensemble_root / f"fold_{fold_idx + 1}_hp_search.csv"

    def fold_epoch_curve_csv(self, fold_idx: int) -> Path:
        return self.ensemble_root / f"fold_{fold_idx + 1}_epoch_curve.csv"

    def fold_checkpoint(self, fold_idx: int) -> Path:
        return self.checkpoints_dir / f"fold_{fold_idx + 1}.joblib"

    @property
    def fold_manifest_json(self) -> Path:
        return self.ensemble_root / "fold_manifest.json"

def _build_paths(cfg, root: str = "./") -> TrainingPathBundle:
    ensemble_root = Path(_resolve_ensemble_root(cfg, root=root))
    run_root = ensemble_root.parent
    reproducibility_dir = Path(
        cfg.get("DIR_REPRODUCIBILITY", run_root / "reproducibility_artifacts")
    )
    return TrainingPathBundle(
        ensemble_root        = ensemble_root,
        dir_figures_train    = ensemble_root / "figures" / "train",
        dir_figures_test     = ensemble_root / "figures" / "test",
        checkpoints_dir      = ensemble_root / "checkpoints",
        bundle_path          = ensemble_root / "ensemble_bundle.joblib",
        reproducibility_dir  = reproducibility_dir,
        cv_folds_csv         = ensemble_root / "CV_folds.csv",
        threshold_report_csv = ensemble_root / "threshold_report.csv",
        gpu_summary_json     = ensemble_root.parent / "gpu" / "gpu_utilization.json",
        gpu_chart_png        = ensemble_root.parent / "gpu" / "gpu_utilization.png",
    )


def _setup_output_dirs(paths: TrainingPathBundle, *, resume: bool = False) -> None:
    """Create output directories and remove stale incremental files."""
    paths.ensemble_root.mkdir(parents=True, exist_ok=True)
    paths.dir_figures_train.mkdir(parents=True, exist_ok=True)
    paths.dir_figures_test.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for stale in (paths.cv_folds_csv, paths.threshold_report_csv):
            if stale.exists():
                stale.unlink()



def _write_gpu_artifacts(monitor, paths: TrainingPathBundle) -> None:
    monitor.save_summary(str(paths.gpu_summary_json))
    monitor.save_chart(str(paths.gpu_chart_png))


def _macro_auc_pr(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    aucs = []
    for idx in range(y_true_bin.shape[1]):
        y_col = y_true_bin[:, idx]
        # PR-AUC is undefined for constant targets.
        if np.unique(y_col).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_col, y_prob[:, idx])
        aucs.append(auc(recall, precision))

    return float(np.mean(aucs)) if aucs else float("nan")
def _safe_macro_roc_auc(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    valid_cols = [idx for idx in range(y_true_bin.shape[1]) if np.unique(y_true_bin[:, idx]).size >= 2]
    if not valid_cols:
        return float("nan")
    return float(roc_auc_score(y_true_bin[:, valid_cols], y_prob[:, valid_cols], average="macro"))


def _safe_macro_ap(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    valid_cols = [idx for idx in range(y_true_bin.shape[1]) if np.unique(y_true_bin[:, idx]).size >= 2]
    if not valid_cols:
        return float("nan")
    return float(average_precision_score(y_true_bin[:, valid_cols], y_prob[:, valid_cols], average="macro"))


def _compute_macro_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> dict:
    per_class = {
        "Precision": [],
        "Recall": [],
        "MCC": [],
        "F1": [],
        "AUC ROC": [],
        "AUC PR": [],
        "AP": [],
    }

    for idx in range(y_true.shape[1]):
        y_c = np.asarray(y_true[:, idx], dtype=np.float32)
        p_c = np.asarray(y_prob[:, idx], dtype=np.float32)
        if y_c.size == 0:
            continue

        y_pred = (p_c >= thresholds[idx]).astype(np.float32)
        _s = _soft_scores(y_c, p_c, thresholds[idx])
        tp, fp, tn, fn = _s["TP"], _s["FP"], _s["TN"], _s["FN"]
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        denom_mcc = float(np.sqrt(
            max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-8)
        ))
        mcc = float((tp * tn - fp * fn) / denom_mcc)
        f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)

        per_class["Precision"].append(float(precision))
        per_class["Recall"].append(float(recall))
        per_class["MCC"].append(float(mcc))
        per_class["F1"].append(float(f1))
        per_class["AUC ROC"].append(float(_soft_roc_auc(y_c, p_c)))
        precision_curve, recall_curve = _soft_pr_curve(y_c, p_c)
        per_class["AUC PR"].append(float(np.trapezoid(precision_curve, recall_curve)) if recall_curve.size > 1 else 0.0)
        per_class["AP"].append(float(_soft_average_precision(y_c, p_c)))

    if not per_class["Precision"]:
        return {key: float("nan") for key in per_class}

    return {key: float(np.mean(values)) for key, values in per_class.items()}


def _fit_stage2_head(
    X_rows: np.ndarray,
    patient_ids_rows: np.ndarray,
    y_int_rows: np.ndarray,
    y_survey_rows: np.ndarray,
    y_cf_rows: np.ndarray,
    context_vectors: dict,
    class_list: list,
    cfg: dict,
    *,
    baseline_metrics: dict | None = None,
    metrics_df: pd.DataFrame | None = None,
) -> dict:
    """Fit a Stage 2 fusion head on the given rows and return its artifact.

    Fits the optional Stage 2 PCA on ``X_rows``, builds the configured fusion
    features (prepending ``y_cf_rows`` when ``stage2_use_cf_passthrough``),
    and runs ``fit_stage2_fusion_fold``. The artifact can be passed to
    ``apply_stage2_fusion``.
    """
    from ..stage2.fit_stage2_fusion import fit_stage2_fusion_fold
    from ..stage2.stage2_context import build_fusion_matrix

    fusion_strategy = cfg.get("fusion_strategy", "2d")
    r = int(cfg.get("lowrank_bilinear_r", 8))

    # --- Stage 2 PCA (optional) ---
    pca = None
    pca_k = cfg.get("stage2_pca_n_components", None)
    if pca_k is not None:
        from sklearn.decomposition import PCA as _PCA
        pca = _PCA(n_components=int(pca_k), whiten=True)
        pca.fit(X_rows)
        X_fit = pca.transform(X_rows)
        ctx_fit = {pid: pca.transform(cv.reshape(1, -1))[0]
                   for pid, cv in context_vectors.items()}
    else:
        X_fit = X_rows
        ctx_fit = context_vectors

    Z = build_fusion_matrix(X_fit, patient_ids_rows, ctx_fit, fusion_strategy=fusion_strategy, r=r)
    use_cf_pt = bool(cfg.get("stage2_use_cf_passthrough", False))
    if use_cf_pt:
        Z = np.concatenate([y_cf_rows.astype(np.float32), Z], axis=1)

    model, calibrators = fit_stage2_fusion_fold(
        Z_tr=Z,
        y_int_tr=y_int_rows,
        y_survey_tr=y_survey_rows,
        y_cf_tr=y_cf_rows,
        patient_ids_tr=patient_ids_rows,
        class_list=class_list,
        cfg=cfg,
        baseline_metrics=baseline_metrics,
        metrics_df=metrics_df,
    )
    return {
        "model": model,
        "pca": pca,
        "use_cf_passthrough": use_cf_pt,
        "fusion_strategy": fusion_strategy,
        "r": r,
        "calibrators": calibrators,
        "class_list": class_list,
    }


def _leave_pair_out_context_swap(
    X_test: np.ndarray,
    Y_test_interview: np.ndarray,
    Y_test_survey: np.ndarray,
    patient_ids_test: np.ndarray,
    context_vectors: dict,
    y_cf: np.ndarray,
    class_list: list,
    cfg: dict,
) -> dict | None:
    """Score the shuffled-context control with heads that saw neither context.

    Test patients with a context vector are grouped in pairs (see
    ``pair_patients_for_context_swap``). For each group, a Stage 2 head is
    fitted without any of the group's patients and scores each member's
    rows twice: with the member's own context and with its swap partner's.
    Both scores come from the same head, so their difference is due to the
    context alone. A fold-pure head would have been trained on the partner's
    rows, and a context it was trained on biases the comparison.

    Args:
        y_cf: (n_test, n_classes) fold-pure Stage 1 predictions; used as the
            heads' ŷ_cf and passed to ``apply_stage2_fusion``. Rows with NaN
            are left out of training and scoring.

    Returns:
        ``None`` when fewer than two patients have a context vector, else a
        dict with ``correct`` and ``shuffled`` ((n_test, n_classes), NaN for
        rows not scored), ``donors`` ((n_test,) partner patient, -1 if not
        scored) and ``groups``.
    """
    from ..stage2.fit_stage2_fusion import apply_stage2_fusion, project_stage2_embeddings
    from ..stage2.stage2_context import build_fusion_matrix, pair_patients_for_context_swap

    pids = np.array([int(float(p)) for p in patient_ids_test])
    with_ctx = [p for p in np.unique(pids) if int(p) in context_vectors]
    groups = pair_patients_for_context_swap(with_ctx, seed=int(cfg.get("shuffled_context_seed", 123)))
    if not groups:
        logger.warning(
            "Shuffled-context control skipped: %d test patient(s) have a context vector.",
            len(with_ctx),
        )
        return None

    correct = np.full(Y_test_interview.shape, np.nan, dtype=np.float32)
    shuffled = np.full(Y_test_interview.shape, np.nan, dtype=np.float32)
    donors = np.full(len(pids), -1, dtype=int)
    usable = ~np.isnan(Y_test_interview).all(axis=1) & ~np.isnan(y_cf).any(axis=1)
    for group in groups:
        train = usable & ~np.isin(pids, group)
        if not train.any():
            logger.warning("Shuffled-context group %s: no other patients to fit a head on.", group)
            continue
        head = _fit_stage2_head(
            X_test[train], pids[train], Y_test_interview[train], Y_test_survey[train],
            y_cf[train], context_vectors, class_list, cfg,
        )
        for i, patient in enumerate(group):
            partner = group[(i + 1) % len(group)]
            rows = np.flatnonzero((pids == patient) & ~np.isnan(y_cf).any(axis=1))
            if rows.size == 0:
                continue
            X_p, ctx_p = project_stage2_embeddings(head, X_test[rows], context_vectors)
            for out, ctx_pid in ((correct, patient), (shuffled, partner)):
                Z = build_fusion_matrix(
                    X_p, np.full(rows.size, ctx_pid), ctx_p,
                    fusion_strategy=head["fusion_strategy"], r=head["r"],
                )
                out[rows] = apply_stage2_fusion(head, Z, y_cf=y_cf[rows])
            donors[rows] = partner
    logger.info(
        "Shuffled-context control: %d leave-pair-out Stage 2 heads, %d rows scored.",
        len(groups), int((donors >= 0).sum()),
    )
    return {"correct": correct, "shuffled": shuffled, "donors": donors, "groups": groups}


def train_ensemble_pipeline(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    patient_ids_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    cfg: dict,
    *,
    item_strings_train: np.ndarray | None = None,
    item_strings_test: np.ndarray | None = None,
    unresolved_mask_train: np.ndarray | None = None,
    unresolved_mask_test: np.ndarray | None = None,
    embedding_cache: dict | None = None,
    resume_from_checkpoint: bool = False,
    Y_test_interview: np.ndarray | None = None,
    Y_test_survey: np.ndarray | None = None,
    patient_ids_test: np.ndarray | None = None,
    context_vectors: dict | None = None,
):
    """Patient-grouped CV ensemble training pipeline.

    Args:
        X_train, Y_train:         Training embeddings and labels.
        patient_ids_train:        Per-row patient IDs for the outer splits (``cv.outer``).
        X_test, Y_test:           Held-out test embeddings and labels.
        cfg:                      Pipeline configuration dict.
        item_strings_train:       Optional array of item strings parallel to
                                  ``X_train``/``Y_train``.  When provided,
                                  per-fold stratum labels are computed and
                                  returned as ``'oof_strata'``.
        item_strings_test:        Optional array of item strings parallel to
                                  ``X_test``/``Y_test``.
        unresolved_mask_train:    Optional bool array (shape: n_train) — True
                                  for rows whose item was NOT resolved by JSON
                                  and needs train-only fuzzy fallback.
        unresolved_mask_test:     Optional bool array (shape: n_test) — True
                                  for rows whose item needs fuzzy fallback.
        embedding_cache:          Optional {item_string: embedding_vector} dict
                                  covering all unique item strings in train and
                                  test.  When provided together with unresolved
                                  masks and ``cfg['fuzzy_threshold']``, fold-
                                  local fuzzy standardization is applied and
                                  embeddings are re-looked-up from this cache
                                  so X_tr/X_val/X_test reflect the post-fuzzy
                                  standardized item strings.
    """
    X_train = _as_float32_array(X_train, "X_train")
    Y_train = _as_float32_array(Y_train, "Y_train")
    X_test = _as_float32_array(X_test, "X_test")
    Y_test = _as_float32_array(Y_test, "Y_test")
    patient_ids_train = np.asarray(patient_ids_train)
    _use_fusion_stage2 = (
        context_vectors is not None
        and Y_test_interview is not None
        and Y_test_survey is not None
        and patient_ids_test is not None
        and bool(cfg.get("stage2_alpha_options"))
    )
    if _use_fusion_stage2:
        patient_ids_test = np.asarray(patient_ids_test)
        Y_test_interview = np.asarray(Y_test_interview, dtype=np.float32)
        Y_test_survey    = np.asarray(Y_test_survey,    dtype=np.float32)
    if item_strings_train is not None:
        item_strings_train = np.asarray(item_strings_train)
    if item_strings_test is not None:
        item_strings_test = np.asarray(item_strings_test)
    if unresolved_mask_train is not None:
        unresolved_mask_train = np.asarray(unresolved_mask_train, dtype=bool)
    if unresolved_mask_test is not None:
        unresolved_mask_test = np.asarray(unresolved_mask_test, dtype=bool)
    # patient_ids_test is always required — used by predict_proba and fold-pure CF tracking.
    if patient_ids_test is None:
        raise ValueError("patient_ids_test must not be None")
    # Y_test_interview and context_vectors are only required when Stage 2 fusion is active.
    if _use_fusion_stage2:
        if Y_test_interview is None:
            raise ValueError("Y_test_interview must not be None when Stage 2 fusion is enabled")
        if context_vectors is None:
            raise ValueError("context_vectors must not be None when Stage 2 fusion is enabled")

    class_list = _resolve_effective_class_list(cfg, target_dim=int(Y_train.shape[1]))

    if int(Y_test.shape[1]) != len(class_list):
        raise ValueError(
            f"Y_test width ({Y_test.shape[1]}) does not match resolved class_list length ({len(class_list)})."
        )

    set_seeds(cfg['global_seed'], deterministic_algorithms=True)
    paths = _build_paths(cfg)

    # outer_splits must be known before resume detection so n_folds is available.
    outer_cv = resolve_cv(cfg, "outer")
    inner_cv = resolve_cv(cfg, "inner")
    outer_splits = patient_splits(patient_ids_train, outer_cv)
    # Patients held out by each outer fold, and the fold that held out each
    # patient. Fold-pure scoring uses these to score every test row with the
    # fold whose model never saw that row's patient.
    _outer_held_out = [
        [int(float(p)) for p in _patients]
        for _patients in held_out_patients(patient_ids_train, outer_splits)
    ]
    _patient_to_fold = {
        int(float(_pid)): _fi
        for _pid, _fi in patient_to_fold(patient_ids_train, outer_splits).items()
    }
    # Inner splits depend only on each outer fold's training patients; build
    # them now so an infeasible inner scheme fails before any training.
    _inner_splits_by_fold = [
        patient_splits(patient_ids_train[_tr_ix], inner_cv)
        for _tr_ix, _ in outer_splits
    ]
    logging.info(
        "Cross-validation: outer %s (%d folds), inner %s.",
        outer_cv.to_dict(), len(outer_splits), inner_cv.to_dict(),
    )

    _setup_output_dirs(paths, resume=resume_from_checkpoint)
    ensemble_root = str(paths.ensemble_root)  # kept for legacy string uses below

    # --- Resume detection ---
    _completed_folds: dict[int, dict] = {}
    if resume_from_checkpoint:
        if paths.bundle_path.exists():
            raise RuntimeError(
                f"Cannot resume: ensemble_bundle.joblib already exists at {paths.bundle_path}. "
                "The run appears complete. Delete the bundle to force a fresh run."
            )
        _completed_folds = detect_completed_folds(
            paths.checkpoints_dir, len(outer_splits), len(class_list)
        )
        if _completed_folds:
            logging.info(
                "Resume: %d completed fold(s) found. Skipping folds %s.",
                len(_completed_folds),
                sorted(f + 1 for f in _completed_folds),
            )
        else:
            logging.info("Resume requested but no valid checkpoints found. Starting from fold 0.")

    ensemble_artifacts = {
        'models': [], 'preps': [], 'calibs': [], 'thresh': [],
        'thresh_inner': [],          # per-fold thresholds from winning HP's inner-CV OOF (leak-free)
        'pos_weights': [],
        'stage2': [],                # per-fold Ridge meta-models (or None entries)
    }
    fold_metrics_frames = []
    threshold_report_frames = []
    oof_preds = np.full((X_train.shape[0], len(class_list)), -1.0)
    # fold-local stratum array, populated only when item_strings_train is provided.
    oof_strata: np.ndarray | None = (
        np.full(X_train.shape[0], None, dtype=object)
        if item_strings_train is not None
        else None
    )
    # Accumulate per-fold Stage-1-only (context-free) test predictions for statistical analysis.
    # Keyed by fold_idx so that resume runs (which skip completed folds) still index correctly.
    _test_probs_cf_folds: dict[int, np.ndarray] = {}
    # Per-fold predictions for each alternative fusion architecture and baseline.
    # Keyed by fold_idx so fold-pure assembly can pick the right model per patient.
    _arch_fold_preds: dict[str, dict[int, np.ndarray]] = {}

    # Validate resumed checkpoints against current splits to catch data-change mismatches.
    if _completed_folds:
        _split_mismatch = []
        for _ri, _ckpt in _completed_folds.items():
            _expected_val_ix = outer_splits[_ri][1]
            _ckpt_val_ix = np.asarray(_ckpt["val_ix"])
            if not np.array_equal(np.sort(_ckpt_val_ix), np.sort(_expected_val_ix)):
                logging.warning(
                    "Resume: fold %d checkpoint val_ix does not match current splits "
                    "(checkpoint %r vs current %r). Will re-train fold.",
                    _ri + 1, _ckpt_val_ix.tolist(), _expected_val_ix.tolist(),
                )
                _split_mismatch.append(_ri)
        for _ri in _split_mismatch:
            del _completed_folds[_ri]
        if _split_mismatch:
            logging.warning(
                "Resume: %d fold(s) discarded due to split mismatch: %s. "
                "These folds will be retrained from scratch.",
                len(_split_mismatch), sorted(f + 1 for f in _split_mismatch),
            )

    # Reconstruct accumulated state from completed fold checkpoints.
    if _completed_folds:
        for _resume_fold_idx in sorted(_completed_folds):
            _ckpt = _completed_folds[_resume_fold_idx]
            ensemble_artifacts["models"].append(_ckpt["model_state"])
            ensemble_artifacts["preps"].append(_ckpt["preprocessor"])
            ensemble_artifacts["calibs"].append(_ckpt["calibrators"])
            ensemble_artifacts["thresh"].append(_ckpt["thresholds"])
            ensemble_artifacts["thresh_inner"].append(_ckpt["thresh_inner"])
            ensemble_artifacts["pos_weights"].append(_ckpt["pos_weights"])
            ensemble_artifacts["stage2"].append(_ckpt.get("stage2"))
            oof_preds[_ckpt["val_ix"]] = _ckpt["oof_probs"]
            # Restore CF test predictions so fold-pure and averaged CF computations
            # include resumed folds, not just newly trained ones.
            _ycf_resume = _ckpt.get("ycf_test_probs")
            if _ycf_resume is not None:
                _test_probs_cf_folds[_resume_fold_idx] = _ycf_resume
            if "ycf_test_probs" not in ensemble_artifacts:
                ensemble_artifacts["ycf_test_probs"] = []
            ensemble_artifacts["ycf_test_probs"].append(_ycf_resume)

    param_grid = [(cfg.get("hidden_dims", []),)]

    # ── Fuzzy fallback helpers ────────────────────────────────────────────────
    # Active when embedding_cache, item strings, and fuzzy_threshold are all
    # present.  The cache must contain embeddings for every unique item string
    # (pre- and post-fuzzy) so we can re-assemble X matrices from post-fuzzy
    # standardized strings without re-running the embedding model.
    _fuzzy_enabled = (
        embedding_cache is not None
        and cfg.get("fuzzy_threshold") is not None
        and item_strings_train is not None
    )
    _item_col = cfg.get("item_col", "Item")

    def _reassemble_X(strings: np.ndarray, original_strings: np.ndarray) -> np.ndarray:
        """Look up embedding vectors from cache; fall back to original string."""
        rows = []
        for i, s in enumerate(strings):
            vec = embedding_cache.get(s)  # type: ignore[union-attr]
            if vec is None:
                vec = embedding_cache.get(original_strings[i])  # type: ignore[union-attr]
            rows.append(vec)
        return _as_float32_array(np.array(rows), "reassembled_X")

    # ── Pre-fold: apply full-train fuzzy fallback to held-out test set ────────
    # Build one matcher from the full training set so test items can be matched
    # against training-only candidates.  Held-out test strings are never added
    # to the matcher's candidate space.
    if _fuzzy_enabled and item_strings_test is not None:
        from shared.preprocessing.fuzzy_mapping import (
            build_fuzzy_matcher_from_train,
            apply_fuzzy_fallback,
        )
        _train_unres = (
            pd.Series(unresolved_mask_train) if unresolved_mask_train is not None else None
        )
        _full_matcher = build_fuzzy_matcher_from_train(
            pd.DataFrame({_item_col: item_strings_train}),
            _item_col,
            float(cfg["fuzzy_threshold"]),
            _train_unres,
        )
        _test_unres = (
            pd.Series(unresolved_mask_test) if unresolved_mask_test is not None else None
        )
        _test_df_std, _ = apply_fuzzy_fallback(
            pd.DataFrame({_item_col: item_strings_test}),
            _item_col,
            _full_matcher,
            _test_unres,
        )
        _test_strings_final = _test_df_std[_item_col].values
        X_test = _reassemble_X(_test_strings_final, item_strings_test)
        logging.info(
            "Pre-fold fuzzy fallback applied to test set: %d unresolved test items processed.",
            int(np.sum(unresolved_mask_test)) if unresolved_mask_test is not None else 0,
        )

    logging.info(f"Starting patient-grouped CV training on X_train: {X_train.shape}...")
    _pipeline_t0 = time.perf_counter()

    if X_train.shape[0] < 2:
        raise ValueError("train_ensemble_pipeline requires at least 2 training samples.")

    # GPU monitor: read config and start background polling
    _gpu_cfg = cfg.get("GPU_MONITOR", {}) or {}
    _gpu_monitor = GpuMonitor(
        enabled               = bool(_gpu_cfg.get("enabled", True)),
        poll_interval         = float(_gpu_cfg.get("poll_interval_s", 1.0)),
        active_util_threshold = int(_gpu_cfg.get("active_util_threshold_pct", 10)),
        upgrade_sm_threshold  = int(_gpu_cfg.get("upgrade_sm_threshold_pct", 70)),
        save_chart_enabled    = bool(_gpu_cfg.get("save_chart", True)),
    )
    _gpu_monitor.start()

    # Executor for async fold postprocessing.
    # max_workers=1: postprocessing runs in one background thread while the
    # next fold's HP search runs on GPU. Avoids concurrent list mutations.
    _post_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
    _pending_post: "Future | None" = None

    for fold_idx, (train_ix, val_ix) in enumerate(outer_splits):
        fold_id = str(fold_idx + 1)

        # Skip folds whose checkpoint was loaded on resume.
        if fold_idx in _completed_folds:
            if item_strings_train is not None and oof_strata is not None:
                _resume_train_norms = set(
                    pd.Series(item_strings_train[train_ix]).map(normalize_for_matching)
                )
                oof_strata[val_ix] = np.where(
                    pd.Series(item_strings_train[val_ix])
                      .map(normalize_for_matching)
                      .isin(_resume_train_norms),
                    "repeated",
                    "novel",
                )
            logging.info("Resume: skipping fold %d (checkpoint valid).", fold_idx + 1)
            continue

        X_tr, Y_tr = X_train[train_ix], Y_train[train_ix]
        X_val, Y_val = X_train[val_ix], Y_train[val_ix]
        patient_ids_tr = patient_ids_train[train_ix]
        _gpu_monitor.mark_fold(fold_idx + 1)

        # ── Fold-local fuzzy fallback ─────────────────────────────────────────
        # Build fuzzy matcher from this fold's training strings ONLY.  The
        # held-out validation strings are never added to the candidate space.
        # After standardization, re-assemble X_tr/X_val from the embedding
        # cache so model inputs reflect the fold-standardized item text.
        if _fuzzy_enabled:
            from shared.preprocessing.fuzzy_mapping import (
                build_fuzzy_matcher_from_train,
                apply_fuzzy_fallback,
            )
            _fold_train_strings = item_strings_train[train_ix]  # type: ignore[index]
            _fold_val_strings   = item_strings_train[val_ix]    # type: ignore[index]

            _fold_train_unres = (
                pd.Series(unresolved_mask_train[train_ix])
                if unresolved_mask_train is not None else None
            )
            _fold_val_unres = (
                pd.Series(unresolved_mask_train[val_ix])
                if unresolved_mask_train is not None else None
            )

            # Matcher built from fold-training unresolved items only.
            _fold_matcher = build_fuzzy_matcher_from_train(
                pd.DataFrame({_item_col: _fold_train_strings}),
                _item_col,
                float(cfg["fuzzy_threshold"]),
                _fold_train_unres,
            )

            _fold_tr_df_std, _ = apply_fuzzy_fallback(
                pd.DataFrame({_item_col: _fold_train_strings}),
                _item_col, _fold_matcher, _fold_train_unres,
            )
            _fold_val_df_std, _ = apply_fuzzy_fallback(
                pd.DataFrame({_item_col: _fold_val_strings}),
                _item_col, _fold_matcher, _fold_val_unres,
            )

            X_tr  = _reassemble_X(_fold_tr_df_std[_item_col].values,  _fold_train_strings)
            X_val = _reassemble_X(_fold_val_df_std[_item_col].values, _fold_val_strings)

        # Logging: outer fold start
        _fold_t0 = time.perf_counter()
        _held_out_label = (
            f"held-out patient: {_outer_held_out[fold_idx][0]}"
            if len(_outer_held_out[fold_idx]) == 1
            else f"held-out patients: {_outer_held_out[fold_idx]}"
        )
        logging.info(
            f"Outer fold {fold_idx + 1} / {len(outer_splits)} — "
            f"{_held_out_label} "
            f"({len(val_ix)} val rows, {len(train_ix)} train rows) — "
            f"starting HP search."
        )
        _zero_pos_classes = [
            class_list[c] for c in range(Y_tr.shape[1])
            if float(Y_tr[:, c].sum()) == 0.0
        ]
        if _zero_pos_classes:
            logging.warning(
                f"Outer fold {fold_idx + 1} — "
                f"{len(_zero_pos_classes)} class(es) with zero positive training "
                f"examples: {_zero_pos_classes}. "
                f"Metrics for these classes will be NaN this fold."
            )

        # Fold-local stratum: items seen in THIS fold's training set are
        # "repeated"; others are "unique".  Differs from the global stratum
        # column (which counts all training patients) because the held-out
        # patient's items are absent from this fold's training vocabulary.
        if item_strings_train is not None and oof_strata is not None:
            train_item_norms = set(
                pd.Series(item_strings_train[train_ix]).map(normalize_for_matching)
            )
            fold_stratum = np.where(
                pd.Series(item_strings_train[val_ix])
                  .map(normalize_for_matching)
                  .isin(train_item_norms),
                "repeated",
                "novel",
            )
            oof_strata[val_ix] = fold_stratum

        # Collect the previous fold's postprocessing result before starting
        # this fold's GPU work. This is always a no-op on fold 0.
        if _pending_post is not None:
            _apply_fold_result(
                _pending_post.result(),
                ensemble_artifacts, oof_preds,
                paths=paths,
            )
            _pending_post = None

        # Inner patient-grouped CV for hyperparameter selection
        inner_splits = _inner_splits_by_fold[fold_idx]
        best_score = np.inf
        best_hp = None
        # best_inner_oof: Stage 1 OOF from the winning HP candidate's inner CV.
        # Used for thresh_inner derivation and Stage 2 Ridge fitting. None when
        # HP is loaded from a crash-recovery CSV (no OOF available); Ridge is
        # skipped and thresh_inner falls back to 0.5 for that fold.
        best_inner_oof: np.ndarray | None = None

        # If an HP search CSV already exists (e.g. crash after HP search but
        # before the fold checkpoint was written), skip the search and reload
        # the best HP directly to avoid re-spending GPU time.
        _hp_csv = paths.fold_hp_search_csv(fold_idx)
        if _hp_csv.exists():
            try:
                _hp_df_cached = pd.read_csv(_hp_csv)
                if not _hp_df_cached.empty:
                    _best_row = _hp_df_cached.nsmallest(1, "inner_brier").iloc[0]
                    import json as _json
                    best_hp = {
                        "selected_stage1_head_config": _json.loads(_best_row["selected_stage1_head_config"]),
                        "lr":           float(cfg.get("lr", cfg.get("lr_options", [0.001])[0])),
                        "weight_decay": float(cfg.get("weight_decay", 0.0001)),
                        "batch_size":   int(cfg.get("batch_size", 2048)),
                        "weight_cap":   float(cfg.get("weight_cap", cfg.get("weight_cap_options", [10.0])[0])),
                    }
                    best_score = float(_best_row["inner_brier"])
                    logging.info(
                        "Fold %d — HP search skipped (loaded from CSV). "
                        "Best: selected_stage1_head_config=%r, inner_Brier=%.4f. "
                        "Stage 2 Ridge will be skipped for this fold (no OOF available).",
                        fold_idx + 1, best_hp["selected_stage1_head_config"], best_score,
                    )
            except Exception as _exc:
                logging.warning(
                    "Fold %d — HP CSV read failed (%s); re-running HP search.",
                    fold_idx + 1, _exc,
                )
                best_hp = None

        if best_hp is None:
            # --- Parallel inner HP grid search ---
            # Each HP candidate runs all inner folds in a separate thread.
            # Threads share the CUDA context safely; unique per-thread seeds
            # prevent global RNG collisions between concurrent workers.
            n_hp_workers = max(1, int(cfg.get("n_hp_workers", 4)))
            n_hp_workers = min(n_hp_workers, len(param_grid))

            with ThreadPoolExecutor(max_workers=n_hp_workers) as hp_pool:
                hp_futures = [
                    hp_pool.submit(
                        _run_hp_candidate,
                        hp_t,
                        X_tr, Y_tr,
                        inner_splits,
                        cfg,
                        seed_base=(hp_idx + 1) * 1000,   # unique per HP candidate
                    )
                    for hp_idx, hp_t in enumerate(param_grid)
                ]
                hp_results = [f.result() for f in hp_futures]

            for hp_candidate, avg_s, inner_oof in hp_results:
                if avg_s < best_score:
                    best_score     = avg_s
                    best_hp        = hp_candidate
                    best_inner_oof = inner_oof

            # Logging: HP search result (improved)
            _hp_elapsed = time.perf_counter() - _fold_t0
            logger.info(
                f"Outer fold {fold_idx + 1} — HP search complete "
                f"({_hp_elapsed:.1f}s). "
                f"Best: selected_stage1_head_config={best_hp['selected_stage1_head_config']!r}, "
                f"inner_Brier={best_score:.4f} "
                f"(grid size: {len(param_grid)})."
            )
            # Write HP grid search results to CSV for diagnostics.
            if bool(cfg.get("log_hp_search_grid", False)):
                import json as _json
                _hp_rows = [
                    {
                        "selected_stage1_head_config": _json.dumps(hp_c["selected_stage1_head_config"]),
                        "inner_brier": round(float(avg_s), 6),
                    }
                    for hp_c, avg_s, _ in hp_results
                ]
                _hp_df = pd.DataFrame(_hp_rows).sort_values("inner_brier").reset_index(drop=True)
                _hp_df.to_csv(paths.fold_hp_search_csv(fold_idx), index=False)
        assert best_hp is not None, f"HP search failed for fold {fold_idx + 1}"

        # Derive per-fold thresholds and fit calibrators from the winning HP candidate's inner-CV OOF.
        # best_inner_oof covers all outer-training rows and never touches the held-out patient.
        if best_inner_oof is not None:
            fold_thresh_inner = threshold_tuning(best_inner_oof, Y_tr, class_list, cfg)
            fold_calibrators = fit_calibrators(best_inner_oof, Y_tr, class_list, cfg)
        else:
            # HP was loaded from a crash-recovery CSV; inner OOF is unavailable.
            # Fall back to 0.5 and warn — this only occurs in resume scenarios.
            fold_thresh_inner = np.full(len(class_list), 0.5, dtype=np.float32)
            fold_calibrators = None
            logging.warning(
                "Fold %d — thresh_inner defaulting to 0.5 (HP loaded from CSV; "
                "inner OOF unavailable for threshold derivation).",
                fold_idx + 1,
            )
        ensemble_artifacts["thresh_inner"].append(fold_thresh_inner)
        # Always store inner OOF calibrators for use in main pipeline
        if "calibrators_inner" not in ensemble_artifacts:
            ensemble_artifacts["calibrators_inner"] = []
        ensemble_artifacts["calibrators_inner"].append(fold_calibrators)

        # --- Final Fold Training ---
        prep    = Preprocessor(
            n_components=int(cfg.get("pca_n_components", 200)),
            whiten=bool(cfg.get("pca_whiten", True)),
        ).fit(X_tr)
        X_tr_p  = prep.transform(X_tr)
        X_val_p = prep.transform(X_val)

        weight_mode = cfg.get("class_weight_mode", "cui")
        beta = float(cfg.get("class_weight_beta", 0.999))
        valid_counts = Y_tr.shape[0]
        pos_mass     = Y_tr.sum(axis=0)
        if weight_mode == "inverse_frequency":
            pos_w = np.clip(
                valid_counts / (2.0 * np.maximum(pos_mass, 1.0)),
                1.0,
                best_hp["weight_cap"],
            )
        else:  # Cui 2019 (default)
            pos_w = (1 - beta) / (1 - np.power(beta, np.maximum(pos_mass, 1)))
            pos_w = np.clip(pos_w, 1.0, best_hp["weight_cap"])

        # Logging: final fold retraining start
        _retrain_t0 = time.perf_counter()
        logging.info(
            f"Outer fold {fold_idx + 1} — final retraining: "
            f"selected_stage1_head_config={best_hp['selected_stage1_head_config']!r}, "
            f"lr={best_hp['lr']:.5f}, "
            f"weight_decay={best_hp['weight_decay']:.5f}, "
            f"batch_size={best_hp['batch_size']}, "
            f"weight_cap={best_hp['weight_cap']:.1f}."
        )
        final_cfg = dict(cfg)
        final_cfg["hidden_dims"] = best_hp.get("selected_stage1_head_config", cfg.get("hidden_dims", []))
        model, raw_val_probs, _fold_thresh_outer = train_single_model(
            X_tr_p, Y_tr, X_val_p, Y_val, final_cfg, best_hp, pos_w,
            log_context=f"Fold {fold_idx + 1} / {len(outer_splits)} final",
            epoch_curve_callback=_make_epoch_writer(str(paths.fold_epoch_curve_csv(fold_idx))),
        )
        # _fold_thresh_outer is tuned on the outer held-out patient — not used for avg_thresh.
        # Logging: final fold retraining complete
        _retrain_elapsed = time.perf_counter() - _retrain_t0
        logging.info(
            f"Outer fold {fold_idx + 1} — final retraining complete "
            f"({_retrain_elapsed:.1f}s). Postprocessing queued."
        )

        # Save artifacts that don't depend on postprocessing immediately.
        # Ensure model is a PyTorch nn.Module, not a function
        from tracks.representation.models.MultiLabelModel import MultiLabelModel
        if not isinstance(model, MultiLabelModel):
            raise TypeError(f"Returned model is not a MultiLabelModel instance, got {type(model)}")
        ensemble_artifacts["models"].append(deepcopy(model.state_dict()))
        ensemble_artifacts["preps"].append(prep)
        ensemble_artifacts["pos_weights"].append(pos_w)

        # Compute Stage 1 context-free predictions on test items for this fold.
        # Always computed (not just for Stage 2) so that test_stage1_vs_survey_metrics.csv
        # can be written at end of pipeline regardless of Stage 2 enablement.
        # Apply inner-OOF calibrators so ŷ_cf is calibrated, matching the oof_probs contract.
        from ..shared.apply_calibrators import apply_calibrators as _apply_cal
        _X_test_p = prep.transform(X_test)
        if not hasattr(model, "eval") or not hasattr(model, "parameters"):
            raise TypeError("Returned model is not a PyTorch nn.Module instance")
        model.eval()
        _device = next(model.parameters()).device
        with torch.no_grad():
            _raw_cf = torch.sigmoid(
                model(torch.tensor(_X_test_p, dtype=torch.float32, device=_device))
            ).cpu().numpy()
        if fold_calibrators is not None:
            _y_cf_test_fold = _apply_cal(fold_calibrators, _raw_cf, class_list).astype(np.float32)
        else:
            _y_cf_test_fold = np.clip(_raw_cf, 0.0, 1.0).astype(np.float32)

        # Accumulate Stage-1-only test predictions across folds for statistical analysis
        # and Stage 2 artifact persistence (calibrated ŷ_cf saved in fold checkpoint).
        # Keyed by fold_idx (not list-appended) so that resume runs index correctly.
        _test_probs_cf_folds[fold_idx] = _y_cf_test_fold
        if "ycf_test_probs" not in ensemble_artifacts:
            ensemble_artifacts["ycf_test_probs"] = []
        ensemble_artifacts["ycf_test_probs"].append(_y_cf_test_fold)

        _fold_stage2 = None
        if _use_fusion_stage2 and _y_cf_test_fold is not None:
            # Stage 2 training items: test items from patients this outer fold
            # did not hold out that have at least one non-NaN interview label.
            _s2_mask = ~np.isin(
                np.array([int(float(p)) for p in patient_ids_test]),
                _outer_held_out[fold_idx],
            )
            if Y_test_interview is None:
                raise ValueError("Y_test_interview must not be None for Stage 2 training")
            _paired = ~np.isnan(Y_test_interview[_s2_mask]).all(axis=1)
            _s2_idx = np.where(_s2_mask)[0][_paired] if _s2_mask is not None else np.array([], dtype=int)

            if len(_s2_idx) > 0:
                # --- ŷ_cf passthrough ---
                _use_cf_pt = bool(cfg.get("stage2_use_cf_passthrough", False))
                # Default: outer fold's transductive CF (may be biased for inner val).
                _y_cf_s2 = _y_cf_test_fold[_s2_idx]
                if _use_cf_pt:
                    # Inner Stage 2 validation patient Q's ŷ_cf should come from a Stage 1
                    # model NOT trained on Q. Substitute fold-pure CF from Q's own outer
                    # fold where that fold has already been computed; otherwise fall back.
                    _y_cf_s2 = _y_cf_test_fold[_s2_idx].copy()
                    _n_xfit = 0
                    for _qi, _global_i in enumerate(_s2_idx):
                        _qpid = int(float(patient_ids_test[_global_i]))
                        _qfold = _patient_to_fold.get(_qpid)
                        if _qfold is not None and _qfold in _test_probs_cf_folds:
                            _y_cf_s2[_qi] = _test_probs_cf_folds[_qfold][_global_i]
                            _n_xfit += 1
                    _n_s2_total = len(_s2_idx)
                    if _n_xfit < _n_s2_total:
                        logger.warning(
                            "Outer fold %d — Stage 2 passthrough: %d/%d items use "
                            "fold-pure cross-fitted ŷ_cf; %d fall back to transductive CF "
                            "(their outer folds not yet computed).",
                            fold_idx + 1, _n_xfit, _n_s2_total, _n_s2_total - _n_xfit,
                        )
                    else:
                        logger.info(
                            "Outer fold %d — Stage 2 passthrough: all %d items use "
                            "fold-pure cross-fitted ŷ_cf.",
                            fold_idx + 1, _n_xfit,
                        )
                # Compute per-class metrics for this fold (F1, Brier, ECE)
                fold_metrics_df = compute_metrics_df(
                    _y_cf_test_fold[_s2_idx],
                    Y_test_interview[_s2_idx],
                    fold_thresh_inner,
                    class_list,
                    cfg,
                    patient_ids_test[_s2_idx],
                )
                # Convert DataFrame to dict of per-class metrics
                baseline_metrics = {
                    row["Class"]: {
                        "F1": row["F1"],
                        "Brier Score": row["Brier Score"],
                        "ECE": row["ECE"],
                    }
                    for _, row in fold_metrics_df.iterrows() if row["Class"] not in ("Macro Average", "Micro Aggregate")
                }
                # Pass the full metrics DataFrame as well as legacy baseline_metrics
                _fold_stage2 = _fit_stage2_head(
                    X_test[_s2_idx],
                    patient_ids_test[_s2_idx],
                    Y_test_interview[_s2_idx],
                    Y_test_survey[_s2_idx],
                    _y_cf_s2,
                    context_vectors,
                    class_list,
                    cfg,
                    baseline_metrics=baseline_metrics,
                    metrics_df=fold_metrics_df,
                )
                logger.info(
                    "Outer fold %d — Stage 2 fusion model fitted on %d items.",
                    fold_idx + 1, len(_s2_idx),
                )
            else:
                logger.warning(
                    "Outer fold %d — no paired test items for Stage 2 training.", fold_idx + 1
                )
        ensemble_artifacts["stage2"].append(_fold_stage2)

        # ── Alternative fusion architectures and baselines ─────────────────────
        # Trains 2d, 3d, lowrank_bilinear, and patient_id under their own inner
        # inner-CV alpha search (same outer fold, independently regularised).
        # Passthrough and stage1_only predictions require no training and are
        # assembled from _y_cf_test_fold at the end of the fold loop.
        if _use_fusion_stage2 and _fold_stage2 is not None and '_s2_idx' in locals() and len(_s2_idx) > 0:
            from ..stage2.fit_stage2_fusion import (
                fit_all_stage2_architectures_fold as _fit_all_archs,
                apply_stage2_fusion as _apply_s2,
                _build_patient_id_baseline_features as _pid_feats,
            )
            from ..stage2.stage2_context import build_fusion_matrix as _bfm_alt

            _r_alt = int(cfg.get("lowrank_bilinear_r", 8))
            _training_pids = patient_ids_test[_s2_idx]
            try:
                _alt_models = _fit_all_archs(
                    X_items_tr=X_test[_s2_idx],
                    patient_ids_tr=_training_pids,
                    context_vectors=context_vectors,
                    y_int_tr=Y_test_interview[_s2_idx],
                    y_survey_tr=Y_test_survey[_s2_idx],
                    y_cf_tr=_y_cf_s2,
                    class_list=class_list,
                    cfg=cfg,
                    r=_r_alt,
                )
                for arch_name, alt_model in _alt_models.items():
                    if arch_name in ("2d", "3d", "lowrank_bilinear"):
                        _Z_test_alt = _bfm_alt(
                            X_test, patient_ids_test, context_vectors,
                            fusion_strategy=arch_name, r=_r_alt,
                        )
                        _preds_alt = _apply_s2(alt_model, _Z_test_alt)
                    elif arch_name == "patient_id":
                        _X_pid_test = _pid_feats(
                            X_test, patient_ids_test, reference_ids=_training_pids,
                        )
                        _preds_alt = _apply_s2(alt_model, _X_pid_test)
                    else:
                        continue
                    _arch_fold_preds.setdefault(arch_name, {})[fold_idx] = _preds_alt
                logger.info(
                    "Outer fold %d — alternative architectures fitted: %s.",
                    fold_idx + 1, sorted(_alt_models.keys()),
                )
            except Exception as _alt_exc:
                logger.warning(
                    "Outer fold %d — alternative architecture training failed: %s. "
                    "arch_predictions will be incomplete.",
                    fold_idx + 1, _alt_exc, exc_info=True,
                )

        # Write per-fold Stage 2 context-aware predictions to disk for inspection.
        if _use_fusion_stage2 and _fold_stage2 is not None:
            from ..stage2.stage2_context import build_fusion_matrix as _bfm_save
            from ..stage2.fit_stage2_fusion import apply_stage2_fusion as _apply_s2_save, project_stage2_embeddings as _proj_save
            try:
                _fusion_strat_save = cfg.get("fusion_strategy", "2d")
                _r_save = int(cfg.get("lowrank_bilinear_r", 8))
                _X_save, _ctx_save = _proj_save(_fold_stage2, X_test, context_vectors)
                _Z_full = _bfm_save(
                    _X_save, patient_ids_test, _ctx_save,
                    fusion_strategy=_fusion_strat_save, r=_r_save,
                )
                _fold_ca_preds = _apply_s2_save(_fold_stage2, _Z_full, y_cf=_y_cf_test_fold)
                _fold_ca_path = paths.ensemble_root / f"fold_{fold_idx + 1}_stage2_preds_ca.npy"
                np.save(str(_fold_ca_path), _fold_ca_preds)
                np.save(str(paths.ensemble_root / f"fold_{fold_idx + 1}_stage2_preds_cf.npy"),
                        _y_cf_test_fold)
            except Exception as _save_exc:
                logger.warning(
                    "Outer fold %d — failed to write per-fold Stage 2 predictions: %s",
                    fold_idx + 1, _save_exc,
                )

        # Submit CPU postprocessing asynchronously.
        # This runs fit_calibrators + apply_calibrators + threshold_tuning +
        # compute_metrics_df in a background thread while the next fold's
        # inner HP search occupies the GPU.
        _pending_post = _post_executor.submit(
            _postprocess_fold,
            fold_idx, val_ix, raw_val_probs, Y_val, class_list, cfg,
            calibrators=fold_calibrators,
            best_hp=best_hp,
            patient_ids_val=patient_ids_train[val_ix],
        )

    # Collect the last fold's postprocessing result and shut down the executor.
    if _pending_post is not None:
        _apply_fold_result(
            _pending_post.result(),
            ensemble_artifacts, oof_preds,
            paths=paths,
            use_inner_calibrators=True,  # Always prefer inner calibrators
        )
    _post_executor.shutdown(wait=True)

    # GPU monitor: stop polling, write JSON summary and optional chart
    _gpu_monitor.stop()
    _write_gpu_artifacts(_gpu_monitor, paths)

    # Fusion Stage 2 is applied at inference time (inside predict_proba), not here.
    # Report per-fold Stage 2 coverage so callers know whether predict_proba will
    # produce a pure Stage 2 average or a mixed Stage 1 / Stage 2 average.
    if _use_fusion_stage2:
        _s2_list = ensemble_artifacts["stage2"]
        _s2_fitted = sum(1 for m in _s2_list if m is not None)
        _s2_total  = len(_s2_list)
        if _s2_fitted == _s2_total:
            logger.info(
                "Stage 2 fusion: all %d folds fitted — predict_proba will "
                "produce a pure Stage 2 ensemble average.",
                _s2_total,
            )
        elif _s2_fitted == 0:
            logger.warning(
                "Stage 2 fusion: 0 of %d folds fitted — predict_proba will "
                "fall back entirely to Stage 1 for all folds.",
                _s2_total,
            )
        else:
            logger.warning(
                "Stage 2 fusion: only %d of %d folds fitted — predict_proba "
                "will average a mix of Stage 2 (folds with a model) and "
                "Stage 1 (folds without one). The resulting test_probs is "
                "not a pure Stage 2 estimate; treat it as a degraded hybrid.",
                _s2_fitted, _s2_total,
            )
    oof_preds_final = oof_preds

    # ── Write fold alignment manifest ────────────────────────────────────────
    # Ties fold IDs → held-out patient IDs → val_ix row indices so any downstream
    # stage can verify alignment without relying on in-memory state.
    try:
        _fold_manifest: dict = {
            "schema_version": "pdm_fold_manifest_v1",
            "n_folds": len(outer_splits),
            "n_classes": len(class_list),
            "class_list": class_list,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "cv": {"outer": outer_cv.to_dict(), "inner": inner_cv.to_dict()},
            "folds": [
                {
                    "fold_id": _fi + 1,
                    "fold_idx": _fi,
                    # The single held-out patient; None when the fold holds out several.
                    "held_out_patient_id": (
                        _outer_held_out[_fi][0] if len(_outer_held_out[_fi]) == 1 else None
                    ),
                    "held_out_patient_ids": _outer_held_out[_fi],
                    "val_ix": outer_splits[_fi][1].tolist(),
                    "n_val": int(len(outer_splits[_fi][1])),
                    "n_train_fold": int(len(outer_splits[_fi][0])),
                    "resumed_from_checkpoint": _fi in _completed_folds,
                    "stage2_fitted": ensemble_artifacts["stage2"][_fi] is not None
                        if _fi < len(ensemble_artifacts["stage2"]) else False,
                }
                for _fi in range(len(outer_splits))
            ],
            "test_patient_ids": [int(float(p)) for p in patient_ids_test],
            "patient_to_fold": _patient_to_fold,
        }
        with open(paths.fold_manifest_json, "w", encoding="utf-8") as _mf:
            _json_mod.dump(_fold_manifest, _mf, indent=2)
        logger.info("Fold alignment manifest written to %s", paths.fold_manifest_json)
    except Exception as _manifest_exc:
        logger.warning("Failed to write fold alignment manifest: %s", _manifest_exc)

    unfilled = (oof_preds_final == -1.0).all(axis=1)
    if unfilled.any():
        raise RuntimeError(
            f"{unfilled.sum()} of {len(unfilled)} training rows have unfilled OOF predictions "
            f"(sentinel -1.0 intact). A fold likely crashed before writing its results."
        )

    # Save and Report
    model_arch = {
        "hidden_dims": list(cfg.get("hidden_dims", [])),
        "dropout":     float(cfg.get("dropout", 0.3)),
        "activation":  str(cfg.get("activation", "gelu")),
    }

    ens_model = EnsemblePredictor(
        ensemble_artifacts['models'],
        ensemble_artifacts['preps'],
        ensemble_artifacts['calibs'],
        ensemble_artifacts['thresh'],
        device=str(DEVICE),
        class_list=class_list,
        model_arch=model_arch,
        stage2=ensemble_artifacts["stage2"],
        stage2_context_vectors=context_vectors,
    )
    ens_model.save_bundle(str(paths.bundle_path), cfg_snapshot=cfg, metadata={"class_list": class_list})
    reproducibility_cfg = cfg.get("REPRODUCIBILITY", {}) or {}
    artifact_toggles = reproducibility_cfg.get("artifacts", {}) or {}
    if bool(artifact_toggles.get("ensemble_manifest_json", True)):
        paths.reproducibility_dir.mkdir(parents=True, exist_ok=True)
        write_ensemble_manifest(
            paths.reproducibility_dir / "ensemble_manifest.json",
            {
                "bundle_format": "pdm_ensemble_v1",
                "bundle_path": str(paths.bundle_path.name),
                "class_list": class_list,
            },
        )
    # avg_thresh: mean of per-fold thresholds derived from each fold's inner-CV OOF.
    # Used as a secondary robustness check only — not the primary reported threshold.
    # ensemble_artifacts['thresh'] (outer fold calibrated thresholds) is retained only
    # for per-fold diagnostic reporting in CV_folds.csv.
    avg_thresh = np.mean(np.array(ensemble_artifacts['thresh_inner']), axis=0)

    # Resolve final thresholds for metrics
    _tau_thresh, threshold_meta = resolve_thresholds(cfg, class_list, tuned_thresholds=avg_thresh, logger=logger)
    # Log per-class thresholds and meta
    logger.info(f"Thresholding mode: {threshold_meta}")
    for i, cname in enumerate(class_list):
        logger.info(f"Class '{cname}': threshold={_tau_thresh[i]:.4f}")

    # ── Stage 1 OOF metrics (train_stage1_oof_vs_survey_metrics.csv) ─────────
    # OOF predictions are patient-held-out estimates against survey labels (Y_train).
    # This is the primary Stage 1 evaluation — a clean, unbiased estimate.
    logger.warning(
        "[METRIC][LABEL SOURCE] Stage 1 OOF metrics use SURVEY labels (Y_train) as ground truth. "
        "These are patient-grouped out-of-fold predictions — a clean held-out estimate within the survey label space."
    )
    _n_uninit = int(np.sum(np.all(oof_preds_final == -1.0, axis=1)))
    if _n_uninit > 0:
        logger.warning(
            "[METRIC][SANITY] %d rows in oof_preds are still at the sentinel value (-1). "
            "Check for missing or crashed folds.", _n_uninit,
        )
    _s1_oof_path = paths.ensemble_root / "train_stage1_oof_vs_survey_metrics.csv"
    _oof_source = f"oof_{outer_cv.scheme}"
    train_metrics_df = compute_metrics_df(oof_preds_final, Y_train, _tau_thresh, class_list, cfg, patient_ids_train)
    write_metrics_csv(train_metrics_df, _s1_oof_path)
    _write_metric_metadata(
        str(_s1_oof_path) + ".meta.json",
        stage="stage1", prediction_source=_oof_source,
        target_label_source="survey",
        is_clean_heldout_estimate=True, is_diagnostic=False,
        thresholding_mode=str(threshold_meta),
        thresholds=_tau_thresh.tolist(),
        n_patients=int(len(np.unique(patient_ids_train))),
        n_folds=int(len(outer_splits)),
    )
    _log_metric_summary(logger, _s1_oof_path.name, _oof_source, "survey",
                        str(threshold_meta), train_metrics_df, is_primary=True)

    write_ensemble_figures(compute_ensemble_curves(oof_preds_final, Y_train, class_list, cfg), str(paths.dir_figures_train))

    if patient_ids_test is None:
        raise ValueError("patient_ids_test required for test-set bootstrap CI")
    test_probs = ens_model.predict_proba(X_test, patient_ids=patient_ids_test)

    # ── Stage 1 test metrics (test_stage1_vs_survey_metrics.csv) ─────────────
    # Averaged context-free predictions across all outer folds vs survey labels.
    # NOTE: transductive — each test item is predicted by ALL folds then averaged.
    # This is a DIAGNOSTIC file, not a clean held-out estimate.
    # Build ordered list here so it can be used in the log message and averaging below.
    _cf_folds_ordered = [_test_probs_cf_folds[i] for i in sorted(_test_probs_cf_folds)
                         if _test_probs_cf_folds[i] is not None]
    logger.warning(
        "[METRIC][LABEL SOURCE] Stage 1 test metrics use SURVEY labels (Y_test) as ground truth. "
        "Predictions are averaged CF across %d folds (transductive — NOT a clean held-out estimate).",
        len(_cf_folds_ordered),
    )
    _s1_test_path = paths.ensemble_root / "test_stage1_vs_survey_metrics.csv"
    # Use averaged per-fold CF predictions rather than ensemble predict_proba,
    # which may include Stage 2 when enabled.
    _test_probs_cf_stage1: np.ndarray | None = (
        np.mean(np.stack(_cf_folds_ordered, axis=0), axis=0).astype(np.float32)
        if _cf_folds_ordered else None
    )
    _s1_test_preds = _test_probs_cf_stage1 if _test_probs_cf_stage1 is not None else test_probs
    test_metrics_df = compute_metrics_df(_s1_test_preds, Y_test, _tau_thresh, class_list, cfg, patient_ids_test)
    write_metrics_csv(test_metrics_df, _s1_test_path)
    _write_metric_metadata(
        str(_s1_test_path) + ".meta.json",
        stage="stage1", prediction_source="transductive_avg_cf",
        target_label_source="survey",
        is_clean_heldout_estimate=False, is_diagnostic=True,
        thresholding_mode=str(threshold_meta),
        thresholds=_tau_thresh.tolist(),
        n_patients=int(len(np.unique(patient_ids_test))),
        n_folds=int(len(_cf_folds_ordered)),
    )
    _log_metric_summary(logger, _s1_test_path.name, "transductive_avg_cf", "survey",
                        str(threshold_meta), test_metrics_df, is_primary=False)
    logger.warning(
        "[METRIC][SANITY] test_stage1_vs_survey_metrics.csv uses transductive-averaged CF predictions. "
        "Do NOT use for unbiased test evaluation — use train_stage1_oof_vs_survey_metrics.csv instead."
    )
    logger.warning(
        "[METRIC][SANITY] Do NOT directly compare Stage 1 OOF (survey labels) "
        "vs Stage 2 test (interview labels) metrics — they use different target label sources."
    )

    write_ensemble_figures(compute_ensemble_curves(test_probs, Y_test, class_list, cfg), str(paths.dir_figures_test))

    # Write metadata sidecar for the incremental CV_folds.csv produced during the fold loop.
    _write_metric_metadata(
        str(paths.cv_folds_csv) + ".meta.json",
        stage="stage1", prediction_source=f"{_oof_source}_per_fold",
        target_label_source="survey",
        is_clean_heldout_estimate=True, is_diagnostic=False,
        thresholding_mode=str(threshold_meta),
        thresholds=_tau_thresh.tolist(),
        n_patients=int(len(np.unique(patient_ids_train))),
        n_folds=int(len(outer_splits)),
        note="Per-fold metrics appended incrementally; each row is the held-out fold for that patient.",
    )

    # ── Stage 2 fold-pure and transductive test metrics ───────────────────────
    # Only written when Stage 2 is enabled (interview/survey test labels available).
    #
    # Additionally, always assemble fold-pure context-free (CF) predictions for H1 analysis.

    # --- Fold-pure context-free (CF) predictions for H1 ---
    # For each test patient, use the CF prediction from the fold where that patient was held out.
    # Use Y_test_survey for shape when available; fall back to Y_test (same n_test × n_classes shape).
    _cf_fp_shape_ref = Y_test_survey if Y_test_survey is not None else Y_test
    _cf_fold_pure = np.full_like(_cf_fp_shape_ref, np.nan)
    _cf_fold_pure_fold_ids = np.full(_cf_fp_shape_ref.shape[0], -1, dtype=int)
    _failed_cf_fold_pure = 0
    for _pi, _pid in enumerate(patient_ids_test):
        _fi = _patient_to_fold.get(int(float(_pid)), None)
        if _fi is not None:
            _cf_pred = _test_probs_cf_folds.get(_fi)
            if _cf_pred is not None and _pi < _cf_pred.shape[0]:
                _cf_fold_pure[_pi] = _cf_pred[_pi]
                _cf_fold_pure_fold_ids[_pi] = _fi
            else:
                _failed_cf_fold_pure += 1
        else:
            _failed_cf_fold_pure += 1

    # Existing Stage 2 fold-pure logic follows
    if _use_fusion_stage2:
        _s2_meta_base = dict(
            thresholding_mode=str(threshold_meta),
            thresholds=_tau_thresh.tolist(),
            n_patients=int(len(np.unique(patient_ids_test))),
            n_folds=int(len(outer_splits)),
        )
        # Fold-pure predictions: each test patient is predicted by the fold that excluded them.
        _s2_fold_pure = np.full_like(Y_test_survey, np.nan)
        _s2_fold_pure_fold_ids = np.full(Y_test_survey.shape[0], -1, dtype=int)
        _failed_fold_pure = 0
        for _pi, _pid in enumerate(patient_ids_test):
            _fi = _patient_to_fold.get(int(float(_pid)), None)
            if _fi is not None:
                _s2m = ensemble_artifacts["stage2"][_fi]
                if _s2m is not None:
                    from ..stage2.stage2_context import build_fusion_matrix as _bfm_fp
                    from ..stage2.fit_stage2_fusion import (
                        apply_stage2_fusion as _apply_fp,
                        project_stage2_embeddings as _proj_fp,
                    )
                    _fs = cfg.get("fusion_strategy", "2d")
                    _r = int(cfg.get("lowrank_bilinear_r", 8))
                    _X_pi, _ctx_pi = _proj_fp(_s2m, X_test[_pi:_pi+1], context_vectors)
                    _Z_pi = _bfm_fp(_X_pi, [_pid], _ctx_pi, fusion_strategy=_fs, r=_r)
                    _ycf_pi = ensemble_artifacts["ycf_test_probs"][_fi][_pi:_pi+1]
                    _s2_fold_pure[_pi:_pi+1] = _apply_fp(_s2m, _Z_pi, y_cf=_ycf_pi)
                    _s2_fold_pure_fold_ids[_pi] = _fi
                else:
                    _failed_fold_pure += 1
            else:
                _failed_fold_pure += 1
        _s2_meta_base["fold_pure_verification_failed"] = bool(_failed_fold_pure)

        # Shuffled-context control (context_specificity): leave-pair-out heads.
        _context_swap = _leave_pair_out_context_swap(
            X_test, Y_test_interview, Y_test_survey, patient_ids_test,
            context_vectors, _cf_fold_pure, class_list, cfg,
        )
        if _context_swap is not None:
            try:
                pd.DataFrame({
                    "row": np.arange(len(patient_ids_test)),
                    "patient_id": [int(float(p)) for p in patient_ids_test],
                    "shuffled_patient_id": _context_swap["donors"],
                }).to_csv(paths.ensemble_root / "test_shuffled_context_map.csv", index=False)
            except Exception as _map_exc:
                logger.warning("Failed to write shuffled-context map: %s", _map_exc)

        # Transductive/averaged: average across all Stage 2 models (diagnostic only).
        _n_s2_models = sum(1 for m in ensemble_artifacts["stage2"] if m is not None)
        _s2_transductive: np.ndarray | None = None
        if _n_s2_models > 0:
            from ..stage2.stage2_context import build_fusion_matrix as _bfm_tr
            from ..stage2.fit_stage2_fusion import (
                apply_stage2_fusion as _apply_tr,
                project_stage2_embeddings as _proj_tr,
            )
            _fs = cfg.get("fusion_strategy", "2d")
            _r = int(cfg.get("lowrank_bilinear_r", 8))
            _cf_mean = np.mean(np.stack(ensemble_artifacts["ycf_test_probs"]), axis=0)
            _all_s2 = []
            for _s2m in ensemble_artifacts["stage2"]:
                if _s2m is not None:
                    _X_all, _ctx_all = _proj_tr(_s2m, X_test, context_vectors)
                    _Z_all = _bfm_tr(_X_all, patient_ids_test, _ctx_all, fusion_strategy=_fs, r=_r)
                    _all_s2.append(_apply_tr(_s2m, _Z_all, y_cf=_cf_mean))
            if _all_s2:
                _s2_transductive = np.mean(np.stack(_all_s2, axis=0), axis=0)

        # Write fold-pure Stage 2 metrics (primary for Stage 2).
        _fp_interview_path = paths.ensemble_root / "test_stage2_fold_pure_vs_interview_metrics.csv"
        _fp_survey_path    = paths.ensemble_root / "test_stage2_fold_pure_vs_survey_metrics.csv"

        _fp_vs_interview = compute_metrics_df(_s2_fold_pure, Y_test_interview, _tau_thresh, class_list, cfg, patient_ids_test)
        write_metrics_csv(_fp_vs_interview, _fp_interview_path)
        _fp_vs_survey = compute_metrics_df(_s2_fold_pure, Y_test_survey, _tau_thresh, class_list, cfg, patient_ids_test)
        write_metrics_csv(_fp_vs_survey, _fp_survey_path)
        _write_metric_metadata(str(_fp_interview_path) + ".meta.json",
            stage="stage2", prediction_source="fold_pure", target_label_source="interview",
            is_clean_heldout_estimate=True, is_diagnostic=False, **_s2_meta_base)
        _write_metric_metadata(str(_fp_survey_path) + ".meta.json",
            stage="stage2", prediction_source="fold_pure", target_label_source="survey",
            is_clean_heldout_estimate=False, is_diagnostic=True, **_s2_meta_base)
        _log_metric_summary(logger, _fp_interview_path.name, "fold_pure", "interview",
                            str(threshold_meta), _fp_vs_interview, is_primary=True)
        _log_metric_summary(logger, _fp_survey_path.name, "fold_pure", "survey",
                            str(threshold_meta), _fp_vs_survey, is_primary=False)

        # Write transductive Stage 2 metrics (diagnostic).
        if _s2_transductive is not None:
            _tr_interview_path = paths.ensemble_root / "test_stage2_transductive_avg_vs_interview_metrics.csv"
            _tr_survey_path    = paths.ensemble_root / "test_stage2_transductive_avg_vs_survey_metrics.csv"
            _tr_vs_interview = compute_metrics_df(_s2_transductive, Y_test_interview, _tau_thresh, class_list, cfg, patient_ids_test)
            write_metrics_csv(_tr_vs_interview, _tr_interview_path)
            _tr_vs_survey = compute_metrics_df(_s2_transductive, Y_test_survey, _tau_thresh, class_list, cfg, patient_ids_test)
            write_metrics_csv(_tr_vs_survey, _tr_survey_path)
            _write_metric_metadata(str(_tr_interview_path) + ".meta.json",
                stage="stage2", prediction_source="transductive_avg", target_label_source="interview",
                is_clean_heldout_estimate=False, is_diagnostic=True, **_s2_meta_base)
            _write_metric_metadata(str(_tr_survey_path) + ".meta.json",
                stage="stage2", prediction_source="transductive_avg", target_label_source="survey",
                is_clean_heldout_estimate=False, is_diagnostic=True, **_s2_meta_base)
            _log_metric_summary(logger, _tr_interview_path.name, "transductive_avg", "interview",
                                str(threshold_meta), _tr_vs_interview, is_primary=False)
            _log_metric_summary(logger, _tr_survey_path.name, "transductive_avg", "survey",
                                str(threshold_meta), _tr_vs_survey, is_primary=False)

        # Stage 2 sanity warnings.
        if _failed_fold_pure > 0:
            logger.warning("[METRIC][SANITY] %d rows could not be predicted fold-pure — not clean held-out estimates.", _failed_fold_pure)
        if _n_s2_models > 0:
            logger.warning("[METRIC][SANITY] Transductive/averaged Stage 2 predictions are DIAGNOSTIC only — not held-out.")
        logger.warning("[METRIC][SANITY] Stage 2 vs SURVEY labels is DIAGNOSTIC only. Interview labels are the primary Stage 2 target.")

    macro_train = _compute_macro_metrics(
        y_true=Y_train,
        y_prob=oof_preds_final,
        thresholds=_tau_thresh,
    )
    macro_test = _compute_macro_metrics(
        y_true=Y_test,
        y_prob=test_probs,
        thresholds=_tau_thresh,
    )

    logging.info(
        f"Pipeline complete — total elapsed: "
        f"{time.perf_counter() - _pipeline_t0:.1f}s."
    )

    # Build arch_predictions dict for statistical comparison.
    # Uses fold-pure predictions: for each row, the model from the fold that
    # excluded that row's patient is used — no fold head averaging.
    arch_predictions: dict | None = None
    if _use_fusion_stage2:
        arch_predictions = {}
        # Main configured strategy: use the already-computed fold-pure Stage 2 predictions.
        _main_arch_label = cfg.get("fusion_strategy", "2d")
        if _s2_fold_pure is not None and not np.all(np.isnan(_s2_fold_pure)):
            arch_predictions[_main_arch_label] = _s2_fold_pure.astype(np.float32)
        else:
            arch_predictions[_main_arch_label] = test_probs  # fallback: ensemble-averaged
        # passthrough / stage1_only: fold-pure Stage 1 (context-free) predictions.
        _cf_arch_source = (
            _cf_fold_pure
            if not np.all(np.isnan(_cf_fold_pure))
            else _test_probs_cf_stage1
        )
        if _cf_arch_source is not None:
            arch_predictions["passthrough"] = np.asarray(_cf_arch_source, dtype=np.float32)
            arch_predictions["stage1_only"] = np.asarray(_cf_arch_source, dtype=np.float32)
        # Alternative fusion architectures: assemble fold-pure row by row.
        if _arch_fold_preds and _patient_to_fold:
            _fp_shape = (
                Y_test_interview if Y_test_interview is not None else Y_test
            ).shape
            for arch_name, fold_preds_map in _arch_fold_preds.items():
                _fp_arr = np.full(_fp_shape, np.nan, dtype=np.float32)
                for _pi, _pid in enumerate(patient_ids_test):
                    _fi = _patient_to_fold.get(int(float(_pid)), None)
                    if _fi is not None and _fi in fold_preds_map:
                        _fp_arr[_pi] = fold_preds_map[_fi][_pi]
                arch_predictions[arch_name] = _fp_arr
        logger.info(
            "arch_predictions assembled fold-pure for statistical comparison: %s",
            sorted(arch_predictions.keys()),
        )

    # Ensure _s2_fold_pure and _s2_transductive are always defined
    if '_s2_fold_pure' not in locals():
        _s2_fold_pure = None
    if '_s2_transductive' not in locals():
        _s2_transductive = None
    if '_context_swap' not in locals() or _context_swap is None:
        _context_swap = {"correct": None, "shuffled": None, "donors": None}
    result: dict = {
        "train": macro_train,
        "test": macro_test,
        "oof_probs_cf": oof_preds_final,  # shape (n_train, n_classes) — OOF CF probabilities
        "test_probs_cf": _test_probs_cf_stage1,        # Stage 1 only, averaged across folds (diagnostic)
        "test_probs_cf_fold_pure": _cf_fold_pure,      # Stage 1, fold-pure: each row from its held-out fold
        "test_probs_ca": test_probs,                   # Stage 2 context-aware (or Stage 1 if no Stage 2)
        "test_probs_ca_fold_pure": _s2_fold_pure,      # Clean fold-pure Stage 2 predictions
        "test_probs_ca_transductive": _s2_transductive, # Diagnostic transductive Stage 2 predictions
        # Shuffled-context control: leave-pair-out heads, own and partner's context.
        "test_probs_ca_leave_pair_out": _context_swap["correct"],
        "test_probs_shuffled_leave_pair_out": _context_swap["shuffled"],
        "shuffled_context_donors": _context_swap["donors"],  # (n_test,) partner patient, -1 if not scored
        "avg_thresh_f1opt": avg_thresh,                # (n_classes,) mean inner-fold F1-optimal thresholds
        "arch_predictions": arch_predictions,          # {arch_name: (n_test, n_classes)} or None
        "cv": _cv_record(
            outer_cv, inner_cv, outer_splits, _outer_held_out,
            patient_ids_train, _inner_splits_by_fold,
        ),
    }
    if oof_strata is not None:
        result["oof_strata"] = oof_strata
    return result
