# LM-ContextProbe Training Orchestrator

**Canonical path:** `tracks/representation/training/orchestrator`

This directory acts as the control center for the LM-ContextProbe representation model training pipeline. It manages the end-to-end nested Leave-One-Patient-Out cross-validation (LOPO-CV) process, seamlessly integrating hyperparameter search, model training, calibration, threshold tuning, metric computation, figure generation, GPU monitoring, and crash-recovery checkpointing.

For the high-level pipeline architecture and how this submodule connects to Stage 1, Stage 2, and the shared utilities, see the parent [training README](../README.md).

## File Index

| File                          | Purpose                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| `train_ensemble_pipeline.py`  | LOPO-CV training pipeline — sole public entry point of the entire training package   |
| `compute_metrics_and_save.py` | Per-class and aggregate metric DataFrame builder + CSV writer                        |
| `fold_postprocessing.py`      | Async fold postprocessing, checkpoint validation, and resume detection               |
| `gpu_monitor.py`              | Background `nvidia-smi` polling thread, summary JSON, two-panel chart                |
| `plot_ensemble_figures.py`    | ROC and PR curve generation (per-class and micro-averaged)                           |

## `train_ensemble_pipeline.py`

The primary entry point for training. The single function `train_ensemble_pipeline(...)` drives the full nested LOPO-CV flow.

### Signature (abridged)

```python
def train_ensemble_pipeline(
    X_train, Y_train, patient_ids_train,
    X_test, Y_test, cfg,
    *,
    item_strings_train=None, item_strings_test=None,
    unresolved_mask_train=None, unresolved_mask_test=None,
    embedding_cache=None,
    resume_from_checkpoint=False,
    Y_test_interview=None, Y_test_survey=None,
    patient_ids_test=None,
    context_vectors=None,
) -> dict
```

`patient_ids_test` is always required (used by `EnsemblePredictor.predict_proba` and fold-pure CF tracking). `Y_test_interview`, `Y_test_survey`, and `context_vectors` are required only when Stage 2 fusion is active.

### High-level flow

1. **Setup**
   - Coerce all inputs to `float32` numpy arrays via `shared.utils.array_utils.as_float32_array`.
   - Resolve `class_list` from one of `cfg["default_classes"]`, `cfg["classes"]`, or `cfg["class_list"]`, with a width-mismatch fallback to generic `class_i` names (logged as a warning).
   - Call `set_seeds(cfg['global_seed'], deterministic_algorithms=True)` once at the top.
   - Build `TrainingPathBundle` (a frozen dataclass) with every output path the run will need.
2. **Resume detection** (if `resume_from_checkpoint=True`)
   - Refuse to resume when `ensemble_bundle.joblib` already exists (the run was complete).
   - Scan `checkpoints_dir` via `detect_completed_folds`, validate every checkpoint against the current `n_classes` and `outer_splits`, discard mismatches, and replay accumulator state from the survivors.
3. **Pre-fold fuzzy fallback**
   - When `embedding_cache`, `item_strings_*`, and `cfg["fuzzy_threshold"]` are all present, build a fuzzy matcher from the full training set, standardize held-out test item strings, and re-assemble `X_test` from the embedding cache. This applies the post-fuzzy standardization without re-running the embedding model.
4. **GPU monitor + async postprocessing executor**
   - Start a `GpuMonitor` background thread.
   - Open a single-worker `ThreadPoolExecutor` for fold postprocessing (calibration, threshold tuning, metric DataFrame, checkpoint write) so that I/O happens during the next fold's HP search rather than starving the GPU.
5. **Per-fold loop** — for each `(train_ix, val_ix)` from `patient_splits(patient_ids_train, cv.outer)` (`shared/cv/`). The inner splits for every fold are built before the loop, so an infeasible `cv.inner` fails before any training:
   - Skip if the fold was loaded from a checkpoint.
   - Apply fold-local fuzzy fallback (matcher built from the fold's training strings only — held-out validation strings never enter the candidate space).
   - Drain the previous fold's pending postprocessing future before the GPU work begins.
   - **Inner HP search**: parallel over `cfg["n_hp_workers"]` (default 4) threads, each evaluating one HP candidate across all inner folds (`cv.inner`); selection criterion is mean inner Brier (lower is better). When `<run_dir>/fold_{i+1}_hp_search.csv` already exists (crash-recovery scenario), the best HP is loaded from CSV and Stage 2 Ridge is skipped for that fold (no inner OOF available).
   - Derive per-class thresholds and isotonic calibrators from the winning candidate's inner OOF — these are leak-free because inner OOF never touches the outer fold's held-out patients.
   - Compute per-class positive weights (`class_weight_mode`: `"cui"` (default, Cui 2019 Effective Number of Samples) or `"inverse_frequency"`) capped at `weight_cap`.
   - Final retrain on the outer-train fold via `stage1.train_single_model`.
   - Compute Stage 1 context-free predictions (`ŷ_cf`) on test items, calibrated with the inner-OOF calibrators.
   - **If Stage 2 active**:
     - Build the fusion matrix `Z` via `stage2.stage2_context.build_fusion_matrix`.
     - Optional `cfg["stage2_use_cf_passthrough"]`: substitute fold-pure cross-fitted ŷ_cf for inner-LOPO val patients (using the patient's own outer fold's predictions) instead of the transductive ŷ_cf, avoiding optimism bias.
     - Optional `cfg["stage2_pca_n_components"]`: fit a Stage 2-internal PCA on item embeddings and project context vectors through the same.
     - Call `stage2.fit_stage2_fusion.fit_stage2_fusion_fold` to perform the inner-α search and final retraining; if the passthrough sentinel is returned, no learned head is used for that fold.
     - Train alternative Stage 2 architectures (`2d`, `3d`, `lowrank_bilinear`, `patient_id`) under their own independent inner LOPO α searches via `fit_all_stage2_architectures_fold` for architecture comparison.
     - Save per-fold ŷ_cf and ŷ_ca npy files for inspection.
   - Submit the postprocessing future (`_postprocess_fold` → `_apply_fold_result` → `write_fold_artifacts`).
6. **Pipeline finalization**
   - Drain the last pending postprocessing future and shut down the executor.
   - Stop the GPU monitor and write its JSON summary + chart.
   - Write the fold alignment manifest (`fold_manifest.json`) tying fold ID → held-out patient ID → val_ix → Stage 2 fitted boolean.
   - Verify no row in `oof_preds` still holds the sentinel `-1.0`; raise if any fold crashed before writing its results.
7. **Final reporting**
   - Build the `EnsemblePredictor` bundle and persist it to `ensemble_bundle.joblib` together with a reproducibility manifest.
   - Average the per-fold inner-OOF thresholds into `avg_thresh`; resolve final reporting thresholds via `shared.resolve_thresholds`.
   - Compute primary Stage 1 OOF-vs-survey metrics (`train_stage1_oof_vs_survey_metrics.csv` — clean held-out estimate).
   - Compute diagnostic Stage 1 test-vs-survey metrics (`test_stage1_vs_survey_metrics.csv`) using the averaged context-free per-fold predictions (transductive — explicitly logged as diagnostic).
   - Generate ROC/PR figures for both train OOF and test.
   - When Stage 2 is active, compute fold-pure and transductive Stage 2 metrics against both interview and survey labels (interview is the primary Stage 2 target; survey is diagnostic).
   - When Stage 2 is active, also score every test row fold-pure with a donor patient's context vector (shuffled-context control). Donors come from `stage2.stage2_context.draw_shuffled_context_donors`, seeded by `shuffled_context_seed`.
   - Assemble the `arch_predictions` dict (fold-pure across architectures) for downstream statistical comparison.

### Output artifacts

Inside `ensemble_root` (default `./output/model`):

| Artifact                                        | Contents                                                          |
| ----------------------------------------------- | ----------------------------------------------------------------- |
| `ensemble_bundle.joblib`                        | Serialized `EnsemblePredictor` with all fold heads / preprocessors / calibrators / thresholds / Stage 2 |
| `fold_manifest.json`                            | Schema-versioned alignment manifest                               |
| `CV_folds.csv`                                  | Per-fold per-class metrics (appended incrementally)               |
| `threshold_report.csv`                          | Per-fold threshold tuning diagnostics                             |
| `train_stage1_oof_vs_survey_metrics.csv`        | Primary Stage 1 metrics (LOPO OOF vs survey)                      |
| `test_stage1_vs_survey_metrics.csv`             | Diagnostic Stage 1 metrics (transductive avg vs survey)           |
| `test_stage2_fold_pure_vs_interview_metrics.csv`| Primary Stage 2 metrics (fold-pure vs interview)                  |
| `test_stage2_fold_pure_vs_survey_metrics.csv`   | Diagnostic Stage 2 metrics (fold-pure vs survey)                  |
| `test_stage2_transductive_avg_vs_*_metrics.csv` | Diagnostic transductive averaged Stage 2 metrics                  |
| `test_shuffled_context_map.csv`                 | Each test row's patient and the donor whose context it was given in the shuffled-context condition |
| `*.meta.json`                                   | Sidecar metadata for every metrics CSV (label source, prediction source, thresholding mode, n_patients, n_folds, ...) |
| `fold_{i+1}_hp_search.csv`                      | HP grid scan diagnostics (when `log_hp_search_grid=True`)         |
| `fold_{i+1}_epoch_curve.csv`                    | Per-epoch loss / val score / patience trace                       |
| `fold_{i+1}_stage2_preds_cf.npy`                | Per-fold Stage 1 context-free test predictions                    |
| `fold_{i+1}_stage2_preds_ca.npy`                | Per-fold Stage 2 context-aware test predictions                   |
| `checkpoints/fold_{i+1}.joblib`                 | `pdm_fold_v3` checkpoint dict (used for resume)                   |
| `figures/train/{ROC,PR}/`                       | OOF ROC/PR figures                                                |
| `figures/test/{ROC,PR}/`                        | Test ROC/PR figures                                               |
| `../gpu/gpu_utilization.json` and `.png`        | GPU monitor summary and chart                                     |

### Returned dict

`train_ensemble_pipeline` returns a dict with:

| Key                            | Shape / type                          | Description                                                       |
| ------------------------------ | ------------------------------------- | ----------------------------------------------------------------- |
| `train`                        | dict of float                         | Macro train metrics (Precision, Recall, MCC, F1, AUC ROC, AUC PR, AP) |
| `test`                         | dict of float                         | Macro test metrics                                                |
| `oof_probs_cf`                 | `(n_train, n_classes)`                | OOF context-free probabilities (LOPO held-out)                    |
| `test_probs_cf`                | `(n_test, n_classes)` or None         | Stage 1 averaged-across-folds CF test predictions (diagnostic)    |
| `test_probs_cf_fold_pure`      | `(n_test, n_classes)`                 | Stage 1 fold-pure CF predictions (each row from its held-out fold)|
| `test_probs_ca`                | `(n_test, n_classes)`                 | Stage 2 context-aware test predictions (or Stage 1 if no Stage 2) |
| `test_probs_ca_fold_pure`      | `(n_test, n_classes)` or None         | Clean fold-pure Stage 2 predictions                               |
| `test_probs_ca_transductive`   | `(n_test, n_classes)` or None         | Diagnostic transductive Stage 2 predictions                       |
| `test_probs_shuffled_fold_pure`| `(n_test, n_classes)` or None         | Fold-pure Stage 2 predictions with a donor patient's context      |
| `shuffled_context_donors`      | `(n_test,)` int or None               | Donor patient of each test row                                    |
| `avg_thresh_f1opt`             | `(n_classes,)`                        | Mean inner-fold thresholds                                        |
| `arch_predictions`             | `{arch_name: (n_test, n_classes)}` or None | Fold-pure predictions for every fusion architecture / baseline  |
| `oof_strata`                   | `(n_train,) of object` or absent      | Fold-local `"repeated"` / `"novel"` stratum labels (only when `item_strings_train` is provided) |

### Helper functions

- `_resolve_class_list(cfg)` — pick from `default_classes`, `classes`, or `class_list`.
- `_resolve_effective_class_list(cfg, target_dim)` — fall back to `active_classes` or generic names if widths mismatch.
- `_make_epoch_writer(path)` — return a callback that appends `epoch, loss, val_score, patience, is_best` to the supplied CSV; the header is written immediately at closure construction so the file exists even if epoch 0 crashes.
- `_resolve_ensemble_root(cfg, root)` — `cfg["DIR_MODEL"]` or default `output/model`.
- `_build_paths(cfg, root)` → `TrainingPathBundle`, a `@dataclass(frozen=True)` carrying every output path.
- `_setup_output_dirs(paths, *, resume)` — create directories; clear stale `CV_folds.csv` and `threshold_report.csv` when **not** resuming.
- `_write_gpu_artifacts(monitor, paths)` — write GPU JSON summary + chart.
- `_macro_auc_pr(y_true_bin, y_prob)`, `_safe_macro_roc_auc`, `_safe_macro_ap` — sklearn-based macro metrics that drop classes with constant targets.
- `_compute_macro_metrics(y_true, y_prob, thresholds)` — returns `{Precision, Recall, MCC, F1, AUC ROC, AUC PR, AP}` from the soft metric set.
- `_write_metric_metadata(meta_path, **kwargs)` — JSON sidecar writer for metric CSV provenance.
- `_log_metric_summary(...)` — emits a one-line `[METRIC][PRIMARY|DIAGNOSTIC] ...` summary including macro F1 / macro AUROC / macro AUPRC.

## `compute_metrics_and_save.py`

### `compute_metrics_df(probs, y_true, thresholds, class_list, cfg, patient_ids) -> pd.DataFrame`

Builds the canonical metric DataFrame. One row per class plus two summary rows: `"Macro Average"` and `"Micro Aggregate"`.

Columns: `Class`, `Precision`, `Recall`, `MCC`, `AUC ROC`, `AUC PR`, `AP`, `F1`, `TP`, `FP`, `TN`, `FN`, `Threshold`, `Valid Count`, `Prevalence`, `Pred Pos Rate`, `ECE`, `ECE CI Lower`, `ECE CI Upper`, `Brier Score`, `Brier CI Lower`, `Brier CI Upper`.

Per-class behaviour:
- Calls `shared.soft_label_utils.masked_class_data` to apply the soft positive-band mask.
- Logs `[THRESH]` info per class (max prob, prevalence, pred_pos_rate at the chosen threshold).
- Emits warnings on threshold/prediction pathologies: `max_prob < threshold` (all-negative predictions), `pred_pos_rate == 0` (all-negative), `pred_pos_rate > 0.5` (overprediction relative to prevalence), `threshold > 0.8 with prevalence < 0.1` (high threshold relative to prevalence).
- Computes ECE and Brier point estimates with patient-clustered bootstrap CIs (`cfg["ece_n_bins"]`, `cfg["ece_n_resamples"]`, defaults 15 and 1000).

The `Macro Average` row is computed as the column-wise mean across the per-class rows. `TP/FP/TN/FN/Threshold` are NaN on the macro row to avoid implying a meaningful aggregation.

The `Micro Aggregate` row is computed from confusion-count totals (sum of per-class soft TP/FP/TN/FN). AUC ROC, AUC PR, AP, ECE, and Brier are NaN on the micro row because they cannot be meaningfully aggregated from confusion counts alone.

After assembly, count columns (`TP`, `TN`, `FP`, `FN`, `Valid Count`) are cast to int (Macro Average row left as NaN for `TP/FP/TN/FN`); decimal columns are rounded to 4 places.

### `write_metrics_csv(df, path) -> None`
Writes the metrics DataFrame to a CSV file via `shared.utils.file_utils.write_csv_file`, creating parent directories as needed. Always overwrites (never appends).

## `fold_postprocessing.py`

Encapsulates async-safe per-fold work and the resume / checkpoint validation contract.

### Checkpoint schema

`CHECKPOINT_VERSION = "pdm_fold_v3"`. Required keys:

```
checkpoint_version, fold_idx, val_ix,
model_state, preprocessor, calibrators,
thresholds, pos_weights,
thresh_inner, oof_probs, oof_labels, best_hp
```

Plus optional keys: `stage2`, `ycf_test_probs`.

### Functions

#### `validate_fold_checkpoint(ckpt, n_classes) -> (is_valid, reason)`
Verifies dict-ness, version match, required keys, and that `oof_probs.shape == oof_labels.shape == (len(val_ix), n_classes)`.

#### `load_fold_checkpoint(path, n_classes) -> dict | None`
Loads + validates a checkpoint. Returns `None` (with a warning) on missing file, joblib decode failure, or schema mismatch — the orchestrator will re-train that fold.

#### `detect_completed_folds(checkpoints_dir, n_folds, n_classes) -> dict[int, dict]`
Scans the directory for valid checkpoints and returns the loaded dicts keyed by 0-based fold index. Used at resume time.

#### `_postprocess_fold(fold_idx, val_ix, raw_val_probs, Y_val, class_list, cfg, *, calibrators, best_hp, patient_ids_val) -> dict`
CPU-bound per-fold work that runs in a background thread:

1. Apply the inner-OOF calibrators to `raw_val_probs` (or fall back to identity per class when calibrators are unavailable, never to outer-val-fitted calibrators — that would leak the held-out patient).
2. Run `threshold_tuning(..., return_report=True)` on the calibrated probabilities.
3. Compute `compute_metrics_df` for the fold.

Returns a dict consumed by `_apply_fold_result` on the main thread.

#### `write_fold_artifacts(result, ensemble_artifacts, paths) -> None`
Main-thread I/O: appends the fold's metric row to `CV_folds.csv` and threshold report (skipping duplicate writes via `_fold_already_in_csv`), then atomically writes the per-fold checkpoint via `mkstemp` + `os.replace`.

#### `_apply_fold_result(result, ensemble_artifacts, oof_preds, *, paths, use_inner_calibrators=False) -> None`
Applies the postprocessing result on the main thread. Mutates accumulator lists (`calibs`, `thresh`) and `oof_preds[val_ix]`, then delegates artifact writes to `write_fold_artifacts`. Always called on the main thread to avoid concurrent list mutation.

## `gpu_monitor.py`

Background `nvidia-smi` polling thread for training diagnostics.

### `GpuMonitor`

Constructor parameters:

| Parameter                | Default | Effect                                                                |
| ------------------------ | ------- | --------------------------------------------------------------------- |
| `enabled`                | `True`  | Master switch. When `False`, all methods are no-ops.                  |
| `poll_interval`          | `1.0`   | Seconds between `nvidia-smi` polls.                                   |
| `active_util_threshold`  | `10`    | SM utilization (%) above which a sample counts toward `sm_util_mean_active_pct`. |
| `upgrade_sm_threshold`   | `70`    | Active-mean SM utilization (%) at or above which `upgrade_recommended` is `True`. |
| `save_chart_enabled`     | `True`  | When `False`, `save_chart()` skips PNG generation but still logs the verdict. |

Public API:

- `start()` — start a daemon polling thread. No-op (with a log message) if disabled, CUDA absent, or `nvidia-smi` not callable.
- `mark_fold(fold_num)` — record a fold boundary at the current wall-clock time; dashed lines are drawn for these on the chart.
- `stop()` — signal the thread to stop and join (5 s timeout).
- `summary() -> dict` — structured utilization summary.
- `save_summary(output_path)` — JSON dump of `summary()`.
- `save_chart(output_path)` — two-panel utilization chart (top: SM util with active + upgrade reference lines; bottom: memory used with total capacity and 85 % warning lines).

### Decision logic

The key decision metric is `sm_util_mean_active_pct`: mean SM utilization filtered to samples where the GPU was actually doing work (above `active_util_threshold_pct`). This filters out CPU-bound idle periods between epochs so the number reflects true GPU efficiency rather than pipeline structure.

`upgrade_recommended = True` when **either**:
- `sm_util_mean_active_pct >= upgrade_sm_threshold_pct`, **or**
- peak memory ≥ 85 % of total capacity.

Verdict tiers:

| `sm_util_mean_active_pct` | Verdict     |
| ------------------------- | ----------- |
| `>= upgrade_sm_threshold` | `"high"`    |
| `>= 50`                   | `"moderate"`|
| `< 50`                    | `"low"`     |

Each tier has a human-readable `upgrade_reason` explaining whether the bottleneck is GPU compute, GPU memory, or CPU-side work between epochs.

### Configuration keys consumed

All under `cfg["GPU_MONITOR"]`:

| Key                          | Default |
| ---------------------------- | ------- |
| `enabled`                    | `True`  |
| `poll_interval_s`            | `1.0`   |
| `active_util_threshold_pct`  | `10`    |
| `upgrade_sm_threshold_pct`   | `70`    |
| `save_chart`                 | `True`  |

### Failure modes

- CUDA absent / `nvidia-smi` not on PATH / `nvidia-smi` returns non-zero / `nvidia-smi` times out → monitor disables itself silently.
- `subprocess.run` exception inside the polling loop → that single sample is dropped; the loop never crashes training.

## `plot_ensemble_figures.py`

### `compute_ensemble_curves(probs, y_true, class_list, cfg) -> dict`

Pre-computes ROC and PR curve data with no filesystem I/O, returning a dict containing:

- `y_micro`, `p_micro` — concatenated masked arrays for micro-average curves.
- `per_class` — list of `{cls, y_c, probs_c}` for classes with non-empty masked data.
- `colors` — `tab10` colormap rows, one per class.
- `class_list` — copy of the input class list.

Classes with no valid samples after `masked_class_data` are silently dropped from the per-class list.

### `write_ensemble_figures(curves, figures_dir) -> None`

Writes:

- `figures_dir/ROC/ROC_Ensemble_Total.png` — micro-averaged ROC + per-class overlay.
- `figures_dir/PR/PR_Ensemble_Total.png` — micro-averaged PR + per-class overlay.
- `figures_dir/ROC/ROC_{cls}.png` and `figures_dir/PR/PR_{cls}.png` — one figure per class.

When the masked input is empty, a placeholder figure with a `"No valid ROC data"` / `"No valid PR data"` text label is written instead, so downstream report generators always have a file to embed.

All curves use `shared.soft_label_utils._soft_roc_curve` / `_soft_pr_curve` / `_soft_roc_auc` / `_soft_average_precision` so the soft physician labels are honoured throughout.
