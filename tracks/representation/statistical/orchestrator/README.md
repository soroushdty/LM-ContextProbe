# Orchestrator (`tracks/representation/statistical/orchestrator`)

Canonical location: `tracks/representation/statistical/orchestrator/`

This module is the sole entry point for the entire statistical analysis phase. It
sequences all hypothesis tests, diagnostic checks, sub-field ablation, architecture
comparison, and figure generation in a fixed, reproducible order and serializes every
artefact to disk.

---

## Entry point: `run_statistical_analysis`

```python
run_statistical_analysis(
    # Core arrays (one row per paired observation)
    y_survey: np.ndarray,               # (n_pairs, n_classes) pair-aggregated survey labels {0, 0.5, 1}
    y_interview: np.ndarray,            # (n_pairs, n_classes) pair-aggregated interview labels {0, 0.5, 1}
    y_hat_cf: np.ndarray,               # (n_pairs, n_classes) context-free predictions [0, 1]
    y_hat_ca: np.ndarray,               # (n_pairs, n_classes) context-aware predictions [0, 1]
    patient_ids: np.ndarray,            # (n_pairs,) patient IDs
    item_texts: np.ndarray,             # (n_pairs,) item text strings
    class_list: list[str],              # class names, length n_classes
    # Threshold info
    avg_thresh_f1opt: np.ndarray,       # (n_classes,) mean inner-fold F1-optimal thresholds
    tau_fixed: float,                   # Fixed threshold (cfg["tau"], typically 0.5)
    # Data source (for individual physician labels)
    dataset_path: Path,                 # Path to dataset.xlsx
    sheet_names: dict,                  # {"train": ..., "test": ..., "interview": ...}
    # Context and encoding (for ablation)
    context_json: dict | None,          # {patient_id_str → {sub_field → value}}
    llm: str,                           # Encoder model ID (for ablation re-encoding)
    cfg: dict,                          # Full pipeline config dict
    # Output
    output_dir: Path,                   # Root directory for all statistical outputs
    # Optional controls
    n_resamples: int = 1000,
    n_permutations: int = 10_000,
    strata: np.ndarray | None = None,   # (n_pairs,) pre-assigned 'repeated'/'novel'; computed if None
    arch_predictions: dict | None = None,  # {arch_name: (n_pairs, n_classes) preds} for arch comparison
    ensemble_bundle_path: Path | None = None,  # For sub-field ablation
    item_texts_train: np.ndarray | None = None, # Training item strings for strata assignment
    rng: np.random.Generator | None = None,
    X_items_test: np.ndarray | None = None,    # (n_pairs, d) item embeddings for ablation
    y_hat_shuffled: np.ndarray | None = None,  # (n_pairs, n_classes) Stage 2 with another patient's context
) -> None
```

The function returns nothing. All results are written to `output_dir`.

---

## Config keys consumed

The orchestrator reads the following keys from `cfg`:

| Key                                             | Source                 | Default             | Purpose                             |
| ----------------------------------------------- | ---------------------- | ------------------- | ----------------------------------- |
| `global_seed`                                   | `training_config.yaml` | `42`                | Seeds the numpy Generator           |
| `statistical_analysis.confirmatory_min_nonzero` | `training_config.yaml` | `15`                | Eligibility threshold               |
| `statistical_analysis.figure_dpi`               | `training_config.yaml` | `150`               | Figure output resolution            |
| `patient_col`                                   | `main_config.yaml`     | `"Patient"`         | Excel column for patient IDs        |
| `physician_col`                                 | `main_config.yaml`     | `"Physician"`       | Excel column for physician IDs      |
| `item_col`                                      | `main_config.yaml`     | `"Item"`            | Excel column for item text          |
| `DIR_JSON_MAP`                                  | `main_config.yaml`     | `data/mapping.json` | Item-text canonical mapping for ICC |

See [`configs/training_config.yaml`](../../../../configs/training_config.yaml) for the full
`statistical_analysis` block.

---

## Pipeline steps

### Step 1 — Core deltas and confirmatory filter

Computes `Δ_p = y_interview − y_survey` and `Δ_m = y_hat_ca − y_hat_cf` via
`shared.statistical.delta`. Partitions classes into confirmatory-eligible (≥ `min_nonzero` non-zero
`Δ_p` values) and descriptive-only. The two lists are passed to every subsequent step that
needs them.

### Step 2 — τ robustness check

Calls `shared.statistical.tau_check.tau_robustness_check(tau_fixed, avg_thresh_f1opt, class_list)`.
A one-line summary is written to the log. Classes with `|τ_fixed − τ_F1opt| > 0.05` are
flagged in the output CSV.

**Output:** `tau_robustness.csv`

### Step 3 — Load individual physician labels

Builds a raw-item-to-canonical-item standardizer from `DIR_JSON_MAP` using the same
`normalize_for_matching()` logic applied during preprocessing. This ensures that items
read from the raw Excel sheet will match keys in `item_texts_test`. Falls back to raw
strings with a warning if the mapping file is unavailable or the build fails.

Calls `shared.statistical.icc.load_individual_physician_labels()`. Used in Steps 9 and 14.

### Step 4 — Directional alignment per class (formerly H1)

Calls `tracks.representation.statistical.hypotheses.h1.h1_binomial_per_class`. The confirmatory p-value (`perm_p`) is a
within-patient permutation test of sign agreement against its chance level (`null_rate`): the class's model deltas are
shuffled among each patient's items. The sign-flip test of agreement > 0.5 (`cluster_p`) and the exact binomial p-value
(`binom_p`) are kept for reference only. BH FDR correction is applied to `perm_p` across confirmatory-eligible classes;
all classes are reported with their `confirmatory` flag.

**Output:** `directional_alignment_per_class.csv`

### Step 5 — Directional alignment pooled (permutation test)

Calls `tracks.representation.statistical.hypotheses.h1.h1_permutation_test` with `n_permutations = 10_000` (configurable
via `statistical_analysis.n_permutations`). Patient rows of `delta_m` are shuffled within
each patient block, pooled across confirmatory-eligible classes.

**Output:** `directional_alignment_pooled.json`

### Step 6 — Directional alignment across classes (CMH)

Calls `tracks.representation.statistical.hypotheses.h1.h1_cmh_test`. Builds one 2×2 table per confirmatory-eligible class
(stratified by physician-delta direction) and combines them via the Mantel-Haenszel
common-odds-ratio statistic. `p_permutation` is a within-patient permutation test of the common odds ratio > 1 (each
patient's model delta rows are shuffled among that patient's items). `p_cmh` treats cells as independent and `p_cluster`
tests pooled sign agreement > 0.5; both are kept for reference only.

**Output:** `directional_alignment_cmh.json`

### Step 6.5 — Context specificity (correct vs shuffled context)

Runs if `y_hat_shuffled` is provided (the pipeline's `test_probs_shuffled_fold_pure`). Calls
`tracks.representation.statistical.hypotheses.context_specificity`, which uses the same statistic as Track 3
(`shared.evaluation.hypothesis_analyzer.context_specificity_test`): on cells with a non-zero physician delta, the mean
of `sign(Δ_p)·(Δ_m − Δ_shuffled)`, with a patient-cluster bootstrap CI and a one-sided patient-cluster sign-flip
test. The pooled result over confirmatory-eligible classes is confirmatory; the per-class p-values get BH correction
across eligible classes. Rows with a missing prediction are left out. Uses its own generator (`global_seed + 3`), so
the other outputs are unchanged.

**Outputs:** `context_specificity_pooled.json`, `context_specificity_per_class.csv`

### Step 7 — Brier improvement per class and macro summary (formerly H2)

Calls `tracks.representation.statistical.hypotheses.h2.h2_wilcoxon_per_class` then `h2_macro_summary`. The confirmatory
p-value (`cluster_p`) is a one-sided patient-cluster sign-flip test of whether context-aware Brier scores are lower; the
Wilcoxon signed-rank p-value (`wilcoxon_p`) treats items as independent and is kept for reference only. BH FDR
correction is applied to `cluster_p` across all classes.

**Outputs:** `brier_improvement_per_class.csv`, `brier_improvement_summary.json`

### Step 8 — Wasserstein distance (descriptive)

Calls `tracks.representation.statistical.hypotheses.h2.h2_wasserstein_per_class`. Secondary/sensitivity measure; no FDR
correction applied.

**Output:** `wasserstein_distance.csv`

### Step 9 — ICC rater-level analysis

Calls `shared.statistical.icc.rater_icc_analysis` using the interview split of `individual_labels`
from Step 3. Computes one model-vs-physician ICC(2,1) per (patient, physician) and one
within-pair human-human ICC(2,1) per patient, each with Lin's CCC as a secondary metric. Failures are
caught and logged as warnings (ICC is skipped rather than aborting the pipeline).

**Outputs:** `icc_results.csv`, `icc_summary.json`

### Step 9.5 — Calibration ECE per class

Calls `shared.evaluation.calibration.calibration_ece_per_class` against `y_hat_ca` and `y_interview`
using 15 equal-width bins. Classes with ECE > 0.10 are flagged and logged as warnings.

**Output:** `calibration_ece.csv`

### Step 10 — Context-induced entropy change

Computes `physician_entropy_change` and `model_entropy_change` per (item, class).
Serializes the full per-(item, patient, class) table and the aggregate Pearson r summary.

**Outputs:** `entropy_change.csv`, `entropy_pearson.json`

### Step 11 — Stratum analysis

If `strata` is not pre-supplied, calls `tracks.representation.statistical.reporting.stratum.assign_test_strata` using
`item_texts` and `item_texts_train`. Then calls `stratum_comparison` to compute sign
agreement rate and Brier improvement for each stratum with Mann-Whitney U tests. Skipped
with a warning if neither `strata` nor `item_texts_train` is provided.

**Output:** `stratum_analysis.csv`

### Step 12 — Sub-field ablation

Runs if both `ensemble_bundle_path` and `context_json` are provided. See
[Sub-field ablation](#sub-field-ablation) below.

**Output:** `ablation_scores.csv`

### Step 13 — Architectural comparison

Runs if `arch_predictions` is provided (a dict mapping architecture names to `(n_pairs, n_classes)`
prediction arrays). Calls `tracks.representation.statistical.reporting.arch_compare.arch_comparison_table` with paired
bootstrap differences against the `"2d"` reference architecture.

**Output:** `arch_comparison.csv`

### Step 14 — Figures

Generates all four output figures. Each is wrapped in a try/except so a single figure
failure does not abort the pipeline.

- **Figure 1** — always generated (requires only `delta_p`)
- **Figure 2** — skipped if `individual_labels` is empty
- **Figure 3** — skipped if `icc_results` is empty
- **Figure 4** — always generated (requires only entropy deltas)

**Outputs:** `figures/figure1_delta_histogram.png`, `figures/figure2_interphysician_agreement.png`,
`figures/figure3_icc_horizontal_plot.png`, `figures/figure4_entropy_scatter.png`

---

## Output manifest

| File | Content |
| --- | --- |
| `tau_robustness.csv` | τ=0.5 vs F1-optimal per-class |
| `directional_alignment_per_class.csv` | Directional alignment (formerly H1): sign agreement per class, BH-adjusted p |
| `directional_alignment_pooled.json` | Directional alignment pooled over classes: permutation test |
| `directional_alignment_cmh.json` | Directional alignment across classes: CMH common odds ratio and cluster p |
| `context_specificity_pooled.json` | Context specificity pooled over eligible classes: correct vs shuffled alignment, CI, sign-flip p |
| `context_specificity_per_class.csv` | Context specificity per class, BH-adjusted p |
| `brier_improvement_per_class.csv` | Brier improvement (formerly H2) per class |
| `brier_improvement_summary.json` | Brier improvement macro-average summary |
| `wasserstein_distance.csv` | Wasserstein-1 distance per class (descriptive) |
| `icc_results.csv`       | Per-pair ICC(2,1) and CCC values              |
| `icc_summary.json`      | ICC aggregate means and bootstrap CIs         |
| `calibration_ece.csv`   | ECE per class with >0.10 flag                 |
| `entropy_change.csv`    | Per-(item, patient, class) entropy change     |
| `entropy_pearson.json`  | Pearson r physician vs model entropy          |
| `stratum_analysis.csv`  | Repeated vs novel sub-analysis                |
| `ablation_scores.csv`   | Sub-field attribution A(k,c)                  |
| `arch_comparison.csv`   | Architecture benchmark vs reference           |
| `figures/figure1_*.png` | Per-class Δ_p histogram                       |
| `figures/figure2_*.png` | Inter-physician agreement survey vs interview |
| `figures/figure3_*.png` | ICC(2,1) horizontal dot plot                  |
| `figures/figure4_*.png` | Physician vs model entropy scatter            |
