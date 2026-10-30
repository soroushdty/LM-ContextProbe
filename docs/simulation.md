# Simulation study of the endpoints

How often does each endpoint test reject when nothing is there, how much power does it have when something is, and do its confidence intervals cover the truth? None of this can be read off one study, so this page reports a simulation in which the truth is planted. It describes protocol v1 as of [#36](https://github.com/soroushdty/LM-ContextProbe/issues/36).

The study feeds simulated scores straight to the endpoint code: Track 3's `HypothesisAnalyzer` and Track 1's `statistical/hypotheses`. It validates the statistics and tests, not the pipelines that produce the scores. In particular it says nothing about Track 1's Stage 2 training, apart from the separate leave-pair-out simulation described under Track 1's shuffled-context condition in [`methodology.md`](methodology.md).

## How to reproduce

```bash
python scripts/simulate_endpoints.py --out output/simulation          # full study, ~1 h on 16 cores
python scripts/simulate_endpoints.py --quick --out /tmp/sim            # smoke test, seconds
python scripts/simulate_endpoints.py --figures-only --out output/simulation
```

The run on this page used seed 2026 and took 58 minutes on 16 cores. Its settings, the generator's parameters and the Monte Carlo truths used for coverage are in [`simulation/design.json`](simulation/design.json). The summary tables are [`simulation/rejection_rates.csv`](simulation/rejection_rates.csv) and [`simulation/ci_coverage.csv`](simulation/ci_coverage.csv). The per-replicate file (`replicates.csv`, 40 MB) is not committed; the command above recreates it exactly.

## Design

The generator is [`shared/simulation/`](../shared/simulation/README.md).

- **Reference labels.** Two observers label each item without and with context. A context shift has a fixed direction per category (most observers shift up), a patient effect, an item effect and a patient-context × item effect. With the defaults, about a third of reference deltas are non-zero, about two thirds of those are upward, and 80% of them are half-steps from observer disagreement.
- **Model scores.** Context-free, correct-context and shuffled-context scores are rounded to 0.05, like LLM outputs, so many model deltas are exactly zero.
- **Scenarios:**

  | Scenario | The model's context shift | Should reject |
  |----------|---------------------------|---------------|
  | `unrelated` | Unrelated to the reference (its own patient effect plus noise) | Nothing |
  | `aligned` | Follows the shift that the context it is given implies, with effect size *d* | `directional_alignment`, `class_correspondence`, `context_specificity`, `brier_improvement` |
  | `nonspecific` | The same whatever the context: it has learned which items move and in which direction | `directional_alignment`, `class_correspondence`; **not** `context_specificity` |
  | `partial` | As `aligned`, but reversed in half the categories | Per-class tests, not pooled ones |

- **Grid.** 6, 10, 20 and 40 patients; 10 and 30 items per patient; *d* = 0.25, 0.5 and 1.0. There are 500 replicates per null design and 200 per other design, and each test uses 1,000 permutations and 500 bootstrap resamples. At a true rate of 5%, the Monte Carlo standard error of a rejection rate is 1.0 percentage point with 500 replicates.
- **Clustering sub-study.** The `unrelated` scenario is repeated with 30 items per patient and a stronger model patient effect (SD 1.0 and 2.0 instead of 0.3), so a patient's cells are more correlated.

## Results

### False-positive rate

![Rejection rate of every test in the unrelated scenario](simulation/figures/false_positive_rate.png)

Every confirmatory test keeps its nominal rate. Across the eight designs at α = 0.05:

| Test | Range |
|------|-------|
| Track 3 `context_sensitivity` (sign-flip vs shuffled) | 1.0–4.2% |
| Track 3 `directional_alignment` (within-patient permutation) | 1.8–2.8% |
| Track 3 `class_correspondence` (category permutation) | 3.4–7.0% |
| `context_specificity`, Track 3 / Track 1 (sign-flip) | 3.0–6.2% / 3.0–6.8% |
| Track 1 `directional_alignment`, pooled / CMH permutation | 1.6–2.6% / 1.8–4.0% |
| Track 1 `directional_alignment`, per class (any class after BH) | 0.4–1.4% |
| Track 1 `brier_improvement`, per class (any class after BH) | 0–1.0% |

The directional-alignment and Brier tests are conservative, which costs some power (below).
- **Brier improvement:** a model whose shift is unrelated to the reference makes the Brier score *worse*, so this scenario sits inside its one-sided null rather than on its boundary.
- **Directional alignment:** why its permutation tests stay at about 2% rather than 5% has not been worked out.

The reference tests are the ones the outputs keep for comparison, and several exceed the nominal rate:
- **Track 1 per-class binomial test:** up to 19.4%.
- **CMH asymptotic test:** up to 11.0%.
- **Sign-flip tests of agreement against 0.5:** up to 12.4%.

Both effects that the confirmatory tests were designed around are visible here. The reference deltas mostly go one way, so agreement above 0.5 is expected by chance. And cells from one patient are not independent.

**Before #36** the directional-alignment tests counted a zero model delta as disagreement. They then rejected in 96–100% of these null replicates (Track 1 per class: 40–100%), because models and observers both move on the same ambiguous items. See [#36](https://github.com/soroushdty/LM-ContextProbe/issues/36) for the before-and-after table.

### Power

![Power of the confirmatory tests, aligned scenario](simulation/figures/power_aligned.png)

Rejection rates for the aligned scenario with 10 items per patient. Track 1's pooled and CMH permutation tests of directional alignment are within 0.08 of the Track 3 test in every design.

| Test | *d* | 6 patients | 10 | 20 | 40 |
|------|-----|-----------|----|----|----|
| Track 3 `directional_alignment` | 0.25 | 0.48 | 0.62 | 0.92 | 1.00 |
| | 0.5 | 0.82 | 0.98 | 1.00 | 1.00 |
| `context_specificity` (Track 3) | 0.25 | 0.29 | 0.51 | 0.80 | 0.96 |
| | 0.5 | 0.62 | 0.90 | 0.99 | 1.00 |
| `class_correspondence` | 0.5 | 0.74 | 0.88 | 0.98 | 1.00 |
| Track 1 `directional_alignment`, per class (BH) | 0.5 | 0.09 | 0.22 | 0.66 | 0.98 |
| Track 1 `brier_improvement`, per class (BH) | 0.5 | 0.06 | 0.84 | 1.00 | 1.00 |

With 30 items per patient, the pooled tests of directional alignment, `class_correspondence` and `context_specificity` all have power ≥ 0.92 at *d* = 0.5, even with 6 patients.

The per-class tests split the data ten ways and correct for multiplicity, so they need about four times as many patients as the pooled tests. With 6 patients, Brier improvement has essentially no power. Its sign-flip test cannot go below p = 1/64 ≈ 0.016, so after BH across ten classes a class can be rejected only when at least four classes reach that floor.

### `context_specificity` separates aligned from non-specific models

![Power of the confirmatory tests, non-specific scenario](simulation/figures/power_nonspecific.png)

A non-specific model shifts every item the way observers tend to shift it, whatever context it is given. It passes directional alignment: at *d* = 0.5 with 20 patients and 10 items, the Track 3 test rejects in 86% of replicates and the Track 1 pooled test in 84%. It also passes class correspondence. That is correct, because it does agree with the observers item by item. `context_specificity` stays at its nominal rate for this model (2–6% in every design) but has power 0.62–1.00 for the aligned model at *d* = 0.5. So `context_specificity` is the endpoint that tells "uses *this* patient's context" apart from "reacts to having context", as [#18](https://github.com/soroushdty/LM-ContextProbe/issues/18) asked.

### Partial alignment

![Power of the confirmatory tests, partial scenario](simulation/figures/power_partial.png)

When the model is aligned in half the categories and anti-aligned in the rest, the pooled tests average the two halves away:
- Track 3 directional alignment, Track 1 pooled and Track 1 CMH reject in at most 15% of replicates.
- Class correspondence rejects in ≤ 2%.

Only the per-class tests find it. Track 1's per-class directional alignment reaches 0.78 with 40 patients × 10 items at *d* = 0.5, and 0.98 with 20 patients × 30 items. A pooled null result does not mean "no alignment in any category".

### Confidence-interval coverage

![Coverage of nominal 95% CIs](simulation/figures/ci_coverage.png)

The context-specificity difference has a known true value (0 when correct and shuffled context are exchangeable; otherwise a Monte Carlo estimate from 4,000 patients). Coverage of its nominal 95% patient-cluster bootstrap CI, averaged over scenarios:

| Items per patient | 6 patients | 10 | 20 | 40 |
|-------------------|-----------|----|----|----|
| 10 | 0.875 | 0.893 | 0.934 | 0.936 |
| 30 | 0.874 | 0.916 | 0.938 | 0.943 |

With 6–10 patients, the percentile bootstrap over patients undercovers. Read a "95%" interval from such a study as roughly a 90% interval.

The context-sensitivity difference covers at 0.93–0.99 with the patient-cluster bootstrap, which is conservative with 30 items per patient.

### Why the patient clustering matters

![Clustering sub-study](simulation/figures/clustering.png)

With the default, weak model patient effect, ignoring the clustering costs little here. Item-level bootstrap CIs cover 0.92–0.94, and a cell-level t-test has a false-positive rate of 4–7%. As a patient's cells become more correlated (model patient-effect SD 1.0 and 2.0; 30 items per patient), the tests that treat cells as independent stop holding their level:

| Test (unrelated scenario, 30 items) | SD 0.3 | SD 1.0 | SD 2.0 |
|-------------------------------------|--------|--------|--------|
| Track 1 per-class binomial (BH), 40 patients | 0.194 | 0.450 | 0.538 |
| CMH asymptotic, 40 patients | 0.110 | 0.128 | 0.150 |
| Cell-level t-test of `context_specificity`, 10 patients | 0.038 | 0.098 | 0.106 |
| **Patient-cluster sign-flip test of `context_specificity`, 10 patients** | 0.032 | 0.034 | 0.026 |
| **CMH within-patient permutation, 40 patients** | 0.040 | 0.056 | 0.070 |

Item-level CI coverage falls to 0.89–0.91. Patient-cluster coverage reaches 0.95–0.97 once there are 20 or more patients.

## Limitations

- **One generative model.** The rates above hold for this model of observers and scores. Different variance components, label rates or score rounding give different numbers; the script's parameters make that easy to explore.
- **`context_sensitivity`'s power is not measured.** No scenario makes the correct context move the scores *further* than another patient's context does, only differently. The test's false-positive rate is checked (1.0–4.2%).
- **Small-sample CIs.** With fewer than about 20 patients the patient-cluster percentile bootstrap undercovers (above). A bias-corrected or studentized cluster bootstrap could do better; this has not been tried.
- **Scores only.** Pipeline effects, such as Track 1's Stage 2 training or LLM parse failures, are outside this study.
