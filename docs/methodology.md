# Methodology

LM-ContextProbe is an evaluation framework, not a benchmark: it defines a protocol and runs it on whatever model and dataset you supply. This document specifies that protocol. It states what LM-ContextProbe measures, the study design it assumes, how each endpoint is computed and tested, and which design decisions protect the results from known pitfalls. It describes the code as it is in this repository. Where the code falls short of the intended design, the gap is listed under [Known limitations](#known-limitations).

## The question

When a clinician sees an EHR item (a diagnosis, a lab result, a medication) on its own, they may file it under one sensitive-data category. With the patient's chart in view, they may file it differently. LM-ContextProbe asks whether a language model shows **the same context-induced shifts**:

1. Does the model's judgment change when patient context is added?
2. Does it change in the same direction as the human judgment?
3. Do the categories that shift most for humans also shift most for the model?
4. Does the alignment depend on the context belonging to *this* patient, or would any plausible clinical text produce it?

The last question is what separates *appropriate* context sensitivity from mere sensitivity. A model whose outputs move whenever a chart is pasted into the prompt is context-sensitive. It is only appropriately sensitive if the movement tracks the human shift and disappears when the chart belongs to someone else.

LM-ContextProbe does not measure classification accuracy. A model can be accurate and still ignore context, or shift correctly while being poorly calibrated.

## Study design the framework assumes

The built-in `paired_context` adapter ([`data_format.md`](data_format.md)) expects a **paired counterfactual design**:

- A set of **patients** (context entities), each with a structured clinical snapshot.
- For each patient, a set of **items** (task instances): EHR strings drawn from that patient's record.
- **Reference observers** (for example physicians) who label each item into the output categories twice: first **context-free** (item text only), then **correct-context** (item text plus the patient's snapshot). Each patient is assigned a fixed group of observers.

Other designs can be supported by writing another adapter ([`adapters.md`](adapters.md)); the endpoints below only need the canonical arrays the adapter produces.

### Unit of analysis and clustering

The **unit of analysis is the (patient, item) pair** after observer aggregation. Pairs from the same patient share a context snapshot and an observer group, so they are not independent. **The patient is the cluster**: confidence intervals resample whole patients, and the effective sample size for any between-patient claim is the number of patients, not the number of pairs.

### Reference labels and the reference delta

For each (patient, item) pair and category *c*, the reference label is the **mean of the observers' binary labels** (`shared/reference/aggregation/paired_reference_mean.py`). With two observers per patient, this takes values in {0, 0.5, 1}. **0.5 means the two observers disagreed.** It is not a probability of category membership. The aggregation fails loudly or deduplicates (with a warning) when a group's observer count differs from the configured count.

The reference delta is

```
Δ_ref(i, c) = y_correct_context(i, c) − y_context_free(i, c)      ∈ {−1, −0.5, 0, 0.5, 1}
```

In typical data most Δ_ref cells are zero. Endpoints that compare directions are therefore computed only on cells where Δ_ref ≠ 0.

### Model conditions and the model delta

Each candidate model scores each (patient, item) pair under up to three conditions:

| Condition | Input | Role |
|-----------|-------|------|
| `context_free` | Item text only | Baseline |
| `correct_context` | Item text + this patient's snapshot | Treatment |
| `shuffled_context` | Item text + a different patient's snapshot | Negative control |

The shuffled patient is drawn uniformly from the other patients, separately for each (patient, item) pair, with a fixed seed (`shuffled_context_seed`); the mapping is saved for audit. Track 1 has no prompt: its shuffled-context score feeds the Stage 2 head another patient's context vector, and the patients are paired rather than drawn per item (see [Track 1](#track-1-tracksrepresentationstatistical)). Scores are per-category values in [0, 1]. The model delta is

```
Δ_model(i, c) = score_correct_context(i, c) − score_context_free(i, c)
```

and `Δ_shuffled` is defined the same way with the shuffled-context score.

## Tracks

Each track is a functional analogue of one way a physician could reach a context-dependent judgment. The analogy organizes the comparison; it is not a claim about how physicians think.

| Track | Physician analogue | How the model sees context | Status |
|-------|--------------------|----------------------------|--------|
| 1 – Representation | General clinical knowledge | A frozen clinical encoder embeds item and context separately; a small trained head combines them | Available |
| 2 – Adaptation | Learning from supervised cases | Fine-tuning on labelled (context, item) cases from training patients | Planned ([#13](https://github.com/soroushdty/LM-ContextProbe/issues/13)) |
| 3 – Reasoning | Deliberating over the chart | An LLM reads item and context in one prompt and returns category scores | Available |

**Track 1** (`tracks/representation/`) trains a context-free multilabel classifier on frozen embeddings (Stage 1) from an encoder (default Bio_ClinicalBERT) or a decoder LLM ([`shared/embeddings/`](../shared/embeddings/README.md)), then a regularized linear fusion head over the item embedding *e* and context embedding *c* (Stage 2). The fusion is configurable (`fusion_strategy`): `2d` = `[e, c]` (default), `3d` adds `e⊙c`, `4_vector` adds `|e−c|`, plus a low-rank bilinear option. The head is fitted to training patients' correct-context labels. All fitting uses nested patient-grouped cross-validation, so every prediction is for a patient the model never saw. Both loops are leave-one-patient-out by default; `cv.outer` and `cv.inner` in `configs/training_config.yaml` can switch either to grouped k-fold (`shared/cv/`). Stage 2 is trained only to fit the labels; it has no loss term that rewards agreement with the reference *delta*, so the delta endpoints are not optimized directly.

**Track 3** (`tracks/reasoning/`) prompts an LLM under all three conditions with deterministic prompts (default `temperature: 0`) and parses a JSON object of per-category scores. Responses are cached per model, condition, patient and item. There is no training.

## Endpoints

Endpoints are named once, in `shared/endpoints.py`, and every track writes its results under those names. Each track declares which endpoints it computes (`tracks/reasoning/endpoints.py`, `tracks/representation/statistical/endpoints.py`). The statistic and test can differ between tracks; the tables below give each track's.

Earlier versions numbered hypotheses per track, and the numbers clashed: Track 1's H1 was Track 3's H2. The old numbers appear only as "(formerly H*n*)" in report headings.

| Endpoint | Question | Track 3 | Track 1 |
|----------|----------|---------|---------|
| `context_sensitivity` | Does the correct patient's context change the model's scores more than another patient's context does? | ✓ (formerly H1) | — |
| `directional_alignment` | Does the score change in the same direction as the reference observers' judgment? | ✓ (formerly H2) | ✓ (formerly H1) |
| `class_correspondence` | Do the categories that shift most for the reference observers also shift most for the model? | ✓ (formerly H3) | — |
| `context_specificity` | Is the alignment specific to the correct patient's context, compared with a shuffled context? | ✓ (formerly H4) | ✓ |
| `brier_improvement` | Does context bring the scores closer to the correct-context reference labels? | — | ✓ (formerly H2) |

### Track 3 (`shared/evaluation/hypothesis_analyzer.py`)

| | Statistic | Uncertainty / test |
|---|---|---|
| **`context_sensitivity`** | Mean \|Δ_model\| and mean \|Δ_shuffled\| over all cells, overall and per category; their paired difference `|Δ_model| − |Δ_shuffled|` | Patient-cluster bootstrap 95% CIs; one-sided patient-cluster sign-flip test of the difference > 0 |
| **`directional_alignment`** | On cells with Δ_ref ≠ 0: mean alignment `sign(Δ_ref)·Δ_model`; on cells where Δ_model ≠ 0 too: sign-agreement rate `sign(Δ_model) = sign(Δ_ref)`; the same for Δ_shuffled | Patient-cluster bootstrap 95% CIs; one-sided permutation test of the sign-agreement rate against a null that shuffles each patient's model deltas among that patient's items |
| **`class_correspondence`** | Pearson *r* between the per-category mean Δ_ref and mean Δ_model | One-sided permutation test over category labels |
| **`context_specificity`** | On cells with Δ_ref ≠ 0: mean of `sign(Δ_ref)·(Δ_model − Δ_shuffled)` | Patient-cluster bootstrap 95% CI; one-sided patient-cluster sign-flip test |

**Why the references.** Mean \|Δ_model\| is above zero for almost any model that reads its input, so a CI that excludes zero shows only that the prompt changed. The test is therefore whether the correct patient's context moves the scores more than another patient's does.

The sign-agreement rate counts only cells where the model moved (Δ_model ≠ 0, or `|Δ_model| ≥ epsilon` in the post-hoc bundle). Models and reference observers both tend to move on the same items, the ambiguous ones whose score is not pinned near 0 or 1. When a zero model delta counted as disagreement, a model that moved exactly where the reference moved, in random directions, beat the permutation null below almost every time: in simulation with scores rounded to 0.05, the false-positive rate was 100% ([#36](https://github.com/soroushdty/LM-ContextProbe/issues/36)). How often the model does not move is reported separately: the share of Δ_ref ≠ 0 cells with a zero model delta (`zero_delta_rate`), or below `epsilon` in the bundle (`unchanged_rate_*`).

Chance agreement is still not 0.5: it depends on which direction the model and the reference each tend to move in. The permutation null keeps that, because it moves each item's whole row of model deltas to another item of the same patient, and breaks only the match between the model's and the reference's shift on the same item. Its mean is reported as the chance agreement rate.

### Track 1 (`tracks/representation/statistical/`)

Confirmatory analyses are restricted to categories with at least `confirmatory_min_nonzero` (default 15) non-zero reference deltas. Other categories are reported descriptively.

| | Statistic | Uncertainty / test |
|---|---|---|
| **`directional_alignment`, per class** | Sign-agreement rate on cells with Δ_ref ≠ 0 and Δ_model ≠ 0 (`n_moved`) | One-sided within-patient permutation test of the rate against its chance level (`perm_p`; the null mean is `null_rate`); Benjamini–Hochberg across eligible classes; patient-cluster bootstrap CI. The sign-flip test of rate > 0.5 (`cluster_p`) and the exact binomial p-value (`binom_p`) are kept for reference only |
| **`directional_alignment`, pooled** | Sign-agreement rate pooled over eligible classes, on the same cells | Permutation test that shuffles model deltas among the items of the same patient (tests item-level specificity within patients); patient-cluster bootstrap CI |
| **`directional_alignment`, across classes** | Mantel–Haenszel common odds ratio of model up vs down by reference up vs down, stratified by class; cells with Δ_model = 0 are left out | One-sided within-patient permutation test of the odds ratio > 1 (`p_permutation`); the CMH p-value (`p_cmh`) and the sign-flip test of pooled agreement > 0.5 (`p_cluster`) are kept for reference only |
| **`brier_improvement`** | Per-class Brier improvement `Brier(context_free) − Brier(correct_context)` against correct-context labels | One-sided patient-cluster sign-flip test (`cluster_p`); Benjamini–Hochberg; patient-cluster bootstrap CI; macro summary. The Wilcoxon signed-rank p-value (`wilcoxon_p`) is kept for reference only |
| **`context_specificity`** | Track 3's statistic (the same function, `context_specificity_test`): on cells with Δ_ref ≠ 0, mean of `sign(Δ_ref)·(Δ_model − Δ_shuffled)`, with both deltas from the leave-pair-out heads described below; pooled over eligible classes (confirmatory) and per class | One-sided patient-cluster sign-flip test; patient-cluster bootstrap CI; Benjamini–Hochberg across eligible classes for the per-class p-values |

**Track 1's shuffled-context condition.** Track 1's Stage 2 head is trained, so a shuffled context cannot simply be fed to the fold-pure head. That head was fitted on every patient except the one it scores, so another patient's context would be one it was trained on, while the patient's own context is new to it. In a simulation with eight patients (the real Stage 2 training code, 150 datasets per scenario), that asymmetry biased `context_specificity` in a direction that depended on the geometry of the context vectors. It gave 8.7% false positives at α = 0.05 when the context vectors shared a common direction, as BERT embeddings do, and it halved the power to detect real specificity (about 9% against 21%).

Track 1 therefore uses **leave-pair-out** heads. Test patients are grouped in random pairs (one group of three when the number is odd; `shuffled_context_seed` in `configs/training_config.yaml`). For each pair, a Stage 2 head is fitted, with the same procedure as a fold head, on all other patients, using fold-pure Stage 1 predictions as ŷ_cf. It scores each patient's items twice: with the patient's own context vector and with the partner's. Both scores come from one head that has seen neither patient, so their difference is due to the context alone. In the same simulation this design had no detectable bias and 4.7% false positives. `context_specificity` compares these two scores; the other Track 1 endpoints still use the fold-pure predictions. This costs one extra Stage 2 fit per pair. Each row's partner is written to `model/test_shuffled_context_map.csv`.

Secondary and descriptive outputs: Wasserstein distance to the label distribution, ICC(2,1) and Lin's CCC of the model against each observer compared with observer–observer agreement, expected calibration error, context-induced entropy change, repeated vs novel item strata, per-field context ablation, and fusion-architecture comparison.

## Inference

- **Bootstrap.** All confidence intervals are percentile intervals from a patient-cluster bootstrap (`shared/statistical/bootstrap.py`): whole patients are resampled with replacement, default 1,000 resamples. With fewer than two patients the CI is reported as NaN.
- **Permutation tests.** Default 10,000 permutations. p-values are one-sided, in the direction of the hypothesis.
- **Within-patient permutation test** (`within_cluster_permutation_test` in `shared/statistical/cluster_tests.py`). Used for every `directional_alignment` test in both tracks (Track 1's pooled test has its own implementation of the same null in `h1.py`). A test against 0.5 would be fooled by a model whose deltas mostly go the way most reference deltas go. Model delta rows are shuffled among the items of each patient while the reference deltas stay in place, so the test asks whether the model matches the reference on *the same item*, not only on the patient's items in general. Patients with a single item contribute nothing to the null.
- **Patient-cluster sign-flip test** (`shared/statistical/cluster_tests.py`). Used wherever a test compares cell-level values with zero (`context_sensitivity` in Track 3; `context_specificity` in both tracks; `brier_improvement` in Track 1, and the reference-only `cluster_p` / `p_cluster` of Track 1's `directional_alignment`). Under the null, each patient's summed contribution is symmetric about zero, so its sign is flipped as a block; the statistic is the mean over cells. When `2^(number of patients)` is at most the permutation count, all sign patterns are enumerated and the test is exact. **The smallest attainable p-value is then `2^−(number of patients)`**: 1/64 with six patients, 1/1024 with ten. This floor is the real limit of a design with that many patients, not an artifact of the test. Tests that treat cells as independent (`binom_p`, `p_cmh`, `wilcoxon_p`) are still written to the outputs for comparison but are not used for decisions.
- **Multiplicity.** Track 1 applies Benjamini–Hochberg (q = 0.05) across classes. Track 3 reports per-category results descriptively and does not correct them.
- **Seeds.** Bootstrap, permutation, shuffled-context and generation seeds are set in config and recorded in the run manifest. `PYTHONHASHSEED` must be set before Python starts (see the README).

## Protocol version

The evaluation protocol (the endpoints, their statistics and tests, and the defaults that change reported values) is versioned separately from the code, as `PROTOCOL_VERSION` in `shared/endpoints.py`. Every `run_manifest.json` records it as `protocol_version`, and the Track 3 run report and the post-hoc bundle report print it. Results can then say "LM-ContextProbe protocol v1", and a code release that leaves the numbers unchanged does not invalidate them. The version is bumped whenever a change would alter reported values for the same data and model.

| Protocol | First release | Changes |
|----------|---------------|---------|
| 1 | 0.2.0 | First versioned protocol: the endpoints and tests described on this page, including the shuffled-context reference for `context_sensitivity`, the within-patient permutation null for `directional_alignment` ([#17](https://github.com/soroushdty/LM-ContextProbe/issues/17)), Track 1's shuffled-context condition and `context_specificity` ([#11](https://github.com/soroushdty/LM-ContextProbe/issues/11)), and sign agreement computed only on cells where the model moved ([#36](https://github.com/soroushdty/LM-ContextProbe/issues/36)) |

## Design decisions and what they guard against

Most of the machinery in LM-ContextProbe exists because a simpler version gave a misleading answer during development. Each row names the pitfall and the safeguard.

| Pitfall | Safeguard |
|---------|-----------|
| Treating (patient, item) pairs as independent understates uncertainty, because pairs from one patient share context and observers | Patient-cluster bootstrap for every CI; patient-grouped cross-validation in Track 1 |
| Training a model on the quantity you then test it on (for example, a loss that rewards matching the reference delta direction) makes the endpoint circular | Track 1's Stage 2 fits labels only; Track 3 involves no training |
| Mistaking any response to extra text for a response to *this patient's* context | Shuffled-context control and `context_specificity` |
| The item's own text appearing in the context snapshot, so the "context" effect is really the item being repeated | `LabelLeakageError`: a context build fails if the item text appears in any context field, unless the check is explicitly disabled |
| Text-standardization rules learned from evaluation items leak information into training | Fuzzy item matching is built from training items only (`build_fuzzy_matcher_from_train`) |
| An item string seen during training (under a different patient) is easier than a novel one | Track 1 reports results separately for repeated and novel items |
| Reading an aggregated 0.5 label as a calibrated probability | 0.5 is documented and handled as observer disagreement; delta endpoints compare directions, not magnitudes |
| Mismatched or duplicated observer rows silently distorting the reference | The adapter checks that context-free and correct-context (patient, observer, item) triplets correspond and that each pair has the configured observer count. With `mismatch_error: true` a mismatch stops the run; otherwise the rows are reported and skipped |
| Claiming a model is "indistinguishable from a physician" because its agreement falls inside the human range | ICC is reported descriptively next to the observer–observer distribution; no equivalence claim is made |
| Over-reading categories with very few shifts | Track 1's confirmatory eligibility threshold |

## Scope of claims

Results are claims about **the patients and reference observers in the evaluated dataset**. With few patients, between-patient generalization is weak by construction, and the cluster bootstrap makes that visible in the interval widths. A null result is informative: the framework is designed to show the absence of appropriate context sensitivity as readily as its presence.

## Known limitations

These are known gaps between the intended design and the current code.

- **`class_correspondence` is a correlation over the number of categories** (ten by default), so it has little power and is best read descriptively.
- **No simulation-based validation yet.** The false-positive rate and power of the endpoints under known effects have not been measured. ([#18](https://github.com/soroushdty/LM-ContextProbe/issues/18))
- **One dataset format so far** (the paired-context workbook; [#6](https://github.com/soroushdty/LM-ContextProbe/issues/6)). The label taxonomy is configurable ([`docs/adapters.md`](adapters.md#label-space-sharedlabel_spacepy)).
