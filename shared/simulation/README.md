# shared/simulation/

**Canonical path:** `shared/simulation/`

Simulated paired-context studies with planted effects. The endpoints' false-positive rate, power and CI coverage cannot be read off any one study, so this module generates datasets in which the truth is known. `scripts/simulate_endpoints.py` runs every endpoint test on them; the results are in [`docs/simulation.md`](../../docs/simulation.md).

The output is the arrays the endpoint code takes (scores and reference labels), not a dataset workbook. No model is trained and no LLM is called.

## Contents

- `generator.py`
  - `SimulationDesign`: the parameters of one simulated study (scenario, effect size, numbers of patients, items, categories and observers, and the variance components below).
  - `simulate_study(design, rng)`: draws one `SimulatedStudy`: patient IDs, reference labels with and without context, and the candidate's context-free, correct-context and shuffled-context scores.
  - `SCENARIOS`: `unrelated`, `aligned`, `nonspecific`, `partial`, `attenuated`.

## Model

Everything is on the logit scale. A row is one (patient, item) pair.

- **Reference observers.** Adding the patient's context shifts the item's logit by `s = mu[c] + b[patient, c] + h[row, c] + g[row, c]`: a fixed common direction per category (most observers shift up), a patient effect shared by the patient's items, how much this item moves whenever any context is added, and the part specific to this patient's context and this item. Each observer labels the item once without and once with context, each label Bernoulli with observer noise; labels are averaged over observers, so two observers give 0, 0.5 or 1.
- **Candidate model.** The context-free score is the item's logit plus model error. Given a context (the patient's own, or another patient's for the shuffled condition), the model shifts the logit by a scenario-dependent amount plus its own patient effect (`model_patient_sd`) and noise, and the score is rounded to `score_round` (0.05 by default, like LLM outputs; `None` keeps scores continuous).

| Scenario | Model shift given context `k` | What should happen |
|----------|-------------------------------|--------------------|
| `unrelated` | Model patient effect + noise | Every endpoint null |
| `aligned` | `effect ·` the shift context `k` implies for this item | Directional alignment, context specificity and class correspondence reject |
| `nonspecific` | `effect · (mu + h)`, whatever the context | Directional alignment and class correspondence reject; context specificity must not |
| `partial` | As `aligned`, sign reversed in the second half of the categories | Pooled directional alignment and class correspondence lose power |
| `attenuated` | As `unrelated` with the patient's own context; with another patient's context, the whole shift times `1 − effect` | Only `context_sensitivity` rejects |

Only `attenuated` makes the correct context move the scores *further* than another patient's context, so it is the scenario that measures `context_sensitivity`'s power (#40).
