"""Simulate paired-context reference labels and candidate scores with planted effects.

The endpoints' operating characteristics (false-positive rate, power,
confidence-interval coverage) cannot be read off any one study, so this
module generates datasets in which the truth is known.  Everything happens
on the logit scale.

Reference observers
-------------------
Each row is one (patient, item) pair.  Its context-free logit for category c
is ``a[r, c]``.  Adding the patient's context shifts it by

    s[r, c] = mu[c] + b[p, c] + h[r, c] + g[r, c]

``mu`` is the category's common direction (most observers shift up; fixed by
``category_seed``), ``b`` a
patient effect shared by the patient's items (the source of within-patient
correlation), ``h`` how much this item tends to move whenever any context is
added, and ``g`` the part that depends on this patient's context and this
item together.  Each observer labels the item once without context and once
with it, each label Bernoulli with an observer-specific logit perturbation;
labels are averaged over observers, so two observers give {0, 0.5, 1}.

Candidate model
---------------
The context-free score is ``sigmoid(a + model error)``.  Given a context
``k`` (the row's own patient, or another patient for the shuffled
condition), the model shifts the logit by ``delta(k)`` and the score is
rounded to ``score_round`` (LLM-like outputs, which produce zero deltas).
The context ``k`` implies its own shift for the row,
``s_k[r, c] = mu[c] + b[k, c] + h[r, c] + g_k[r, c]``, where ``g_k`` is
redrawn for another patient's context.  Scenarios differ only in
``delta(k)``:

``unrelated`` (the null scenario; not called ``null`` because pandas reads that as missing)
    ``delta = m[k, c] + e``: a model-specific patient effect and noise,
    unrelated to the reference.  Correct and shuffled context are
    exchangeable, so every endpoint is null.
``aligned``
    ``delta = effect · s_k + m[k, c] + e``: the model shifts the way the
    context it is given implies.  Correct context aligns with the reference
    more than shuffled context does.
``nonspecific``
    ``delta = effect · (mu + h) + m[k, c] + e``: the model has learned which
    items tend to move and moves them the same way whatever context it is
    given.  It agrees with the reference item by item, but correct and
    shuffled context are exchangeable, so only ``context_specificity`` (and
    ``context_sensitivity``) should stay null.
``partial``
    As ``aligned`` with the sign of the effect reversed in the second half
    of the categories.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SCENARIOS = ("unrelated", "aligned", "nonspecific", "partial")


@dataclass(frozen=True)
class SimulationDesign:
    """Parameters of one simulated study."""

    scenario: str = "unrelated"
    effect: float = 0.0
    n_patients: int = 10
    items_per_patient: int = 20
    n_categories: int = 10
    n_observers: int = 2
    # Reference observers
    base_logit_mean: float = -2.0
    base_logit_sd: float = 3.0
    shift_mean: float = 0.5
    shift_category_sd: float = 0.5
    patient_sd: float = 0.7
    item_sd: float = 0.7
    interaction_sd: float = 0.7
    observer_sd: float = 0.5
    # The categories' common shift directions ``mu`` are a fixed property of
    # the taxonomy, drawn once from this seed, so that every replicate of a
    # design has the same estimands.
    category_seed: int = 0
    # Candidate model
    model_cf_sd: float = 0.5
    model_patient_sd: float = 0.3
    model_noise_sd: float = 0.5
    score_round: float | None = 0.05

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {self.scenario!r}; expected one of {SCENARIOS}.")
        if self.n_patients < 2:
            raise ValueError("A shuffled context needs at least two patients.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulatedStudy:
    """One simulated dataset, in the arrays the endpoint code takes.

    All score and label arrays have shape ``(n_rows, n_categories)``.
    """

    patient_ids: np.ndarray
    category_names: list[str]
    reference_context_free: np.ndarray
    reference_correct_context: np.ndarray
    context_free_scores: np.ndarray
    correct_context_scores: np.ndarray
    shuffled_context_scores: np.ndarray
    shuffled_patient_ids: np.ndarray

    @property
    def delta_reference(self) -> np.ndarray:
        return self.reference_correct_context - self.reference_context_free


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _round(scores: np.ndarray, step: float | None) -> np.ndarray:
    if step is None:
        return scores
    return np.clip(np.round(scores / step) * step, 0.0, 1.0)


def simulate_study(design: SimulationDesign, rng: np.random.Generator) -> SimulatedStudy:
    """Draw one dataset from ``design``."""
    d = design
    P, C = d.n_patients, d.n_categories
    pids = np.repeat(np.arange(P), d.items_per_patient)
    n = pids.size

    # Reference shift components.
    mu = np.random.default_rng(d.category_seed).normal(d.shift_mean, d.shift_category_sd, size=C)
    b = rng.normal(0.0, d.patient_sd, size=(P, C))
    h = rng.normal(0.0, d.item_sd, size=(n, C))
    g = rng.normal(0.0, d.interaction_sd, size=(n, C))
    a = rng.normal(d.base_logit_mean, d.base_logit_sd, size=(n, C))
    s = mu + b[pids] + h + g

    def _labels(logit: np.ndarray) -> np.ndarray:
        noise = rng.normal(0.0, d.observer_sd, size=(d.n_observers, *logit.shape))
        return (rng.random(noise.shape) < _sigmoid(logit + noise)).mean(axis=0)

    ref_cf = _labels(a)
    ref_cc = _labels(a + s)

    # Shuffled context: another patient, drawn per row.
    offsets = rng.integers(1, P, size=n)
    donors = (pids + offsets) % P

    # The shift each context implies for the row.
    implied_correct = s
    implied_shuffled = mu + b[donors] + h + rng.normal(0.0, d.interaction_sd, size=(n, C))

    m = rng.normal(0.0, d.model_patient_sd, size=(P, C))  # model's own patient effect
    sign = np.ones(C)
    if d.scenario == "partial":
        sign[C // 2:] = -1.0

    def _model_shift(context_ids: np.ndarray, implied: np.ndarray) -> np.ndarray:
        base = m[context_ids] + rng.normal(0.0, d.model_noise_sd, size=(n, C))
        if d.scenario in ("aligned", "partial"):
            return base + d.effect * sign * implied
        if d.scenario == "nonspecific":
            return base + d.effect * (mu + h)
        return base

    model_cf_logit = a + rng.normal(0.0, d.model_cf_sd, size=(n, C))
    cf = _round(_sigmoid(model_cf_logit), d.score_round)
    cc = _round(_sigmoid(model_cf_logit + _model_shift(pids, implied_correct)), d.score_round)
    sh = _round(_sigmoid(model_cf_logit + _model_shift(donors, implied_shuffled)), d.score_round)

    return SimulatedStudy(
        patient_ids=pids,
        category_names=[f"c{i}" for i in range(C)],
        reference_context_free=ref_cf,
        reference_correct_context=ref_cc,
        context_free_scores=cf,
        correct_context_scores=cc,
        shuffled_context_scores=sh,
        shuffled_patient_ids=donors,
    )
