#!/usr/bin/env python3
"""
Simulation study of the endpoints: false-positive rate, power and CI coverage.

Generates paired-context studies with planted effects
(``shared.simulation``), runs every endpoint test of Track 3
(``HypothesisAnalyzer``) and Track 1 (``statistical/hypotheses``) on each,
and summarises how often each test rejects and how often each CI covers the
true value.  No model is trained and no LLM is called: the simulated scores
go straight to the endpoint code, so the study validates the statistics and
tests, not the pipelines that produce the scores.

Each test is reported with a role:

* ``confirmatory``: the test the endpoint is decided on.
* ``reference``: a test the outputs keep for comparison (for example the
  sign-flip test of agreement against 0.5, or tests that treat cells as
  independent).

Per-class tests (Track 1) are reported as a family: the family rejects when
any confirmatory-eligible class does after Benjamini–Hochberg correction.

Outputs (``--out``, default ``output/simulation``):
    replicates.csv        one row per replicate and test
    rejection_rates.csv   rejection rate at ``--alpha`` with its Monte Carlo SE
    ci_coverage.csv       coverage and width of patient-cluster and item-level CIs
    design.json           the designs, seeds and settings used
    figures/*.png

Usage:
    python scripts/simulate_endpoints.py                       # full study
    python scripts/simulate_endpoints.py --quick --out /tmp/s  # a few minutes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the path when run as a script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scipy import stats

from shared.evaluation.hypothesis_analyzer import (
    HypothesisAnalyzer,
    context_specificity_test,
)
from shared.simulation import SimulationDesign, simulate_study
from shared.statistical.delta import confirmatory_eligible_classes
from tracks.representation.statistical.hypotheses.context_specificity import (
    context_specificity_pooled,
)
from tracks.representation.statistical.hypotheses.h1 import (
    _bh_correct,
    h1_binomial_per_class,
    h1_cmh_test,
    h1_permutation_test,
)
from tracks.representation.statistical.hypotheses.h2 import (
    h2_wilcoxon_per_class,
)

# (track, endpoint, test, role); the order is the order of the tables.
TESTS: list[tuple[str, str, str, str]] = [
    ("track3", "context_sensitivity", "sign_flip_vs_shuffled", "confirmatory"),
    ("track3", "directional_alignment", "within_patient_permutation", "confirmatory"),
    ("track3", "class_correspondence", "category_permutation", "confirmatory"),
    ("track3", "context_specificity", "sign_flip", "confirmatory"),
    ("track3", "context_specificity", "cell_level_t_test", "reference"),
    ("track1", "directional_alignment", "per_class_permutation_bh", "confirmatory"),
    ("track1", "directional_alignment", "per_class_sign_flip_vs_half_bh", "reference"),
    ("track1", "directional_alignment", "per_class_binomial_bh", "reference"),
    ("track1", "directional_alignment", "pooled_permutation", "confirmatory"),
    ("track1", "directional_alignment", "cmh_permutation", "confirmatory"),
    ("track1", "directional_alignment", "cmh_sign_flip_vs_half", "reference"),
    ("track1", "directional_alignment", "cmh_asymptotic", "reference"),
    ("track1", "brier_improvement", "per_class_sign_flip_bh", "confirmatory"),
    ("track1", "brier_improvement", "per_class_wilcoxon_bh", "reference"),
    ("track1", "context_specificity", "pooled_sign_flip", "confirmatory"),
]
_ROLE = {(t, e, n): r for t, e, n, r in TESTS}

# Quantities whose CIs are checked, with the statistic computed from a study.
CI_QUANTITIES = ("context_specificity_difference", "context_sensitivity_difference")


class _Paired:
    """The two attributes ``HypothesisAnalyzer`` reads from a paired dataset."""

    def __init__(self, patient_ids, category_names):
        self.patient_ids = patient_ids
        self.category_names = category_names


def _family_p(df: pd.DataFrame, column: str) -> float:
    """Smallest BH-adjusted p over confirmatory-eligible classes (1 if none)."""
    rows = df[df["confirmatory"].astype(bool)]
    p = rows[column].to_numpy(dtype=float)
    p = np.where(np.isnan(p), 1.0, p)
    if p.size == 0:
        return 1.0
    return float(np.min(_bh_correct(p)))


def _cells(study):
    """Cell-level arrays for the CI quantities, restricted as the endpoints are."""
    dp = study.delta_reference
    dm = study.correct_context_scores - study.context_free_scores
    ds = study.shuffled_context_scores - study.context_free_scores
    pids = np.broadcast_to(study.patient_ids[:, None], dp.shape)
    nz = dp != 0
    spec = (np.sign(dp) * (dm - ds))[nz]
    sens = (np.abs(dm) - np.abs(ds)).ravel()
    return {
        "context_specificity_difference": (spec, pids[nz]),
        "context_sensitivity_difference": (sens, pids.ravel()),
    }


def _item_level_ci(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap CI that resamples cells as if they were independent."""
    if values.size < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_study(study, *, n_bootstrap: int, n_permutations: int, seed: int,
                   min_nonzero: int = 15) -> tuple[list[dict], list[dict]]:
    """Run every test on one study.

    Returns ``(test_rows, ci_rows)``: one row per test with its estimate and
    p-value, and one row per CI quantity and method with the CI bounds.
    """
    cf = study.context_free_scores
    cc = study.correct_context_scores
    sh = study.shuffled_context_scores
    dp = study.delta_reference
    dm = cc - cf
    ds = sh - cf
    pids = study.patient_ids
    classes = study.category_names
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(8)]

    rows: list[dict] = []

    def add(track, endpoint, test, estimate, p_value):
        rows.append({
            "track": track, "endpoint": endpoint, "test": test,
            "role": _ROLE[(track, endpoint, test)],
            "estimate": float(estimate), "p_value": float(p_value),
        })

    # --- Track 3 ---
    seeds = rngs[0].integers(0, 2**31 - 1, size=2)
    analyzer = HypothesisAnalyzer(_Paired(pids, classes), {
        "n_bootstrap_resamples": n_bootstrap, "n_permutations": n_permutations,
        "bootstrap_seed": int(seeds[0]), "permutation_seed": int(seeds[1]),
    })
    h1 = analyzer.compute_h1(cf, cc, sh)
    add("track3", "context_sensitivity", "sign_flip_vs_shuffled", h1["mean_abs_delta_difference"], h1["p_value"])
    h2 = analyzer.compute_h2(cf, cc, dp, sh)
    add("track3", "directional_alignment", "within_patient_permutation", h2["sign_agreement_rate"], h2["p_value"])
    h3 = analyzer.compute_h3(cf, cc, dp)
    add("track3", "class_correspondence", "category_permutation", h3["pearson_r"], h3["p_value"])
    h4 = context_specificity_test(
        dm, ds, dp, pids, n_bootstrap=n_bootstrap, n_permutations=n_permutations,
        bootstrap_rng=rngs[1], permutation_rng=rngs[2],
    )
    add("track3", "context_specificity", "sign_flip", h4["mean_diff"], h4["p_value"])
    spec_cells, _ = _cells(study)["context_specificity_difference"]
    t_p = stats.ttest_1samp(spec_cells, 0.0, alternative="greater").pvalue if spec_cells.size > 1 else np.nan
    add("track3", "context_specificity", "cell_level_t_test", h4["mean_diff"], t_p)

    # --- Track 1 ---
    eligible, _ = confirmatory_eligible_classes(dp, classes, min_nonzero=min_nonzero)
    per_class = h1_binomial_per_class(
        dp, dm, classes, eligible, pids, n_bootstrap, rngs[3],
        n_permutations=n_permutations, perm_rng=rngs[4], null_rng=rngs[5],
    )
    rate = float(per_class["sign_agree_rate"].mean())
    add("track1", "directional_alignment", "per_class_permutation_bh", rate, _family_p(per_class, "perm_p"))
    add("track1", "directional_alignment", "per_class_sign_flip_vs_half_bh", rate, _family_p(per_class, "cluster_p"))
    add("track1", "directional_alignment", "per_class_binomial_bh", rate, _family_p(per_class, "binom_p"))
    if eligible:
        pooled = h1_permutation_test(
            dp, dm, pids, n_permutations=n_permutations, eligible_classes=eligible,
            class_list=classes, rng=rngs[3], n_resamples=n_bootstrap,
        )
        cmh = h1_cmh_test(
            dp, dm, eligible, classes, patient_ids=pids, n_permutations=n_permutations,
            perm_rng=rngs[4], null_rng=rngs[5],
        )
        spec1 = context_specificity_pooled(
            dp, dm, ds, pids, eligible, classes, n_resamples=n_bootstrap,
            n_permutations=n_permutations, rng=rngs[6],
        )
        add("track1", "directional_alignment", "pooled_permutation", pooled["aggregate_rate"], pooled["p_value"])
        add("track1", "directional_alignment", "cmh_permutation", cmh["common_odds_ratio"], cmh["p_permutation"])
        add("track1", "directional_alignment", "cmh_sign_flip_vs_half", cmh["common_odds_ratio"], cmh["p_cluster"])
        add("track1", "directional_alignment", "cmh_asymptotic", cmh["common_odds_ratio"], cmh["p_cmh"])
        add("track1", "context_specificity", "pooled_sign_flip", spec1["mean_alignment_difference"], spec1["p_value"])
    brier = h2_wilcoxon_per_class(
        cf, cc, study.reference_correct_context, classes, pids, n_bootstrap, rngs[3],
        n_permutations=n_permutations, perm_rng=rngs[4],
    )
    brier["confirmatory"] = True  # Brier improvement is tested in every class
    improvement = float(brier["improvement"].mean())
    add("track1", "brier_improvement", "per_class_sign_flip_bh", improvement, _family_p(brier, "cluster_p"))
    add("track1", "brier_improvement", "per_class_wilcoxon_bh", improvement, _family_p(brier, "wilcoxon_p"))

    # --- CIs ---
    ci_rows: list[dict] = []
    cells = _cells(study)
    cluster_ci = {
        "context_specificity_difference": (h4["mean_diff"], h4["ci_lower"], h4["ci_upper"]),
        "context_sensitivity_difference": (
            h1["mean_abs_delta_difference"], h1["ci_lower_difference"], h1["ci_upper_difference"],
        ),
    }
    for quantity in CI_QUANTITIES:
        values, _ = cells[quantity]
        est, lo, hi = cluster_ci[quantity]
        ci_rows.append({"quantity": quantity, "method": "patient_cluster", "estimate": est,
                        "ci_lower": lo, "ci_upper": hi})
        lo_i, hi_i = _item_level_ci(values, n_bootstrap, rngs[7])
        ci_rows.append({"quantity": quantity, "method": "item_level", "estimate": est,
                        "ci_lower": lo_i, "ci_upper": hi_i})
    return rows, ci_rows


def population_value(design: SimulationDesign, n_patients: int = 4000, seed: int = 12345) -> dict[str, float]:
    """Monte Carlo estimate of each CI quantity's true value under ``design``.

    Under ``unrelated`` and ``nonspecific`` correct and shuffled context are
    exchangeable, so both values are exactly 0.
    """
    if design.scenario in ("unrelated", "nonspecific"):
        return dict.fromkeys(CI_QUANTITIES, 0.0)
    big = replace(design, n_patients=n_patients)
    cells = _cells(simulate_study(big, np.random.default_rng(seed)))
    return {q: float(np.mean(cells[q][0])) for q in CI_QUANTITIES}


# ---------------------------------------------------------------------------
# Study grid
# ---------------------------------------------------------------------------


def build_designs(patients, items, effects, base: SimulationDesign,
                  clustering_sds=(), attenuations=()) -> list[SimulationDesign]:
    """The main grid, a clustering sub-study and the ``attenuated`` scenario.

    The sub-study repeats the ``unrelated`` scenario at the largest
    items-per-patient with a stronger model patient effect
    (``model_patient_sd``), which correlates a patient's cells more and so
    shows what ignoring the clustering costs.
    """
    designs = []
    for n_p in patients:
        for n_i in items:
            designs.append(replace(base, scenario="unrelated", effect=0.0, n_patients=n_p, items_per_patient=n_i))
            for scenario in ("aligned", "nonspecific", "partial"):
                for eff in effects:
                    designs.append(replace(base, scenario=scenario, effect=eff,
                                           n_patients=n_p, items_per_patient=n_i))
    for sd in clustering_sds:
        for n_p in patients:
            designs.append(replace(base, scenario="unrelated", effect=0.0, n_patients=n_p,
                                   items_per_patient=max(items), model_patient_sd=sd))
    # Last, so that the designs above keep the seeds they had before it was added.
    # Its effect is the attenuation fraction (1.0: no shift at all with another
    # patient's context).
    for n_p in patients:
        for n_i in items:
            for att in attenuations:
                designs.append(replace(base, scenario="attenuated", effect=att,
                                       n_patients=n_p, items_per_patient=n_i))
    return designs


def _run_job(job):
    design, rep, seed, n_bootstrap, n_permutations = job
    study = simulate_study(design, np.random.default_rng(np.random.SeedSequence([seed, 0])))
    rows, ci_rows = evaluate_study(study, n_bootstrap=n_bootstrap, n_permutations=n_permutations,
                                   seed=int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]))
    keys = {"scenario": design.scenario, "effect": design.effect,
            "n_patients": design.n_patients, "items_per_patient": design.items_per_patient,
            "model_patient_sd": design.model_patient_sd, "rep": rep}
    return [{**keys, **r} for r in rows], [{**keys, **r} for r in ci_rows]


_GROUP = ["scenario", "effect", "n_patients", "items_per_patient", "model_patient_sd"]


def summarise_rejections(replicates: pd.DataFrame, alpha: float) -> pd.DataFrame:
    df = replicates.assign(reject=replicates["p_value"] < alpha)
    out = (df.groupby(_GROUP + ["track", "endpoint", "test", "role"], sort=False)
             .agg(n_reps=("reject", "size"), rejection_rate=("reject", "mean"),
                  mean_estimate=("estimate", "mean"))
             .reset_index())
    out["mc_se"] = np.sqrt(out["rejection_rate"] * (1 - out["rejection_rate"]) / out["n_reps"])
    return out


def summarise_coverage(ci: pd.DataFrame, truths: dict) -> pd.DataFrame:
    ci = ci.copy()
    ci["truth"] = [truths[(s, e, m, q)] for s, e, m, q in
                   zip(ci["scenario"], ci["effect"], ci["model_patient_sd"], ci["quantity"])]
    ci["covered"] = (ci["ci_lower"] <= ci["truth"]) & (ci["truth"] <= ci["ci_upper"])
    ci["width"] = ci["ci_upper"] - ci["ci_lower"]
    out = (ci.groupby(_GROUP + ["quantity", "method"], sort=False)
             .agg(n_reps=("covered", "size"), coverage=("covered", "mean"),
                  mean_width=("width", "mean"), truth=("truth", "first"))
             .reset_index())
    out["mc_se"] = np.sqrt(out["coverage"] * (1 - out["coverage"]) / out["n_reps"])
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(rej: pd.DataFrame, cov: pd.DataFrame, fig_dir: Path, alpha: float) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    # Figures 1-3 show the main grid; the clustering sub-study has its own.
    base_sd = SimulationDesign().model_patient_sd
    clustering_rej = rej[(rej["scenario"] == "unrelated")]
    clustering_cov = cov[(cov["scenario"] == "unrelated")]
    rej = rej[rej["model_patient_sd"] == base_sd]
    cov = cov[cov["model_patient_sd"] == base_sd]
    tests = [(t, e, n) for t, e, n, _ in TESTS]
    labels = [f"{t[-1]}·{e}·{n}" for t, e, n in tests]
    items = sorted(rej["items_per_patient"].unique())
    patients = sorted(rej["n_patients"].unique())

    # 1. False-positive rate under the null, every test.
    null = rej[rej["scenario"] == "unrelated"]
    fig, axes = plt.subplots(1, len(items), figsize=(7 * len(items), 7), sharey=True, squeeze=False)
    for ax, n_i in zip(axes[0], items):
        sub = null[null["items_per_patient"] == n_i]
        for n_p, marker in zip(patients, "os^Dv<>"):
            vals = [sub[(sub["n_patients"] == n_p) & (sub["track"] == t) & (sub["endpoint"] == e)
                        & (sub["test"] == n)]["rejection_rate"].mean() for t, e, n in tests]
            ax.plot(vals, range(len(tests)), marker, label=f"{n_p} patients")
        ax.axvline(alpha, color="k", lw=0.8, ls="--")
        ax.set_yticks(range(len(tests)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(f"Rejection rate at α = {alpha} (unrelated scenario: every endpoint null)")
        ax.set_title(f"{n_i} items per patient")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "false_positive_rate.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # 2. Power by scenario, confirmatory tests.
    confirm = [(t, e, n) for t, e, n, r in TESTS if r == "confirmatory"]
    for scenario in ("aligned", "nonspecific", "partial", "attenuated"):
        sub = rej[rej["scenario"] == scenario]
        if sub.empty:
            continue
        effects = sorted(sub["effect"].unique())
        fig, axes = plt.subplots(len(items), len(confirm), figsize=(2.6 * len(confirm), 2.6 * len(items)),
                                 sharex=True, sharey=True, squeeze=False)
        for row, n_i in enumerate(items):
            for col, (t, e, n) in enumerate(confirm):
                ax = axes[row][col]
                for eff in effects:
                    s = sub[(sub["items_per_patient"] == n_i) & (sub["effect"] == eff) & (sub["track"] == t)
                            & (sub["endpoint"] == e) & (sub["test"] == n)].sort_values("n_patients")
                    ax.plot(s["n_patients"], s["rejection_rate"], "o-", ms=3, label=f"effect {eff}")
                ax.axhline(alpha, color="k", lw=0.6, ls="--")
                ax.set_xscale("log")
                ax.minorticks_off()
                ax.set_xticks(patients, [str(p) for p in patients], fontsize=7)
                ax.set_ylim(-0.02, 1.02)
                if row == 0:
                    ax.set_title(f"{t[-1]}·{e}\n{n}", fontsize=7)
                if col == 0:
                    ax.set_ylabel(f"{n_i} items/patient\nrejection rate", fontsize=8)
                if row == len(items) - 1:
                    ax.set_xlabel("patients", fontsize=8)
        axes[0][0].legend(fontsize=6)
        fig.suptitle(f"Scenario: {scenario}")
        fig.tight_layout()
        path = fig_dir / f"power_{scenario}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    # 3. CI coverage, patient-cluster vs item-level.
    fig, axes = plt.subplots(1, len(CI_QUANTITIES), figsize=(6 * len(CI_QUANTITIES), 4), sharey=True)
    for ax, q in zip(np.atleast_1d(axes), CI_QUANTITIES):
        sub = cov[cov["quantity"] == q]
        for method, color in (("patient_cluster", "C0"), ("item_level", "C3")):
            for n_i, ls in zip(items, ("-", "--", ":")):
                s = (sub[(sub["method"] == method) & (sub["items_per_patient"] == n_i)]
                     .groupby("n_patients")["coverage"].mean())
                ax.plot(s.index, s.values, ls, marker="o", color=color, label=f"{method}, {n_i} items")
        ax.axhline(0.95, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(patients, [str(p) for p in patients])
        ax.set_xlabel("patients")
        ax.set_title(q)
    np.atleast_1d(axes)[0].set_ylabel("coverage of nominal 95% CI\n(mean over scenarios)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    path = fig_dir / "ci_coverage.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # 4. Clustering sub-study: what ignoring the patient clusters costs.
    sds = sorted(clustering_cov["model_patient_sd"].unique())
    n_i = clustering_cov.loc[clustering_cov["model_patient_sd"] == sds[-1], "items_per_patient"].max()
    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(11, 4))
    for sd, ls in zip(sds, ("-", "--", ":", "-.")):
        sub = clustering_cov[(clustering_cov["model_patient_sd"] == sd)
                             & (clustering_cov["items_per_patient"] == n_i)
                             & (clustering_cov["quantity"] == "context_specificity_difference")]
        for method, color in (("patient_cluster", "C0"), ("item_level", "C3")):
            s_ = sub[sub["method"] == method].sort_values("n_patients")
            ax_c.plot(s_["n_patients"], s_["coverage"], ls, marker="o", color=color,
                      label=f"{method}, patient SD {sd}")
        r_ = clustering_rej[(clustering_rej["model_patient_sd"] == sd)
                            & (clustering_rej["items_per_patient"] == n_i)
                            & (clustering_rej["endpoint"] == "context_specificity")
                            & (clustering_rej["track"] == "track3")]
        for test, color in (("sign_flip", "C0"), ("cell_level_t_test", "C3")):
            s_ = r_[r_["test"] == test].sort_values("n_patients")
            ax_r.plot(s_["n_patients"], s_["rejection_rate"], ls, marker="o", color=color,
                      label=f"{test}, patient SD {sd}")
    ax_c.axhline(0.95, color="k", lw=0.8, ls="--")
    ax_r.axhline(alpha, color="k", lw=0.8, ls="--")
    for ax, title in ((ax_c, "CI coverage, context_specificity"),
                      (ax_r, "False-positive rate, context_specificity")):
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(patients, [str(p) for p in patients])
        ax.set_xlabel(f"patients ({n_i} items each)")
        ax.set_title(title)
        ax.legend(fontsize=6)
    fig.tight_layout()
    path = fig_dir / "clustering.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="output/simulation")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--patients", type=_ints, default=[6, 10, 20, 40])
    ap.add_argument("--items", type=_ints, default=[10, 30])
    ap.add_argument("--effects", type=_floats, default=[0.25, 0.5, 1.0])
    ap.add_argument("--clustering-sds", type=_floats, default=[1.0, 2.0],
                    help="Model patient-effect SDs for the clustering sub-study (unrelated scenario).")
    ap.add_argument("--attenuations", type=_floats, default=[0.05, 0.1, 0.25, 1.0],
                    help="Attenuation fractions for the attenuated scenario (context_sensitivity power).")
    ap.add_argument("--reps-null", type=int, default=500)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--quick", action="store_true",
                    help="Small grid and few replicates, for a smoke test.")
    ap.add_argument("--figures-only", action="store_true",
                    help="Redraw the figures from the CSVs already in --out.")
    args = ap.parse_args(argv)
    if args.figures_only:
        out = Path(args.out)
        rej = pd.read_csv(out / "rejection_rates.csv", keep_default_na=False)
        cov = pd.read_csv(out / "ci_coverage.csv", keep_default_na=False)
        for path in make_figures(rej, cov, out / "figures", args.alpha):
            print(f"wrote {path}")
        return 0
    if args.quick:
        args.patients, args.items, args.effects, args.clustering_sds = [6, 12], [10], [1.0], [2.0]
        args.attenuations = [0.25]
        args.reps_null, args.reps, args.n_bootstrap, args.n_permutations = 20, 10, 100, 200

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = SimulationDesign()
    designs = build_designs(args.patients, args.items, args.effects, base, args.clustering_sds,
                            args.attenuations)
    seeds = np.random.SeedSequence(args.seed)
    jobs = []
    for design in designs:
        reps = args.reps_null if design.scenario == "unrelated" else args.reps
        for rep, child in enumerate(seeds.spawn(reps)):
            jobs.append((design, rep, int(child.generate_state(1)[0]), args.n_bootstrap, args.n_permutations))

    t0 = time.perf_counter()
    print(f"{len(designs)} designs, {len(jobs)} replicates, {args.workers} workers", flush=True)
    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker) as pool:
            results = pool.map(_run_job, jobs, chunksize=4)
    else:
        results = [_run_job(j) for j in jobs]
    rep_rows = [r for rows, _ in results for r in rows]
    ci_rows = [r for _, rows in results for r in rows]
    print(f"replicates done in {time.perf_counter() - t0:.0f}s", flush=True)

    replicates = pd.DataFrame(rep_rows)
    replicates.to_csv(out / "replicates.csv", index=False)
    rej = summarise_rejections(replicates, args.alpha)
    rej.to_csv(out / "rejection_rates.csv", index=False)

    truths = {}
    for design in designs:
        key = (design.scenario, design.effect, design.model_patient_sd)
        if (*key, CI_QUANTITIES[0]) not in truths:
            for q, v in population_value(design).items():
                truths[(*key, q)] = v
    cov = summarise_coverage(pd.DataFrame(ci_rows), truths)
    cov.to_csv(out / "ci_coverage.csv", index=False)

    (out / "design.json").write_text(json.dumps({
        "seed": args.seed, "alpha": args.alpha, "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations, "reps_null": args.reps_null, "reps": args.reps,
        "patients": args.patients, "items_per_patient": args.items, "effects": args.effects,
        "base_design": base.to_dict(),
        "clustering_sds": args.clustering_sds, "attenuations": args.attenuations,
        "population_values": {f"{s}|{e}|{m}|{q}": v for (s, e, m, q), v in truths.items()},
    }, indent=2))
    for path in make_figures(rej, cov, out / "figures", args.alpha):
        print(f"wrote {path}")
    print(f"wrote {out / 'rejection_rates.csv'} and {out / 'ci_coverage.csv'}")
    return 0


def _init_worker() -> None:
    # One BLAS / torch thread per worker; the parallelism is across replicates.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
