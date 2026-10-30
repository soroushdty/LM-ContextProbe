# LM-ContextProbe


**LM-ContextProbe** is an evaluation framework for testing whether language models change their judgments with patient context the way physicians do. The built-in task is classifying EHR items into sensitive-data categories; the label set is configurable.

The framework has three parts:

- **A study design.** Human reference observers label each item twice: first **without** patient context, then **with** the patient's clinical snapshot. Both tracks add a shuffled (wrong-patient) context as a control.
- **Endpoints with statistical tests.** How much a model's judgments move when context is added, whether they move the same way the human judgments did, and whether that depends on the context belonging to the right patient. Inference is clustered by patient ([`docs/methodology.md`](docs/methodology.md)).
- **Pipelines that run the design** on any model you configure and any dataset in the paired-context format, or in another format through an adapter.

It is not a benchmark: there is no fixed dataset or leaderboard. You bring the data and the model, and the results describe that dataset's patients and reference observers.

The framework is a **functional analogy to physician decision-making**. Each evaluation track stands for one way a physician could reach a context-dependent judgment: drawing on general clinical knowledge (representation), learning from supervised case experience (adaptation), or deliberating over the chart at decision time (reasoning). Comparing the tracks against the physician reference tests which mechanism reproduces the physicians' context shifts. It does not claim to model how physicians think.

> This repository contains code and a fully **synthetic** example dataset only. It does not include data from any human-subjects study.

## Quickstart

```bash
python -m pip install -r requirements.txt
export PYTHONHASHSEED=42

# Track 3 (reasoning) on the synthetic example, no LLM calls
python main.py --track reasoning -- --config configs/reasoning_config.yaml --dry_run --model dry_run_example
```

Outputs go to `output/reasoning/<run_id>/`. To use your own data, see [`docs/data_format.md`](docs/data_format.md).

## Architecture

The code is organized into three top-level packages (full tree under [Project Structure](#project-structure)):

- **`shared/`**: dataset-agnostic utilities, preprocessing, statistics, evaluation, and the adapter contracts (`shared/adapters/base.py`).
- **`tracks/`**: the evaluation pipelines: Track 1 representation and Track 3 reasoning. Track 2 (adaptation) is planned ([#13](https://github.com/soroushdty/LM-ContextProbe/issues/13)).
- **`adapters/`**: dataset adapters that translate a dataset into the framework's canonical objects. The built-in `paired_context` adapter serves both tracks.

Dependencies point one way: `tracks/` and `adapters/` import from `shared/`; `shared/` imports from neither, and `adapters/` never imports from `tracks/`. Architecture tests in `tests/architecture/` enforce this.

### Tracks

| Track | Physician analogue | Question | Status |
|-------|--------------------|----------|--------|
| 1 – Representation | General clinical knowledge, before any task-specific training | Is the context shift already latent in pretrained knowledge? | Available |
| 2 – Adaptation | Learning from supervised case experience (clinical training, feedback on real cases) | Can the shift be learned from labelled cases and carried over to new patients? | Planned ([#13](https://github.com/soroushdty/LM-ContextProbe/issues/13)) |
| 3 – Reasoning | Deliberating over the chart at decision time, without changing what the model knows | Does the shift come from reasoning at decision time? | Available |

- **[Track 1 — Representation](tracks/representation/README.md)** (`tracks/representation/`): embedding pipeline (default model Bio_ClinicalBERT; decoder LLMs also work, see [`shared/embeddings/`](shared/embeddings/README.md)), multilabel classifier, context-aware fusion head, patient-grouped cross-validation (leave-one-patient-out or grouped k-fold), and statistical analysis of the `directional_alignment`, `brier_improvement` and `context_specificity` endpoints. For the shuffled-context control, patients are paired and each item is also scored with the partner's context vector, by a Stage 2 head trained without either patient.
- **[Track 3 — Reasoning](tracks/reasoning/README.md)** (`tracks/reasoning/`): LLM prompting pipeline. Runs each EHR item under three conditions (context-free, correct-context, shuffled-context) and computes the four Track 3 endpoints against physician judgment shifts.

### Shared Layer

The [`shared/`](shared/) directory contains cross-track utilities, preprocessing, statistics, evaluation, and adapter contracts. Both tracks import from `shared.*` for general-purpose logic. See the subdirectory READMEs for details on each module.

---

## Research Overview

Every track uses the same **paired counterfactual design**. Human reference observers (for example, physicians working in fixed pairs) label EHR item strings into sensitive-data categories under two conditions. First they see the item without patient information (context-free, `reference_context_free`). Then they see it with a structured clinical snapshot of the patient (correct-context, `reference_correct_context`). These matched counterfactual labels allow a direct measurement of context-induced judgment shifts (`delta_reference`).

The central question is: **Does a model change its privacy-category judgment when patient context is added, and do those changes match the human judgment shifts?**

The endpoints have the same names in every track (`shared/endpoints.py`; statistics and tests per track are in [`docs/methodology.md`](docs/methodology.md#endpoints)):

- **`context_sensitivity`**: Does the correct patient's context move the model's scores more than another patient's (shuffled) context does? Compares mean absolute deltas (context minus context-free) under the two contexts. *(Track 3; formerly H1)*
- **`directional_alignment`**: Do the model's context-induced deltas agree in sign with the physicians' judgment deltas? *(Track 3, formerly H2; Track 1, formerly H1)*
- **`class_correspondence`**: Do class-level mean model deltas correlate with class-level mean physician deltas across the categories? *(Track 3; formerly H3)*
- **`context_specificity`**: Does the correct patient's context produce stronger alignment with the physician deltas than a shuffled (wrong-patient) context? *(Track 3, formerly H4; Track 1)*
- **`brier_improvement`**: Does context lower the Brier score against the correct-context reference labels? *(Track 1; formerly H2)*

The default label space is the ten sensitive-data categories used in the SHARES project: behavioral_health, diagnoses, disabilities, infectious_diseases, genetics, medications, sexual_reproductive_health, social_determinants_of_health, violence, other. Other taxonomies are set in config (`classes` and `class_definitions`; see [`docs/adapters.md`](docs/adapters.md#label-space-sharedlabel_spacepy)).

**Scope**: Claims are scoped to the patients and reference observers of the dataset being evaluated. The endpoints measure context-induced shifts and their agreement with the reference shifts, not classification accuracy. The one exception is Track 1's `brier_improvement`, which tests whether context lowers the Brier score against the correct-context reference labels.

Formal definitions of every endpoint, the unit of analysis, the statistical tests, and the design decisions behind them are in [`docs/methodology.md`](docs/methodology.md).

## Origins and Acknowledgments

LM-ContextProbe grew out of work in Dr. Adela Grando's **SHARES** project at Arizona State University on patient-controlled, granular segmentation of sensitive health data. Its paired context-free / correct-context design is modeled on the study by Kaufman et al., in which physicians categorized EHR items first without and then with additional patient context:

- Kaufman HJ, Banerjee I, Wei M, et al.; Grando MA. *New Automated Granular Data Segmentation Approach: Context Impacts Categorization by Physicians.* JMIR Preprints #66059, 2024. [doi:10.2196/preprints.66059](https://doi.org/10.2196/preprints.66059)
- Lee P, Dhadwal AS, Kaiser M, Dianaty S, et al.; Grando A. *Assessing the Effectiveness and Scalability of FHIR-Based Granular Data Segmentation Technology.* Appl Clin Inform. 2026;17(3):423–433. [doi:10.1055/a-2863-4129](https://doi.org/10.1055/a-2863-4129)

An early version of this analysis was presented as: Dianaty S, Kaiser M, Murcko A, Grando A. *Early Evidence for Context-Aware Large Language Models (LLMs) in Sensitive Health Data Classification.* AcademyHealth Annual Research Meeting 2026, Seattle, WA, May 30 – Jun 2, 2026.

The study data is not distributed with LM-ContextProbe. The bundled example dataset is synthetic.

## Project Status

LM-ContextProbe 0.2.0 is an early public release of an evaluation framework, and the first archived one. Its endpoints and tests are protocol v1 (see [`docs/methodology.md`](docs/methodology.md#protocol-version)). It has not yet been validated on real data or by simulation.

- **Tested:** both tracks run end to end on the bundled synthetic dataset, under CI.
- **Not yet available:** results on real data, simulation studies of the endpoints' false-positive rate and power ([#18](https://github.com/soroushdty/LM-ContextProbe/issues/18)), and a second dataset format. Known gaps in the current statistics are listed in [`docs/methodology.md`](docs/methodology.md#known-limitations).

**Development history.** LM-ContextProbe began as the analysis code for the study described under [Origins and Acknowledgments](#origins-and-acknowledgments) and was developed in a private repository from January to June 2026 (about 800 commits). It was then generalized into a dataset-agnostic framework. This public repository was created from a sanitized snapshot, so that study data and private development material are not in its git history. A short public commit history does not reflect the age of the code.

## Track 3 — Reasoning

Track 3 runs each EHR item through a Hugging Face LLM under three conditions:

1. **Context-free**: Item text only, no patient information.
2. **Correct-context**: Item text plus the correct patient's clinical snapshot.
3. **Shuffled-context**: Item text plus a randomly selected different patient's clinical snapshot (control).

For each condition the LLM outputs a JSON object with a probability score for every category in the label space (the ten SHARES categories by default). Track 3 then computes physician and LLM judgment deltas and the four Track 3 endpoints.

Canonical entry point: `tracks/reasoning/run_reasoning.py`. See [`tracks/reasoning/README.md`](tracks/reasoning/README.md) for full details.

## Installation

```bash
python -m pip install -r requirements.txt
```

Requires Python >= 3.11 (tested on 3.11 and 3.12).

## Usage

### Track 3 — Reasoning (dry run)

```bash
python main.py --track reasoning -- --config configs/reasoning_config.yaml --dry_run --model dry_run_example
```

The dry run generates deterministic mock scores and complete artifact trees without calling an LLM. See [`docs/running_example.md`](docs/running_example.md) for full guidance.

### Track 1 — Representation

```bash
python main.py --track representation -- --config configs/main_config.yaml
```

Runs the representation learning pipeline: preprocessing, embedding computation, ensemble training with LOPO-CV, and statistical hypothesis analysis.

The first run downloads the `emilyalsentzer/Bio_ClinicalBERT` embedding model (~440 MB) from Hugging Face.

### Track 3 with a real LLM

```bash
python main.py --track reasoning -- --config configs/reasoning_config.yaml --model meta-llama/Llama-3.1-8B-Instruct
```

Requires a valid `HF_TOKEN` environment variable.

### Track 1 general usage

```bash
python main.py --track representation -- [--dir .] [--hf_token TOKEN] [--config PATH]
```

Config resolution order when `--config` is omitted:

1. A directory named `configs/` in the repo root — if it contains `main_config.yaml`, use it.
2. Recursive search for `main_config.yaml` across the repo.
3. Error if not found.

Tokens may also be supplied via environment variables:

- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` — HuggingFace API token

CLI args take precedence over environment variables when both are provided.

### Inject a custom Track 1 strategy

`tracks/representation/framework.py` composes a dataset adapter, a training strategy, and zero or more artifact writers through dependency injection. To plug in a custom strategy:

```python
from tracks.representation.framework import RepresentationTrack
from tracks.representation.strategies.base import TrainingStrategy, RepresentationDataset
from adapters.paired_context import PairedContextRepresentationAdapter

class MyStrategy:
    def fit_and_evaluate(self, dataset: RepresentationDataset, config: dict) -> dict:
        # ... your training/evaluation code ...
        return {"metrics": {...}, "predictions": {...}}

track = RepresentationTrack(
    adapter=PairedContextRepresentationAdapter(),
    strategy=MyStrategy(),
    artifact_writers=[...],
)
track.run(config)
```

See [`docs/architecture.md`](docs/architecture.md) for the adapter/strategy protocols and [`docs/adapters.md`](docs/adapters.md) for writing a new dataset adapter.

### Notebook

For Google Colab, Jupyter, or HPC: open `main_notebook.ipynb`.

## Track 3 Pipeline Stages

| Stage                    | Description                                                                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Preprocessing**        | Loads the dataset workbook, standardizes item text via fuzzy matching, aggregates physician labels by pair, generates summary artifacts          |
| **Dataset Loading**      | Constructs paired physician consensus labels (survey and interview) and computes Delta_Physician (`delta_reference`) for all patient–item pairs  |
| **Context Building**     | Builds correct-context and shuffled-context patient snapshots with label leakage prevention                                                     |
| **LLM Prompting**        | Runs each item under three conditions (context-free, correct-context, shuffled-context) through a Hugging Face LLM                              |
| **Score Parsing**        | Validates LLM JSON outputs against the privacy-category schema and exports per-condition score CSVs                                             |
| **Hypothesis Analysis**  | Computes the four Track 3 endpoints with patient-cluster bootstrap CIs and permutation tests                                                    |
| **Report Generation**    | Produces a markdown analysis report with all results, per-category breakdowns, and a limitations section                                        |

## Configuration

All configuration files live in `configs/`.

### Track 3 Configuration

`configs/reasoning_config.yaml` controls Track 3:

| Key | Purpose |
|-----|---------|
| `backend` | `huggingface`, `local_transformers`, or `dry_run` |
| `model_ids` | List of Hugging Face model IDs to evaluate |
| `hf_token_env` | Environment variable name for the HF token |
| `temperature`, `max_tokens`, `top_p`, `seed` | LLM generation parameters |
| `shuffled_context_seed`, `bootstrap_seed`, `permutation_seed` | Reproducibility seeds |
| `n_bootstrap_resamples`, `n_permutations` | Statistical test parameters |
| `output_dir`, `cache_dir`, `scores_dir`, `reports_dir` | Output paths |

### Track 1 / Representation Configuration

Three YAML files control the Stage-1/Stage-2 pipeline:

| File                             | Purpose                                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| `configs/main_config.yaml`       | Dataset paths, schema column names, label space, embedding model, preprocessing controls |
| `configs/training_config.yaml`   | Model architecture, training hyperparameters, threshold strategy, HP search settings     |
| `configs/summary_config.yaml`    | Which preprocessing summary artifacts to generate (individual toggles per output)        |

## Roadmap

Planned work is tracked in [GitHub issues](https://github.com/soroushdty/LM-ContextProbe/issues). None of it is available yet:

- **Context perturbation suite**: field-level ablation, irrelevant-context insertion, field reordering, and item paraphrasing for the reasoning track, to test whether context shifts come from clinically relevant content.
- **TRIPOD-LLM report**: an auto-filled TRIPOD-LLM checklist per run, with links to the supporting artifacts and explicit TODOs for author-only items.
- **Multi-agent reference emulation**: blind LLM rater pairs with consensus and adjudication, mirroring the paired-physician reference design (aligned with the EviTrace multi-agent roadmap).
- **Track 2 — Adaptation** ([#13](https://github.com/soroushdty/LM-ContextProbe/issues/13)): the analogue of a physician learning from supervised case experience. A model is fine-tuned on labelled (patient context, item) cases from training patients, then evaluated on new patients under the same three conditions as Track 3. A learning curve shows how alignment with physicians grows with the number of patients learned from.
- **Same-model comparison across tracks** ([#12](https://github.com/soroushdty/LM-ContextProbe/issues/12)): one model family run frozen, fine-tuned, and prompted, so differences between tracks reflect the mechanism rather than the model.
- **Distribution-shift evaluation**: requires a second dataset in the paired-context format.
- **Other inputs and backends**: FHIR/OMOP ingestion, confidence-weighted or adjudicated reference aggregation, and hosted-API LLM backends.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — Package boundaries and dependency rules
- [`docs/data_format.md`](docs/data_format.md) — Input files expected by the paired-context adapter
- [`docs/adapters.md`](docs/adapters.md) — Adapter responsibilities and the paired-context adapter
- [`docs/running_example.md`](docs/running_example.md) — Commands, outputs, and caveats for the synthetic example
- [`docs/artifact_contracts.md`](docs/artifact_contracts.md) — Required output artifacts and schemas
- [`docs/data_model.md`](docs/data_model.md) — Canonical vocabulary and data model
- [`docs/methodology.md`](docs/methodology.md) — Endpoints, statistical tests, protocol version, design rationale, and known limitations
- [`docs/releasing.md`](docs/releasing.md) — Release checklist: protocol version, Zenodo archiving and DOI

## Project Structure

```
LM-ContextProbe/
├── shared/                    # Dataset-agnostic code shared by both tracks
│   ├── adapters/              # Adapter contracts (protocols, RepresentationDataset)
│   ├── io/                    # Dataset-neutral Excel reading
│   ├── utils/                 # General-purpose utilities (array, file, JSON, mapping, path, text)
│   ├── prerequisites/         # Config loading, reproducibility guards, logging setup
│   ├── preprocessing/         # Shared preprocessing (fuzzy matching, index mapping, standardization)
│   ├── embeddings/            # Embedding computation
│   ├── reference/             # Reference observer aggregation and audit diagnostics
│   ├── reporting/             # Input summary reporting infrastructure
│   ├── statistical/           # Shared metrics (bootstrap, ICC, entropy, delta, tau)
│   └── evaluation/            # Shared evaluation (hypothesis analysis, report generation, calibration)
├── tracks/                    # Track-specific pipelines
│   ├── representation/        # Track 1: embedding + training + statistical analysis
│   │   ├── strategies/        # Injectable training strategies
│   │   ├── training/          # Ensemble pipeline, calibration, CV, threshold tuning
│   │   ├── models/            # EnsemblePredictor, MultiLabelModel, ModelRegistry
│   │   └── statistical/       # Endpoint tests, run_analysis orchestrator, reporting
│   └── reasoning/             # Track 3: LLM reasoning pipeline
│       └── bundle/            # Post-hoc analysis bundle (deltas, endpoints, report)
├── adapters/                  # Dataset adapters
│   └── paired_context/        # Paired-context adapter (column maps, labels, Track 1 and Track 3 adapters)
├── configs/                   # All YAML configuration files
├── examples/synthetic/        # Synthetic example dataset
├── scripts/                   # Synthetic dataset generator; post-hoc Track 3 analysis bundle
├── docs/                      # Architecture, data format, adapters, artifact contracts
├── tests/                     # Test suite (architecture, shared, tracks, adapters, audits)
├── data/                      # Local input data (git-ignored)
├── output/                    # Pipeline outputs (git-ignored)
├── main.py                    # CLI dispatcher (--track representation|reasoning)
└── main_notebook.ipynb        # Colab/Jupyter launcher
```

## Requirements

- Python >= 3.11
- PyTorch 2.x, Transformers 5.x, scikit-learn, pandas, numpy, scipy
- See `requirements.txt` for pinned versions

## Reproducibility

For full reproducibility, **PYTHONHASHSEED must be set as an environment variable before launching Python**:

```bash
export PYTHONHASHSEED=42
python main.py ...
```

Setting `PYTHONHASHSEED` at runtime (e.g., via `os.environ` in code) has no effect. This is required for deterministic hashing and reproducible results.

## Citation

If you use LM-ContextProbe, please cite it using the metadata in [`CITATION.cff`](CITATION.cff), along with the Kaufman et al. study listed under [Origins and Acknowledgments](#origins-and-acknowledgments).

## License

Licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026 Soroush Dianaty.
