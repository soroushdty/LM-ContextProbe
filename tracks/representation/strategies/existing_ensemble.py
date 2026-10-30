"""Existing ensemble training strategy wrapper.

Wraps the existing LOPO-CV ensemble training pipeline as a TrainingStrategy
implementation. Delegates all training logic to ``train_ensemble_pipeline()``,
builds the Stage 2 context vectors it needs, and runs the statistical analysis
over its predictions when ``statistical_analysis.enabled`` is set.

Canonical path: tracks/representation/strategies/existing_ensemble.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import RepresentationDataset

logger = logging.getLogger(__name__)


class ExistingEnsembleTrainingStrategy:
    """Wraps the existing LOPO-CV ensemble training pipeline as a TrainingStrategy.

    Delegates all training logic to ``train_ensemble_pipeline()``. This wrapper
    handles the translation between ``RepresentationDataset`` fields and the
    legacy function signature, Stage 2 context-vector construction, and the
    post-training statistical analysis.

    Field mapping
    -------------
    dataset.X_train               -> X_train
    dataset.Y_train               -> Y_train
    dataset.patient_ids_train     -> patient_ids_train
    dataset.X_test                -> X_test
    dataset.Y_test                -> Y_test
    dataset.patient_ids_test      -> patient_ids_test
    dataset.Y_test_context_free   -> Y_test_survey
    dataset.Y_test_correct_context -> Y_test_interview
    dataset.context_vectors       -> context_vectors (else built from
                                     dataset.context_records)

    When ``config['fuzzy_threshold']`` is set, the item strings, unresolved
    masks, and embedding cache are also passed so the pipeline can run its
    train-only fuzzy fallback.

    Parameters
    ----------
    resume_from_checkpoint : bool
        Resume a partial run from the checkpoints under ``config['DIR_MODEL']``.
    """

    def __init__(self, *, resume_from_checkpoint: bool = False) -> None:
        self._resume_from_checkpoint = resume_from_checkpoint

    def fit_and_evaluate(
        self, dataset: RepresentationDataset, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Train the ensemble, then run the statistical analysis if enabled.

        Parameters
        ----------
        dataset : RepresentationDataset
            The canonical data container produced by an adapter.
        config : dict[str, Any]
            Pipeline configuration dict (passed as ``cfg``).

        Returns
        -------
        dict[str, Any]
            The result dict from ``train_ensemble_pipeline``.
        """
        # Deferred import to avoid heavyweight dependencies at module load time
        from tracks.representation.training.orchestrator.train_ensemble_pipeline import (
            train_ensemble_pipeline,
        )

        context_vectors = self._context_vectors(dataset, config)

        fuzzy = bool(config.get("fuzzy_threshold"))
        result = train_ensemble_pipeline(
            X_train=dataset.X_train,
            Y_train=dataset.Y_train,
            patient_ids_train=dataset.patient_ids_train,
            X_test=dataset.X_test,
            Y_test=dataset.Y_test,
            cfg=config,
            item_strings_train=dataset.item_strings_train if fuzzy else None,
            item_strings_test=dataset.item_strings_test if fuzzy else None,
            unresolved_mask_train=dataset.unresolved_mask_train if fuzzy else None,
            unresolved_mask_test=dataset.unresolved_mask_test if fuzzy else None,
            embedding_cache=dataset.embedding_cache if fuzzy else None,
            resume_from_checkpoint=self._resume_from_checkpoint,
            patient_ids_test=dataset.patient_ids_test,
            Y_test_interview=dataset.Y_test_correct_context,
            Y_test_survey=dataset.Y_test_context_free,
            context_vectors=context_vectors,
        )

        stat_cfg = config.get("statistical_analysis", {}) or {}
        if stat_cfg.get("enabled", False):
            self._run_statistical_analysis(dataset, config, result, stat_cfg)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _context_vectors(
        dataset: RepresentationDataset, config: dict[str, Any]
    ) -> dict | None:
        """Return the dataset's context vectors, building them from records if needed."""
        if dataset.context_vectors is not None or dataset.context_records is None:
            return dataset.context_vectors

        from tracks.representation.training.stage2.stage2_context import (
            build_context_vectors,
        )

        return build_context_vectors(dataset.context_records, config["llm"], config)

    def _run_statistical_analysis(
        self,
        dataset: RepresentationDataset,
        config: dict[str, Any],
        result: dict[str, Any],
        stat_cfg: dict[str, Any],
    ) -> None:
        """Run the confirmatory statistical analysis over the ensemble predictions.

        Setup errors (before the analysis starts) are logged and swallowed;
        errors raised by the analysis itself propagate.
        """
        logger.info("Starting statistical analysis...")
        hard_error = False
        try:
            from tracks.representation.statistical.orchestrator.run_analysis import (
                run_statistical_analysis,
            )

            model_dir = Path(config.get("DIR_MODEL", "output/model"))
            output_dir = model_dir.parent / "statistical_analysis"
            y_hat_cf = result.get("test_probs_cf_fold_pure")
            if y_hat_cf is None:
                y_hat_cf = result.get("test_probs_cf")
            y_hat_ca = result.get("test_probs_ca_fold_pure")
            avg_thresh = result.get("avg_thresh_f1opt")

            if y_hat_ca is None:
                raise RuntimeError(
                    "Statistical analysis requires fold-pure Stage 2 predictions "
                    "(test_probs_ca_fold_pure).\n"
                    "These are required for confirmatory statistics.\n"
                    "Ensure the training pipeline completed successfully and "
                    "fold-pure predictions are available."
                )
            if y_hat_cf is None:
                logger.warning(
                    "test_probs_cf_fold_pure and test_probs_cf are both None — "
                    "Stage 2 may not have been enabled. "
                    "Using fold-pure Stage 2 predictions as context-free fallback."
                )
                y_hat_cf = y_hat_ca
            if avg_thresh is None:
                avg_thresh = np.full(
                    len(dataset.output_dimensions), float(config.get("tau", 0.5))
                )

            metadata = dataset.metadata or {}
            hard_error = True

            run_statistical_analysis(
                y_survey=dataset.Y_test_context_free,
                y_interview=dataset.Y_test_correct_context,
                y_hat_cf=y_hat_cf,
                y_hat_ca=y_hat_ca,
                patient_ids=dataset.patient_ids_test,
                item_texts=dataset.item_strings_test,
                class_list=dataset.output_dimensions,
                avg_thresh_f1opt=avg_thresh,
                tau_fixed=float(config.get("tau", 0.5)),
                dataset_path=Path(metadata["dataset_path"]),
                sheet_names=metadata["sheet_names"],
                context_json=dataset.context_records,
                llm=config["llm"],
                cfg=config,
                output_dir=output_dir,
                n_resamples=int(stat_cfg.get("n_resamples", 1000)),
                n_permutations=int(stat_cfg.get("n_permutations", 10000)),
                item_texts_train=(
                    dataset.item_strings_train
                    if config.get("fuzzy_threshold")
                    else None
                ),
                ensemble_bundle_path=model_dir / "ensemble_bundle.joblib",
                arch_predictions=result.get("arch_predictions"),
                X_items_test=dataset.X_test,
                y_hat_ca_leave_pair_out=result.get("test_probs_ca_leave_pair_out"),
                y_hat_shuffled=result.get("test_probs_shuffled_leave_pair_out"),
            )
            logger.info("Statistical analysis complete. Outputs at: %s", output_dir)
        except Exception:
            if hard_error:
                logger.critical(
                    "Statistical analysis failed mid-run — required outputs may be incomplete.",
                    exc_info=True,
                )
                raise
            logger.exception(
                "Statistical analysis could not start (setup error). "
                "Check that the training pipeline produced valid outputs."
            )
