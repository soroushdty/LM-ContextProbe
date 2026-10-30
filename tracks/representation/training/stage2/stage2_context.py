import logging
from copy import deepcopy
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ablate_field(ctx_dict: dict, field_name: str) -> dict:
    """Return a copy of ctx_dict with ctx_dict[field_name] replaced by an empty string."""
    ctx_copy = deepcopy(ctx_dict)
    if field_name in ctx_copy:
        ctx_copy[field_name] = ""
    return ctx_copy


def build_ablated_context_vector(patient_ctx: dict, field_name: str, model_id: str, cfg: dict) -> np.ndarray:
    """
    Return the context embedding for a single patient with the specified field ablated (set to "").
    This re-encodes the context string using the user-configured fusion.
    """
    from shared.embeddings.compute_embeddings import compute_embeddings
    ablated_ctx = ablate_field(patient_ctx, field_name)
    ctx_str = build_context_string(ablated_ctx)
    emb_dict = compute_embeddings(model_id, pd.Series([ctx_str]), schema="pandas", cfg=cfg)
    return emb_dict[ctx_str]

_CONTEXT_FIELDS = [
    "summary", "medical_history", "allergies", "medication_history",
    "social_history", "labs", "radiology", "procedures",
]

_CONTEXT_TEMPLATE = (
    "Summary: {summary}. Medical History: {medical_history}. "
    "Allergies: {allergies}. Medication History: {medication_history}. "
    "Social History: {social_history}. Labs: {labs}. "
    "Radiology: {radiology}. Procedures: {procedures}."
)


def build_context_string(patient_ctx: dict) -> str:
    """Serialize eight sub-fields into one canonical string."""
    return _CONTEXT_TEMPLATE.format(
        **{k: str(patient_ctx.get(k, "")) for k in _CONTEXT_FIELDS}
    )


def build_context_vectors(
    context_json: dict,   # {str(patient_id): {"summary": ..., ...}}
    model_id: str,
    cfg: dict,
) -> dict:                # {patient_id_int: np.ndarray shape (d,)}
    """Encode every patient's context string through the frozen encoder.

    Calls compute_embeddings with schema='pandas' and pooling='mean'.
    Called ONCE before the outer fold loop; result cached for all folds.
    """
    from shared.embeddings.compute_embeddings import compute_embeddings

    pid_to_str = {}
    for pid_str, ctx in context_json.items():
        try:
            pid_int = int(float(pid_str))
        except (ValueError, TypeError):
            logger.warning("Skipping non-numeric patient ID in context JSON: %r", pid_str)
            continue
        pid_to_str[pid_int] = build_context_string(ctx)

    if not pid_to_str:
        return {}

    series = pd.Series(list(pid_to_str.values()))
    emb_dict = compute_embeddings(model_id, series, schema="pandas", cfg=cfg)

    result = {}
    for pid_int, ctx_str in pid_to_str.items():
        if ctx_str in emb_dict:
            result[pid_int] = emb_dict[ctx_str]
        else:
            logger.warning("No embedding found for patient %d context string.", pid_int)
    return result


def pair_patients_for_context_swap(patient_ids, seed: int) -> list[list[int]]:
    """Group patients for the shuffled-context control, which swaps contexts within a group.

    Patients are shuffled with ``seed`` and dealt into pairs; with an odd
    number, the last group has three. Within a group, each patient is given
    the next patient's context (the last is given the first's), so no
    patient gets its own. Track 1 fits one Stage 2 head per group without any
    of the group's patients, so the head has seen neither the patient's own
    context nor the one it is swapped for.

    Args:
        patient_ids: Patients to group (duplicates are ignored).
        seed:        Seed for the grouping (``shuffled_context_seed``).

    Returns:
        Groups of patient IDs, each of length 2 or 3. Empty with fewer than
        two patients.
    """
    pool = np.array(sorted({int(float(p)) for p in patient_ids}), dtype=int)
    if pool.size < 2:
        return []
    order = np.random.default_rng(seed).permutation(pool).tolist()
    groups = [order[i:i + 2] for i in range(0, len(order) - len(order) % 2, 2)]
    if len(order) % 2:
        groups[-1].append(order[-1])
    return groups


def _build_2d_fusion(e_i: np.ndarray, c_p: np.ndarray) -> np.ndarray:
    return np.concatenate([e_i, c_p])


def _build_3d_fusion(e_i: np.ndarray, c_p: np.ndarray) -> np.ndarray:
    return np.concatenate([e_i, c_p, e_i * c_p])


def _build_lowrank_bilinear_fusion(e_i: np.ndarray, c_p: np.ndarray, r: int = 8) -> np.ndarray:
    # Return [e_i ‖ c_p] so the LowRankBilinearFusionModel can split them and
    # learn U_c, V_c per class. Random projections are NOT used here because
    # ŷ_c = e_i^T (U_c V_c^T) c_p requires learned (not fixed) per-class factors.
    return np.concatenate([e_i, c_p])


def build_fusion_features(
    e_i: np.ndarray,
    c_p: np.ndarray,
    fusion_strategy: str = "2d",
    r: int = 8,
) -> np.ndarray:
    if fusion_strategy == "3d":
        return _build_3d_fusion(e_i, c_p)
    elif fusion_strategy == "lowrank_bilinear":
        return _build_lowrank_bilinear_fusion(e_i, c_p, r)
    elif fusion_strategy == "4_vector":
        return np.concatenate([e_i, c_p, e_i * c_p, np.abs(e_i - c_p)])
    return _build_2d_fusion(e_i, c_p)


def build_fusion_matrix(
    X_items: np.ndarray,
    patient_ids: np.ndarray,
    context_vectors: dict,
    fusion_strategy: str = "2d",
    r: int = 8,
) -> np.ndarray:
    """Build the full fusion matrix row by row.

    If a patient's context vector is missing, use zeros and log a warning.
    """
    n, d = X_items.shape
    rows = []
    warned = set()
    for idx in range(n):
        pid = int(float(patient_ids[idx]))
        c_p = context_vectors.get(pid)
        if c_p is None:
            if pid not in warned:
                logger.warning(
                    "No context vector for patient %d — using zeros.", pid
                )
                warned.add(pid)
            c_p = np.zeros(d, dtype=X_items.dtype)
        if c_p.shape[0] != d:
            raise ValueError(
                f"Context vector dimension {c_p.shape[0]} != item embedding "
                f"dimension {d} for patient {pid}. Both must use the same "
                "frozen encoder and pooling."
            )
        rows.append(build_fusion_features(X_items[idx], c_p, fusion_strategy, r))
    return np.vstack(rows).astype(np.float32)
