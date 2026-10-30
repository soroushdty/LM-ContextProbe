"""Endpoints computed by Track 1 (representation).

Track 1 used to number these H1 and H2; the numbers are kept as display
aliases. Track 1's H1 is the same question as Track 3's H2. Context
specificity was added after the renaming and has no Track 1 number.
"""

from __future__ import annotations

from shared.endpoints import (
    BRIER_IMPROVEMENT,
    CONTEXT_SPECIFICITY,
    DIRECTIONAL_ALIGNMENT,
    TrackEndpoint,
)

DIRECTIONAL_ALIGNMENT_T1 = TrackEndpoint(DIRECTIONAL_ALIGNMENT, legacy_alias="H1")
BRIER_IMPROVEMENT_T1 = TrackEndpoint(BRIER_IMPROVEMENT, legacy_alias="H2")
CONTEXT_SPECIFICITY_T1 = TrackEndpoint(CONTEXT_SPECIFICITY)

TRACK_ENDPOINTS: tuple[TrackEndpoint, ...] = (
    DIRECTIONAL_ALIGNMENT_T1,
    BRIER_IMPROVEMENT_T1,
    CONTEXT_SPECIFICITY_T1,
)
