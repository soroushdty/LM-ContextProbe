"""
shared.simulation — Simulated paired-context studies with planted effects.

Canonical path: shared/simulation/
"""

from .generator import (  # noqa: F401
    SCENARIOS,
    SimulatedStudy,
    SimulationDesign,
    simulate_study,
)

__all__ = [
    "SCENARIOS",
    "SimulatedStudy",
    "SimulationDesign",
    "simulate_study",
]
