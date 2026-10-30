import matplotlib
matplotlib.use("Agg")

import os

import pytest
import torch

from tracks.representation.models.ModelRegistry import get_registry

# Under pytest-xdist each worker would start one torch thread per core, so N
# workers oversubscribe the CPU N-fold and timing tests (and everything else)
# slow down by an order of magnitude. One thread per worker keeps the total at
# one per core.
if os.environ.get("PYTEST_XDIST_WORKER"):
    torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def clear_model_registry():
    """Clear the ModelRegistry singleton before every test.

    Prevents cross-test contamination: bundles written to the same output
    path by different tests could produce a stale cache hit when file mtimes
    coincide (possible on NTFS whose resolution is 100 ns).
    """
    get_registry().clear()
    yield
    get_registry().clear()
