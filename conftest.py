"""Repository-wide pytest configuration."""

import os
import sys
from pathlib import Path

import pytest


# Set this before pytest imports test modules that may import matplotlib.pyplot.
# The GUI backends can leave event-loop processes alive after the suite exits.
os.environ["MPLBACKEND"] = "Agg"


def pytest_sessionfinish() -> None:
    """Release any figures left open by plotting tests."""
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


@pytest.fixture(scope="session")
def shared_cache_root():
    """A persistent scratch cache root shared by every test, worker and run.

    Metrics on a legacy-scenario config read the real student tables through the
    content-addressed student cache. A per-test cache root misses it every time
    and re-parses seven years of students (~35 s a test). Entries are keyed by
    source contents and written under flock, so sharing one root across
    pytest-xdist workers and across runs is safe and a stale entry cannot hit.
    Override the location with SFUSD_TEST_CACHE_ROOT. Tests that exercise cache
    isolation itself keep their own root.
    """
    default = Path.home() / ".cache" / "sfusd-zone" / "test-cache"
    root = Path(os.environ.get("SFUSD_TEST_CACHE_ROOT", default))
    root.mkdir(parents=True, exist_ok=True)
    return root
