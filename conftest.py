"""Repository-wide pytest configuration."""

import os
import sys


# Set this before pytest imports test modules that may import matplotlib.pyplot.
# The GUI backends can leave event-loop processes alive after the suite exits.
os.environ["MPLBACKEND"] = "Agg"


def pytest_sessionfinish() -> None:
    """Release any figures left open by plotting tests."""
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")
