"""Unit tests for the Pareto-frontier extraction logic."""

import sys
from pathlib import Path

import pandas as pd

# scripts/analysis is not a package; add it to the path to import the module.
_SCRIPTS_ANALYSIS = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
if str(_SCRIPTS_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ANALYSIS))

from plot_simulation_frontier import compute_pareto_frontier  # noqa: E402


def _points():
    """A small scatter with a known lower-left frontier.

    Returns:
        DataFrame with columns label/x/y.
    """
    return pd.DataFrame(
        {
            "label": ["A", "B", "C", "D", "E"],
            "x": [1.0, 2.0, 3.0, 4.0, 2.0],
            "y": [5.0, 3.0, 4.0, 1.0, 4.0],
        }
    )


def test_frontier_min_min():
    """Both axes minimized: keep the lower-left non-dominated points."""
    frontier = compute_pareto_frontier(_points(), "x", "y")
    assert set(frontier["label"]) == {"A", "B", "D"}
    # Output is ordered by ascending x.
    assert list(frontier["x"]) == [1.0, 2.0, 4.0]


def test_frontier_drops_dominated_and_ties():
    """C is dominated by B; E ties B on x with worse y -> both dropped."""
    frontier = compute_pareto_frontier(_points(), "x", "y")
    assert "C" not in set(frontier["label"])
    assert "E" not in set(frontier["label"])


def test_frontier_maximize_y():
    """Maximizing y flips which points are non-dominated."""
    frontier = compute_pareto_frontier(
        _points(), "x", "y", x_minimize=True, y_minimize=False
    )
    # Lower x is better, higher y is better: A (x=1,y=5) dominates everything
    # with larger x and smaller y; C (x=3,y=4) is dominated by A.
    assert "A" in set(frontier["label"])
    assert set(frontier["label"]) == {"A"}


def test_frontier_ignores_nan():
    """Rows with NaN on either axis are excluded."""
    points = _points()
    points.loc[len(points)] = {"label": "F", "x": float("nan"), "y": 0.0}
    frontier = compute_pareto_frontier(points, "x", "y")
    assert "F" not in set(frontier["label"])
