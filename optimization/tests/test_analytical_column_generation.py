"""Tests for restricted analytical complete-zone column generation."""

from __future__ import annotations

import pytest

from optimization.analytical_column_generation import (
    AnalyticalZoneColumn,
    _solve_restricted_integer_master,
    _solve_restricted_lp,
)
from optimization.tests.synthetic import make_grid_problem


def test_restricted_pattern_master_selects_best_partition():
    pytest.importorskip("gurobipy")
    problem = make_grid_problem(2, 2, boundary_prop=1.0)
    columns = [
        AnalyticalZoneColumn(0, frozenset({0, 1}), 5.0, 2),
        AnalyticalZoneColumn(0, frozenset({0, 2}), 6.0, 2),
        AnalyticalZoneColumn(1, frozenset({2, 3}), 7.0, 2),
        AnalyticalZoneColumn(1, frozenset({1, 3}), 8.0, 2),
    ]

    lp = _solve_restricted_lp(problem, columns)
    selected, objective = _solve_restricted_integer_master(problem, columns)

    assert -lp.fun == pytest.approx(14.0)
    assert objective == pytest.approx(14.0)
    assert {(column.label, column.nodes) for column in selected} == {
        (0, frozenset({0, 2})),
        (1, frozenset({1, 3})),
    }
