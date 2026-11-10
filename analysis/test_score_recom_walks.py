"""The walk scorer's constraint flags agree with check_zoning."""

from __future__ import annotations

import random

import numpy as np
import pytest

from analysis.score_recom_walks import constraint_columns, zone_sums
from optimization.data.feasibility import check_zoning
from optimization.solvers import get_solver
from optimization.tests.synthetic import make_grid_problem

SCHOOLS = {0: 100, 5: 200, 30: 300, 35: 400, 14: 500, 21: 600}


def _walk_states(problem, moves=300, seed=3):
    nodes = list(problem.nodes)
    states = []
    get_solver("recom", seed=seed, recom_iterations=moves).sample_feasible(
        problem,
        lambda a: states.append([a[n] for n in nodes]),
        pair_selection="uniform_boundary",
        visit_rejected=True,
    )
    return np.asarray(states, dtype=np.uint8)


def _graded_problem(**overrides):
    """A 6x6 grid whose FRL and seats vary by node, so the rows bind."""
    problem = make_grid_problem(
        6, 6, schools={0: 100, 5: 200, 30: 300, 35: 400}, **overrides
    )
    rng = random.Random(0)
    for node in problem.nodes:
        attrs = problem.G.nodes[node]
        attrs["FRL"] = rng.choice([0.0, 0.2, 0.5, 0.9, 1.0])
        attrs["ge_capacity"] = attrs["all_prog_capacity"] = rng.choice([0, 0.5, 1, 2])
    total = sum(problem.G.nodes[n]["FRL"] for n in problem.nodes)
    problem.G.graph["F"] = total / sum(problem.students(n) for n in problem.nodes)
    return problem


def test_zone_sums_match_a_loop() -> None:
    states = np.asarray([[0, 1, 1, 2], [2, 2, 0, 1]], dtype=np.uint8)
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert zone_sums(states, values, 3).tolist() == [[1, 5, 4], [3, 4, 3]]


@pytest.mark.parametrize("frl_dev, capacity_dev", [(0.15, 0.3), (0.1, 0.5), (0.3, 0.2)])
def test_frl_and_capacity_flags_match_check_zoning(frl_dev, capacity_dev) -> None:
    walk = _graded_problem(overage=-1, shortage=-1, frl_dev=-1)
    states = _walk_states(walk)
    cols = constraint_columns(
        walk, states, frl_dev=frl_dev, school_slack=99, capacity_dev=capacity_dev
    )
    for kind, flag, overrides in (
        ("frl", "frl_ok", {"frl_dev": frl_dev}),
        ("capacity", "capacity_ok", {"overage": capacity_dev, "shortage": capacity_dev}),
    ):
        checked = _graded_problem(**{"overage": -1, "shortage": -1, "frl_dev": -1, **overrides})
        nodes = list(checked.nodes)
        expected = [
            not any(
                v.kind.startswith(kind)
                for v in check_zoning(checked, dict(zip(nodes, map(int, s)))).violations
            )
            for s in states
        ]
        assert cols[flag].tolist() == expected
        # The walk visits both sides of each row, so this is a real check.
        assert 0 < sum(expected) < len(expected) or kind == "capacity"


def test_schools_flag_is_within_one_of_the_average() -> None:
    problem = make_grid_problem(
        6, 6, schools={0: 100, 5: 200, 30: 300, 35: 400}, overage=-1, shortage=-1
    )
    # Two more school nodes that are not centroids: four zones holding 6
    # schools, average 1.5, so 1 or 2 pass and 3 fails.
    for node in (14, 21):
        problem.G.nodes[node]["num_schools"] = 1
    centroids_first = {0: 0, 5: 1, 30: 2, 35: 3}
    nodes = list(problem.nodes)
    zone_of = lambda n, extra: centroids_first.get(n, extra.get(n, 0))
    ok = np.asarray([[zone_of(n, {14: 0, 21: 1}) for n in nodes]], dtype=np.uint8)
    bad = np.asarray([[zone_of(n, {14: 0, 21: 0}) for n in nodes]], dtype=np.uint8)
    for states, expected in ((ok, True), (bad, False)):
        cols = constraint_columns(
            problem, states, frl_dev=1, school_slack=1, capacity_dev=99
        )
        assert bool(cols["schools_ok"][0]) is expected
    assert cols["schools_max_dev"][0] == pytest.approx(1.5)


def test_the_cell_packs_the_three_flags() -> None:
    problem = make_grid_problem(6, 6, schools=SCHOOLS, overage=-1, shortage=-1)
    states = _walk_states(problem, moves=50)
    cols = constraint_columns(
        problem, states, frl_dev=0.15, school_slack=1, capacity_dev=0.3
    )
    assert (
        cols["cell"]
        == cols["frl_ok"] + 2 * cols["schools_ok"] + 4 * cols["capacity_ok"]
    ).all()
