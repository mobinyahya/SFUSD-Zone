"""Contract tests for the partial, single-zone CP-SAT solver."""

from __future__ import annotations

import json

import networkx as nx
import pytest

from Config.Constants import AREA_ETHNICITIES
from Config.metrics_config import MetricColumns
from metrics import MetricsCalculator
from optimization.config import OptimizationConfig
from optimization.data.contiguity import boundary_edges
from optimization.solvers import get_solver
from optimization.tests.synthetic import make_single_zone_problem


def _solve(problem=None, **options):
    return get_solver(
        "cp_single_zone",
        solve_time_limit=5,
        workers=1,
        seed=1,
        **options,
    ).solve(problem or make_single_zone_problem())


def test_single_zone_solver_selects_optimal_connected_zone():
    problem = make_single_zone_problem()

    solution = _solve(problem)

    assert solution.status == "OPTIMAL"
    assert len(solution.assignment) == 3
    assert set(solution.assignment.values()) == {0}
    assert problem.centroids[0] in solution.assignment
    assert nx.is_connected(problem.G.subgraph(solution.assignment))
    assert {0, 6}.isdisjoint(solution.assignment)
    assert solution.objective == 2
    assert solution.objective == boundary_edges(problem.G, solution.assignment)
    assert solution.metadata["partial_assignment"] is True
    assert solution.metadata["centroid_school_id"] == 100
    assert solution.metadata["selected_node_count"] == 3
    assert solution.metadata["omitted_node_count"] == 4

    students = sum(problem.students(node) for node in solution.assignment)
    capacity = sum(problem.capacity(node) for node in solution.assignment)
    frl = sum(problem.frl(node) for node in solution.assignment)
    assert capacity == students
    assert frl == problem.district_frl * students
    for ethnicity in AREA_ETHNICITIES:
        count = sum(problem.ethnicity(node, ethnicity) for node in solution.assignment)
        assert count == problem.district_racial[ethnicity] * students


def test_single_zone_solver_enforces_max_distance():
    problem = make_single_zone_problem(max_distance=1.0)

    solution = _solve(problem)

    assert solution.status == "OPTIMAL"
    assert set(solution.assignment) == {2, 3, 4}


def test_distance_exempt_node_is_candidate_beyond_max_distance():
    problem = make_single_zone_problem(max_distance=1.0)
    problem.G.nodes[0]["max_distance_exempt"] = True

    assert problem.candidate_zones(0) == {0}
    assert problem.candidate_zones(1) == set()


def test_single_zone_solver_requires_nodes_within_centroid_neighbor_radius():
    problem = make_single_zone_problem()
    problem.G.nodes[3]["ge_capacity"] = 2.0

    assert _solve(problem, centroid_neighbor_radius=0).status == "OPTIMAL"
    assert _solve(problem, centroid_neighbor_radius=1).status == "INFEASIBLE"


def test_single_zone_solver_excludes_nodes_within_other_school_radius():
    problem = make_single_zone_problem()
    problem.G.nodes[3]["ge_capacity"] = 5.0

    assert _solve(problem, centroid_neighbor_radius=0).status == "OPTIMAL"
    assert _solve(problem, centroid_neighbor_radius=1).status == "INFEASIBLE"


def test_single_zone_solver_rejects_invalid_centroid_neighbor_radius():
    with pytest.raises(ValueError, match="non-negative integer"):
        _solve(centroid_neighbor_radius=-1)


@pytest.mark.parametrize(
    ("school_ids", "num_schools"),
    [([400], 0), ([], 1)],
)
def test_single_zone_solver_excludes_all_other_school_node_schemas(
    school_ids, num_schools
):
    problem = make_single_zone_problem(max_distance=1.0)
    problem.G.nodes[2]["school_ids"] = school_ids
    problem.G.nodes[2]["num_schools"] = num_schools

    solution = _solve(problem)

    assert solution.status == "INFEASIBLE"


@pytest.mark.parametrize("constraint", ["capacity", "frl", "racial"])
def test_single_zone_solver_reports_balance_infeasibility(constraint):
    problem = make_single_zone_problem()
    if constraint == "capacity":
        problem.G.nodes[3]["ge_capacity"] = 2.5
    elif constraint == "frl":
        problem.G.nodes[3]["FRL"] = 1.0
        problem.G.graph["F"] = 0.0
    else:
        ethnicity = AREA_ETHNICITIES[0]
        problem.G.nodes[3][ethnicity] = 1.0
        problem.G.graph["R"][ethnicity] = 0.0

    solution = _solve(problem)

    assert solution.status == "INFEASIBLE"
    assert solution.assignment == {}
    assert solution.objective is None


def test_single_zone_solver_requires_one_centroid():
    problem = make_single_zone_problem()
    problem.centroids.append(1)

    with pytest.raises(ValueError, match="exactly one centroid"):
        _solve(problem)


@pytest.mark.parametrize(
    ("school_ids", "num_schools"),
    [([], 0), ([100, 101], 2), ([100], 2)],
)
def test_single_zone_solver_requires_exactly_one_school_at_centroid(
    school_ids, num_schools
):
    problem = make_single_zone_problem()
    problem.G.nodes[3]["school_ids"] = school_ids
    problem.G.nodes[3]["num_schools"] = num_schools

    with pytest.raises(ValueError, match="exactly one school"):
        _solve(problem)


def test_single_zone_solver_rejects_solver_progress():
    with pytest.raises(ValueError, match="save_solver_progress"):
        _solve(save_solver_progress=True)


def test_single_zone_config_requires_single_strategy():
    with pytest.raises(ValueError, match="requires strategy='single'"):
        OptimizationConfig(
            levels=["Block_1", "Block_0"],
            solver="cp_single_zone",
            strategy="recursive",
        )


def test_single_zone_config_passes_centroid_neighbor_radius():
    config = OptimizationConfig(
        levels=["Block_0"],
        solver="cp_single_zone",
        centroid_neighbor_radius=2,
    )

    assert config.make_solver().options["centroid_neighbor_radius"] == 2


def test_config_rejects_negative_centroid_neighbor_radius():
    with pytest.raises(ValueError, match="centroid_neighbor_radius"):
        OptimizationConfig(
            levels=["Block_0"],
            solver="cp_single_zone",
            centroid_neighbor_radius=-1,
        )


def test_single_zone_solution_saves_only_selected_nodes(tmp_path):
    solution = _solve()

    solution.save(str(tmp_path))

    assignment = json.loads(
        (tmp_path / "zone_dict_Block_0.json").read_text(encoding="utf-8")
    )
    info = json.loads((tmp_path / "solution_Block_0.json").read_text(encoding="utf-8"))
    assert {int(node) for node in assignment} == set(solution.assignment)
    assert len(assignment) == 3
    assert info["num_zones"] == 1
    assert info["contiguous"] is True
    assert info["metadata"]["partial_assignment"] is True


def test_single_zone_metrics_count_selected_boundary():
    solution = _solve()

    result = MetricsCalculator(solution, config=_metrics_config()).compute()

    assert result.metrics["num_zones"] == 1
    assert result.metrics["cut_edges"] == solution.objective
    assert result.metrics["final_cut_edges"] == solution.objective
    assert set(result.zone_data) == {0}


def test_single_zone_metrics_use_district_frl_and_outside_schools():
    problem = make_single_zone_problem(frl_dev=0.5)
    problem.G.graph["F"] = 0.25
    solution = _solve(problem)
    for node in solution.assignment:
        problem.G.graph["distance_dict"][node][0] = 0.25

    result = MetricsCalculator(solution, config=_metrics_config()).compute()

    assert result.metrics[MetricColumns.FRL_MAD] == pytest.approx(0.25)
    assert result.metrics[MetricColumns.AVG_OUT_OF_ZONE_GE_SCHOOLS] >= 1.0


def _metrics_config():
    return {
        "strategy": "single",
        "choice_model": "distance",
        "data": {"scenario": "legacy", "overrides": {}},
    }
