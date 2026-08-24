"""Synthetic tests for the ReCom-family solvers."""

from __future__ import annotations

import random

import networkx as nx
import pytest

from optimization.config import OptimizationConfig
from optimization.data.contiguity import boundary_cost, boundary_edges
from optimization.data.edge_weights import BOUNDARY_WEIGHT_ATTR
from optimization.solvers import get_solver
from optimization.solvers.balance import balance_constraints
from optimization.solvers.base import available_solvers
from optimization.solvers.recom import (
    _CutCandidate,
    _DynamicMaxNormalizer,
    _ReComContext,
    _ReComKernel,
    _ZoneStats,
)
from optimization.tests.synthetic import make_grid_problem, make_solver_contract_problem

RECOM_SOLVERS = ("recom", "relaxed_recom", "short_bursts")


def test_recom_solvers_are_registered() -> None:
    assert set(RECOM_SOLVERS) <= set(available_solvers())


def test_config_passes_recom_options_to_solver() -> None:
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="short_bursts",
        recom_iterations=321,
        short_bursts_length=17,
        short_bursts_method="relaxed_recom",
    )

    solver = config.make_solver()

    assert solver.options["recom_iterations"] == 321
    assert solver.options["short_bursts_length"] == 17
    assert solver.options["short_bursts_method"] == "relaxed_recom"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"short_bursts_length": 0}, "short_bursts_length"),
        ({"short_bursts_method": "bad"}, "short_bursts_method"),
        (
            {"recom_iterations": -1, "solve_time_limits": []},
            "solve_time_limits",
        ),
    ],
)
def test_config_rejects_invalid_recom_options(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        OptimizationConfig(levels=["BlockGroup_0"], **overrides)


@pytest.mark.parametrize("solver_name", RECOM_SOLVERS)
def test_recom_solver_requires_a_hint(solver_name: str) -> None:
    problem = make_solver_contract_problem(hint=None)

    solution = get_solver(
        solver_name,
        hints="none",
        recom_iterations=1,
    ).solve(problem)

    assert solution.status == "ERROR"
    assert solution.assignment == {}
    assert "hint" in solution.metadata["error_message"].lower()


@pytest.mark.parametrize("solver_name", RECOM_SOLVERS)
def test_recom_solver_can_generate_voronoi_hint(solver_name: str) -> None:
    problem = make_solver_contract_problem(hint=None, max_distance=0.0)

    solution = get_solver(
        solver_name,
        hints="voronoi",
        recom_iterations=0,
    ).solve(problem)

    _assert_valid_recom_solution(problem, solution)
    assert solution.metadata["hint_source"] == "generated"


@pytest.mark.parametrize("solver_name", RECOM_SOLVERS)
def test_recom_solver_allows_negative_capacity_tolerances(solver_name: str) -> None:
    problem = make_solver_contract_problem(overage=-1, shortage=-1)
    for node in problem.nodes:
        problem.G.nodes[node]["ge_capacity"] = 0.0

    solution = get_solver(solver_name, recom_iterations=0).solve(problem)

    _assert_valid_recom_solution(problem, solution)


@pytest.mark.parametrize("solver_name", RECOM_SOLVERS)
def test_recom_ignores_centroid_anchors_and_max_distance(solver_name: str) -> None:
    swapped = {0: 1, 1: 1, 2: 0, 3: 0}
    problem = make_solver_contract_problem(hint=swapped, max_distance=0.0)

    solution = get_solver(solver_name, recom_iterations=0).solve(problem)

    _assert_valid_recom_solution(problem, solution)
    assert solution.assignment == swapped
    assert solution.assignment[problem.centroids[0]] != 0
    assert solution.assignment[problem.centroids[1]] != 1


@pytest.mark.parametrize("solver_name", RECOM_SOLVERS)
def test_recom_rejects_structurally_invalid_hint(solver_name: str) -> None:
    problem = make_solver_contract_problem(hint={0: 0, 1: 1, 2: 0, 3: 1})

    solution = get_solver(solver_name, recom_iterations=1).solve(problem)

    assert solution.status == "ERROR"
    assert "contiguous" in solution.metadata["error_message"]


@pytest.mark.parametrize("solver_name", RECOM_SOLVERS)
def test_recom_honors_explicit_candidates(solver_name: str) -> None:
    problem = make_solver_contract_problem(
        candidates={1: {0}, 2: {1}},
        max_distance=0.0,
    )

    solution = get_solver(
        solver_name,
        recom_iterations=20,
        short_bursts_length=4,
        seed=7,
    ).solve(problem)

    _assert_valid_recom_solution(problem, solution)
    assert solution.assignment[1] == 0
    assert solution.assignment[2] == 1


def test_base_recom_rejects_infeasible_proposals() -> None:
    problem = _problem_without_feasible_contiguous_partition()

    solution = get_solver("recom", recom_iterations=10, seed=3).solve(problem)

    assert solution.status == "UNKNOWN"
    assert solution.assignment == {}
    assert solution.metadata["accepted_moves"] == 0
    assert solution.metadata["rejected_moves"] == 10


@pytest.mark.parametrize("solver_name", ["relaxed_recom", "short_bursts"])
def test_unrejected_walks_accept_infeasible_proposals(solver_name: str) -> None:
    problem = _problem_without_feasible_contiguous_partition()

    solution = get_solver(
        solver_name,
        recom_iterations=7,
        short_bursts_length=3,
        seed=3,
    ).solve(problem)

    assert solution.status == "UNKNOWN"
    assert solution.metadata["attempted_moves"] == 7
    assert solution.metadata["accepted_moves"] == 7
    assert solution.metadata["rejected_moves"] == 0


def test_cut_selection_prefers_pair_feasible_cuts() -> None:
    problem = make_solver_contract_problem()
    for node, schools in enumerate([0, 2, 2, 0]):
        problem.G.nodes[node]["num_schools"] = schools
    context = _ReComContext(problem)
    state = context.build_state(context.validate_hint(problem.hint or {}))

    for seed in range(10):
        move = _ReComKernel(context, random.Random(seed), None).propose(
            state, "uniform"
        )
        assert move.stats_a.schools == 2
        assert move.stats_b.schools == 2
        assert all(value == pytest.approx(0.0) for value in move.violations_a)
        assert all(value == pytest.approx(0.0) for value in move.violations_b)


def test_wilson_sampler_returns_a_spanning_tree() -> None:
    problem = make_solver_contract_problem()
    context = _ReComContext(problem)
    kernel = _ReComKernel(context, random.Random(4), None)
    nodes = set(range(len(context.nodes)))
    adjacency, _ = kernel._pair_graph(nodes)

    tree = kernel._random_spanning_tree(sorted(nodes), adjacency)

    assert set(tree) == nodes
    assert sum(len(neighbors) for neighbors in tree.values()) // 2 == len(nodes) - 1
    graph = nx.Graph()
    graph.add_nodes_from(tree)
    graph.add_edges_from(
        (node, neighbor) for node, neighbors in tree.items() for neighbor in neighbors
    )
    assert nx.is_tree(graph)


def test_incremental_state_matches_full_rebuild_after_grid_walk() -> None:
    hint = {node: 0 if node % 5 < 2 else 1 for node in range(25)}
    problem = make_grid_problem(5, 5, hint=hint)
    context = _ReComContext(problem)
    state = context.build_state(context.validate_hint(hint))
    kernel = _ReComKernel(context, random.Random(11), None)

    for _ in range(50):
        kernel.apply(state, kernel.propose(state, "relaxed"))
        rebuilt = context.build_state(list(state.assignment))

        assert state.zone_nodes == rebuilt.zone_nodes
        assert state.boundary_pairs == rebuilt.boundary_pairs
        assert state.boundary_costs == rebuilt.boundary_costs
        assert state.boundary_cost == rebuilt.boundary_cost
        assert state.violations == pytest.approx(rebuilt.violations)
        for actual, expected in zip(state.zone_stats, rebuilt.zone_stats):
            assert actual.node_count == expected.node_count
            assert actual.students == pytest.approx(expected.students)
            assert actual.values == pytest.approx(expected.values)
            assert actual.schools == pytest.approx(expected.schools)
            assert actual.internal_edges == expected.internal_edges


def test_relaxed_probabilities_use_metric_products_and_tree_proxy() -> None:
    context = _ReComContext(make_solver_contract_problem())
    kernel = _ReComKernel(context, random.Random(1), None)
    balanced_schools = _candidate(
        _stats(schools=2, internal_edges=3),
        _stats(schools=2, internal_edges=3),
    )
    unbalanced_schools = _candidate(
        _stats(schools=1, internal_edges=3),
        _stats(schools=3, internal_edges=3),
    )
    more_cycles = _candidate(
        _stats(schools=2, internal_edges=4),
        _stats(schools=2, internal_edges=4),
    )

    probabilities = kernel._relaxed_probabilities(
        [balanced_schools, unbalanced_schools, more_cycles]
    )

    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1]
    assert probabilities[2] > probabilities[0]


def test_relaxed_shortage_metric_is_absolute_student_relative_percent() -> None:
    metrics = _ReComKernel._relaxed_metrics(_stats(students=100, seats=80, schools=2))
    equal_metrics = _ReComKernel._relaxed_metrics(
        _stats(students=100, seats=100, schools=2)
    )

    assert metrics["shortage%"] == pytest.approx(0.2)
    assert equal_metrics["shortage%"] > 0


def test_dynamic_max_normalizer_uses_running_component_maxima() -> None:
    normalizer = _DynamicMaxNormalizer(2)
    normalizer.observe((2.0, 10.0))
    assert normalizer.penalty((2.0, 10.0)) == pytest.approx(2.0)

    normalizer.observe((4.0, 5.0))

    assert normalizer.maxima == [4.0, 10.0]
    assert normalizer.penalty((2.0, 10.0)) == pytest.approx(1.5)
    assert normalizer.penalty((4.0, 5.0)) == pytest.approx(1.5)


@pytest.mark.parametrize("method", ["recom", "relaxed_recom"])
def test_short_bursts_uses_shared_iteration_budget(method: str) -> None:
    problem = make_solver_contract_problem()

    solution = get_solver(
        "short_bursts",
        recom_iterations=7,
        short_bursts_length=3,
        short_bursts_method=method,
        seed=5,
    ).solve(problem)

    _assert_valid_recom_solution(problem, solution)
    assert solution.metadata["attempted_moves"] == 7
    assert solution.metadata["accepted_moves"] == 7
    assert solution.metadata["completed_bursts"] == 3
    assert solution.metadata["short_bursts_method"] == method


def test_negative_iterations_use_only_time_limit() -> None:
    problem = make_solver_contract_problem()

    solution = get_solver(
        "recom",
        recom_iterations=-1,
        solve_time_limit=0.0,
    ).solve(problem)

    _assert_valid_recom_solution(problem, solution)
    assert solution.metadata["attempted_moves"] == 0
    assert solution.metadata["stop_reason"] == "time_limit"


@pytest.mark.parametrize("solver_name", RECOM_SOLVERS)
def test_recom_rejects_choice_objective(solver_name: str) -> None:
    problem = make_solver_contract_problem(choice_objective=object())

    with pytest.raises(NotImplementedError, match=solver_name):
        get_solver(solver_name).solve(problem)


@pytest.mark.parametrize("solver_name", RECOM_SOLVERS)
def test_recom_uses_weighted_boundary_cost(solver_name: str) -> None:
    problem = make_solver_contract_problem(weight_edges=True)
    problem.G.graph["weight_edges"] = True
    for edge, weight in zip(problem.G.edges(), [3, 11, 5]):
        problem.G.edges[edge][BOUNDARY_WEIGHT_ATTR] = weight

    solution = get_solver(
        solver_name,
        recom_iterations=5,
        short_bursts_length=2,
        seed=3,
    ).solve(problem)

    _assert_valid_recom_solution(problem, solution)
    assert solution.objective == boundary_cost(
        problem.G,
        solution.assignment,
        weight_edges=True,
    )
    assert solution.objective != boundary_edges(problem.G, solution.assignment)
    assert solution.metadata["objective_kind"] == "weighted_boundary_length"
    assert solution.metadata["objective_unit"] == "meter"


def _problem_without_feasible_contiguous_partition():
    problem = make_solver_contract_problem()
    for node, value in enumerate([1.0, 1.0, 0.0, 0.0]):
        problem.G.nodes[node]["FRL"] = value
    problem.G.graph["F"] = 0.5
    problem.frl_dev = 0.1
    return problem


def _stats(
    *,
    students: float = 100.0,
    seats: float = 80.0,
    frl: float = 50.0,
    schools: float = 2.0,
    internal_edges: int = 3,
) -> _ZoneStats:
    return _ZoneStats(
        node_count=3,
        students=students,
        values=(seats, frl),
        schools=schools,
        internal_edges=internal_edges,
    )


def _candidate(stats_a: _ZoneStats, stats_b: _ZoneStats) -> _CutCandidate:
    return _CutCandidate(
        tin=1,
        size=1,
        subtree_to_a=True,
        stats_a=stats_a,
        stats_b=stats_b,
        violations_a=(),
        violations_b=(),
        global_violations=(),
        boundary_cost=1,
    )


def _assert_valid_recom_solution(problem, solution) -> None:
    assert solution.status == "FEASIBLE"
    assert set(solution.assignment) == set(problem.nodes)
    assert set(solution.assignment.values()) == set(range(problem.Z))
    for node, zone in solution.assignment.items():
        if problem.candidates is not None and node in problem.candidates:
            assert zone in problem.candidates[node]
        elif problem.fixed is not None and node in problem.fixed:
            assert zone == problem.fixed[node]

    for zone in range(problem.Z):
        nodes = [
            node for node, assigned in solution.assignment.items() if assigned == zone
        ]
        assert nx.is_connected(problem.G.subgraph(nodes))
        students = sum(problem.students(node) for node in nodes)
        for constraint in balance_constraints(problem):
            value = sum(constraint.value(node) for node in nodes)
            if constraint.lower_ratio is not None:
                assert value >= constraint.lower_ratio * students - 1e-6
            if constraint.upper_ratio is not None:
                assert value <= constraint.upper_ratio * students + 1e-6

    total_schools = sum(problem.num_schools(node) for node in problem.nodes)
    if total_schools:
        average = total_schools / problem.Z
        for zone in range(problem.Z):
            schools = sum(
                problem.num_schools(node)
                for node, assigned in solution.assignment.items()
                if assigned == zone
            )
            assert schools >= max(0.0, average - 1.0) - 1e-6
            assert schools <= average + 1.0 + 1e-6

    assert solution.objective == boundary_cost(
        problem.G,
        solution.assignment,
        weight_edges=problem.weight_edges,
    )
    assert solution.is_contiguous()
