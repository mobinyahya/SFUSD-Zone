"""Production solver contract tests on small, data-free problems."""

from __future__ import annotations

import pytest

from Config.Constants import AREA_ETHNICITIES
from optimization.problem import NoCandidateZonesError
from optimization.tests.solver_contract import (
    CONTRACT_SOLVERS,
    assert_no_feasible_solution,
    assert_valid_solution,
    solve_contract_problem,
)
from optimization.tests.synthetic import make_solver_contract_problem


@pytest.mark.parametrize("solver_name", CONTRACT_SOLVERS)
def test_solver_satisfies_complete_contract(solver_name: str) -> None:
    problem = make_solver_contract_problem()

    solution = solve_contract_problem(solver_name, problem)

    assert_valid_solution(problem, solution)
    assert solution.metadata["solver"] == solver_name


@pytest.mark.parametrize("solver_name", CONTRACT_SOLVERS)
@pytest.mark.parametrize(
    "restrictions",
    [
        {"candidates": {1: {0}, 2: {1}}},
        {"fixed": {1: 0, 2: 1}},
    ],
    ids=["explicit-candidates", "fixed-assignments"],
)
def test_solver_honors_assignment_restrictions(
    solver_name: str,
    restrictions: dict,
) -> None:
    problem = make_solver_contract_problem(max_distance=float("inf"), **restrictions)

    solution = solve_contract_problem(solver_name, problem)

    assert_valid_solution(problem, solution)


@pytest.mark.parametrize("solver_name", CONTRACT_SOLVERS)
def test_solver_repairs_hint_that_violates_distance_candidates(
    solver_name: str,
) -> None:
    problem = make_solver_contract_problem(hint={0: 1, 1: 1, 2: 0, 3: 0})

    solution = solve_contract_problem(solver_name, problem)

    assert_valid_solution(problem, solution)
    assert solution.assignment == {0: 0, 1: 0, 2: 1, 3: 1}


@pytest.mark.parametrize("solver_name", CONTRACT_SOLVERS)
def test_solver_rejects_node_without_candidate_zone(solver_name: str) -> None:
    problem = make_solver_contract_problem(
        max_distance=float("inf"),
        candidates={1: set()},
    )

    with pytest.raises(NoCandidateZonesError):
        solve_contract_problem(solver_name, problem)


@pytest.mark.parametrize("solver_name", CONTRACT_SOLVERS)
def test_solver_does_not_report_forced_disconnection_as_feasible(
    solver_name: str,
) -> None:
    problem = make_solver_contract_problem(
        max_distance=float("inf"),
        candidates={1: {1}, 2: {0}},
        hint={0: 0, 1: 1, 2: 0, 3: 1},
    )

    solution = solve_contract_problem(solver_name, problem)

    assert_no_feasible_solution(solution)


@pytest.mark.parametrize("solver_name", CONTRACT_SOLVERS)
@pytest.mark.parametrize(
    "constraint_case",
    ["capacity", "frl", "racial", "schools"],
)
def test_solver_does_not_report_hard_constraint_violation_as_feasible(
    solver_name: str,
    constraint_case: str,
) -> None:
    problem = make_solver_contract_problem()

    if constraint_case == "capacity":
        problem.G.nodes[0]["ge_capacity"] = 0.0
        problem.G.nodes[1]["ge_capacity"] = 0.0
        problem.overage = 10.0
    elif constraint_case == "frl":
        for node, value in enumerate([1.0, 1.0, 0.0, 0.0]):
            problem.G.nodes[node]["FRL"] = value
        problem.G.graph["F"] = 0.5
        problem.frl_dev = 0.1
    elif constraint_case == "racial":
        target = AREA_ETHNICITIES[0]
        for ethnicity in AREA_ETHNICITIES:
            for node in problem.nodes:
                problem.G.nodes[node][ethnicity] = 0.0
            problem.G.graph["R"][ethnicity] = 0.0
        for node, value in enumerate([1.0, 1.0, 0.0, 0.0]):
            problem.G.nodes[node][target] = value
        problem.G.graph["R"][target] = 0.5
        problem.racial_dev = 0.1
    else:
        counts = [3, 2, 1, 0]
        for node, count in enumerate(counts):
            problem.G.nodes[node]["num_schools"] = count

    solution = solve_contract_problem(solver_name, problem)

    assert_no_feasible_solution(solution)
