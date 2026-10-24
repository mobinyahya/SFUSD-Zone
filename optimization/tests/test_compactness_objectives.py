"""Named distance/perimeter objectives share costs across all search methods."""

from dataclasses import replace
from itertools import product

import pytest

from benchmark.config import optimization_config_hash
from optimization.config import OptimizationConfig
from optimization.data.edge_weights import boundary_cost
from optimization.data.feasibility import check_zoning
from optimization.objectives import METHODS, heuristic_scorer, solve_objective
from optimization.pipeline import step_config
from optimization.tests.synthetic import make_grid_problem


def _problem():
    problem = make_grid_problem(2, 3, hint={n: int(n >= 3) for n in range(6)})
    for node in problem.nodes:
        problem.G.nodes[node]["ge_students"] = node + 1.0
        problem.G.nodes[node]["ge_capacity"] = node + 1.0
        problem.G.nodes[node]["land_area_m2"] = 7.0 - node
    for index, (u, v) in enumerate(problem.G.edges):
        problem.G.edges[u, v]["boundary_weight"] = 11 * (index + 1)
    return problem


def _cost(problem, assignment, objective):
    if objective == "boundary_perimeter":
        return boundary_cost(problem.G, assignment, weighted=True)
    if objective == "average_distance":
        students = sum(problem.students(n) for n in problem.nodes)
        return (
            sum(
                problem.students(n) * problem.distance(problem.centroids[z], n)
                for n, z in assignment.items()
            )
            / students
        )
    weights = problem.school_distance_weights()
    return sum(
        weights[n] * problem.distance(problem.centroids[z], n) ** 2
        for n, z in assignment.items()
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "objective,unit",
    [
        ("average_distance", "mile"),
        ("average_squared_distance", "mile^2"),
        ("boundary_perimeter", "meter"),
    ],
)
def test_methods_minimize_and_report_the_same_physical_cost(method, objective, unit):
    if method == "mip":
        pytest.importorskip("gurobipy")
    problem = _problem()
    solution = solve_objective(
        problem,
        method,
        objective,
        options={
            "solve_time_limit": 3,
            "recom_iterations": 30,
            "recom_burst_length": 5,
            "workers": 1,
            "seed": 17,
            "hints": "none",
        },
    )
    assert solution.feasible
    assert check_zoning(solution.problem, solution.assignment).feasible
    assert solution.objective == pytest.approx(
        _cost(problem, solution.assignment, objective), abs=1e-7
    )
    assert solution.metadata["objective_kind"] == objective
    assert solution.metadata["objective_unit"] == unit
    assert solution.metadata["method"] == method
    assert problem.weight_edges is False  # Dispatch leaves the source untouched.
    if method in {"cp_bool", "cp_int", "mip"}:
        feasible_costs = [
            _cost(problem, assignment, objective)
            for labels in product(range(problem.Z), repeat=problem.A)
            if check_zoning(
                problem, assignment := dict(zip(problem.nodes, labels))
            ).feasible
        ]
        assert solution.status == "OPTIMAL"
        assert solution.objective == pytest.approx(min(feasible_costs), abs=1e-7)


def test_average_distance_is_linear_and_uses_only_student_shares():
    problem = replace(_problem(), compactness_objective="average_distance")
    for node in problem.nodes:
        problem.G.nodes[node].pop("land_area_m2")
    costs = problem.distance_objective_costs()
    total = sum(problem.students(n) for n in problem.nodes)
    for (zone, node), cost in costs.items():
        assert cost == pytest.approx(
            problem.students(node)
            / total
            * problem.distance(problem.centroids[zone], node)
        )
    assert (
        problem.distance_objective_metadata()["distance_weighting"] == "student_share"
    )
    scorer = heuristic_scorer(
        replace(problem, compactness_objective="cut_edges"), "average_distance", {}
    )
    assert scorer((problem.hint,)) == pytest.approx(
        (_cost(problem, problem.hint, "average_distance"),)
    )


def test_average_distance_rejects_an_empty_student_population():
    problem = _problem()
    for node in problem.nodes:
        problem.G.nodes[node]["ge_students"] = 0.0
    with pytest.raises(ValueError, match="positive student total"):
        solve_objective(problem, "cp_bool", "average_distance")


@pytest.mark.parametrize("method", METHODS)
def test_squared_distance_name_reuses_existing_area_and_student_weights(method):
    config = OptimizationConfig(method=method, objective="average_squared_distance")
    effective = step_config(config, config.steps[0])
    underlying = (
        effective.compactness_objective
        if method in {"cp_bool", "cp_int", "mip"}
        else effective.recom_objective
    )
    assert underlying == "school_distance"
    assert effective.school_distance_area_weight == 0.75
    assert effective.school_distance_student_weight == 0.25


def test_perimeter_loads_weights_without_changing_other_step_objectives():
    config = OptimizationConfig(
        steps=[
            {"name": "count", "objective": "cut_edges"},
            {"name": "length", "objective": "boundary_perimeter"},
            {"name": "distance", "objective": "average_distance"},
        ]
    )
    effective = [step_config(config, s, i) for i, s in enumerate(config.steps)]
    assert config.weight_edges is False
    assert config.graph_weight_edges is True
    assert [c.weight_edges for c in effective] == [False, True, False]
    assert [c.compactness_objective for c in effective] == [
        "cut_edges",
        "cut_edges",
        "average_distance",
    ]


def test_mixed_objectives_preserve_units_costs_and_saved_problem_settings(tmp_path):
    from optimization.tests.test_pipeline import MemoryDataset

    names = [
        "cut_edges",
        "boundary_perimeter",
        "average_distance",
        "average_squared_distance",
    ]
    config = OptimizationConfig(
        hints="none",
        workers=1,
        solve_time_limits=[2],
        frl_dev=1,
        racial_dev=1,
        overage=5,
        shortage=0,
        steps=[{"name": name, "objective": name} for name in names],
    )
    dataset = MemoryDataset(config)
    dataset.base = _problem()
    strategy = config.make_strategy()
    results = strategy.run(dataset, config.make_solver())
    assert [s.metadata["objective_unit"] for s in results[1:]] == [
        "meter",
        "mile",
        "mile^2",
    ]
    assert len(results) == len(names)
    for solution, name in zip(results, names):
        assert solution.feasible
        solution.save(str(tmp_path / name))
        if name == "cut_edges":
            assert solution.objective == boundary_cost(
                solution.problem.G, solution.assignment, weighted=False
            )
        else:
            assert solution.objective == pytest.approx(
                _cost(dataset.base, solution.assignment, name), abs=1e-7
            )
        restored = strategy.problem_for_stage(dataset, solution.metadata)
        assert restored.weight_edges is (name == "boundary_perimeter")
        assert restored.compactness_objective == solution.problem.compactness_objective


@pytest.mark.parametrize(
    "objective,weights_matter",
    [("average_squared_distance", True), ("average_distance", False)],
)
def test_benchmark_hash_tracks_only_weights_used_by_the_objective(
    objective, weights_matter
):
    a = OptimizationConfig(
        objective=objective,
        school_distance_area_weight=0.5,
        school_distance_student_weight=0.5,
    )
    b = OptimizationConfig(
        objective=objective,
        school_distance_area_weight=0.25,
        school_distance_student_weight=0.75,
    )
    assert (
        optimization_config_hash(a) != optimization_config_hash(b)
    ) is weights_matter
