"""Tests for pairwise same-zone access cuts and cut aggregation in iterative choice."""

import pytest

from choice.objective import (
    ChoiceCut,
    ChoiceEvaluation,
    ChoiceObjective,
    ChoiceTerm,
)
from optimization.solvers import get_solver
from optimization.strategies import get_strategy
from optimization.strategies import iterative_choice as iterative_choice_module
from optimization.tests.synthetic import FakeDataset, make_grid_problem


from optimization.solvers.base import available_solvers

SOLVERS = [s for s in ["cp_int", "cp_bool", "mip"] if s in available_solvers()]


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_access_variable_co_zoning_incentive(solver_name):
    problem = make_grid_problem(2, 2)
    cuts = tuple(
        ChoiceCut(
            node=n,
            constant=0.0,
            terms=(
                (
                    ChoiceTerm(coefficient=100.0, node=0),
                    ChoiceTerm(coefficient=-100.0, node=3),
                )
                if n == 1
                else ()
            ),
        )
        for n in problem.nodes
    )
    problem.choice_objective = ChoiceObjective(
        cuts=cuts,
        lower_bound=-500.0,
        upper_bound=500.0,
        scale=100,
        aggregate_cuts=False,
    )

    solver = get_solver(solver_name, solve_time_limit=10, workers=1)
    solution = solver.solve(problem)

    assert solution.status in ("OPTIMAL", "FEASIBLE")
    assert solution.assignment[1] == solution.assignment[0]
    assert solution.objective == pytest.approx(100.0, abs=0.1)


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_aggregated_cut_objective(solver_name):
    problem = make_grid_problem(2, 2)
    agg_cut = ChoiceCut(
        node=None,
        constant=20.0,
        terms=(ChoiceTerm(coefficient=50.0, node=0, student_node=1),),
    )
    problem.choice_objective = ChoiceObjective(
        cuts=(agg_cut,),
        lower_bound=-500.0,
        upper_bound=500.0,
        scale=100,
        aggregate_cuts=True,
    )

    solver = get_solver(solver_name, solve_time_limit=10, workers=1)
    solution = solver.solve(problem)

    assert solution.status in ("OPTIMAL", "FEASIBLE")
    assert solution.assignment[1] == solution.assignment[0]
    assert solution.objective == pytest.approx(70.0, abs=0.1)


class SyntheticIterativeModel:
    def __init__(self):
        self.calls = 0

    def utility_bounds(self, problem):
        return -500.0, 500.0

    def evaluate_with_cuts(self, problem, assignment):
        self.calls += 1
        cuts = tuple(
            ChoiceCut(
                node=node,
                constant=5.0,
                terms=(ChoiceTerm(coefficient=10.0, node=0),),
            )
            for node in problem.nodes
        )
        total_utility = sum(
            15.0 if assignment.get(n) == assignment.get(0) else 5.0
            for n in problem.nodes
        )
        return ChoiceEvaluation(utility=total_utility, cuts=cuts)


def test_iterative_choice_strategy_adds_per_node_cuts(monkeypatch):
    problem = make_grid_problem(2, 2)
    dataset = FakeDataset(problem)
    solver = get_solver("cp_int", solve_time_limit=10, workers=1)
    model = SyntheticIterativeModel()
    monkeypatch.setattr(
        iterative_choice_module,
        "build_mnl_choice_model",
        lambda data, **kwargs: model,
    )

    strat = get_strategy(
        "iterative_choice",
        levels=["BlockGroup_0"],
        max_iterations=2,
        boundary_prop=-1.0,
    )

    solutions = strat.run(dataset, solver)
    assert len(solutions) >= 1
    sol = solutions[-1]
    assert sol.feasible
    assert "choice_utility" in sol.metadata
    assert sol.metadata["choice_cuts_added"] == len(problem.nodes)


def test_config_rejects_unknown_choice_aggregate_cut_key():
    """Choice cut aggregation has no config toggle; stray keys must not be ignored."""

    from benchmark.config import optimization_config_from_dict

    dict_cfg = {
        "levels": ["BlockGroup_0"],
        "strategy": "iterative_choice",
        "solver": "cp_int",
        "data": {
            "scenario": "legacy",
            "overrides": {"filters": {"optimization": {"program_population": "GE"}}},
        },
        "choice_aggregate_cuts": True,
    }

    with pytest.raises(ValueError, match="Unknown optimization config keys"):
        optimization_config_from_dict(dict_cfg)


def _total_utility_problem(cut: ChoiceCut, scale: float = 100):
    problem = make_grid_problem(2, 2)
    problem.choice_objective = ChoiceObjective(
        cuts=(cut,),
        lower_bound=-500.0,
        upper_bound=500.0,
        scale=scale,
        aggregate_cuts=True,
    )
    return problem


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_subresolution_coefficient_keeps_a_valid_outer_bound(solver_name):
    """A coefficient finer than 1/scale must round outward, not vanish.

    Rounding ``0.004 * 100`` to the nearest integer yields ``0``, which drops
    the incentive entirely and reports ``0.0`` instead of a valid bound.
    """

    coefficient = 0.004
    problem = _total_utility_problem(
        ChoiceCut(
            node=None,
            constant=0.0,
            terms=(ChoiceTerm(coefficient=coefficient, node=0, student_node=1),),
        )
    )

    solution = get_solver(solver_name, solve_time_limit=10, workers=1).solve(problem)

    assert solution.status in ("OPTIMAL", "FEASIBLE")
    # The incentive must survive the conversion to integer coefficients.
    assert solution.assignment[1] == solution.assignment[0]
    assert solution.objective > 0.0
    # And the reported bound must never fall below the true cut value.
    assert solution.objective >= coefficient - 1e-9


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_anchor_active_coefficient_rounds_outward_but_stays_tight(solver_name):
    """At its anchor a cut must stay tight to within one scaled unit."""

    scale = 100
    constant = 0.1234567
    active, inactive = 0.3333333, -0.2222222
    cut = ChoiceCut(
        node=None,
        constant=constant,
        terms=(
            ChoiceTerm(coefficient=active, node=0, student_node=1),
            ChoiceTerm(coefficient=inactive, node=3, student_node=1),
        ),
        anchor_access=(((1, 0), 1), ((1, 3), 0)),
    )

    solution = get_solver(solver_name, solve_time_limit=10, workers=1).solve(
        _total_utility_problem(cut, scale=scale)
    )

    assert solution.status in ("OPTIMAL", "FEASIBLE")
    # The anchor is node 1 co-zoned with node 0 but not with node 3.
    assert solution.assignment[1] == solution.assignment[0]
    assert solution.assignment[1] != solution.assignment[3]
    anchor_value = constant + active
    assert solution.objective >= anchor_value - 1e-9
    assert solution.objective <= anchor_value + 1.0 / scale + 1e-9


@pytest.mark.skipif(len(SOLVERS) < 2, reason="needs at least two backends")
def test_cp_and_mip_agree_on_exactly_representable_choice_objective():
    cut = ChoiceCut(
        node=None,
        constant=20.0,
        terms=(ChoiceTerm(coefficient=50.0, node=0, student_node=1),),
    )
    solutions = {
        name: get_solver(name, solve_time_limit=10, workers=1).solve(
            _total_utility_problem(cut)
        )
        for name in SOLVERS
    }

    for name, solution in solutions.items():
        assert solution.status in ("OPTIMAL", "FEASIBLE"), name
        assert solution.assignment[1] == solution.assignment[0], name
        assert solution.objective == pytest.approx(70.0, abs=1e-6), name


@pytest.mark.skipif("mip" not in SOLVERS, reason="needs the MIP backend")
def test_cp_bounds_dominate_the_exact_mip_value_under_rounding():
    """CP-SAT rounding may only loosen, never tighten, the MIP-exact bound."""

    cut = ChoiceCut(
        node=None,
        constant=0.1234567,
        terms=(ChoiceTerm(coefficient=0.3333333, node=0, student_node=1),),
    )
    exact = get_solver("mip", solve_time_limit=10, workers=1).solve(
        _total_utility_problem(cut)
    )
    assert exact.status in ("OPTIMAL", "FEASIBLE")

    for name in (n for n in SOLVERS if n != "mip"):
        rounded = get_solver(name, solve_time_limit=10, workers=1).solve(
            _total_utility_problem(cut)
        )
        assert rounded.status in ("OPTIMAL", "FEASIBLE"), name
        assert rounded.assignment[1] == rounded.assignment[0], name
        assert rounded.objective >= exact.objective - 1e-9, name
