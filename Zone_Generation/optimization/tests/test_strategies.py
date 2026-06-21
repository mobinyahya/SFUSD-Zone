"""Strategy tests using a FakeDataset (no SFUSD data required)."""

import pytest

from Zone_Generation.choice.objective import ChoiceCut, ChoiceEvaluation
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers import get_solver
from Zone_Generation.optimization.strategies import iterative_choice as iterative_choice_module
from Zone_Generation.optimization.strategies import get_strategy
from Zone_Generation.optimization.tests.synthetic import FakeDataset, make_grid_problem


def test_single_strategy():
    problem = make_grid_problem(3, 3)
    dataset = FakeDataset(problem)
    solver = get_solver("cp_int", solve_time_limit=10, workers=1)
    strat = get_strategy("single", levels=["BlockGroup_0"])
    solutions = strat.run(dataset, solver)
    assert len(solutions) == 1
    assert solutions[-1].status in ("OPTIMAL", "FEASIBLE")
    assert solutions[-1].is_contiguous()


def test_iterative_choice_strategy_terminates():
    problem = make_grid_problem(3, 3)
    dataset = FakeDataset(problem)
    solver = get_solver("cp_int", solve_time_limit=10, workers=1)
    strat = get_strategy(
        "iterative_choice",
        levels=["BlockGroup_0"],
        max_iterations=3,
        choice_model="distance",
    )
    solutions = strat.run(dataset, solver)
    assert 1 <= len(solutions) <= 3
    last = solutions[-1]
    assert "choice_utility" in last.metadata
    assert solutions[0].metadata["choice_objective_cuts"] == 0
    assert solutions[0].metadata["choice_cuts_added"] > 0
    assert last.is_contiguous()


def test_iterative_choice_stops_on_absolute_model_objective_change(monkeypatch):
    problem = make_grid_problem(2, 2)
    dataset = FakeDataset(problem)
    solver = ObjectiveSequenceSolver([100.0, 10.0, 9.0, 9.2, 50.0])
    model = DecreasingRealUtilityModel()
    monkeypatch.setattr(
        iterative_choice_module,
        "get_configured_choice_model",
        lambda options: model,
    )
    strat = get_strategy(
        "iterative_choice",
        levels=["BlockGroup_0"],
        max_iterations=5,
        choice_model="distance",
        tolerance=0.25,
    )

    solutions = strat.run(dataset, solver)

    assert [solution.objective for solution in solutions] == [100.0, 10.0, 9.0, 9.2]
    assert [solution.metadata["choice_utility"] for solution in solutions] == [
        100.0,
        99.0,
        98.0,
        97.0,
    ]
    assert solutions[2].metadata["choice_model_utility_change"] == 1.0
    assert solutions[3].metadata["choice_model_utility_change"] == pytest.approx(0.2)


class ObjectiveSequenceSolver:
    def __init__(self, objectives):
        self.objectives = list(objectives)
        self.calls = 0

    def solve(self, problem):
        objective = self.objectives[self.calls]
        self.calls += 1
        assignment = {0: 0, 1: 0, 2: 1, 3: 1}
        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status="FEASIBLE",
            objective=objective,
            wall_time=0.0,
        )


class DecreasingRealUtilityModel:
    def __init__(self):
        self.calls = 0

    def utility_bounds(self, problem):
        return -1_000.0, 1_000.0

    def evaluate_with_cuts(self, problem, assignment):
        utility = 100.0 - self.calls
        self.calls += 1
        cuts = tuple(
            ChoiceCut(node=node, zone=zone, constant=0.0)
            for node in problem.nodes
            for zone in problem.candidate_zones(node)
        )
        return ChoiceEvaluation(utility=utility, cuts=cuts)
