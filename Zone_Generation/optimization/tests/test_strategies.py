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


def test_recursive_carry_over_compute_disabled_by_default():
    problem = make_grid_problem(2, 2)
    dataset = FakeDataset(problem)
    solver = TimedSequenceSolver(
        statuses=["OPTIMAL", "OPTIMAL"],
        wall_times=[1.0, 2.0],
    )
    strat = get_strategy(
        "recursive",
        levels=["BlockGroup_0", "BlockGroup_0"],
        solve_time_limits=[5.0, 10.0],
    )

    solutions = strat.run(dataset, solver)

    assert solver.solve_time_limits == [5.0, 10.0]
    assert "effective_time_limit_seconds" not in solutions[0].metadata


def test_recursive_carry_over_compute_adds_unused_time_for_feasible_stage():
    problem = make_grid_problem(2, 2)
    dataset = FakeDataset(problem)
    solver = TimedSequenceSolver(
        statuses=["FEASIBLE", "OPTIMAL"],
        wall_times=[1.25, 2.0],
    )
    strat = get_strategy(
        "recursive",
        levels=["BlockGroup_0", "BlockGroup_0"],
        solve_time_limits=[5.0, 10.0],
        carry_over_compute=True,
    )

    solutions = strat.run(dataset, solver)

    assert solver.solve_time_limits == [5.0, 13.75]
    assert solutions[0].metadata["configured_time_limit_seconds"] == 5.0
    assert solutions[0].metadata["effective_time_limit_seconds"] == 5.0
    assert solutions[0].metadata["unused_time_carried_forward_seconds"] == pytest.approx(
        3.75
    )
    assert solutions[1].metadata["carry_over_time_received_seconds"] == pytest.approx(
        3.75
    )
    assert solutions[1].metadata["effective_time_limit_seconds"] == pytest.approx(
        13.75
    )


def test_recursive_carry_over_compute_adds_unused_time_after_infeasible_stage():
    problem = make_grid_problem(2, 2)
    dataset = FakeDataset(problem)
    solver = TimedSequenceSolver(
        statuses=["INFEASIBLE", "OPTIMAL"],
        wall_times=[0.5, 1.0],
        assignments=[{}, None],
    )
    strat = get_strategy(
        "recursive",
        levels=["BlockGroup_0", "BlockGroup_0"],
        solve_time_limits=[4.0, 7.0],
        carry_over_compute=True,
    )

    solutions = strat.run(dataset, solver)

    assert solver.solve_time_limits == [4.0, 10.5]
    assert solutions[0].metadata["unused_time_carried_forward_seconds"] == pytest.approx(
        3.5
    )
    assert solutions[1].metadata["carry_over_time_received_seconds"] == pytest.approx(
        3.5
    )
    assert solver.problems[1].hint is None


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


class TimedSequenceSolver:
    def __init__(self, statuses, wall_times, assignments=None):
        self.statuses = list(statuses)
        self.wall_times = list(wall_times)
        self.assignments = list(assignments) if assignments is not None else None
        self.options = {}
        self.problems = []
        self.solve_time_limits = []

    def solve(self, problem):
        idx = len(self.problems)
        self.problems.append(problem)
        self.solve_time_limits.append(self.options.get("solve_time_limit"))
        assignment = None
        if self.assignments is not None:
            assignment = self.assignments[idx]
        if assignment is None:
            assignment = _split_assignment(problem)
        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status=self.statuses[idx],
            objective=0.0,
            wall_time=self.wall_times[idx],
            metadata={"solver": "timed_sequence"},
        )


def _split_assignment(problem):
    nodes = sorted(problem.nodes)
    midpoint = len(nodes) // 2
    return {node: 0 if idx < midpoint else 1 for idx, node in enumerate(nodes)}


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
