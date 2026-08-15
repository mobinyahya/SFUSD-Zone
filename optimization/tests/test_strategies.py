"""Strategy tests using a FakeDataset (no SFUSD data required)."""

import pytest

from choice.objective import ChoiceCut, ChoiceEvaluation
from optimization.config import OptimizationConfig
from optimization.data.initial_solutions import InitialSolution
from optimization.problem import DuplicateCentroidError
from optimization.solution import ZoneSolution
from optimization.solvers import get_solver
from optimization.strategies import get_strategy
from optimization.strategies import iterative_choice as iterative_choice_module
from optimization.strategies import single as single_module
from optimization.strategies.base import available_strategies
from optimization.tests.synthetic import FakeDataset, make_grid_problem


def test_only_supported_strategies_are_registered():
    assert available_strategies() == ["iterative_choice", "recursive", "single"]


@pytest.mark.parametrize(
    "strategy",
    [
        "overlapping",
        "cutoffs",
        "welfare",
        "approximate_welfare",
        "zoned_column_generation",
        "zoned_benders",
    ],
)
def test_config_rejects_removed_strategies(strategy):
    with pytest.raises(ValueError, match="strategy must be one of"):
        OptimizationConfig(levels=["BlockGroup_0"], strategy=strategy)


def test_single_strategy():
    problem = make_grid_problem(3, 3)
    dataset = FakeDataset(problem)
    solver = get_solver("cp_int", solve_time_limit=10, workers=1)
    strat = get_strategy("single", levels=["BlockGroup_0"], boundary_prop=0.0)

    solutions = strat.run(dataset, solver)

    assert len(solutions) == 1
    assert solutions[-1].status in ("OPTIMAL", "FEASIBLE")
    assert solutions[-1].is_contiguous()
    assert solutions[-1].problem.boundary_prop < 0


def test_single_math_programming_solver_uses_generated_hint(monkeypatch):
    problem = make_grid_problem(3, 3)
    dataset = FakeDataset(problem)
    solver = TimedSequenceSolver(statuses=["OPTIMAL"], wall_times=[0.0])
    solver.name = "cp_int"
    solver.options["hints"] = "voronoi"
    hint = {node: 0 if node < 4 else 1 for node in problem.nodes}

    monkeypatch.setattr(
        single_module,
        "initial_solution",
        lambda problem_arg, hints: InitialSolution(
            assignment=hint,
            metadata={"hints": hints},
        ),
    )
    strat = get_strategy(
        "single",
        levels=["BlockGroup_0"],
        hints="voronoi",
    )

    strat.run(dataset, solver)

    assert solver.problems[0].hint == hint


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
    assert solutions[0].metadata[
        "unused_time_carried_forward_seconds"
    ] == pytest.approx(3.75)
    assert solutions[1].metadata["carry_over_time_received_seconds"] == pytest.approx(
        3.75
    )
    assert solutions[1].metadata["effective_time_limit_seconds"] == pytest.approx(13.75)


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
    assert solutions[0].metadata[
        "unused_time_carried_forward_seconds"
    ] == pytest.approx(3.5)
    assert solutions[1].metadata["carry_over_time_received_seconds"] == pytest.approx(
        3.5
    )
    assert solver.problems[1].hint is None


def test_recursive_skips_nonfinal_duplicate_centroid_stage():
    problem = make_grid_problem(2, 2)
    dataset = FakeDataset(problem)
    solver = DuplicateCentroidSequenceSolver([True, False])
    strat = get_strategy(
        "recursive",
        levels=["BlockGroup_1", "BlockGroup_0"],
        solve_time_limits=[4.0, 7.0],
        carry_over_compute=True,
    )

    solutions = strat.run(dataset, solver)

    assert [solution.status for solution in solutions] == ["SKIPPED", "OPTIMAL"]
    assert solutions[0].assignment == {}
    assert solutions[0].metadata["skip_reason"] == "duplicate_centroid"
    assert solutions[0].metadata["duplicate_centroid_node"] == 23
    assert solutions[0].metadata["duplicate_centroid_zones"] == [0, 3]
    assert solutions[0].metadata["unused_time_carried_forward_seconds"] == 4.0
    assert solver.solve_time_limits == [4.0, 11.0]
    assert solver.problems[1].hint is None


def test_recursive_raises_final_duplicate_centroid_stage():
    problem = make_grid_problem(2, 2)
    dataset = FakeDataset(problem)
    solver = DuplicateCentroidSequenceSolver([True])
    strat = get_strategy(
        "recursive",
        levels=["BlockGroup_0"],
        solve_time_limits=[4.0],
    )

    with pytest.raises(
        DuplicateCentroidError,
        match="Node 23 is used as multiple centroids",
    ):
        strat.run(dataset, solver)


def test_recursive_looseness_scales_constraints_by_configured_stage():
    problem = make_grid_problem(
        2,
        2,
        frl_dev=0.2,
        racial_dev=0.3,
        overage=0.4,
        shortage=0.1,
    )
    dataset = FakeDataset(problem)
    solver = TimedSequenceSolver(
        statuses=["OPTIMAL", "OPTIMAL", "OPTIMAL"],
        wall_times=[0.0, 0.0, 0.0],
    )
    strat = get_strategy(
        "recursive",
        levels=["BlockGroup_2", "BlockGroup_1", "BlockGroup_0"],
        solve_time_limits=[1.0, 1.0, 1.0],
        looseness=1.2,
    )

    strat.run(dataset, solver)

    multipliers = [1.2**2, 1.2, 1.0]
    assert [p.level.name for p in solver.problems] == [
        "BlockGroup_2",
        "BlockGroup_1",
        "BlockGroup_0",
    ]
    assert [p.frl_dev for p in solver.problems] == pytest.approx(
        [problem.frl_dev * multiplier for multiplier in multipliers]
    )
    assert [p.racial_dev for p in solver.problems] == pytest.approx(
        [problem.racial_dev * multiplier for multiplier in multipliers]
    )
    assert [p.overage for p in solver.problems] == pytest.approx(
        [problem.overage * multiplier for multiplier in multipliers]
    )
    assert [p.shortage for p in solver.problems] == pytest.approx(
        [problem.shortage * multiplier for multiplier in multipliers]
    )


def test_recursive_looseness_scales_by_configured_stages_when_levels_skip():
    problem = make_grid_problem(2, 2, frl_dev=0.2)
    dataset = FakeDataset(problem)
    solver = TimedSequenceSolver(
        statuses=["OPTIMAL", "OPTIMAL"],
        wall_times=[0.0, 0.0],
    )
    strat = get_strategy(
        "recursive",
        levels=["BlockGroup_2", "BlockGroup_0"],
        solve_time_limits=[1.0, 1.0],
        looseness=1.2,
    )

    strat.run(dataset, solver)

    assert [p.frl_dev for p in solver.problems] == pytest.approx([0.24, 0.2])


def test_recursive_rejects_tightening_looseness():
    problem = make_grid_problem(2, 2)
    dataset = FakeDataset(problem)
    solver = TimedSequenceSolver(statuses=["OPTIMAL"], wall_times=[0.0])
    strat = get_strategy(
        "recursive",
        levels=["BlockGroup_0"],
        solve_time_limits=[1.0],
        looseness=0.9,
    )

    with pytest.raises(ValueError, match="looseness"):
        strat.run(dataset, solver)


def test_recursive_anchors_centroids_in_projected_hint():
    problem = make_grid_problem(3, 3)
    dataset = FakeDataset(problem)
    bad_assignment = {node: 1 for node in problem.nodes}
    solver = TimedSequenceSolver(
        statuses=["OPTIMAL", "OPTIMAL"],
        wall_times=[0.0, 0.0],
        assignments=[bad_assignment, None],
    )
    strat = get_strategy(
        "recursive",
        levels=["BlockGroup_0", "BlockGroup_0"],
        solve_time_limits=[1.0, 1.0],
    )

    strat.run(dataset, solver)

    refined = solver.problems[1]
    assert refined.hint[problem.centroids[0]] == 0
    assert refined.hint[problem.centroids[1]] == 1
    assert refined.candidate_zones(problem.centroids[0]) == {0}
    assert refined.candidate_zones(problem.centroids[1]) == {1}


def test_iterative_choice_strategy_terminates():
    problem = make_grid_problem(3, 3)
    dataset = FakeDataset(problem)
    solver = get_solver("cp_int", solve_time_limit=10, workers=1)
    strat = get_strategy(
        "iterative_choice",
        levels=["BlockGroup_0"],
        max_iterations=3,
        choice_model="distance",
        boundary_prop=0.5,
    )

    solutions = strat.run(dataset, solver)

    assert 1 <= len(solutions) <= 3
    last = solutions[-1]
    assert "choice_utility" in last.metadata
    assert solutions[0].metadata["choice_objective_cuts"] == 0
    assert solutions[0].metadata["choice_cuts_added"] > 0
    assert last.is_contiguous()
    assert all(solution.problem.boundary_prop == 0.5 for solution in solutions)


def test_config_passes_choice_utility_hints_to_iterative_strategy():
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        strategy="iterative_choice",
        choice_utility_hints=True,
        boundary_prop=0.25,
    )

    strategy = config.make_strategy()

    assert strategy.options["choice_utility_hints"] is True
    assert strategy.options["boundary_prop"] == 0.25


def test_config_rejects_non_boolean_remove_city_wide():
    with pytest.raises(ValueError, match="remove_city_wide"):
        OptimizationConfig(levels=["BlockGroup_0"], remove_city_wide=1)


@pytest.mark.parametrize("value", [1.01, float("nan"), True, "invalid"])
def test_config_rejects_invalid_boundary_prop(value):
    with pytest.raises(ValueError, match="boundary_prop"):
        OptimizationConfig(levels=["BlockGroup_0"], boundary_prop=value)


@pytest.mark.parametrize("value", [-1, -0.25, 0, 1])
def test_config_accepts_boundary_prop_and_disabled_values(value):
    config = OptimizationConfig(levels=["BlockGroup_0"], boundary_prop=value)

    assert config.boundary_prop == float(value)


def test_iterative_choice_seeds_choice_utility_hint_cuts(monkeypatch):
    problem = make_grid_problem(2, 2)
    dataset = FakeDataset(problem)
    solver = TimedSequenceSolver(statuses=["OPTIMAL"], wall_times=[0.0])
    model = HintCutModel()
    monkeypatch.setattr(
        iterative_choice_module,
        "get_configured_choice_model",
        lambda options: model,
    )
    strat = get_strategy(
        "iterative_choice",
        levels=["BlockGroup_0"],
        max_iterations=1,
        choice_model="mnl",
        choice_utility_hints=True,
    )

    solutions = strat.run(dataset, solver)

    assert solver.problems[0].choice_objective.cuts == model.hint_cuts
    assert solutions[0].metadata["choice_objective_cuts"] == len(model.hint_cuts)
    assert solutions[0].metadata["choice_utility_hint_cuts"] == len(model.hint_cuts)


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


class DuplicateCentroidSequenceSolver:
    def __init__(self, duplicate_by_call):
        self.duplicate_by_call = list(duplicate_by_call)
        self.options = {}
        self.problems = []
        self.solve_time_limits = []

    def solve(self, problem):
        idx = len(self.problems)
        self.problems.append(problem)
        self.solve_time_limits.append(self.options.get("solve_time_limit"))
        if self.duplicate_by_call[idx]:
            raise DuplicateCentroidError(23, {0, 3})
        return ZoneSolution(
            problem=problem,
            assignment=_split_assignment(problem),
            status="OPTIMAL",
            objective=0.0,
            wall_time=1.0,
            metadata={"solver": "duplicate_centroid_sequence"},
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


class HintCutModel:
    def __init__(self):
        self.hint_cuts = (ChoiceCut(node=0, zone=0, constant=1.0),)

    def utility_bounds(self, problem):
        return -1_000.0, 1_000.0

    def choice_utility_hint_cuts(self, problem):
        return self.hint_cuts

    def evaluate_with_cuts(self, problem, assignment):
        return ChoiceEvaluation(utility=1.0, cuts=())
