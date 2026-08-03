"""Strategy tests using a FakeDataset (no SFUSD data required)."""

import threading
import time
from types import SimpleNamespace

import pytest

from choice.objective import ChoiceCut, ChoiceEvaluation
from optimization.config import OptimizationConfig
from optimization.data.initial_solutions import InitialSolution
from optimization.problem import CutoffMarket, DuplicateCentroidError
from optimization.solution import ZoneSolution
from optimization.solvers import get_solver
from optimization.strategies import (
    iterative_choice as iterative_choice_module,
)
from optimization.strategies import cutoffs as cutoffs_module
from optimization.strategies import overlapping as overlapping_module
from optimization.strategies import single as single_module
from optimization.strategies import get_strategy
from optimization.tests.synthetic import (
    FakeDataset,
    make_grid_problem,
    make_single_zone_problem,
)


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
        [problem.frl_dev * m for m in multipliers]
    )
    assert [p.racial_dev for p in solver.problems] == pytest.approx(
        [problem.racial_dev * m for m in multipliers]
    )
    assert [p.overage for p in solver.problems] == pytest.approx(
        [problem.overage * m for m in multipliers]
    )
    assert [p.shortage for p in solver.problems] == pytest.approx(
        [problem.shortage * m for m in multipliers]
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


def test_cutoffs_config_requires_cp_bool_year_23_all_programs():
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        strategy="cutoffs",
        solver="cp_bool",
        years=[23],
        population_type="All",
        boundary_prop=0.5,
    )

    strategy = config.make_strategy()

    assert strategy.name == "cutoffs"
    assert strategy.options["cutoff_lottery_scale"] == 20
    assert strategy.options["boundary_prop"] == 0.5
    assert strategy.options["remove_city_wide"] is False


def test_welfare_config_requires_isolated_year_23_all_program_markets():
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        strategy="welfare",
        solver="cp_bool",
        years=[23],
        population_type="All",
        remove_city_wide=True,
        welfare_utility_scale=10_000,
    )

    strategy = config.make_strategy()

    assert strategy.name == "welfare"
    assert strategy.options["remove_city_wide"] is True
    assert strategy.options["welfare_utility_scale"] == 10_000
    assert strategy.options["welfare_prefix_depth"] == 10


def test_cutoffs_strategy_applies_boundary_prop(monkeypatch):
    problem = make_grid_problem(2, 2, population_type="All")
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(years=[23], population_type="All")
    solver = TimedSequenceSolver(statuses=["OPTIMAL"], wall_times=[0.0])
    solver.name = "cp_bool"
    market = CutoffMarket(
        students=(),
        school_nodes={},
        school_capacities={},
        zone_restricted_schools=frozenset(),
        lottery_scale=10,
    )
    cutoff_options = {}

    def build_market(*args, **kwargs):
        cutoff_options.update(kwargs)
        return market

    monkeypatch.setattr(cutoffs_module, "build_cutoff_market", build_market)
    strategy = OptimizationConfig(
        levels=["BlockGroup_0"],
        strategy="cutoffs",
        solver="cp_bool",
        years=[23],
        population_type="All",
        boundary_prop=0.5,
        remove_city_wide=True,
    ).make_strategy()

    strategy.run(dataset, solver)

    assert solver.problems[0].boundary_prop == 0.5
    assert cutoff_options["remove_city_wide"] is True


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


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"solver": "cp_int"}, "cp_bool"),
        ({"years": [22]}, "years"),
        ({"population_type": "GE"}, "population_type"),
    ],
)
def test_cutoffs_config_rejects_unsupported_inputs(overrides, message):
    params = {
        "levels": ["BlockGroup_0"],
        "strategy": "cutoffs",
        "solver": "cp_bool",
        "years": [23],
        "population_type": "All",
    }
    params.update(overrides)

    with pytest.raises(ValueError, match=message):
        OptimizationConfig(**params)


def test_config_passes_school_solve_time_limit_to_overlapping_strategy():
    config = OptimizationConfig(
        levels=["Block_0"],
        strategy="overlapping",
        school_solve_time_limit=12.5,
    )

    strategy = config.make_strategy()

    assert strategy.options["school_solve_time_limit"] == 12.5


@pytest.mark.parametrize("value", [0, -1, float("inf")])
def test_config_rejects_invalid_school_solve_time_limit(value):
    with pytest.raises(ValueError, match="school_solve_time_limit"):
        OptimizationConfig(
            levels=["Block_0"],
            strategy="overlapping",
            school_solve_time_limit=value,
        )


def test_overlapping_strategy_runs_school_solves_in_parallel(monkeypatch):
    problem = make_grid_problem(3, 3)
    dataset = FakeDataset(problem)
    full_solver = RecordingFullSolver(workers=2)
    child_options = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    class ChildSolver:
        def __init__(self, options):
            self.options = options

        def solve(self, child_problem):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            centroid = child_problem.centroids[0]
            school_id = child_problem.G.nodes[centroid]["school_ids"][0]
            selected = {0, 1, 3, 4} if school_id == 100 else {4, 5, 7, 8}
            return ZoneSolution(
                problem=child_problem,
                assignment={node: 0 for node in selected},
                status="FEASIBLE",
                wall_time=0.01,
                metadata={"centroid_school_id": school_id},
            )

    def fake_get_solver(name, **options):
        assert name == "cp_single_zone"
        child_options.append(options)
        return ChildSolver(options)

    monkeypatch.setattr(overlapping_module, "get_solver", fake_get_solver)
    strategy = get_strategy(
        "overlapping",
        levels=["Block_0"],
        school_solve_time_limit=7.5,
        boundary_radius=0,
    )

    solutions = strategy.run(dataset, full_solver)

    assert len(solutions) == 3
    assert [solution.metadata["centroid_school_id"] for solution in solutions[:-1]] == [
        100,
        200,
    ]
    assert max_active == 2
    assert all(options["workers"] == 1 for options in child_options)
    assert all(options["solve_time_limit"] == 7.5 for options in child_options)
    assert full_solver.options["workers"] == 2
    assert full_solver.problem.centroids == [0, 8]
    assert solutions[-1].metadata["centroid_school_ids"] == [100, 200]
    assert solutions[-1].metadata["school_solve_parallelism"] == 2


def test_overlapping_strategy_solves_synthetic_problem_end_to_end():
    problem = make_grid_problem(2, 2)
    dataset = FakeDataset(problem)
    solver = get_solver("cp_int", solve_time_limit=5, workers=2, seed=1)
    strategy = get_strategy(
        "overlapping",
        levels=["BlockGroup_0"],
        school_solve_time_limit=5,
        boundary_radius=0,
    )

    solutions = strategy.run(dataset, solver)

    assert len(solutions) == 3
    assert all(solution.feasible for solution in solutions)
    assert set(solutions[-1].assignment) == set(problem.nodes)
    assert solutions[-1].problem.centroids == [0, 3]
    assert solutions[-1].metadata["school_solve_feasible_count"] == 2


def test_overlapping_fixed_assignments_exclude_overlap_and_all_boundary_bands():
    problem = make_single_zone_problem()
    left = ZoneSolution(
        problem=problem,
        assignment={node: 0 for node in range(6)},
        status="FEASIBLE",
    )
    right = ZoneSolution(
        problem=problem,
        assignment={node: 0 for node in range(3, 7)},
        status="FEASIBLE",
    )

    fixed, boundary_band, counts = (
        overlapping_module.OverlappingStrategy._fixed_assignments(
            problem.G,
            [left, right],
            radius=0,
        )
    )

    assert boundary_band == {2, 3, 5, 6}
    assert fixed == {0: 0, 1: 0}
    assert counts == {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 1}


def test_overlapping_fixed_assignments_ignore_boundary_for_negative_one_radius():
    problem = make_single_zone_problem()
    left = ZoneSolution(
        problem=problem,
        assignment={node: 0 for node in range(6)},
        status="FEASIBLE",
    )
    right = ZoneSolution(
        problem=problem,
        assignment={node: 0 for node in range(3, 7)},
        status="FEASIBLE",
    )

    fixed, boundary_band, counts = (
        overlapping_module.OverlappingStrategy._fixed_assignments(
            problem.G,
            [left, right],
            radius=-1,
        )
    )

    assert boundary_band == set()
    assert fixed == {0: 0, 1: 0, 2: 0, 6: 1}
    assert counts == {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 1}


def test_overlapping_strategy_rejects_colocated_schools():
    problem = make_grid_problem(2, 2)
    problem.G.nodes[0]["school_ids"] = [100, 200]
    problem.G.nodes[0]["num_schools"] = 2
    problem.G.nodes[3]["school_ids"] = []
    problem.G.nodes[3]["num_schools"] = 0
    dataset = FakeDataset(problem)
    strategy = get_strategy(
        "overlapping",
        levels=["BlockGroup_0"],
        school_solve_time_limit=1,
    )

    with pytest.raises(ValueError, match="Colocated schools"):
        strategy.run(dataset, RecordingFullSolver(workers=1))


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


class RecordingFullSolver:
    name = "cp_int"

    def __init__(self, workers):
        self.options = {"workers": workers, "solve_time_limit": 10}
        self.problem = None

    def solve(self, problem):
        self.problem = problem
        midpoint = len(problem.nodes) // 2
        assignment = {
            node: int(idx >= midpoint) for idx, node in enumerate(problem.nodes)
        }
        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status="FEASIBLE",
            objective=0.0,
            wall_time=0.01,
            metadata={"solver": self.name},
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
