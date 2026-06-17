"""Strategy tests using a FakeDataset (no SFUSD data required)."""

from Zone_Generation.pipeline.solvers import get_solver
from Zone_Generation.pipeline.strategies import get_strategy
from Zone_Generation.pipeline.tests.synthetic import FakeDataset, make_grid_problem


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
        choice_model_options={},
    )
    solutions = strat.run(dataset, solver)
    assert 1 <= len(solutions) <= 3
    last = solutions[-1]
    assert "choice_utility" in last.metadata
    assert last.is_contiguous()
