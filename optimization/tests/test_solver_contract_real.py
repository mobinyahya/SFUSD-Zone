"""Production solver contract tests on a small cached SFUSD graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from optimization.config import OptimizationConfig
from optimization.data.dataset import Dataset
from optimization.data.initial_solutions import initial_solution
from optimization.levels import LevelSpec
from optimization.problem import ZoneProblem
from optimization.tests.solver_contract import (
    CONTRACT_SOLVERS,
    assert_valid_solution,
    solve_contract_problem,
)

REAL_LEVEL = "Block_2"


@dataclass(frozen=True)
class RealContractData:
    config: OptimizationConfig
    dataset: Dataset


@pytest.fixture(scope="module")
def real_contract_data() -> RealContractData:
    config = OptimizationConfig(
        centroids_type="2-zone-mcmc",
        levels=[REAL_LEVEL],
        strategy="single",
        frl_dev=1.0,
        racial_dev=-1.0,
        overage=10.0,
        shortage=1.0,
        max_distance=float("inf"),
        solve_time_limits=[5.0],
        gap_limits=[1.0],
        hints="voronoi",
        seed=1,
        workers=1,
    )
    dataset = Dataset(config)
    graph_path = Path(dataset._graph_path(LevelSpec.parse(REAL_LEVEL)))
    if not graph_path.exists():
        pytest.skip(f"cached real-data graph not available: {graph_path}")
    return RealContractData(config=config, dataset=dataset)


def _real_problem(data: RealContractData) -> ZoneProblem:
    problem = data.dataset.problem_for(REAL_LEVEL)
    initial = initial_solution(problem, "voronoi")
    assert initial is not None
    problem.hint = initial.assignment
    return problem


@pytest.mark.real_data
@pytest.mark.parametrize("solver_name", CONTRACT_SOLVERS)
def test_solver_satisfies_contract_on_real_data(
    solver_name: str,
    real_contract_data: RealContractData,
) -> None:
    problem = _real_problem(real_contract_data)

    solution = solve_contract_problem(
        solver_name,
        problem,
        solve_time_limit=5.0,
    )

    assert_valid_solution(problem, solution)
    assert solution.metadata["solver"] == solver_name
