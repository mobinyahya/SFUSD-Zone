"""Production solver contract tests on a small cached SFUSD graph."""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from dataclasses import dataclass
from pathlib import Path

import pytest

from optimization.config import OptimizationConfig
from optimization.data.dataset import Dataset
from optimization.data.initial_solutions import initial_solution
from optimization.levels import LevelSpec
from optimization.problem import ZoneProblem
from optimization.solution import ZoneSolution
from optimization.solvers import get_solver
from optimization.tests.solver_contract import (
    CONTRACT_SOLVERS,
    RECOM_SOLVERS,
    assert_valid_solution,
    solve_contract_problem,
)

REAL_LEVEL = "Block_2"
PROCESS_TIMEOUT_SECONDS = 15.0


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
        recom_iterations=0,
        recom_cut_attempts=10,
        recom_population_epsilon=10.0,
        recom_balance_metric="nodes",
        short_bursts_length=1,
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
        recom_iterations=0,
    )

    assert_valid_solution(problem, solution)
    assert solution.metadata["solver"] == solver_name


def _run_one_recom_step(
    solver_name: str,
    problem: ZoneProblem,
    output: mp.Queue,
) -> None:
    try:
        solution = get_solver(
            solver_name,
            solve_time_limit=2.0,
            seed=1,
            hints="voronoi",
            recom_iterations=1,
            recom_cut_attempts=10,
            recom_population_epsilon=10.0,
            recom_balance_metric="nodes",
            recom_repair_iterations=0,
            short_bursts_length=1,
            relaxed_recom_min_boundary_edges=0,
        ).solve(problem)
        output.put(
            {
                "ok": True,
                "assignment": solution.assignment,
                "status": solution.status,
                "objective": solution.objective,
                "wall_time": solution.wall_time,
                "metadata": solution.metadata,
            }
        )
    except BaseException:
        output.put({"ok": False, "traceback": traceback.format_exc()})


@pytest.mark.real_data
@pytest.mark.parametrize("solver_name", RECOM_SOLVERS)
def test_recom_solver_completes_one_real_data_step(
    solver_name: str,
    real_contract_data: RealContractData,
) -> None:
    problem = _real_problem(real_contract_data)
    context = mp.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=_run_one_recom_step,
        args=(solver_name, problem, output),
    )
    process.start()
    process.join(PROCESS_TIMEOUT_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail(f"{solver_name} exceeded the real-data process timeout")

    assert process.exitcode == 0
    try:
        result = output.get_nowait()
    except queue.Empty:
        pytest.fail(f"{solver_name} process returned no result")
    assert result["ok"], result.get("traceback")

    solution = ZoneSolution(
        problem=problem,
        assignment=result["assignment"],
        status=result["status"],
        objective=result["objective"],
        wall_time=result["wall_time"],
        metadata=result["metadata"],
    )
    assert_valid_solution(problem, solution)
    assert solution.metadata["attempted_moves"] == 1
