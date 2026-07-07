"""Real-data ReCom regression tests.

These tests exercise the exact short-bursts ReCom cases that previously hung after
writing only the initial progress row. They are skipped when the cached SFUSD
graphs are not available, but when real data is present they verify that the
solver returns under the configured timeout instead of getting trapped inside a
GerryChain proposal.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path

import pytest

from Zone_Generation.optimization.config import OptimizationConfig
from Zone_Generation.optimization.levels import LevelSpec


_FORMERLY_STALLED_CASES = [
    ("6-zone-3", 1),
    ("8-zone-24", 3),
]
_SOLVE_SECONDS = 2.0
_PROCESS_TIMEOUT_SECONDS = 30.0


def _short_bursts_config(centroids_type: str, seed: int) -> OptimizationConfig:
    return OptimizationConfig(
        centroids_type=centroids_type,
        seed=seed,
        levels=["Block_0", "Block_1"],
        solver="short_bursts_recom",
        strategy="single",
        solve_time_limits=[_SOLVE_SECONDS],
        max_distance=4.52,
        hints="gerry_chain",
        workers=1,
        racial_dev=-1,
        frl_dev=0.15,
        overage=0.15,
        shortage=0.15,
        recom_iterations=1_000_000_000,
        recom_cut_attempts=10,
        recom_population_epsilon=10.0,
        recom_balance_metric="nodes",
        short_bursts_length=25,
    )


def _skip_without_cached_real_graphs() -> None:
    config = _short_bursts_config(*_FORMERLY_STALLED_CASES[0])
    dataset = config.make_dataset()
    graph_path = Path(dataset._graph_path(LevelSpec.parse("Block_1")))
    if not graph_path.exists():
        pytest.skip(f"cached real-data graph not available: {graph_path}")


def _run_short_bursts_case(
    centroids_type: str,
    seed: int,
    output: mp.Queue,
) -> None:
    try:
        config = _short_bursts_config(centroids_type, seed)
        start = time.monotonic()
        solution = config.make_strategy().run(config.make_dataset(), config.make_solver())[
            -1
        ]
        output.put(
            {
                "ok": True,
                "elapsed_seconds": time.monotonic() - start,
                "solution_status": solution.status,
                "wall_time": solution.wall_time,
                "attempted_moves": solution.metadata.get("attempted_moves"),
                "accepted_moves": solution.metadata.get("accepted_moves"),
                "proposal_failures": solution.metadata.get("proposal_failures"),
                "recom_balance_metric": solution.metadata.get("recom_balance_metric"),
                "recom_population_epsilon": solution.metadata.get(
                    "recom_population_epsilon"
                ),
            }
        )
    except BaseException:
        output.put({"ok": False, "traceback": traceback.format_exc()})


@pytest.mark.parametrize("centroids_type,seed", _FORMERLY_STALLED_CASES)
def test_short_bursts_recom_real_data_stalled_cases_return(
    centroids_type: str,
    seed: int,
) -> None:
    _skip_without_cached_real_graphs()

    context = mp.get_context("fork")
    output = context.Queue()
    process = context.Process(
        target=_run_short_bursts_case,
        args=(centroids_type, seed, output),
    )
    process.start()
    process.join(_PROCESS_TIMEOUT_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail(
            f"short_bursts_recom stalled on real-data case "
            f"{centroids_type} seed {seed}"
        )

    assert process.exitcode == 0
    try:
        result = output.get_nowait()
    except queue.Empty:
        pytest.fail(
            f"short_bursts_recom process returned no result for "
            f"{centroids_type} seed {seed}"
        )

    assert result["ok"], result.get("traceback")
    assert result["recom_balance_metric"] == "nodes"
    assert result["recom_population_epsilon"] == pytest.approx(10.0)
    assert result["wall_time"] <= _SOLVE_SECONDS + 5.0
    assert result["elapsed_seconds"] <= _PROCESS_TIMEOUT_SECONDS
    assert result["attempted_moves"] > 0
    assert result["accepted_moves"] > 0
