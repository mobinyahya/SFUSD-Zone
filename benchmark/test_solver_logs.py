"""Tests for heuristic progress logs and the solver-log parser."""

from __future__ import annotations

import json
import os

import pytest

from benchmark.solver_logs import (
    collect,
    detect_format,
    parse_gurobi_log,
    parse_log_name,
    parse_recom_log,
)
from optimization.levels import LevelSpec
from optimization.problem import ZoneProblem
from optimization.solvers import get_solver
from optimization.solvers.heuristic_log import HeuristicProgressLog
from optimization.tests.synthetic import make_grid_graph

HEURISTIC_SOLVERS = ("recom", "relaxed_recom", "short_bursts", "adaptive_short_bursts")

GUROBI_LOG = """Gurobi Optimizer version 11.0.0 build v11.0.0rc2 (mac64[arm])

Root relaxation: objective 1.000000e+02, 10 iterations, 0.50 seconds (0.1 work units)

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0          -    0             -  100.0000      -     -    5s
H    0     0                       150.0000  100.0000  33.3%     -   10s
*   12     4               7       120.0000  110.0000  8.33%  25.0   20s

Explored 12 nodes (345 simplex iterations) in 20.13 seconds (16.52 work units)

Time limit reached
Best objective 1.200000000000e+02, best bound 1.100000000000e+02, gap 8.3333%
"""


def skewed_frl_problem(rows: int = 6, cols: int = 6, **overrides) -> ZoneProblem:
    """A grid whose FRL is split left/right, so many partitions are infeasible."""

    G = make_grid_graph(rows, cols)
    for node, attrs in G.nodes(data=True):
        attrs["FRL"] = 1.0 if (node % cols) < cols // 2 else 0.0
    G.graph["F"] = 0.5
    params = dict(
        frl_dev=0.02,
        racial_dev=-1.0,
        overage=5.0,
        shortage=0.0,
        max_distance=float("inf"),
    )
    params.update(overrides)
    return ZoneProblem(
        G=G,
        level=LevelSpec("BlockGroup", 0),
        centroids=[0, rows * cols - 1],
        centroid_school_ids=[100, 200],
        **params,
    )


# ---------------------------------------------------------------------- #
# the log writer
# ---------------------------------------------------------------------- #
def test_log_records_only_strict_improvements(tmp_path) -> None:
    log = HeuristicProgressLog(
        str(tmp_path / "progress.jsonl"),
        solver="recom",
        level="Block_2",
        labels=("capacity_upper",),
        start=0.0,
    )

    # A feasible state, then a worse one, then a tie, then a real improvement.
    assert log.record(feasible=True, boundary_cost=10.0, violations=(0.0,))
    assert not log.record(feasible=True, boundary_cost=11.0, violations=(0.0,))
    assert not log.record(feasible=True, boundary_cost=10.0, violations=(0.0,))
    assert log.record(feasible=True, boundary_cost=9.0, violations=(0.0,))
    # An infeasible state never beats a feasible one, however cheap its cut.
    assert not log.record(feasible=False, boundary_cost=1.0, violations=(5.0,))
    log.finish(stop_reason="iteration_limit")

    records = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    assert [row["record"] for row in records] == [
        "header",
        "improvement",
        "improvement",
        "summary",
    ]
    assert [row["boundary_cost"] for row in records[1:3]] == [10.0, 9.0]
    assert records[-1]["improvements"] == 2


def test_infeasible_states_improve_by_unweighted_lagrangian(tmp_path) -> None:
    log = HeuristicProgressLog(
        str(tmp_path / "progress.jsonl"),
        solver="relaxed_recom",
        level="Block_2",
        labels=("capacity_upper", "frl_upper"),
        start=0.0,
    )

    assert log.record(feasible=False, boundary_cost=10.0, violations=(3.0, 4.0))
    # Cheaper cut but far worse residuals: a higher Lagrangian, so no record.
    assert not log.record(feasible=False, boundary_cost=1.0, violations=(9.0, 0.0))
    assert log.record(
        feasible=False, boundary_cost=10.0, violations=(1.0, 1.0), weights=(2.0, 3.0)
    )
    log.finish()

    records = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    first, second = records[1], records[2]
    assert first["unweighted_lagrangian"] == pytest.approx(10.0 + 9.0 + 16.0)
    assert first["penalties"] == {"capacity_upper": 3.0, "frl_upper": 4.0}
    assert "weighted_lagrangian" not in first
    assert second["unweighted_lagrangian"] == pytest.approx(12.0)
    assert second["weighted_lagrangian"] == pytest.approx(10.0 + 2.0 + 3.0)


def test_log_rejects_mismatched_violation_length(tmp_path) -> None:
    log = HeuristicProgressLog(
        str(tmp_path / "progress.jsonl"),
        solver="recom",
        level="Block_2",
        labels=("capacity_upper",),
        start=0.0,
    )
    with pytest.raises(ValueError, match="labels"):
        log.record(feasible=True, boundary_cost=1.0, violations=(0.0, 0.0))


# ---------------------------------------------------------------------- #
# solver integration
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("solver_name", HEURISTIC_SOLVERS)
def test_heuristic_solvers_write_a_progress_log(solver_name: str, tmp_path) -> None:
    solution = get_solver(
        solver_name,
        save_solver_logs=True,
        output_dir=str(tmp_path),
        recom_iterations=200,
        solve_time_limit=30,
        seed=5,
    ).solve(skewed_frl_problem())

    log_path = solution.metadata["solver_log_path"]
    assert solution.metadata["solver_log_format"] == "jsonl"
    assert log_path == os.path.join(
        "solver_logs", f"solver_00_BlockGroup_0_{solver_name}.jsonl"
    )

    rows = parse_recom_log(str(tmp_path / log_path))
    improvements = [row for row in rows if row["record"] == "improvement"]
    assert len(improvements) == solution.metadata["solver_log_improvements"]
    assert improvements

    # Improvements strictly decrease under the logged ordering: feasible first,
    # then boundary cost (feasible) or unweighted Lagrangian (infeasible).
    keys = [
        (
            0 if row["feasible"] else 1,
            row["boundary_cost"] if row["feasible"] else row["unweighted_lagrangian"],
        )
        for row in improvements
    ]
    assert all(later < earlier for earlier, later in zip(keys, keys[1:]))
    assert all(row["elapsed_seconds"] is not None for row in improvements)
    elapsed = [row["elapsed_seconds"] for row in improvements]
    assert elapsed == sorted(elapsed)

    # Every constraint is named and every penalty is reported.
    labels = {
        "capacity_lower",
        "frl_lower",
        "frl_upper",
        "schools_lower",
        "schools_upper",
    }
    assert labels <= {
        key[len("pen_") :] for key in improvements[0] if key.startswith("pen_")
    }
    if solver_name == "adaptive_short_bursts":
        assert improvements[0]["weighted_lagrangian"] is not None
        assert any(key.startswith("w_") for key in improvements[0])


def test_no_progress_log_without_save_solver_logs(tmp_path) -> None:
    solution = get_solver(
        "recom",
        output_dir=str(tmp_path),
        recom_iterations=50,
        solve_time_limit=30,
        seed=5,
    ).solve(skewed_frl_problem())

    assert "solver_log_path" not in solution.metadata
    assert not (tmp_path / "solver_logs").exists()


def test_violation_labels_cover_every_constraint() -> None:
    from optimization.solvers.recom import _ReComContext

    context = _ReComContext(skewed_frl_problem(racial_dev=0.1))
    labels = context.violation_labels()

    assert len(labels) == context.violation_count
    assert len(set(labels)) == len(labels)
    assert labels[:2] == ("capacity_lower", "capacity_upper")
    assert labels[-2:] == ("schools_lower", "schools_upper")
    assert any(label.startswith("racial_") for label in labels)


# ---------------------------------------------------------------------- #
# the parser
# ---------------------------------------------------------------------- #
def test_parse_log_name_handles_units_with_underscores() -> None:
    assert parse_log_name("solver_03_Block_2_cp_bool.log") == {
        "log_index": 3,
        "level": "Block_2",
        "solver": "cp_bool",
    }
    assert parse_log_name("solver_00_attendance_area_1_relaxed_recom.jsonl") == {
        "log_index": 0,
        "level": "attendance_area_1",
        "solver": "relaxed_recom",
    }


def test_parse_gurobi_log_reads_the_node_table(tmp_path) -> None:
    path = tmp_path / "solver_00_Block_2_mip.log"
    path.write_text(GUROBI_LOG)
    assert detect_format(str(path)) == "gurobi"

    rows = parse_gurobi_log(str(path))
    by_record: dict[str, list[dict]] = {}
    for row in rows:
        by_record.setdefault(row["record"], []).append(row)

    assert by_record["root_relaxation"][0]["bound"] == pytest.approx(100.0)
    nodes = by_record["node"]
    assert [row["elapsed_seconds"] for row in nodes] == [5.0, 10.0, 20.0]
    assert nodes[0]["incumbent"] is None  # the "-" incumbent column
    assert nodes[1]["incumbent"] == pytest.approx(150.0)
    assert nodes[2]["gap_rel"] == pytest.approx(0.0833)
    assert nodes[2]["gap_abs"] == pytest.approx(10.0)

    final = by_record["final"][0]
    assert final["incumbent"] == pytest.approx(120.0)
    assert final["bound"] == pytest.approx(110.0)
    assert final["elapsed_seconds"] == pytest.approx(20.13)


def test_collect_merges_formats_into_one_frame(tmp_path) -> None:
    run_dir = tmp_path / "6-zone-3" / "seed1" / "run"
    (run_dir / "solver_logs").mkdir(parents=True)
    (run_dir / "benchmark_manifest.json").write_text(
        json.dumps({"task_id": "t1", "config_hash": "abc", "stages": []})
    )
    (run_dir / "solver_logs" / "solver_01_Block_2_mip.log").write_text(GUROBI_LOG)

    get_solver(
        "relaxed_recom",
        save_solver_logs=True,
        output_dir=str(run_dir),
        recom_iterations=100,
        solve_time_limit=30,
        seed=5,
    ).solve(skewed_frl_problem())

    frame = collect(str(tmp_path), out_csv="solver_progress.csv")

    assert (tmp_path / "solver_progress.csv").exists()
    assert set(frame["solver"]) == {"mip", "relaxed_recom"}
    assert set(frame["log_format"]) == {"gurobi", "recom"}
    assert set(frame["task_id"]) == {"t1"}
    # The unified columns: heuristics fill incumbent only, the MIP fills both.
    mip = frame[frame["solver"] == "mip"]
    heuristic = frame[frame["solver"] == "relaxed_recom"]
    assert mip["bound"].notna().any()
    assert heuristic["bound"].isna().all()
    assert heuristic["boundary_cost"].notna().any()
    assert frame["elapsed_seconds"].notna().all()


def test_collect_returns_empty_frame_without_logs(tmp_path) -> None:
    assert collect(str(tmp_path)).empty
