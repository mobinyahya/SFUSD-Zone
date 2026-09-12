"""Regression tests for feasible-only, correctly timed cut-edge trajectories."""

import json

import numpy as np
import pandas as pd
import pytest

from benchmark.plot_edges import (
    load_trajectories,
    main,
    prepare_trajectory,
    render_figures,
    summarize_traces,
    visible_trajectories,
)
from benchmark.solver_logs import parse_cpsat_log, parse_gurobi_log


@pytest.mark.parametrize("status", ["UNKNOWN", "INFEASIBLE", "MODEL_INVALID"])
def test_cpsat_unsolved_summary_is_not_a_feasible_point(tmp_path, status):
    path = tmp_path / "cp.log"
    path.write_text(
        f"Starting CP-SAT solver\nCpSolverResponse summary:\nstatus: {status}\n"
        "objective: 21646\nbest_bound: 30\nwalltime: 600.2\n"
    )
    row = parse_cpsat_log(str(path))[-1]
    assert row["incumbent"] is None
    assert row["bound"] == (30 if status == "UNKNOWN" else None)


def test_cpsat_feasible_events_and_bounds(tmp_path):
    path = tmp_path / "cp.log"
    path.write_text(
        "Starting CP-SAT solver\n"
        "#Bound   0.50s best:inf next:[10,100]\n"
        "#1       2.00s best:80 next:[10,79] default_lp\n"
        "#Bound   3.00s best:80 next:[20,79]\n"
        "#2       4.00s best:70 next:[20,69] default_lp\n"
        "CpSolverResponse summary:\nstatus: FEASIBLE\n"
        "objective: 70\nbest_bound: 20\nwalltime: 6\n"
    )
    trace = prepare_trajectory(pd.DataFrame(parse_cpsat_log(str(path))))
    np.testing.assert_allclose(trace["incumbent"], [np.nan, 80, 80, 70, 70])
    assert trace["bound"].tolist() == [10, 10, 20, 20, 20]
    assert trace["elapsed_seconds"].tolist() == [0.5, 2, 3, 4, 6]


def test_gurobi_final_bound_without_incumbent_or_gap(tmp_path):
    path = tmp_path / "mip.log"
    path.write_text(
        "Gurobi Optimizer version 13\n"
        "Explored 1 nodes (10 simplex iterations) in 600.25 seconds\n"
        "Best objective -, best bound 1.340000000000e+02, gap -\n"
    )
    final = parse_gurobi_log(str(path))[-1]
    assert final["incumbent"] is None
    assert final["bound"] == 134
    assert final["elapsed_seconds"] == 600.25


def test_step_trace_does_not_backfill_and_ends_at_logged_summary():
    events = pd.DataFrame(
        [
            {"record": "root_relaxation", "elapsed_seconds": 0.1, "bound": 999},
            {"elapsed_seconds": 1, "feasible": False, "incumbent": 10},
            {"elapsed_seconds": 2, "feasible": True, "incumbent": 80},
            {"elapsed_seconds": 2, "feasible": True, "incumbent": 70},
            {"elapsed_seconds": 4, "feasible": True, "incumbent": 75},
            {"record": "summary", "elapsed_seconds": 6},
        ]
    )
    trace = prepare_trajectory(events)
    assert trace["elapsed_seconds"].tolist() == [1, 2, 4, 6]
    np.testing.assert_allclose(trace["incumbent"], [np.nan, 70, 70, 70])
    assert trace["bound"].isna().all()


def _run(root, name, *, strategy="single", weight_edges=False, seed=1, log=True):
    directory = root / name
    directory.mkdir(parents=True)
    config = {
        "solver": "recom",
        "strategy": strategy,
        "weight_edges": weight_edges,
        "levels": ["Block_3", "Block_2"],
        "centroids_type": "6-zone-3",
        "seed": seed,
        "solve_time_limits": [600],
    }
    (directory / "benchmark_manifest.json").write_text(json.dumps({"config": config}))
    if log:
        logs = directory / "solver_logs"
        logs.mkdir()
        records = [
            {"record": "header", "solver": "recom", "level": "Block_2"},
            {
                "record": "improvement",
                "elapsed_seconds": 1,
                "feasible": False,
                "boundary_cost": 10,
            },
            {
                "record": "improvement",
                "elapsed_seconds": 3,
                "feasible": True,
                "boundary_cost": 80,
            },
            {"record": "summary", "elapsed_seconds": 600},
        ]
        (logs / "solver_00_Block_2_recom.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records)
        )
    return directory


def test_loader_filters_and_preserves_missing_logs(tmp_path):
    _run(tmp_path, "good")
    _run(tmp_path, "recursive", strategy="recursive")
    _run(tmp_path, "weighted", weight_edges=True)
    _run(tmp_path, "other-seed", seed=2)
    _run(tmp_path, "missing-log", log=False)
    events, runs, skipped = load_trajectories(tmp_path, seeds=[1])
    assert set(runs["run_id"]) == {"good", "missing-log"}
    assert set(events["run_id"]) == {"good"}
    assert runs["has_feasible_objective"].sum() == 1
    assert not runs["has_bound"].any()
    assert skipped == {
        "strategy=recursive": 1,
        "objective is not plain cut edges": 1,
        "filtered": 1,
    }


def test_cli_reads_yaml_without_source_data_and_renders(tmp_path):
    root = tmp_path / "results"
    _run(root, "good")
    config = tmp_path / "sweep.yaml"
    config.write_text(f"execution:\n  output_dir: {root}\n")
    output = tmp_path / "figures"
    assert (
        main([str(config), "-o", str(output), "--dpi", "50", "--skip-initial", "0"])
        == 0
    )
    assert (output / "cut_edges_Block_2_tl_600s_median.png").stat().st_size > 1000
    runs = pd.read_csv(output / "cut_edges_runs.csv")
    assert runs.loc[0, "has_feasible_objective"]
    assert (output / "cut_edges_events.csv").exists()
    assert (output / "cut_edges_aggregates.csv").exists()


def test_skip_counts_improvements_not_repeated_logs_or_timestamps():
    events = pd.DataFrame(
        [
            {"elapsed_seconds": 1, "incumbent": 100, "bound": 10},
            {"elapsed_seconds": 1, "incumbent": 90, "bound": 10},
            {"elapsed_seconds": 2, "incumbent": 90, "bound": 20},
            {"elapsed_seconds": 3, "incumbent": 80, "bound": 20},
            {"elapsed_seconds": 4, "incumbent": 80, "bound": 20},
        ]
    )
    trace = prepare_trajectory(events, skip_initial=2)
    np.testing.assert_allclose(trace["incumbent"], [np.nan, np.nan, 80, 80])
    assert prepare_trajectory(events, skip_initial=3)["incumbent"].isna().all()


def test_never_feasible_and_fully_trimmed_runs_cannot_plot_bounds():
    events = pd.DataFrame(
        [
            {"run_id": "failed", "elapsed_seconds": 1, "bound": 999},
            {"run_id": "short", "elapsed_seconds": 1, "incumbent": 100, "bound": 10},
            {"run_id": "long", "elapsed_seconds": 1, "incumbent": 100, "bound": 10},
            {"run_id": "long", "elapsed_seconds": 2, "incumbent": 80, "bound": 20},
        ]
    )
    runs = pd.DataFrame(
        [
            {"run_id": "failed", "has_feasible_objective": False},
            {"run_id": "short", "has_feasible_objective": True},
            {"run_id": "long", "has_feasible_objective": True},
        ]
    )
    assert set(visible_trajectories(events, runs, skip_initial=0)) == {"short", "long"}
    assert set(visible_trajectories(events, runs, skip_initial=1)) == {"long"}


def test_median_waits_for_fixed_cohort_and_never_jumps_up():
    first = prepare_trajectory(
        pd.DataFrame(
            [
                {"elapsed_seconds": 1, "incumbent": 100, "bound": 20},
                {"elapsed_seconds": 2, "incumbent": 80, "bound": 30},
                {"elapsed_seconds": 5, "incumbent": 70},
            ]
        )
    )
    second = prepare_trajectory(
        pd.DataFrame(
            [
                {"elapsed_seconds": 0, "bound": 40},
                {"elapsed_seconds": 3, "incumbent": 200, "bound": 50},
                {"elapsed_seconds": 4, "incumbent": 180},
            ]
        )
    )
    summary = summarize_traces([first, second]).set_index("elapsed_seconds")
    assert summary.index.min() == 0
    assert summary.loc[summary.index < 3, "incumbent"].isna().all()
    assert np.isnan(summary.loc[0, "q25"])
    assert summary.loc[1, "q25"] == summary.loc[1, "q75"] == 100
    assert "bound" not in summary
    assert summary.loc[3, "incumbent"] == 140
    assert summary.loc[3, "q25"] == 110
    assert summary.loc[3, "q75"] == 170
    assert summary.loc[3, "n"] == 2
    assert summary.loc[5, "n"] == 2  # completed runs retain their best logged values
    assert summary.loc[5, "incumbent"] == 125
    assert summary.loc[summary.index >= 3, "n"].eq(2).all()
    for column in ("incumbent", "q25", "q75"):
        assert summary.loc[summary.index >= 3, column].diff().dropna().le(0).all()


def test_cli_splits_figures_by_both_level_and_budget(tmp_path):
    for index, (level, budget) in enumerate(
        [("Block_2", 600), ("Block_1", 600), ("Block_2", 1800)]
    ):
        directory = _run(tmp_path / "results", str(index))
        path = directory / "benchmark_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["config"]["levels"] = [level]
        manifest["config"]["solve_time_limits"] = [budget]
        path.write_text(json.dumps(manifest))
        log = directory / "solver_logs" / "solver_00_Block_2_recom.jsonl"
        text = log.read_text().replace("Block_2", level)
        log.unlink()
        (log.parent / f"solver_00_{level}_recom.jsonl").write_text(text)
    output = tmp_path / "plots"
    assert (
        main(
            [
                str(tmp_path / "results"),
                "-o",
                str(output),
                "--dpi",
                "30",
                "--skip-initial",
                "0",
            ]
        )
        == 0
    )
    assert {path.name for path in output.glob("*.png")} == {
        "cut_edges_Block_2_tl_600s_median.png",
        "cut_edges_Block_1_tl_600s_median.png",
        "cut_edges_Block_2_tl_1800s_median.png",
    }


def test_band_is_available_before_last_run_but_median_is_hidden():
    traces = [
        prepare_trajectory(
            pd.DataFrame(
                [
                    {"elapsed_seconds": start, "incumbent": value},
                    {"elapsed_seconds": 10},
                ]
            )
        )
        for start, value in [(1, 100), (1, 140), (5, 200)]
    ]
    summary = summarize_traces(traces).set_index("elapsed_seconds")
    assert summary.loc[1, "n"] == 2
    assert summary.loc[1, "q25"] == 110
    assert summary.loc[1, "q75"] == 130
    assert np.isnan(summary.loc[1, "incumbent"])
    assert summary.loc[5, "incumbent"] == 140


@pytest.mark.parametrize("aggregate", ["median", "individual"])
def test_render_never_draws_bounds(tmp_path, monkeypatch, aggregate):
    from matplotlib.axes import Axes

    _run(tmp_path, "good")
    events, runs, _ = load_trajectories(tmp_path)
    events["bound"] = 10
    runs["has_bound"] = True
    series = []
    original = Axes.step

    def capture(self, x, y, *args, **kwargs):
        series.append(np.asarray(y))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "step", capture)
    render_figures(
        events, runs, tmp_path / "plots", skip_initial=0, aggregate=aggregate, dpi=30
    )
    assert len(series) == 1
    assert np.isin(series[0][np.isfinite(series[0])], [80]).all()
