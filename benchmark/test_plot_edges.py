"""Regression tests for feasible-only, correctly timed cut-edge trajectories."""

import json

import numpy as np
import pandas as pd
import pytest

from benchmark.plot_edges import (
    DASHES,
    SECONDS_PER_MINUTE,
    METHODS,
    Y_FRAMES,
    frame_ylim,
    load_trajectories,
    late_window,
    main,
    marker_phases,
    minutes,
    prepare_trajectory,
    recursive_events,
    render_figures,
    spread_positions,
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


def test_weighted_mode_selects_metres_and_uses_distinct_outputs(tmp_path):
    _run(tmp_path, "weighted", weight_edges=True)
    _run(tmp_path, "unweighted")
    events, runs, _ = load_trajectories(tmp_path, weighted=True)
    assert set(runs["run_id"]) == {"weighted"}
    assert set(events["objective_unit"]) == {"meter"}
    output = tmp_path / "plots"
    assert (
        main(
            [
                str(tmp_path),
                "--weighted",
                "--skip-initial",
                "0",
                "--dpi",
                "30",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "weighted_boundary_Block_2_tl_600s_median.png").exists()
    assert (output / "weighted_boundary_aggregates.csv").exists()


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
    assert len(series) == (3 if aggregate == "median" else 1)
    for values in series:
        assert np.isin(values[np.isfinite(values)], [80]).all()


def test_zoom_preserves_step_value_at_left_edge_without_backfilling():
    frame = pd.DataFrame(
        {
            "elapsed_seconds": [0, 100, 500, 600],
            "incumbent": [np.nan, np.nan, 80, 70],
            "q25": [np.nan, 90, 75, 65],
            "q75": [np.nan, 110, 85, 75],
        }
    )
    zoom = late_window(frame, 300, 600).reset_index(drop=True)
    assert zoom["elapsed_seconds"].tolist() == [300, 500, 600]
    assert np.isnan(zoom.loc[0, "incumbent"])
    assert zoom.loc[0, "q25"] == 90
    assert zoom.loc[0, "q75"] == 110


def test_recursive_timeline_uses_actual_stage_durations_and_keeps_best_objective():
    stages = [
        {"index": 0, "level": "Block_3", "wall_time": 40},
        {"index": 1, "level": "Block_2", "wall_time": 560},
    ]
    rows = [
        {
            "log_index": 0,
            "level": "Block_3",
            "elapsed_seconds": 2,
            "incumbent": 80,
            "bound": 20,
        },
        {"log_index": 0, "level": "Block_3", "elapsed_seconds": 10, "incumbent": 50},
        {"log_index": 1, "level": "Block_2", "elapsed_seconds": 1, "incumbent": 70},
        {"log_index": 1, "level": "Block_2", "elapsed_seconds": 5, "incumbent": 40},
    ]
    joined = recursive_events(rows, stages)
    assert [row["elapsed_seconds"] for row in joined] == [2, 10, 40, 41, 45, 600]
    assert joined[0]["stage_bound"] == 20
    assert joined[0]["bound"] is None
    trace = prepare_trajectory(pd.DataFrame(joined), skip_initial=1)
    np.testing.assert_allclose(trace["incumbent"], [np.nan, 50, 50, 50, 40, 40])
    assert trace["bound"].isna().all()
    with pytest.raises(ValueError, match="missing recursive stage log"):
        recursive_events(rows[:2], stages)


def test_weighted_recursive_loader_groups_by_total_budget_and_final_level(tmp_path):
    directory = _run(
        tmp_path, "recursive", strategy="recursive", weight_edges=True, log=False
    )
    path = directory / "benchmark_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"].update(solver="cp_bool", solve_time_limits=[450, 150])
    stages = []
    logs = directory / "solver_logs"
    logs.mkdir()
    for index, (level, duration) in enumerate([("Block_3", 40), ("Block_2", 560)]):
        filename = f"solver_{index:02d}_{level}_cp_bool.log"
        (logs / filename).write_text(
            "Starting CP-SAT solver\n"
            "#1 1.00s best:80000 next:[0,79999]\n"
            "#2 5.00s best:40000 next:[0,39999]\n"
            "CpSolverResponse summary:\nstatus: FEASIBLE\n"
            f"objective: 40000\nbest_bound: 20\nwalltime: {duration}\n"
        )
        stages.append(
            {
                "index": index,
                "level": level,
                "wall_time": duration,
                "metadata": {"solver_log_path": f"solver_logs/{filename}"},
            }
        )
    manifest["stages"] = stages
    path.write_text(json.dumps(manifest))
    events, runs, _ = load_trajectories(tmp_path, weighted=True, time_limits=[600])
    assert runs.loc[0, "method"] == "recursive_cp_bool"
    assert runs.loc[0, "level"] == "Block_2"
    assert runs.loc[0, "time_limit"] == 600
    assert set(events["stage_level"]) == {"Block_3", "Block_2"}
    assert set(events["stage_start_seconds"]) == {0, 40}
    assert events["elapsed_seconds"].max() == 600
    assert load_trajectories(tmp_path, weighted=True, single_only=True)[1].empty
    manifest["config"]["looseness"] = 1.2
    path.write_text(json.dumps(manifest))
    assert load_trajectories(tmp_path, weighted=True)[1].empty


@pytest.mark.parametrize("unit,expected", [("km", 0.08), ("m", 80)])
def test_length_unit_scales_objective_independently_of_the_time_axis(
    tmp_path, monkeypatch, unit, expected
):
    from matplotlib.axes import Axes

    _run(tmp_path, "weighted", weight_edges=True)
    events, runs, _ = load_trajectories(tmp_path, weighted=True)
    series = []
    original = Axes.step

    def capture(self, x, y, *args, **kwargs):
        series.append((np.asarray(x), np.asarray(y)))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "step", capture)
    render_figures(
        events,
        runs,
        tmp_path / "plots",
        skip_initial=0,
        weighted=True,
        length_unit=unit,
        dpi=30,
    )
    x, y = series[0]
    np.testing.assert_allclose(y[np.isfinite(y)], expected)
    # The objective divisor must not touch the time axis, which is drawn in
    # minutes while the logs and CSVs stay in seconds.
    assert x[np.isfinite(y)][0] == 3 / SECONDS_PER_MINUTE
    assert x[-1] == 600 / SECONDS_PER_MINUTE


def _axis(frames):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    for frame in frames:
        ax.plot(frame["elapsed_seconds"], frame["incumbent"])
    return figure, ax


def _summary(incumbents, band=200.0):
    return pd.DataFrame(
        {
            "elapsed_seconds": np.arange(len(incumbents), dtype=float),
            "incumbent": incumbents,
            "q25": [value - 1 for value in incumbents],
            "q75": [band] * len(incumbents),
        }
    )


def test_zoom_frame_excludes_early_spikes_but_keeps_every_final_value():
    import matplotlib.pyplot as plt

    frames = [_summary([900.0, 60.0, 50.0]), _summary([800.0, 130.0, 120.0])]
    figure, ax = _axis(frames)
    frame_ylim(ax, frames, mode="zoom")
    low, high = ax.get_ylim()
    plt.close(figure)
    # The transient 800-900 km first solutions and the 200 km band are cropped;
    # both final medians stay inside the panel with room to spare.
    assert low < 50 <= 120 < high
    assert high < 200


def test_zoom_frame_gives_a_flat_series_a_usable_span():
    import matplotlib.pyplot as plt

    frames = [_summary([40.0, 40.0, 40.0])]
    figure, ax = _axis(frames)
    frame_ylim(ax, frames, mode="zoom")
    low, high = ax.get_ylim()
    plt.close(figure)
    assert low < 40 < high


def test_log_frame_keeps_the_full_range_including_the_bands():
    import matplotlib.pyplot as plt

    frames = [_summary([900.0, 60.0, 50.0])]
    figure, ax = _axis(frames)
    frame_ylim(ax, frames, mode="log")
    low, high = ax.get_ylim()
    scale = ax.get_yscale()
    plt.close(figure)
    assert scale == "log"
    assert low < 49 and high > 900


def test_zero_frame_starts_at_zero_and_rejects_unknown_modes():
    import matplotlib.pyplot as plt

    frames = [_summary([900.0, 60.0, 50.0])]
    figure, ax = _axis(frames)
    frame_ylim(ax, frames, mode="zero")
    low, high = ax.get_ylim()
    assert low == 0 and high > 900
    with pytest.raises(ValueError):
        frame_ylim(ax, frames, mode="linear")
    plt.close(figure)


@pytest.mark.parametrize("mode", Y_FRAMES)
def test_every_frame_survives_a_panel_with_no_median_yet(mode):
    import matplotlib.pyplot as plt

    frames = [_summary([np.nan, np.nan])]
    figure, ax = _axis(frames)
    frame_ylim(ax, frames, mode=mode)
    plt.close(figure)


def test_recursive_methods_are_dashed_and_no_two_methods_share_an_encoding():
    recursive = {method for method in METHODS if method.startswith("recursive_")}
    assert recursive and len(recursive) < len(METHODS)
    assert all(DASHES[method] != "solid" for method in recursive)
    assert all(
        DASHES[method] == "solid" for method in METHODS if method not in recursive
    )
    colors = [color for _, color in METHODS.values()]
    assert len(set(colors)) == len(METHODS)


def test_marker_phases_are_distinct_and_stay_inside_one_step():
    phases = marker_phases(["cp_bool", "mip", "recursive_mip"])
    assert len(set(phases.values())) == 3
    assert all(abs(phase) < 0.5 for phase in phases.values())
    assert marker_phases([]) == {}


def test_spread_positions_separates_ties_without_leaving_the_axes():
    positions = spread_positions([0.5, 0.5, 0.5], 0.072)
    assert positions == sorted(positions)
    assert all(0.0 <= position <= 1.0 for position in positions)
    gaps = np.diff(positions)
    assert (gaps >= 0.072 - 1e-9).all()
    # Anchors already far apart are left where they are.
    assert spread_positions([0.1, 0.9], 0.072) == [0.1, 0.9]
    # More labels than the axis can hold fall back to even spacing.
    crowded = spread_positions([0.5] * 20, 0.072)
    assert crowded[0] == 0.0 and crowded[-1] == 1.0


@pytest.mark.parametrize("y_frame", Y_FRAMES)
def test_render_writes_one_file_per_frame_so_variants_coexist(tmp_path, y_frame):
    _run(tmp_path, "good")
    events, runs, _ = load_trajectories(tmp_path)
    paths = render_figures(
        events,
        runs,
        tmp_path / "plots",
        skip_initial=0,
        dpi=30,
        y_frame=y_frame,
    )
    # "_median" here is the aggregate, which is why the frame is not named that.
    suffix = "" if y_frame == "log" else f"_{y_frame}"
    assert all(path.exists() for path in paths)
    assert paths[0].name.endswith(f"_median{suffix}.png")
    with pytest.raises(ValueError):
        render_figures(events, runs, tmp_path / "plots", dpi=30, y_frame="linear")


def test_time_axis_is_minutes_while_the_data_stays_in_seconds(tmp_path):
    from matplotlib.axes import Axes

    _run(tmp_path, "good")
    events, runs, _ = load_trajectories(tmp_path)
    drawn = []
    original = Axes.step

    def capture(self, x, y, *args, **kwargs):
        drawn.append(np.asarray(x, dtype=float))
        return original(self, x, y, *args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(Axes, "step", capture)
    render_figures(events, runs, tmp_path / "plots", skip_initial=0, dpi=30)
    monkeypatch.undo()
    # A 600-second budget is drawn as a 10-minute axis...
    assert drawn and max(values.max() for values in drawn) == 600 / SECONDS_PER_MINUTE
    # ...while the trajectory the figure was built from keeps seconds.
    traces = visible_trajectories(events, runs, skip_initial=0)
    assert max(trace["elapsed_seconds"].max() for trace in traces.values()) == 600
    assert minutes(600) == 10
    assert minutes([0, 30, 600]).tolist() == [0, 0.5, 10]
