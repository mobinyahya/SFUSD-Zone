#!/usr/bin/env python3
"""Tests for the choice-benchmark figure's zoning selection and baseline row."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis.plots.plot_choice_benchmark import (
    BASELINE_LABEL,
    DISTRIBUTION_MIN_POINTS,
    CENTROID_COLORS,
    _describe_baseline,
    _describe_replicates,
    _measure_panel,
    _saa_sample_count,
    append_baseline,
    attach_best_stage,
    iter_results,
    plot_strategy_comparison,
    centroid_channel,
    single_trace_per_centroid,
)


def runs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": ["a", "b"],
            "variant": ["SAA", "MID"],
            "centroid": ["6-zone-3", "6-zone-3"],
            "welfare": [1000.0, 800.0],
            "mid_welfare": [1100.0, 900.0],
            "cut_edges": [500.0, 600.0],
            "budget_pct_of_best": [100.0, 81.8],
        }
    )


def stage_frame() -> pd.DataFrame:
    """Two runs; run "a" saved three zonings, run "b" only the one it returned."""
    return pd.DataFrame(
        {
            "path": ["a", "a", "a", "b"],
            "stage_index": [0, 1, 2, 0],
            "stage_name": ["it_00", "it_01", "it_02", "stage_00"],
            "is_selected": [False, False, True, True],
            "mid_welfare": [1400.0, 1300.0, 1200.0, 900.0],
            "logsum_welfare": [1500.0, 1350.0, 1000.0, 800.0],
            "cut_edges": [700.0, 650.0, 500.0, 600.0],
        }
    )


def chain_frame(**overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "centroid": "6-zone-3",
            "seed": 1,
            "iteration": [0, 1, 2],
            "choice_welfare": [500.0, 600.0, 700.0],
            "budget_welfare": [550.0, 650.0, 750.0],
            "cut_edges": [900.0, 800.0, 700.0],
        }
    )
    return frame.assign(**overrides) if overrides else frame


def test_attach_best_stage_takes_the_budget_best_zoning(tmp_path: Path):
    csv = tmp_path / "mid_welfare_base.csv"
    stage_frame().to_csv(csv, index=False)

    scored = attach_best_stage(runs_frame(), csv).set_index("path")

    # Run "a" is rescored to its it_00 zoning, which the run did not return.
    assert scored.loc["a", "mid_welfare"] == 1400.0
    assert scored.loc["a", "welfare"] == 1500.0
    assert scored.loc["a", "cut_edges"] == 700.0
    assert scored.loc["a", "best_stage"] == "it_00"
    # Run "b" saved one zoning, so nothing moves.
    assert scored.loc["b", "welfare"] == 800.0
    # The ranking measure is renormalized against the rescored budget best.
    assert scored.loc["a", "budget_pct_of_best"] == 100.0


def test_attach_best_stage_keeps_the_returned_stage_on_ties(tmp_path: Path):
    csv = tmp_path / "tied.csv"
    tied = stage_frame()
    tied.loc[tied["path"] == "a", "mid_welfare"] = 1400.0
    tied.to_csv(csv, index=False)

    scored = attach_best_stage(runs_frame(), csv).set_index("path")

    assert scored.loc["a", "best_stage"] == "it_02"


def test_attach_best_stage_without_a_replay_keeps_final_zonings(tmp_path: Path):
    runs = runs_frame()
    assert attach_best_stage(runs, tmp_path / "absent.csv").equals(runs)

    partial = tmp_path / "partial.csv"
    stage_frame().drop(columns=["mid_welfare"]).to_csv(partial, index=False)
    assert attach_best_stage(runs, partial).equals(runs)


def test_append_baseline_adds_one_variant(tmp_path: Path):
    csv = tmp_path / "relaxed_recom_baseline.csv"
    chain_frame().to_csv(csv, index=False)

    combined = append_baseline(runs_frame(), csv)
    baseline = combined[combined["variant"] == BASELINE_LABEL]

    # Iteration 0 is the starting solution, not something the chain produced.
    assert list(baseline["welfare"]) == [600.0, 700.0]
    assert list(baseline["mid_welfare"]) == [650.0, 750.0]
    assert list(baseline["cut_edges"]) == [800.0, 700.0]
    # The chain is ranked on budget welfare, against the benchmark's own
    # per-centroid budget best, so the row sorts among the strategies without
    # moving that yardstick.
    assert list(baseline["budget_pct_of_best"]) == pytest.approx(
        [100.0 * 650.0 / 1100.0, 100.0 * 750.0 / 1100.0]
    )
    assert len(combined) == len(runs_frame()) + 2


def test_append_baseline_ignores_missing_or_unusable_csv(tmp_path: Path):
    runs = runs_frame()
    assert append_baseline(runs, tmp_path / "absent.csv").equals(runs)

    partial = tmp_path / "partial.csv"
    chain_frame().drop(columns=["budget_welfare"]).to_csv(partial, index=False)
    assert append_baseline(runs, partial).equals(runs)

    only_start = tmp_path / "start.csv"
    chain_frame().head(1).to_csv(only_start, index=False)
    assert append_baseline(runs, only_start).equals(runs)


def test_append_baseline_drops_infeasible_iterations(tmp_path: Path):
    csv = tmp_path / "flagged.csv"
    chain_frame(feasible=[True, False, True]).to_csv(csv, index=False)

    combined = append_baseline(runs_frame(), csv)
    baseline = combined[combined["variant"] == BASELINE_LABEL]

    # Iteration 0 by the iteration filter, iteration 1 by the balance flag.
    assert list(baseline["welfare"]) == [700.0]


def test_append_baseline_drops_centroids_the_benchmark_lacks(tmp_path: Path):
    csv = tmp_path / "other_centroid.csv"
    chain_frame(centroid="6-zone-9").to_csv(csv, index=False)
    assert append_baseline(runs_frame(), csv).equals(runs_frame())


def test_dense_row_is_summarized_rather_than_drawn_point_by_point():
    rng = np.random.default_rng(0)
    values = rng.normal(1000.0, 10.0, DISTRIBUTION_MIN_POINTS)
    runs = pd.DataFrame(
        {
            "variant": BASELINE_LABEL,
            "centroid": "6-zone-3",
            "welfare": values,
        }
    )

    figure, ax = plt.subplots()
    try:
        _measure_panel(
            ax, runs, "welfare", [BASELINE_LABEL], ["6-zone-3"],
            centroid_channel(["6-zone-3"]), ",.0f", "T", "x",
        )
        # Range line, middle-half band, mean marker - not one marker per point.
        assert [len(line.get_xdata()) for line in ax.lines] == [2, 2, 1]
        # The range the row reports is the range the panel shows.
        assert ax.get_xlim()[0] < values.min()
        assert ax.get_xlim()[1] > values.max()
    finally:
        plt.close(figure)


def test_append_baseline_needs_budget_welfare_to_rank_against(tmp_path: Path):
    """Budget welfare is the ranking measure, so the row cannot be placed without it."""
    csv = tmp_path / "relaxed_recom_baseline.csv"
    chain_frame().to_csv(csv, index=False)
    runs = runs_frame().drop(columns=["mid_welfare", "budget_pct_of_best"])

    assert append_baseline(runs, csv).equals(runs)


def test_strategy_comparison_refuses_to_plot_without_budget_welfare(tmp_path: Path):
    """The logsum panel is gone, so a missing budget replay leaves nothing to show."""
    runs = runs_frame().drop(columns=["budget_pct_of_best"])
    with pytest.raises(SystemExit, match="recompute_mid_welfare"):
        plot_strategy_comparison(runs, tmp_path)


def iterations_frame() -> pd.DataFrame:
    """A replicated sweep: two price arms x two seeds x two centroids."""
    rows = []
    for centroid in ("6-zone-3", "6-zone-9"):
        for scale in (0.0, 1.0):
            for seed in (2, 3):
                for iteration in (0, 1):
                    rows.append(
                        {
                            "strategy": "priced_access",
                            "centroid": centroid,
                            "run": f"{centroid}/s{seed}/x{scale}",
                            "seed": seed,
                            "price_scale": scale,
                            "iteration": iteration,
                        }
                    )
    rows.append(
        {
            "strategy": "saa",
            "centroid": "6-zone-3",
            "run": "6-zone-3/s2/saa",
            "seed": 2,
            "price_scale": None,
            "iteration": 0,
        }
    )
    return pd.DataFrame(rows)


def test_single_trace_keeps_one_run_per_centroid():
    trace = single_trace_per_centroid(
        iterations_frame(), "priced_access", price_scale=1.0
    )
    assert set(trace["strategy"]) == {"priced_access"}
    assert set(trace["price_scale"]) == {1.0}
    # One run per centroid is what plot_model_vs_actual can draw as one line.
    assert trace.groupby("centroid")["seed"].nunique().eq(1).all()
    assert set(trace["seed"]) == {2}
    assert len(trace) == 4


def test_single_trace_picks_the_lowest_seed_present_at_each_centroid():
    """A seed missing at one centroid must not drop that centroid's panel."""
    frame = iterations_frame()
    frame = frame[~((frame["centroid"] == "6-zone-9") & (frame["seed"] == 2))]
    trace = single_trace_per_centroid(frame, "priced_access", price_scale=1.0)
    kept = trace.groupby("centroid")["seed"].unique().apply(list).to_dict()
    assert kept == {"6-zone-3": [2], "6-zone-9": [3]}


def test_single_trace_on_a_strategy_with_no_iterations_is_empty():
    trace = single_trace_per_centroid(iterations_frame(), "mid_decomp")
    assert trace.empty


def test_centroid_channel_covers_more_centroids_than_the_palette():
    """A four-centroid sweep used to raise IndexError partway through figure 1."""
    centroids = [f"6-zone-{i}" for i in range(1, 5)]
    colors, markers = centroid_channel(centroids)

    assert set(colors) == set(centroids)
    # Four distinct slots in both channels, so no two rows collide.
    assert len(set(colors.values())) == 4
    assert len(set(markers.values())) == 4


def test_centroid_channel_reuses_slots_rather_than_failing():
    """Past the palette the figure degrades; it does not stop drawing."""
    centroids = [f"z{i}" for i in range(len(CENTROID_COLORS) + 2)]
    colors, markers = centroid_channel(centroids)

    assert set(colors) == set(centroids)
    assert colors[centroids[0]] == colors[centroids[len(CENTROID_COLORS)]]


def test_strategy_comparison_draws_four_centroids(tmp_path: Path):
    runs = pd.concat(
        [runs_frame().assign(centroid=f"6-zone-{i}", path=[f"a{i}", f"b{i}"])
         for i in range(1, 5)],
        ignore_index=True,
    )
    assert plot_strategy_comparison(runs, tmp_path).exists()


def test_replicate_caption_counts_the_runs_it_drew():
    """The caption used to hardcode "8 seeds for Short Bursts"."""
    runs = pd.DataFrame(
        {
            "variant": ["Short Bursts"] * 3 + ["MID"] * 2 + [BASELINE_LABEL] * 9,
            "centroid": ["6-zone-3"] * 3 + ["6-zone-3"] * 2 + ["6-zone-3"] * 9,
        }
    )
    note = _describe_replicates(runs)

    assert "3 for Short Bursts" in note
    assert "2 for MID" in note
    # The chain row is captioned separately, not as a replicate count.
    assert BASELINE_LABEL not in note


def test_replicate_caption_reports_a_range_for_a_partial_sweep():
    runs = pd.DataFrame(
        {
            "variant": ["SAA"] * 5,
            "centroid": ["6-zone-3"] * 2 + ["6-zone-9"] * 3,
        }
    )
    assert "2-3 for SAA" in _describe_replicates(runs)


def test_baseline_caption_reports_the_scored_iteration_count():
    runs = pd.DataFrame(
        {
            "variant": [BASELINE_LABEL] * 1200,
            "centroid": ["6-zone-3"] * 600 + ["6-zone-9"] * 600,
        }
    )
    assert "600 feasible iterations" in _describe_baseline(runs)
    assert _describe_baseline(runs_frame()) == ""


def test_saa_sample_count_comes_from_the_run_configs():
    iterations = pd.DataFrame(
        {"strategy": ["saa", "saa", "iterative_choice"], "saa_num_seeds": [10, 10, None]}
    )
    assert _saa_sample_count(iterations) == "10"

    mixed = pd.DataFrame({"strategy": ["saa", "saa"], "saa_num_seeds": [10, 15]})
    assert _saa_sample_count(mixed) == "10 and 15"


def test_attach_best_stage_rejects_a_csv_from_another_benchmark(tmp_path: Path):
    """A replay of a different sweep joins to nothing; say so, don't limp on."""
    csv = tmp_path / "other_sweep.csv"
    stage_frame().assign(path=["x", "x", "x", "y"]).to_csv(csv, index=False)

    with pytest.raises(SystemExit, match="shares no run"):
        attach_best_stage(runs_frame(), csv)


def test_iter_results_drops_errored_runs(tmp_path: Path):
    def write(name: str, status: str) -> None:
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "config": {"solver": "cp_bool", "strategy": "saa"},
                    "run": {"stages": []},
                    "status": status,
                }
            )
        )

    write("good", "FEASIBLE")
    write("bad", "ERROR")

    assert [p.parent.name for p, _ in iter_results(tmp_path)] == ["good"]
    kept = sorted(p.parent.name for p, _ in iter_results(tmp_path, keep_failed=True))
    assert kept == ["bad", "good"]


def test_single_trace_separates_runs_that_share_a_seed():
    """Two SAA arms at one seed differ only by a sweep flag, not by seed.

    Reducing on the seed alone left both in the frame, and the panel drew them
    interleaved as a single zigzag.
    """
    rows = [
        {
            "strategy": "saa",
            "centroid": "6-zone-3",
            "run": f"6-zone-3/seed1/multicut_{multicut}",
            "seed": 1,
            "price_scale": None,
            "iteration": iteration,
        }
        for multicut in (True, False)
        for iteration in (0, 1)
    ]
    trace = single_trace_per_centroid(pd.DataFrame(rows), "saa")

    assert trace["run"].nunique() == 1
    assert len(trace) == 2


def test_append_baseline_warns_when_the_chain_misses_centroids(tmp_path, capsys):
    """A chain from an earlier sweep covers only the centroids the two share."""
    csv = tmp_path / "relaxed_recom_baseline.csv"
    chain_frame().to_csv(csv, index=False)  # 6-zone-3 only
    runs = pd.concat(
        [runs_frame(), runs_frame().assign(centroid="6-zone-9", path=["c", "d"])],
        ignore_index=True,
    )

    combined = append_baseline(runs, csv)

    assert "no chain for ['6-zone-9']" in capsys.readouterr().out
    # The chain it does have is still used, for the centroid it covers.
    baseline = combined[combined["variant"] == BASELINE_LABEL]
    assert set(baseline["centroid"]) == {"6-zone-3"}


def _write_run(root: Path, name: str, **config) -> None:
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "config": {"solver": "cp_bool", "strategy": "saa", **config},
                "run": {"stages": []},
                "status": "FEASIBLE",
            }
        )
    )


def test_config_filter_holds_a_swept_arm_the_strategy_axis_pools(tmp_path: Path):
    """direct_samples' scenario count is a sweep arm that shares one row.

    ``_variant_label`` splits on the price scale but on nothing else, so two
    ``saa_num_seeds`` arms would average into a single ``direct_samples`` row.
    """
    _write_run(tmp_path, "five", strategy="direct_samples", saa_num_seeds=5)
    _write_run(tmp_path, "ten", strategy="direct_samples", saa_num_seeds=10)

    kept = [p.parent.name for p, _ in iter_results(tmp_path)]
    assert sorted(kept) == ["five", "ten"]

    filtered = [
        p.parent.name
        for p, _ in iter_results(tmp_path, config_filters=[("saa_num_seeds", "10")])
    ]
    assert filtered == ["ten"]


def test_config_filter_compares_numbers_not_spellings(tmp_path: Path):
    """The value comes off a command line, the config holds whatever JSON had."""
    _write_run(tmp_path, "int", saa_num_seeds=10)
    _write_run(tmp_path, "float", saa_num_seeds=10.0)

    kept = sorted(
        p.parent.name
        for p, _ in iter_results(tmp_path, config_filters=[("saa_num_seeds", "10")])
    )
    assert kept == ["float", "int"]


def test_config_filter_drops_runs_that_never_set_the_key(tmp_path: Path):
    """A key absent from a config is not a match, so the arm stays clean."""
    _write_run(tmp_path, "set", saa_num_seeds=10)
    _write_run(tmp_path, "unset")

    kept = [
        p.parent.name
        for p, _ in iter_results(tmp_path, config_filters=[("saa_num_seeds", "10")])
    ]
    assert kept == ["set"]


def test_config_filters_combine(tmp_path: Path):
    _write_run(tmp_path, "both", saa_num_seeds=10, seed=1)
    _write_run(tmp_path, "one", saa_num_seeds=10, seed=2)
    _write_run(tmp_path, "other", saa_num_seeds=5, seed=1)

    kept = [
        p.parent.name
        for p, _ in iter_results(
            tmp_path, config_filters=[("saa_num_seeds", "10"), ("seed", "1")]
        )
    ]
    assert kept == ["both"]
