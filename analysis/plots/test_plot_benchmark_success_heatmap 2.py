#!/usr/bin/env python3
"""Tests for benchmark success rate heatmap plotting script."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.plots.plot_benchmark_success_heatmap import (
    SOLVER_DISPLAY_ORDER,
    clean_level_value,
    format_time_limit_minutes,
    clean_total_time_limit_value,
    create_success_heatmap_matrix,
    extract_zone_count_label,
    generate_benchmark_heatmaps,
    is_success_status,
    map_solver_display_name,
    parse_level_sort_key,
    parse_solver_sort_key,
    parse_time_limit_sort_key,
    parse_zone_sort_key,
    prepare_benchmark_dataframe,
)


def test_parse_zone_sort_key():
    zones = ["18 Zones", "6 Zones", "10 Zones", "8 Zones", "13 Zones"]
    sorted_z = sorted(zones, key=parse_zone_sort_key)
    assert sorted_z == ["6 Zones", "8 Zones", "10 Zones", "13 Zones", "18 Zones"]


def test_parse_level_sort_key():
    levels = ["Block_0", "Block_2", "Block_1", "BlockGroup_0"]
    sorted_l = sorted(levels, key=parse_level_sort_key)
    assert sorted_l == ["Block_2", "Block_1", "Block_0", "BlockGroup_0"]


def test_parse_solver_sort_key():
    solvers = ["CP (Multi-Level)", "Adaptive Short Bursts", "Short Bursts", "Recom", "Relaxed Recom", "CP"]
    sorted_s = sorted(solvers, key=parse_solver_sort_key)
    assert sorted_s == ["Recom", "Relaxed Recom", "Short Bursts", "Adaptive Short Bursts", "CP", "CP (Multi-Level)"]


def test_parse_time_limit_sort_key():
    tls = ["1800s", "600s", "30s"]
    sorted_tl = sorted(tls, key=parse_time_limit_sort_key)
    assert sorted_tl == ["30s", "600s", "1800s"]


def test_clean_level_value():
    assert clean_level_value('["Block_0"]') == "Block_0"
    assert clean_level_value('["Block_2", "Block_1", "Block_0"]') == "Block_0"
    assert clean_level_value('["Block_3", "Block_2"]') == "Block_2"
    assert clean_level_value("Block_0") == "Block_0"
    assert clean_level_value(["Block_3", "Block_1"]) == "Block_1"


def test_clean_total_time_limit_value():
    assert clean_total_time_limit_value("[600]") == "600s"
    assert clean_total_time_limit_value("[450, 150]") == "600s"
    assert clean_total_time_limit_value("[1350, 450]") == "1800s"
    assert clean_total_time_limit_value(600) == "600s"
    assert clean_total_time_limit_value("1800") == "1800s"


def test_map_solver_display_name():
    assert map_solver_display_name({"config_solver": "cp_bool", "config_strategy": "single"}) == "CP"
    assert map_solver_display_name({"config_solver": "cp_bool", "config_strategy": "recursive"}) == "CP (Multi-Level)"
    assert map_solver_display_name({"config_solver": "mip", "config_strategy": "recursive"}) == "MIP (Multi-Level)"
    assert map_solver_display_name({"config_solver": "cp_int", "config_strategy": "recursive"}) == "CP (Int, Multi-Level)"
    # Unknown solvers fall through to a generated name, suffixed the same way.
    assert map_solver_display_name({"config_solver": "new_solver", "config_strategy": "recursive"}) == "New Solver (Multi-Level)"
    assert map_solver_display_name({"config_solver": "recom", "config_strategy": "single"}) == "Recom"
    assert map_solver_display_name({"config_solver": "relaxed_recom", "config_strategy": "single"}) == "Relaxed Recom"
    assert map_solver_display_name({"config_solver": "short_bursts", "config_strategy": "single"}) == "Short Bursts"
    assert map_solver_display_name({"config_solver": "adaptive_short_bursts", "config_strategy": "single"}) == "Adaptive Short Bursts"


def test_extract_zone_count_label():
    assert extract_zone_count_label({"num_zones": 6.0}) == "6 Zones"
    assert extract_zone_count_label({"config_centroids_type": "13-zone-6"}) == "13 Zones"
    assert extract_zone_count_label({"centroids_type": "8-zone-22"}) == "8 Zones"


def test_is_success_status():
    assert is_success_status("FEASIBLE") is True
    assert is_success_status("OPTIMAL") is True
    assert is_success_status("feasible") is True
    assert is_success_status("optimal") is True
    assert is_success_status("UNKNOWN") is False
    assert is_success_status("INFEASIBLE") is False
    assert is_success_status("ERROR") is False
    assert is_success_status(None) is False
    assert is_success_status(np.nan) is False


def test_create_success_heatmap_matrix():
    data = [
        {"config_solver": "cp_bool", "config_strategy": "single", "config_centroids_type": "6-zone-3", "config_levels": '["Block_3", "Block_2"]', "config_solve_time_limits": "[600]", "status": "FEASIBLE", "config_seed": 0},
        {"config_solver": "cp_bool", "config_strategy": "single", "config_centroids_type": "6-zone-9", "config_levels": '["Block_3", "Block_2"]', "config_solve_time_limits": "[600]", "status": "OPTIMAL", "config_seed": 1},
        {"config_solver": "cp_bool", "config_strategy": "single", "config_centroids_type": "8-zone-22", "config_levels": '["Block_3", "Block_2"]', "config_solve_time_limits": "[600]", "status": "UNKNOWN", "config_seed": 0},
        {"config_solver": "recom", "config_strategy": "single", "config_centroids_type": "6-zone-3", "config_levels": '["Block_3", "Block_2"]', "config_solve_time_limits": "[600]", "status": "UNKNOWN", "config_seed": 0},
    ]
    df = pd.DataFrame(data)
    df_prepared = prepare_benchmark_dataframe(df)

    matrix, labels, solvers, columns, active_levels = create_success_heatmap_matrix(df_prepared)

    assert solvers == ["Recom", "CP"]
    assert columns == [("Block_2", "6 Zones"), ("Block_2", "8 Zones")]
    assert active_levels == ["Block_2"]

    # Recom @ Block_2 6 Zones: 0 successes / 1 run -> 0.0, label '0 / 1'
    assert matrix[0, 0] == 0.0
    assert labels[0][0] == "0 / 1"

    # Recom @ Block_2 8 Zones: missing -> NaN, label '-'
    assert np.isnan(matrix[0, 1])
    assert labels[0][1] == "-"

    # CP @ Block_2 6 Zones: 2 successes / 2 runs (aggregated across 6-zone-3 and 6-zone-9) -> 1.0, label '2 / 2'
    assert matrix[1, 0] == 1.0
    assert labels[1][0] == "2 / 2"

    # CP @ Block_2 8 Zones: 0 successes / 1 run -> 0.0, label '0 / 1'
    assert matrix[1, 1] == 0.0
    assert labels[1][1] == "0 / 1"


def test_generate_benchmark_heatmaps_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "summary.csv"
        out_dir = Path(tmpdir) / "plots"

        rows = []
        for s, strat, tl in [("cp_bool", "single", "[600]"), ("cp_bool", "recursive", "[450, 150]"), ("recom", "single", "[600]")]:
            for c in ["6-zone-3", "6-zone-9", "8-zone-22"]:
                for lvl in ['["Block_3", "Block_2"]', '["Block_2", "Block_0"]']:
                    for seed in range(2):
                        status = "FEASIBLE" if "cp_bool" in s else "UNKNOWN"
                        rows.append({
                            "config_solver": s,
                            "config_strategy": strat,
                            "config_centroids_type": c,
                            "config_levels": lvl,
                            "config_solve_time_limits": tl,
                            "config_seed": seed,
                            "status": status,
                        })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        generated = generate_benchmark_heatmaps(tmpdir, output_dir=out_dir)
        # Both [600] and [450, 150] sum to 600s, so exactly 1 plot is created
        assert len(generated) == 1
        for f in generated:
            assert f.exists()
            assert f.stat().st_size > 0


def test_solver_labels_and_order_match_the_cut_length_figures():
    """The two figures name the same method identically, or a reader comparing
    them sees two names for one solver. ``plot_edges`` claims this agreement in
    a comment, so assert it rather than trusting the comment."""
    from benchmark.plot_edges import METHODS

    assert SOLVER_DISPLAY_ORDER == [label for label, _ in METHODS.values()]
    for method, (label, _) in METHODS.items():
        recursive = method.startswith("recursive_")
        row = {
            "config_solver": method.removeprefix("recursive_"),
            "config_strategy": "recursive" if recursive else "single",
        }
        assert map_solver_display_name(row) == label


def test_time_limit_is_titled_in_minutes_but_slugged_in_seconds():
    assert format_time_limit_minutes("600s") == "10 min"
    assert format_time_limit_minutes("1800s") == "30 min"
    assert format_time_limit_minutes("30s") == "0.5 min"
    # Anything not in the '<number>s' form is passed through untouched.
    assert format_time_limit_minutes("unknown") == "unknown"
    assert format_time_limit_minutes(600) == "600"
