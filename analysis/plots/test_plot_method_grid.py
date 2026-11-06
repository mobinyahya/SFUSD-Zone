"""Tests for the method x level x zones comparison grid."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from analysis.plots import plot_method_grid as grid
from analysis.plots.plot_benchmark_success_heatmap import SOLVER_DISPLAY_ORDER


def _row(task_id, solver, levels, zones, status, cut, **recom):
    return {
        "task_id": task_id,
        "status": status,
        "final_status": "",
        "config_solver": solver,
        "config_levels": json.dumps(levels),
        "config_centroids_type": f"{zones}-zone-1",
        "cut_edges": "" if cut is None else str(cut),
        "config_recom_burst_length": str(recom.get("burst", 25)),
        "config_recom_reject_infeasible": str(recom.get("reject", False)),
        "config_recom_pair_selection": recom.get("pair", "softmax"),
        "config_recom_cut_selection": recom.get("cut_sel", "softmax"),
    }


def _write(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "summary.csv", index=False)
    return path


def test_method_order_matches_the_success_heatmap():
    assert grid.METHOD_ORDER == SOLVER_DISPLAY_ORDER


@pytest.mark.parametrize(
    "solver, levels, recom, label",
    [
        ("recom", ["Block_0"], {"burst": 0, "reject": True}, "Recom"),
        ("recom", ["Block_0"], {"burst": 0}, "Relaxed Recom"),
        (
            "recom",
            ["Block_0"],
            {"pair": "uniform", "cut_sel": "uniform"},
            "Short Bursts",
        ),
        ("recom", ["Block_0"], {}, "Adaptive Short Bursts"),
        ("mip", ["Block_1"], {}, "MIP"),
        ("cp_bool", ["Block_3", "Block_1"], {}, "CP (Multi-Level)"),
        ("cp_int", ["Block_3", "Block_1"], {}, "CP (Int, Multi-Level)"),
    ],
)
def test_method_label(solver, levels, recom, label):
    row = pd.Series(_row("t", solver, levels, 6, "FEASIBLE", 1, **recom))
    assert grid.method_label(row) == label


def test_later_sweep_replaces_runs_and_cells_compare_to_the_best(tmp_path):
    first = _write(
        tmp_path / "a",
        [
            _row("m1", "mip", ["Block_3", "Block_0"], 6, "UNKNOWN", None),
            _row("c1", "cp_bool", ["Block_0"], 6, "FEASIBLE", 120),
            _row("c2", "cp_bool", ["Block_0"], 6, "FEASIBLE", 100),
        ],
    )
    rerun = _write(
        tmp_path / "b", [_row("m1", "mip", ["Block_3", "Block_0"], 6, "OPTIMAL", 90)]
    )

    runs = grid.load_runs([first, rerun])
    assert len(runs) == 3
    cells = grid.aggregate(runs, min_runs=1).set_index("method")

    assert cells.loc["MIP (Multi-Level)", "feasible"] == 1
    assert cells.loc["MIP (Multi-Level)", "gap"] == pytest.approx(0.0)
    assert cells.loc["CP", "median_cut_edges"] == 110
    assert cells.loc["CP", "gap"] == pytest.approx(110 / 90 - 1)


def test_cells_below_min_runs_never_set_the_best(tmp_path):
    root = _write(
        tmp_path / "a",
        [
            _row("lucky", "mip", ["Block_0"], 6, "FEASIBLE", 10),
            _row("x", "mip", ["Block_0"], 6, "UNKNOWN", None),
            *[
                _row(f"c{i}", "cp_bool", ["Block_0"], 6, "FEASIBLE", 100)
                for i in range(2)
            ],
        ],
    )
    cells = grid.aggregate(grid.load_runs([root]), min_runs=2).set_index("method")
    assert not cells.loc["MIP", "reliable"]
    assert cells.loc["CP", "gap"] == pytest.approx(0.0)
    assert cells.loc["MIP", "gap"] == pytest.approx(-0.9)


def test_main_writes_figure_and_table(tmp_path):
    root = _write(
        tmp_path / "a",
        [
            _row("c1", "cp_bool", ["Block_1"], 6, "FEASIBLE", 100),
            _row("c2", "cp_bool", ["Block_0"], 8, "UNKNOWN", None),
            _row("r1", "recom", ["Block_0"], 8, "FEASIBLE", 140),
        ],
    )
    out = tmp_path / "fig" / "grid.png"
    grid.main([str(root), "-o", str(out), "--min-runs", "1", "--title", "t"])
    for kind in ("feasible", "cut_edges"):
        assert (out.parent / f"grid_{kind}.png").stat().st_size > 0
    assert not out.exists() and not out.with_suffix(".pdf").exists()
    table = pd.read_csv(out.with_suffix(".csv"))
    assert set(table["method"]) == {"CP", "Adaptive Short Bursts"}
