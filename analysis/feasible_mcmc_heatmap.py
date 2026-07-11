#!/usr/bin/env python3
"""Plot feasibility rates by zone count and graph level for an MCMC sweep."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from benchmark.config import BenchmarkTask, SimulationSweep
from optimization.data.loaders import load_centroid_schools
from optimization.levels import LEVEL_NODE_TARGETS, LevelSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_CONFIG = PROJECT_ROOT / "benchmark/configs/sweep.feasible-mcmc.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
SUCCESS_STATUSES = {"FEASIBLE", "OPTIMAL"}


@dataclass(frozen=True)
class FeasibilityCell:
    solver: str
    num_zones: int
    level: str
    num_nodes: int
    feasible: int
    total: int

    @property
    def success_rate(self) -> float:
        return self.feasible / self.total


def main() -> None:
    args = parse_args()
    sweep = SimulationSweep.from_yaml(str(args.config))
    cells = collect_feasibility(sweep.generate_tasks())
    if not cells:
        raise ValueError("The sweep did not generate any tasks.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for solver in sorted({cell.solver for cell in cells}):
        output_path = args.output_dir / f"feasible_mcmc_{solver}_heatmap.png"
        solver_cells = [cell for cell in cells if cell.solver == solver]
        plot_heatmap(solver_cells, output_path)
        print(f"Wrote {output_path}")
        for cell in sorted(
            solver_cells, key=lambda item: (item.num_nodes, item.num_zones)
        ):
            print(
                f"  {cell.level} ({cell.num_nodes} nodes), "
                f"{cell.num_zones} zones: {cell.feasible} / {cell.total}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot feasible-run heatmaps for an MCMC benchmark sweep."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SWEEP_CONFIG,
        help=f"Benchmark sweep YAML. Default: {DEFAULT_SWEEP_CONFIG}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def collect_feasibility(tasks: list[BenchmarkTask]) -> list[FeasibilityCell]:
    grouped: dict[tuple[str, int, str], list[bool]] = defaultdict(list)
    node_counts: dict[str, set[int]] = defaultdict(set)

    for task in tasks:
        solver = str(task.config["solver"])
        num_zones = len(load_centroid_schools(str(task.config["centroids_type"])))
        levels = [str(level) for level in task.config.get("levels", [])]
        if not levels:
            raise ValueError(f"Task {task.task_id} has no configured graph level.")

        payload = load_result(task)
        succeeded = str(payload.get("status") or "").upper() in SUCCESS_STATUSES
        for level in levels:
            grouped[(solver, num_zones, level)].append(succeeded)
            saved_count = saved_node_count(payload, level)
            if saved_count is not None:
                node_counts[level].add(saved_count)

    cells = []
    for (solver, num_zones, level), outcomes in grouped.items():
        cells.append(
            FeasibilityCell(
                solver=solver,
                num_zones=num_zones,
                level=level,
                num_nodes=level_node_count(level, node_counts[level]),
                feasible=sum(outcomes),
                total=len(outcomes),
            )
        )
    return cells


def load_result(task: BenchmarkTask) -> dict[str, Any]:
    result_path = Path(task.output_dir) / "result.json"
    if not result_path.exists():
        return {}
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def saved_node_count(payload: dict[str, Any], level: str) -> int | None:
    for stage in (payload.get("run") or {}).get("stages", []):
        if stage.get("level") == level and stage.get("num_nodes") is not None:
            return int(stage["num_nodes"])
    return None


def level_node_count(level: str, saved_counts: set[int]) -> int:
    spec = LevelSpec.parse(level)
    target = LEVEL_NODE_TARGETS.get(spec.unit, {}).get(spec.depth)
    if target is not None:
        return target
    if len(saved_counts) == 1:
        return next(iter(saved_counts))
    if not saved_counts:
        raise ValueError(
            f"{level} has no graph-builder target and no saved node count."
        )
    raise ValueError(
        f"{level} has inconsistent saved node counts: {sorted(saved_counts)}"
    )


def plot_heatmap(cells: list[FeasibilityCell], output_path: Path) -> None:
    if not cells:
        raise ValueError("Cannot plot an empty feasibility table.")

    solver = cells[0].solver
    zone_counts = sorted({cell.num_zones for cell in cells})
    node_counts = sorted({cell.num_nodes for cell in cells})
    cell_by_position = {(cell.num_nodes, cell.num_zones): cell for cell in cells}
    rates = np.zeros((len(node_counts), len(zone_counts)), dtype=float)

    for row, num_nodes in enumerate(node_counts):
        for column, num_zones in enumerate(zone_counts):
            cell = cell_by_position.get((num_nodes, num_zones))
            if cell is not None:
                rates[row, column] = cell.success_rate

    cmap = plt.colormaps["RdYlGn"].copy()
    cmap.set_bad("black")
    masked_rates = np.ma.masked_equal(rates, 0)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    image = ax.imshow(
        masked_rates,
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="lower",
        aspect="auto",
    )

    ax.set_xticks(range(len(zone_counts)), labels=zone_counts)
    ax.set_yticks(
        range(len(node_counts)), labels=[f"{value:,}" for value in node_counts]
    )
    ax.set_xlabel("Number of zones")
    ax.set_ylabel("Number of nodes")
    ax.set_title(f"Feasible MCMC Runs: {solver.replace('_', ' ').title()}")

    for row, num_nodes in enumerate(node_counts):
        for column, num_zones in enumerate(zone_counts):
            cell = cell_by_position.get((num_nodes, num_zones))
            if cell is None:
                continue
            text_color = (
                "white"
                if cell.success_rate == 0 or cell.success_rate < 0.3
                else "black"
            )
            ax.text(
                column,
                row,
                f"{cell.feasible} / {cell.total}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(-0.5, len(zone_counts), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(node_counts), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, pad=0.03)
    colorbar.set_label("Success rate across configured runs")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
