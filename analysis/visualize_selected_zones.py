#!/usr/bin/env python3
"""Render the selected small- and medium-zone plans side by side.

Usage:
    uv run python analysis/visualize_selected_zones.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Config.Constants import zone_colors  # noqa: E402
from optimization.data.loaders import load_census_shapefile  # noqa: E402

DEFAULT_ZONE_ROOT = Path("/soalnas/share/data/school_choice/simulation-files/zones")
DEFAULT_SMALL_ZONES = (
    DEFAULT_ZONE_ROOT / "Zones_13-FRL_Dev_0.25-Objective_2500_BG.csv"
)
DEFAULT_MEDIUM_ZONES = (
    DEFAULT_ZONE_ROOT / "Zones_6-FRL_Dev_0.10-Objective_1430_BG.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis/plots/selected_zone_plans.png"

ZONE_FILENAME_PATTERN = re.compile(
    r"Zones_(?P<count>\d+)-FRL_Dev_(?P<frl>[\d.]+)-Objective_(?P<objective>[\d.]+)_BG"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-zones", type=Path, default=DEFAULT_SMALL_ZONES)
    parser.add_argument("--medium-zones", type=Path, default=DEFAULT_MEDIUM_ZONES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_zone_assignment(path: Path) -> dict[int, int]:
    """Load a headerless row-per-zone CSV as ``{BlockGroup: zone_id}``."""
    assignment: dict[int, int] = {}
    with path.open(encoding="utf-8", newline="") as zone_file:
        rows = list(csv.reader(zone_file))

    if not rows:
        raise ValueError(f"Zone file is empty: {path}")

    for zone_id, row in enumerate(rows):
        area_ids = [value.strip() for value in row if value.strip()]
        if not area_ids:
            raise ValueError(f"Zone {zone_id} is empty in {path}")
        for value in area_ids:
            area_id = int(value)
            if area_id in assignment:
                raise ValueError(
                    f"BlockGroup {area_id} appears in zones "
                    f"{assignment[area_id]} and {zone_id} in {path}"
                )
            assignment[area_id] = zone_id
    return assignment


def load_block_group_geometry() -> gpd.GeoDataFrame:
    census = load_census_shapefile("BlockGroup")
    geometry = census[["BlockGroup", "geometry"]].dissolve(
        by="BlockGroup", as_index=False
    )
    geometry["BlockGroup"] = geometry["BlockGroup"].astype("int64")
    if geometry.crs is not None:
        geometry = geometry.to_crs(epsg=4326)
    return geometry


def _plan_details(path: Path, zone_count: int) -> str:
    match = ZONE_FILENAME_PATTERN.fullmatch(path.stem)
    if not match:
        return f"{zone_count} zones"
    return (
        f"{zone_count} zones  |  FRL deviation {match['frl']}  |  "
        f"objective {match['objective']}"
    )


def plot_zone_plan(
    ax,
    geometry: gpd.GeoDataFrame,
    assignment: dict[int, int],
    *,
    label: str,
    source: Path,
) -> None:
    assigned = geometry.copy()
    assigned["zone_id"] = assigned["BlockGroup"].map(assignment)
    unknown = set(assignment) - set(assigned["BlockGroup"])
    if unknown:
        examples = ", ".join(str(area_id) for area_id in sorted(unknown)[:5])
        raise ValueError(f"{source} has BlockGroups missing from geometry: {examples}")

    assigned = assigned.dropna(subset=["zone_id"]).copy()
    assigned["zone_id"] = assigned["zone_id"].astype(int)
    zones = sorted(assigned["zone_id"].unique())
    colors = {zone: zone_colors[zone] for zone in zones}
    assigned["zone_color"] = assigned["zone_id"].map(colors)

    assigned.plot(
        ax=ax,
        color=assigned["zone_color"],
        edgecolor="white",
        linewidth=0.22,
    )
    boundaries = assigned.dissolve(by="zone_id")
    boundaries.boundary.plot(ax=ax, color="#17222b", linewidth=1.15)

    label_geometry = boundaries.to_crs(epsg=7131).representative_point().to_crs(
        epsg=4326
    )
    for zone_id, point in label_geometry.items():
        text = ax.text(
            point.x,
            point.y,
            str(int(zone_id)),
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        text.set_path_effects(
            [path_effects.withStroke(linewidth=2.5, foreground="#17222b")]
        )

    ax.set_title(
        f"{label}\n{_plan_details(source, len(zones))}",
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    ax.text(
        0,
        -0.025,
        source.name,
        transform=ax.transAxes,
        fontsize=7.5,
        color="#52606b",
        ha="left",
        va="top",
    )
    ax.set_axis_off()


def render_comparison(
    small_zones: Path,
    medium_zones: Path,
    output: Path,
    *,
    geometry: gpd.GeoDataFrame | None = None,
) -> Path:
    small_zones = small_zones.expanduser().resolve()
    medium_zones = medium_zones.expanduser().resolve()
    output = output.expanduser().resolve()
    small_assignment = load_zone_assignment(small_zones)
    medium_assignment = load_zone_assignment(medium_zones)
    geometry = load_block_group_geometry() if geometry is None else geometry

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), constrained_layout=True)
    fig.patch.set_facecolor("#f6f3ed")
    for ax in axes:
        ax.set_facecolor("#f6f3ed")

    plot_zone_plan(
        axes[0], geometry, small_assignment, label="Small Zones", source=small_zones
    )
    plot_zone_plan(
        axes[1],
        geometry,
        medium_assignment,
        label="Medium Zones",
        source=medium_zones,
    )
    fig.suptitle(
        "Selected SFUSD Zone Plans",
        fontsize=20,
        fontweight="bold",
        color="#17222b",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    output = render_comparison(args.small_zones, args.medium_zones, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
