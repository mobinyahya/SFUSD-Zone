#!/usr/bin/env python3
"""Three maps of one multi-level run: coarse solve, released band, final zoning.

Reads a run folder's ``zonings.json.xz`` and ``zoning_config.yaml`` and draws,
side by side:

1. the first stage's zoning, with that level's units outlined;
2. the refine step's feasible set: the blocks it held fixed (faded, in their
   coarse zone) and the boundary band it released to be re-decided;
3. the final zoning, with the blocks that changed zone outlined.

Every stage is stored per base area (Block_0 block), and the refine stage's
metadata records ``pipeline_fixed_areas``, so nothing is re-solved::

    uv run python -m analysis.plots.plot_multilevel_stages RUN_DIR -o stages.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from optimization.config import OptimizationConfig
from optimization.data.dataset import Dataset
from optimization.data.loaders import load_census_shapefile
from optimization.levels import LevelSpec
from optimization.pipeline import SolutionInput
from optimization.run_archive import area_assignment, read_zonings
from optimization.strategies.pipeline import (
    centroid_neighborhood,
    release_unreachable_assignments,
    selected_nodes,
)

# Reference categorical palette (dataviz skill), fixed order; zones are also
# numbered on the map, so identity never rests on colour alone.
ZONE_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
RELEASED = "#d9d8d2"
PROJECTED_CRS = "EPSG:26910"  # UTM 10N, metres, for dissolving and labels


def _zone_color(zone: int) -> str:
    return ZONE_COLORS[int(zone) % len(ZONE_COLORS)]


def _stage(archive, index: int) -> dict:
    return archive["stages"][index]


def _node_areas(G, node) -> list:
    attrs = G.nodes[node]
    return attrs.get("block_ids") or [attrs["area_id"]]


def run_graph(ds: Dataset, archive, stage):
    """The graph ``stage`` was solved on.

    Level names and graph construction change between code versions (Block
    levels were renumbered, so an old ``Block_3`` is today's ``Block_4``), so
    a run is matched to a graph by content: same node count, and every node's
    areas carry the saved zone of that node. The current code's graph is
    tried first, then every cached graph of that name.
    """
    level = LevelSpec.parse(stage["level"])
    zones = dict(zip(stage["node_ids"], stage["node_zones"]))
    area_zone = area_assignment(archive, stage)

    def matches(G) -> bool:
        return G.number_of_nodes() == len(zones) and all(
            area_zone.get(int(a)) == zones.get(n) for n in G for a in _node_areas(G, n)
        )

    G = ds.graph_for(level)
    if matches(G):
        return G
    graphs = Path(ds._graph_path(level)).parents[2]
    for path in sorted(graphs.glob(f"v*/*/{level.name}.pickle")):
        with open(path, "rb") as handle:
            G = pickle.load(handle)
        if matches(G):
            return G
    raise ValueError(
        f"No cached graph matches the saved {level.name} zoning; it was built "
        "by code whose graphs are not in the local cache."
    )


def current_fixed_areas(config: OptimizationConfig, archive) -> set[int]:
    """The refine step's fixed base areas as the current pipeline code builds them.

    Rebuilds what ``PipelineStrategy._solve_step`` fixes for a
    ``recursive`` step (interior of the radius-``boundary_radius`` band plus
    the kept centroid neighborhood, within distance, then
    :func:`release_unreachable_assignments` with ``release_paths``) from the
    saved coarse zoning, on the graph the run itself used, so a run saved by
    older code can be shown as it would be solved now.
    """
    ds = Dataset(config)
    last = archive["stages"][-1]
    coarse = area_assignment(archive, archive["stages"][0])
    G_run = run_graph(ds, archive, last)
    ds.graph_for = lambda level: G_run
    problem = ds.problem_for(LevelSpec.parse(last["level"]))
    G = problem.G

    def areas(node):
        attrs = G.nodes[node]
        return attrs.get("block_ids") or [attrs["area_id"]]

    projected = {n: coarse[int(areas(n)[0])] for n in G}
    radius = int(config.boundary_radius)
    band = SolutionInput(mode="assign", select="interior", radius=radius)
    fixed = {n: projected[n] for n in selected_nodes(G, projected, band)}
    # The run's own release settings, or the current defaults for a run saved
    # before they existed (keep radius null: the band radius).
    keep = config.keep_centroid_radius
    keep = radius if keep is None else keep
    for node, zone in centroid_neighborhood(
        G, problem.centroids, projected, keep
    ).items():
        fixed.setdefault(node, zone)
    fixed = {n: z for n, z in fixed.items() if z in problem.candidate_zones(n)}
    fixed, _, _ = release_unreachable_assignments(
        problem, fixed, dict(projected), config.release_paths
    )
    return {int(a) for n in fixed for a in areas(n)}


def load_stages(run_dir: Path, *, current_release: bool = False):
    """``(blocks, coarse_units, labels)`` for the run's first and last stage.

    With ``current_release`` the refine step's fixed set is rebuilt with the
    current pipeline code instead of read from the run's saved metadata.
    """
    archive = read_zonings(run_dir)
    if len(archive["stages"]) < 2:
        raise ValueError(f"{run_dir} saved one stage; it is not a multi-level run.")
    first, last = _stage(archive, 0), _stage(archive, -1)
    coarse = area_assignment(archive, first)
    final = area_assignment(archive, last)
    config = OptimizationConfig.from_yaml(str(run_dir / "zoning_config.yaml"))
    if current_release:
        fixed = current_fixed_areas(config, archive)
    else:
        fixed = {
            int(area)
            for area in (last.get("metadata") or {}).get("pipeline_fixed_areas", {})
        }
    unit = LevelSpec.parse(first["level"]).unit

    blocks = load_census_shapefile(unit, config.data_scenario)[[unit, "geometry"]]
    blocks = blocks.dropna().astype({unit: "int64"})
    blocks = blocks.dissolve(by=unit, as_index=False).to_crs(PROJECTED_CRS)
    blocks["coarse"] = blocks[unit].map(coarse)
    blocks["final"] = blocks[unit].map(final)
    blocks = blocks.dropna(subset=["coarse", "final"]).astype(
        {"coarse": int, "final": int}
    )
    blocks["fixed"] = blocks[unit].isin(fixed)
    blocks["changed"] = blocks["coarse"] != blocks["final"]

    # The first stage's own units (aggregates of blocks), to show its grain.
    G = run_graph(Dataset(config), archive, first)
    area_node = {
        int(a): int(n)
        for n, attrs in G.nodes(data=True)
        for a in (attrs.get("block_ids") or [attrs["area_id"]])
    }
    blocks["coarse_unit"] = blocks[unit].map(area_node)
    units = blocks.dissolve(by="coarse_unit", as_index=False)[["geometry"]]
    labels = {
        "first": first["level"].replace("_", " "),
        "last": last["level"].replace("_", " "),
        "centroids": config.centroids_type,
    }
    return blocks, units, labels


def _zone_outlines(blocks: gpd.GeoDataFrame, column: str) -> gpd.GeoDataFrame:
    return blocks.dissolve(by=column, as_index=False)[[column, "geometry"]]


def _label_zones(ax, zones: gpd.GeoDataFrame, column: str) -> None:
    for _, row in zones.iterrows():
        point = row.geometry.representative_point()
        ax.text(
            point.x,
            point.y,
            str(int(row[column]) + 1),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=INK,
            bbox=dict(boxstyle="circle,pad=0.25", fc=SURFACE, ec="none", alpha=0.9),
        )


def plot(blocks, units, labels, output: Path) -> None:
    plt.rcParams.update(
        {"font.family": "sans-serif", "font.size": 9, "hatch.linewidth": 1.6}
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=SURFACE)
    coarse_zones = _zone_outlines(blocks, "coarse")
    final_zones = _zone_outlines(blocks, "final")
    colors_coarse = blocks["coarse"].map(_zone_color)
    colors_final = blocks["final"].map(_zone_color)
    released = int((~blocks["fixed"]).sum())
    changed = int(blocks["changed"].sum())

    ax = axes[0]
    blocks.plot(ax=ax, color=colors_coarse, linewidth=0)
    units.boundary.plot(ax=ax, color=SURFACE, linewidth=0.25, alpha=0.35)
    coarse_zones.boundary.plot(ax=ax, color=INK, linewidth=0.9)
    _label_zones(ax, coarse_zones, "coarse")
    ax.set_title(f"1. {labels['first']} solution", loc="left", fontsize=10.5)

    ax = axes[1]
    fixed = blocks[blocks["fixed"]]
    fixed.plot(ax=ax, color=fixed["coarse"].map(_zone_color), alpha=0.7, linewidth=0)
    blocks[~blocks["fixed"]].plot(ax=ax, color=RELEASED, linewidth=0)
    coarse_zones.boundary.plot(ax=ax, color=INK, linewidth=0.6)
    ax.set_title(
        f"2. Boundary band released ({released / len(blocks):.0%} of blocks)",
        loc="left",
        fontsize=10.5,
    )
    ax.legend(
        handles=[
            mpatches.Patch(color=RELEASED, label="released, re-solved"),
            mpatches.Patch(color=ZONE_COLORS[0], alpha=0.7, label="fixed to its zone"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.0),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )

    ax = axes[2]
    blocks.plot(ax=ax, color=colors_final, linewidth=0)
    # Blocks that changed zone: stripes in the zone they left, over the zone
    # they joined. Dissolved per (old, new) pair so the stripes span whole
    # patches rather than vanishing inside single small blocks.
    moved = blocks[blocks["changed"]].dissolve(by=["coarse", "final"], as_index=False)
    for _, patch in moved.iterrows():
        gpd.GeoSeries([patch.geometry], crs=blocks.crs).plot(
            ax=ax,
            facecolor="none",
            edgecolor=_zone_color(patch["coarse"]),
            hatch="////",
            linewidth=0,
        )
    final_zones.boundary.plot(ax=ax, color=INK, linewidth=0.9)
    _label_zones(ax, final_zones, "final")
    stripe = mpatches.Patch(
        facecolor=ZONE_COLORS[2],
        edgecolor=ZONE_COLORS[1],
        hatch="////",
        linewidth=0,
        label="changed zone (stripes: zone it left)",
    )
    ax.legend(
        handles=[stripe],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.0),
        frameon=False,
        fontsize=8.5,
    )
    ax.set_title(
        f"3. {labels['last']} solution ({changed / len(blocks):.0%} of blocks changed zone)",
        loc="left",
        fontsize=10.5,
    )

    for ax in axes:
        ax.set_axis_off()
        ax.set_aspect("equal")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="A multi-level run folder.")
    parser.add_argument("-o", "--output", default="multilevel_stages.png")
    parser.add_argument(
        "--current-release",
        action="store_true",
        help="Rebuild step 2's fixed set with the current pipeline code "
        "instead of reading the run's saved one.",
    )
    args = parser.parse_args(argv)
    blocks, units, labels = load_stages(
        Path(args.run_dir), current_release=args.current_release
    )
    plot(blocks, units, labels, Path(args.output))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
