"""Pipeline-native zoning visualizations.

This module deliberately does not reuse ``Graphic_Visualization``. The old
visualizers re-read and re-dissolved shapefiles for every plot and mixed legacy
output formats. Here the expensive geometry work is cached in a shared artifact
folder and rendering always writes PNG file artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from Helper_Functions.util import load_census_shapefile
from Zone_Generation.Config.Constants import zone_colors
from Zone_Generation.pipeline.levels import LevelSpec
from Zone_Generation.pipeline.solution import ZoneSolution

GeometryLoader = Callable[[str, bool], gpd.GeoDataFrame]

DEFAULT_ARTIFACT_DIR = Path(
    "/share/data/school_choice/Data/Computed/visualization_artifacts"
)


@dataclass
class RenderResult:
    """Files/artifacts produced for one rendered solution stage."""

    stage: str
    figure_paths: list[Path] = field(default_factory=list)
    geometry_artifact: Path | None = None
    skipped: str | None = None


def selected_stage_indices(num_solutions: int, mode: str) -> list[int]:
    """Return solution indices to visualize for ``final`` or ``all`` mode."""

    if num_solutions <= 0:
        return []
    if mode == "final":
        return [num_solutions - 1]
    if mode == "all":
        return list(range(num_solutions))
    raise ValueError("viz stages must be 'final' or 'all'.")


def stage_name(index: int, solution: ZoneSolution) -> str:
    """Stable unique name for recursive levels and iterative-choice repeats."""

    return f"stage_{index:02d}_{solution.level.name}"


class VisualizationArtifactStore:
    """Build and reuse expensive visualization artifacts under one artifact dir."""

    def __init__(
        self,
        is_local: bool,
        artifact_dir: str | Path | None = None,
        geometry_loader: GeometryLoader | None = None,
    ):
        self.artifact_dir = Path(artifact_dir or DEFAULT_ARTIFACT_DIR)
        self.is_local = is_local
        self.geometry_loader = geometry_loader or load_census_shapefile
        self._memory_cache: dict[Path, gpd.GeoDataFrame] = {}
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def geometry_for(self, level: LevelSpec, G) -> tuple[gpd.GeoDataFrame, Path]:
        """Return cached node geometry for ``G`` at ``level``.

        The cache key includes a graph/partition fingerprint so changing METIS
        aggregation or switching base units naturally creates a new artifact.
        """

        level = LevelSpec.parse(level)
        fingerprint = graph_geometry_fingerprint(G)
        path = self.artifact_dir / f"geometry_{level.name}_{fingerprint}.pkl"
        meta_path = self.artifact_dir / f"geometry_{level.name}_{fingerprint}.json"

        if path in self._memory_cache:
            return self._memory_cache[path].copy(), path
        if path.exists():
            geometry = pd.read_pickle(path)
        else:
            geometry = self._build_geometry(level, G)
            geometry.to_pickle(path)
            with meta_path.open("w") as f:
                json.dump(
                    {
                        "level": level.name,
                        "unit": level.unit,
                        "nodes": int(G.number_of_nodes()),
                        "fingerprint": fingerprint,
                    },
                    f,
                    indent=2,
                )

        self._memory_cache[path] = geometry
        return geometry.copy(), path

    def _build_geometry(self, level: LevelSpec, G) -> gpd.GeoDataFrame:
        base = self.geometry_loader(level.unit, self.is_local)
        if level.unit not in base.columns:
            raise ValueError(
                f"Base geometry for {level.unit!r} must include a {level.unit!r} column."
            )
        if "geometry" not in base.columns:
            raise ValueError("Base geometry must include a geometry column.")

        area_to_node: dict[int, int] = {}
        for node in G.nodes():
            for area_id in _node_area_ids(G, node):
                area_to_node[int(area_id)] = int(node)

        geo = (
            base[[level.unit, "geometry"]]
            .dropna(subset=[level.unit, "geometry"])
            .copy()
        )
        geo[level.unit] = geo[level.unit].astype("int64")
        geo["node"] = geo[level.unit].map(area_to_node)
        geo = geo.dropna(subset=["node"]).copy()
        geo["node"] = geo["node"].astype(int)
        if geo.empty:
            raise ValueError(
                f"No {level.unit} geometries matched graph nodes for {level.name}."
            )

        dissolved = geo.dissolve(by="node", as_index=False)[["node", "geometry"]]
        if dissolved.crs is not None:
            dissolved = dissolved.to_crs(epsg=4326)
        return dissolved


def graph_geometry_fingerprint(G) -> str:
    """Small stable hash of node labels and base ids used by cached geometry."""

    h = hashlib.sha1()
    for node in sorted(G.nodes()):
        h.update(str(int(node)).encode("utf-8"))
        h.update(b":")
        for area_id in sorted(_node_area_ids(G, node)):
            h.update(str(int(area_id)).encode("utf-8"))
            h.update(b",")
        h.update(b";")
    return h.hexdigest()[:12]


def visualize_solutions(
    solutions: Sequence[ZoneSolution],
    is_local: bool,
    stages: str = "final",
    geometry_loader: GeometryLoader | None = None,
    artifact_dir: str | Path | None = None,
) -> list[RenderResult]:
    """Render selected solution stages and save PNG artifacts."""

    store = VisualizationArtifactStore(
        is_local=is_local,
        artifact_dir=artifact_dir,
        geometry_loader=geometry_loader,
    )

    results: list[RenderResult] = []
    for index in selected_stage_indices(len(solutions), stages):
        solution = solutions[index]
        stage = stage_name(index, solution)
        result = RenderResult(stage=stage)

        if not solution.assignment:
            result.skipped = "solution has no assignment"
            results.append(result)
            continue

        geometry, geometry_path = store.geometry_for(solution.level, solution.problem.G)
        result.geometry_artifact = geometry_path
        fig = render_solution_map(solution, geometry, stage)
        result.figure_paths = [_save_figure(fig, store.artifact_dir, stage)]
        plt.close(fig)

        results.append(result)
    return results


def render_solution_map(
    solution: ZoneSolution,
    geometry: gpd.GeoDataFrame,
    stage: str,
):
    assigned = geometry.copy()
    assignment = {int(node): int(zone) for node, zone in solution.assignment.items()}
    assigned["zone_id"] = assigned["node"].map(assignment)
    assigned = assigned.dropna(subset=["zone_id"]).copy()
    assigned["zone_id"] = assigned["zone_id"].astype(int)
    if assigned.empty:
        raise ValueError(f"No geometries matched solution assignments for {stage}.")

    fig, map_ax = plt.subplots(figsize=(10, 10))

    colors = _zone_color_map(assigned["zone_id"].unique())
    assigned["zone_color"] = assigned["zone_id"].map(colors)
    assigned.plot(
        ax=map_ax,
        color=assigned["zone_color"],
        edgecolor="white",
        linewidth=0.2,
    )
    _plot_centroids(map_ax, solution)
    _format_map_axis(map_ax, solution, stage)
    _add_zone_legend(map_ax, colors)

    fig.tight_layout()
    return fig


def _save_figure(fig, output_dir: Path, stage: str) -> Path:
    path = output_dir / f"visualization_{stage}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def _node_area_ids(G, node: int) -> list[int]:
    attrs = G.nodes[node]
    if "area_id" in attrs:
        return [int(attrs["area_id"])]
    return [int(area_id) for area_id in attrs.get("block_ids", [])]


def _zone_color_map(zones) -> dict[int, str]:
    ordered = sorted(int(zone) for zone in zones)
    colors: dict[int, str] = {}
    missing = []
    for zone in ordered:
        if zone in zone_colors:
            colors[zone] = zone_colors[zone]
        else:
            missing.append(zone)

    if missing:
        cmap = plt.get_cmap("tab20", max(len(missing), 1))
        for idx, zone in enumerate(missing):
            colors[zone] = mcolors.to_hex(cmap(idx))
    return colors


def _plot_centroids(ax, solution: ZoneSolution) -> None:
    xs = []
    ys = []
    labels = []
    for zone, node in enumerate(solution.problem.centroids):
        if node not in solution.problem.G:
            continue
        attrs = solution.problem.G.nodes[node]
        lon = attrs.get("lon")
        lat = attrs.get("lat")
        if lon is None or lat is None:
            continue
        xs.append(float(lon))
        ys.append(float(lat))
        labels.append(str(zone))

    if not xs:
        return
    ax.scatter(xs, ys, s=70, c="black", marker="*", linewidths=0.4, zorder=5)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(
            label,
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="black",
            zorder=6,
        )


def _format_map_axis(ax, solution: ZoneSolution, stage: str) -> None:
    objective = "n/a" if solution.objective is None else f"{solution.objective:.2f}"
    wall = "n/a" if solution.wall_time is None else f"{solution.wall_time:.1f}s"
    ax.set_title(
        f"{stage}\nstatus={solution.status} objective={objective} time={wall}",
        fontsize=12,
    )
    ax.set_axis_off()


def _add_zone_legend(ax, colors: dict[int, str]) -> None:
    if len(colors) > 20:
        return
    handles = [
        mpatches.Patch(color=colors[zone], label=f"Zone {zone}")
        for zone in sorted(colors)
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=8,
        frameon=True,
        framealpha=0.85,
    )
