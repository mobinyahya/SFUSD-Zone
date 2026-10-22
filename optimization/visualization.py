"""Optimization-native zoning visualizations.

This module deliberately does not reuse ``Graphic_Visualization``. The old
visualizers re-read and re-dissolved shapefiles for every plot and mixed legacy
output formats. Here the expensive geometry work is cached in a validated shared
namespace and rendered PNGs are written to the optimization output directory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from Config.Constants import zone_colors
from loaders import CacheStore, DataScenario, load_scenario
from optimization.config import OptimizationConfig
from optimization.data.closer_neighbors import graph_geometry_fingerprint
from optimization.data.loaders import (
    OUTPUT_LATLON_CRS,
    census_geometry_roles,
    load_census_shapefile,
)
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution

GeometryLoader = Callable[[str], gpd.GeoDataFrame]
VISUALIZATION_GEOMETRY_CACHE_SCHEMA_VERSION = 4
GEOMETRY_PAYLOAD = "geometry.pkl"


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
    """Build and reuse source-aware geometry in the configured shared cache."""

    def __init__(
        self,
        data: DataScenario,
        artifact_dir: str | Path | None = None,
        geometry_loader: GeometryLoader | None = None,
    ):
        if not isinstance(data, DataScenario):
            raise TypeError("VisualizationArtifactStore data must be a DataScenario.")
        self.data = _with_cache_root(data, artifact_dir) if artifact_dir else data
        self.geometry_loader = geometry_loader or (
            lambda unit: load_census_shapefile(unit, self.data)
        )

    def geometry_for(self, level: LevelSpec, G) -> tuple[gpd.GeoDataFrame, Path]:
        """Return cached node geometry for ``G`` at ``level``.

        The namespace binds graph membership and scenario filters to validated
        census/crosswalk source manifests, including current source checksums.
        """

        level = LevelSpec.parse(level)
        namespace = CacheStore(self.data).namespace(
            "visualization_geometry",
            {
                "level": level.name,
                "unit": level.unit,
                "graph_geometry_fingerprint": graph_geometry_fingerprint(G),
                "node_count": G.number_of_nodes(),
                "optimization_filters": self.data.filters.get("optimization", {}),
                "output_crs": OUTPUT_LATLON_CRS,
                "operation": "dissolve_base_areas_by_graph_node",
            },
            schema_version=VISUALIZATION_GEOMETRY_CACHE_SCHEMA_VERSION,
            roles=census_geometry_roles(self.data, level.unit),
        )
        path = namespace.payload_path(GEOMETRY_PAYLOAD)

        geometry = namespace.load_pickle(GEOMETRY_PAYLOAD)
        if not _valid_cached_geometry(geometry):
            geometry = self._build_geometry(level, G)
            path = namespace.save_pickle(GEOMETRY_PAYLOAD, geometry)

        return geometry.copy(), path

    def _build_geometry(self, level: LevelSpec, G) -> gpd.GeoDataFrame:
        base = self.geometry_loader(level.unit)
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

        if geo.crs is None:
            geo = geo.set_crs(OUTPUT_LATLON_CRS, allow_override=True)
        dissolved = geo.dissolve(by="node", as_index=False)[["node", "geometry"]]
        dissolved = dissolved.to_crs(OUTPUT_LATLON_CRS)
        return dissolved


def _valid_cached_geometry(value: Any) -> bool:
    return (
        isinstance(value, gpd.GeoDataFrame)
        and {"node", "geometry"} <= set(value.columns)
        and not value.empty
        and not value["node"].isna().any()
        and not value["node"].duplicated().any()
    )


def _with_cache_root(data: DataScenario, cache_root: str | Path) -> DataScenario:
    roots = dict(data.roots)
    roots["cache"] = Path(cache_root).expanduser().resolve()
    return replace(data, roots=MappingProxyType(roots))


def _scenario_for_solution(
    solution: ZoneSolution,
    config: OptimizationConfig | DataScenario | Mapping[str, Any] | None,
) -> DataScenario:
    if isinstance(config, DataScenario):
        return config
    if isinstance(config, OptimizationConfig):
        return config.data_scenario
    if isinstance(config, Mapping) and isinstance(config.get("data"), Mapping):
        return load_scenario(config["data"])

    solution_config = solution.problem.optimization_config
    if solution_config is not None:
        return solution_config.data_scenario
    raise ValueError(
        "Visualization requires the solution's strict OptimizationConfig/data scenario."
    )


def visualize_solutions(
    solutions: Sequence[ZoneSolution],
    output_dir: str | Path,
    stages: str = "final",
    config: OptimizationConfig | DataScenario | Mapping[str, Any] | None = None,
    geometry_loader: GeometryLoader | None = None,
    artifact_dir: str | Path | None = None,
) -> list[RenderResult]:
    """Render selected solution stages and save PNG artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[RenderResult] = []
    stores: dict[tuple[str, str], VisualizationArtifactStore] = {}
    for index in selected_stage_indices(len(solutions), stages):
        solution = solutions[index]
        stage = stage_name(index, solution)
        result = RenderResult(stage=stage)

        if not solution.assignment:
            result.skipped = "solution has no assignment"
            results.append(result)
            continue

        scenario = _scenario_for_solution(solution, config)
        store_key = (str(scenario.cache_root), scenario.semantic_fingerprint)
        store = stores.get(store_key)
        if store is None:
            store = VisualizationArtifactStore(
                scenario,
                artifact_dir=artifact_dir,
                geometry_loader=geometry_loader,
            )
            stores[store_key] = store
        geometry, geometry_path = store.geometry_for(solution.level, solution.problem.G)
        result.geometry_artifact = geometry_path
        fig = render_solution_map(solution, geometry, stage)
        result.figure_paths = [_save_figure(fig, output_path, stage)]
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
    _plot_schools(map_ax, solution)
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


def _plot_schools(ax, solution: ZoneSolution) -> None:
    school_data = solution.problem.G.graph.get("school_data", {})
    plotted: set[int] = set()

    for _node, attrs in solution.problem.G.nodes(data=True):
        for school_id in attrs.get("school_ids", []):
            sid = int(school_id)
            if sid in plotted:
                continue

            info = school_data.get(school_id, school_data.get(sid, {}))
            if not isinstance(info, dict):
                info = {}
            lon = _valid_float(info.get("lon", info.get("Lon")))
            lat = _valid_float(info.get("lat", info.get("Lat")))
            if lon is None or lat is None:
                lon = _valid_float(attrs.get("lon"))
                lat = _valid_float(attrs.get("lat"))
            if lon is None or lat is None:
                continue

            ax.text(
                lon,
                lat,
                "S",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=5,
            )
            plotted.add(sid)


def _valid_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(number):
        return None
    return number


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
