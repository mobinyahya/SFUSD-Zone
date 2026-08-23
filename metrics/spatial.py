"""Spatial structure metrics for zoning solutions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from optimization.config import OptimizationConfig
from optimization.data.conversion import LevelConverter
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.visualization import VisualizationArtifactStore

try:  # Shapely 2.x
    from shapely import minimum_bounding_circle
except ImportError:  # pragma: no cover - older shapely fallback
    minimum_bounding_circle = None


PROJECTED_CRS = "EPSG:32610"  # San Francisco is in UTM zone 10N.


@dataclass(frozen=True)
class SpatialMetrics:
    cut_edges: int
    normalized_cut_edges: float
    fractional_cut_edges: float
    avg_reock_score: float
    max_reock_score: float
    avg_polsby_popper_score: float
    max_polsby_popper_score: float


def compute_spatial_metrics(
    solution: ZoneSolution,
    config: Mapping[str, Any] | OptimizationConfig | None = None,
) -> SpatialMetrics:
    """Compute cut-edge and compactness metrics for one solution stage.

    Cut edges, Reock, and Polsby-Popper are computed after converting the
    solution to ``Block_0``.
    """

    config = config if config is not None else {}
    if not solution.assignment:
        return SpatialMetrics(
            cut_edges=0,
            normalized_cut_edges=0.0,
            fractional_cut_edges=0.0,
            avg_reock_score=0.0,
            max_reock_score=0.0,
            avg_polsby_popper_score=0.0,
            max_polsby_popper_score=0.0,
        )

    block_G = _block0_graph(solution, config)
    block_assignment = _assignment_on_block0(solution, block_G, config)
    cut_edges = _cut_edges(
        block_G,
        block_assignment,
        include_unassigned_boundary=bool(
            solution.metadata.get("partial_assignment", False)
        ),
    )
    num_zones = solution.problem.Z
    normalized_cut_edges = cut_edges / num_zones if num_zones else 0.0
    total_edges = block_G.number_of_edges()
    fractional_cut_edges = cut_edges / total_edges if total_edges else 0.0

    (
        avg_reock_score,
        max_reock_score,
        avg_polsby_popper_score,
        max_polsby_popper_score,
    ) = _zone_shape_scores(
        solution,
        config,
        graph=block_G,
        assignment=block_assignment,
        level=LevelSpec("Block", 0),
    )
    return SpatialMetrics(
        cut_edges=cut_edges,
        normalized_cut_edges=normalized_cut_edges,
        fractional_cut_edges=fractional_cut_edges,
        avg_reock_score=avg_reock_score,
        max_reock_score=max_reock_score,
        avg_polsby_popper_score=avg_polsby_popper_score,
        max_polsby_popper_score=max_polsby_popper_score,
    )


def _block0_graph(
    solution: ZoneSolution,
    config: Mapping[str, Any] | OptimizationConfig,
):
    injected = config.get("block0_graph") if isinstance(config, Mapping) else None
    if injected is not None:
        return injected

    if solution.level.unit == "Block" and solution.level.depth == 0:
        return solution.problem.G

    source_config = _optimization_config_for_solution(solution, config)
    block_config = OptimizationConfig(levels=["Block_0"], data=source_config.data)
    return block_config.make_dataset().graph_for(LevelSpec("Block", 0))


def _assignment_on_block0(
    solution: ZoneSolution,
    block_G,
    config: Mapping[str, Any] | OptimizationConfig,
) -> dict[int, int]:
    if (
        solution.problem.G is block_G
        and solution.level.unit == "Block"
        and solution.level.depth == 0
    ):
        return dict(solution.assignment)
    data = None
    if solution.level.unit != "Block":
        data = _optimization_config_for_solution(solution, config).data_scenario
    return LevelConverter(data=data).between(
        solution.problem.G,
        solution.assignment,
        solution.level,
        block_G,
        LevelSpec("Block", 0),
    )


def _cut_edges(
    G,
    assignment: Mapping[int, int],
    *,
    include_unassigned_boundary: bool = False,
) -> int:
    cut = 0
    for u, v in G.edges():
        zone_u = assignment.get(u)
        zone_v = assignment.get(v)
        if include_unassigned_boundary:
            cut += int(zone_u != zone_v)
            continue
        if zone_u is None or zone_v is None:
            continue
        if zone_u != zone_v:
            cut += 1
    return cut


def _zone_shape_scores(
    solution: ZoneSolution,
    config: Mapping[str, Any] | OptimizationConfig,
    *,
    graph=None,
    assignment: Mapping[int, int] | None = None,
    level: LevelSpec | None = None,
) -> tuple[float, float, float, float]:
    assignment = solution.assignment if assignment is None else assignment
    areas = _node_area_metrics(solution, config, graph=graph, level=level)
    if areas.empty:
        return 0.0, 0.0, 0.0, 0.0

    assigned = areas[["node", "geometry"]].copy()
    assigned["zone_id"] = assigned["node"].map(
        {int(node): int(zone) for node, zone in assignment.items()}
    )
    assigned = assigned.dropna(subset=["zone_id", "geometry"]).copy()
    if assigned.empty:
        return 0.0, 0.0, 0.0, 0.0
    assigned["zone_id"] = assigned["zone_id"].astype(int)

    zones = assigned.dissolve(by="zone_id", as_index=False)[["zone_id", "geometry"]]
    reock_scores: list[float] = []
    polsby_popper_scores: list[float] = []
    for geometry in zones.geometry:
        geometry = _clean_geometry(geometry)
        if geometry is None or geometry.is_empty:
            continue
        area = float(geometry.area)
        perimeter = float(geometry.length)
        if area <= 0:
            continue
        reock_scores.append(_reock_score(geometry, area))
        polsby_popper_scores.append(_polsby_popper_score(area, perimeter))

    return (
        _mean(reock_scores),
        max(reock_scores, default=0.0),
        _mean(polsby_popper_scores),
        max(polsby_popper_scores, default=0.0),
    )


def _node_area_metrics(
    solution: ZoneSolution,
    config: Mapping[str, Any] | OptimizationConfig,
    *,
    graph=None,
    level: LevelSpec | None = None,
) -> gpd.GeoDataFrame:
    G = solution.problem.G if graph is None else graph
    level = solution.level if level is None else level

    injected = (
        config.get("geometry_metrics_gdf") if isinstance(config, Mapping) else None
    )
    if injected is not None:
        return _prepare_injected_geometry(injected)

    graph_geometry = _geometry_from_graph_attrs(G)
    if graph_geometry is not None:
        return graph_geometry

    source_config = _optimization_config_for_solution(solution, config)
    geometry, _ = VisualizationArtifactStore(source_config.data_scenario).geometry_for(
        level, G
    )
    return _prepare_injected_geometry(geometry)


def _optimization_config_for_solution(
    solution: ZoneSolution,
    config: Mapping[str, Any] | OptimizationConfig,
) -> OptimizationConfig:
    if isinstance(config, OptimizationConfig):
        return config
    if solution.problem.optimization_config is not None:
        return solution.problem.optimization_config
    data = config.get("data") if isinstance(config, Mapping) else None
    if isinstance(data, Mapping):
        return OptimizationConfig(levels=[solution.level.name], data=dict(data))
    raise ValueError(
        "Spatial metrics require the solution's strict OptimizationConfig/data scenario."
    )


def _prepare_injected_geometry(value: Any) -> gpd.GeoDataFrame:
    gdf = value.copy()
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
    if gdf.crs is None:
        gdf = gdf.set_crs(PROJECTED_CRS, allow_override=True)
    elif str(gdf.crs).upper() != PROJECTED_CRS:
        gdf = gdf.to_crs(PROJECTED_CRS)
    gdf = gdf[["node", "geometry"]].copy()
    gdf["node"] = gdf["node"].astype(int)
    gdf["geometry"] = gdf["geometry"].apply(_clean_geometry)
    gdf["area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length
    return gdf


def _geometry_from_graph_attrs(G) -> gpd.GeoDataFrame | None:
    rows = []
    for node, attrs in G.nodes(data=True):
        geometry = attrs.get("geometry")
        if geometry is None:
            return None
        rows.append({"node": int(node), "geometry": geometry})
    if not rows:
        return None
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=PROJECTED_CRS)
    gdf["geometry"] = gdf["geometry"].apply(_clean_geometry)
    gdf["area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length
    return gdf


def _clean_geometry(geometry: BaseGeometry | None) -> BaseGeometry | None:
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.is_valid:
        return geometry
    return geometry.buffer(0)


def _reock_score(geometry: BaseGeometry, area: float) -> float:
    if minimum_bounding_circle is not None:
        circle = minimum_bounding_circle(geometry)
        circle_area = float(circle.area) if circle is not None else 0.0
    else:  # pragma: no cover - used only with older shapely versions
        circle_area = _fallback_bounding_circle_area(geometry)
    return _bounded_score(area / circle_area) if circle_area > 0 else 0.0


def _fallback_bounding_circle_area(geometry: BaseGeometry) -> float:
    minx, miny, maxx, maxy = geometry.bounds
    radius = math.hypot(maxx - minx, maxy - miny) / 2
    return math.pi * radius * radius


def _polsby_popper_score(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return _bounded_score(4 * math.pi * area / (perimeter * perimeter))


def _bounded_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
