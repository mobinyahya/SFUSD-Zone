"""Spatial structure metrics for zoning solutions."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from Config.Constants import get_dropbox_path
from optimization.data.conversion import LevelConverter
from optimization.data.loaders import load_census_shapefile
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution

try:  # Shapely 2.x
    from shapely import minimum_bounding_circle
except ImportError:  # pragma: no cover - older shapely fallback
    minimum_bounding_circle = None


DEFAULT_ARTIFACT_DIR = Path(
    "/share/data/school_choice/Data/Computed/shape_metric_artifacts"
)
DEFAULT_COMPUTED_GRAPH_DIR = Path("/share/data/school_choice/Data/Computed/Graphs")
PROJECTED_CRS = "EPSG:32610"  # San Francisco is in UTM zone 10N.


@dataclass(frozen=True)
class SpatialMetrics:
    cut_edges: int
    normalized_cut_edges: float
    fractional_cut_edges: float
    avg_reock_score: float
    avg_polsby_popper_score: float


_BLOCK0_GRAPH_CACHE: dict[Path, Any] = {}
_GEOMETRY_CACHE: dict[Path, gpd.GeoDataFrame] = {}


def compute_spatial_metrics(
    solution: ZoneSolution,
    config: Mapping[str, Any] | None = None,
) -> SpatialMetrics:
    """Compute cut-edge and compactness metrics for one solution stage.

    Cut edges, Reock, and Polsby-Popper are computed after converting the
    solution to ``Block_0``.
    """

    config = config or {}
    if not solution.assignment:
        return SpatialMetrics(
            cut_edges=0,
            normalized_cut_edges=0.0,
            fractional_cut_edges=0.0,
            avg_reock_score=0.0,
            avg_polsby_popper_score=0.0,
        )

    block_G = _block0_graph(solution, config)
    block_assignment = _assignment_on_block0(solution, block_G)
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

    avg_reock_score, avg_polsby_popper_score = _average_zone_shape_scores(
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
        avg_polsby_popper_score=avg_polsby_popper_score,
    )


def _block0_graph(solution: ZoneSolution, config: Mapping[str, Any]):
    injected = config.get("block0_graph")
    if injected is not None:
        return injected

    if solution.level.unit == "Block" and solution.level.depth == 0:
        return solution.problem.G

    path = _block0_graph_path(config)
    if path is None:
        raise FileNotFoundError(
            "Block_0.pickle is required for normalized cut_edges metrics. "
            "Set config['block0_graph_path'] or ensure the graph exists under "
            "/share/data/school_choice/Data/Computed/Graphs."
        )
    if path not in _BLOCK0_GRAPH_CACHE:
        with path.open("rb") as f:
            _BLOCK0_GRAPH_CACHE[path] = pickle.load(f)
    return _BLOCK0_GRAPH_CACHE[path]


def _block0_graph_path(config: Mapping[str, Any]) -> Path | None:
    explicit = config.get("block0_graph_path")
    if explicit:
        path = Path(str(explicit)).expanduser()
        return path if path.exists() else None

    candidates: list[Path] = []
    graphs_dir = config.get("graphs_dir")
    if graphs_dir:
        graphs = Path(str(graphs_dir)).expanduser()
        candidates.extend(
            [
                graphs / "Block_0.pickle",
                graphs.parent / "Block_0.pickle",
            ]
        )

    candidates.extend(
        [
            DEFAULT_COMPUTED_GRAPH_DIR / "Block_0.pickle",
            Path(get_dropbox_path(False)).expanduser()
            / "Optimization"
            / "Zones"
            / "Graphs"
            / "Block_0.pickle",
            Path(get_dropbox_path(False)).expanduser()
            / "Optimization"
            / "Zones"
            / "Graphs"
            / "optimization"
            / "Block_0.pickle",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _assignment_on_block0(solution: ZoneSolution, block_G) -> dict[int, int]:
    if (
        solution.problem.G is block_G
        and solution.level.unit == "Block"
        and solution.level.depth == 0
    ):
        return dict(solution.assignment)
    return LevelConverter().between(
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


def _average_zone_shape_scores(
    solution: ZoneSolution,
    config: Mapping[str, Any],
    *,
    graph=None,
    assignment: Mapping[int, int] | None = None,
    level: LevelSpec | None = None,
) -> tuple[float, float]:
    assignment = solution.assignment if assignment is None else assignment
    areas = _node_area_metrics(solution, config, graph=graph, level=level)
    if areas.empty:
        return 0.0, 0.0

    assigned = areas[["node", "geometry"]].copy()
    assigned["zone_id"] = assigned["node"].map(
        {int(node): int(zone) for node, zone in assignment.items()}
    )
    assigned = assigned.dropna(subset=["zone_id", "geometry"]).copy()
    if assigned.empty:
        return 0.0, 0.0
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

    return _mean(reock_scores), _mean(polsby_popper_scores)


def _node_area_metrics(
    solution: ZoneSolution,
    config: Mapping[str, Any],
    *,
    graph=None,
    level: LevelSpec | None = None,
) -> gpd.GeoDataFrame:
    G = solution.problem.G if graph is None else graph
    level = solution.level if level is None else level

    injected = config.get("geometry_metrics_gdf")
    if injected is not None:
        return _prepare_injected_geometry(injected)

    graph_geometry = _geometry_from_graph_attrs(G)
    if graph_geometry is not None:
        return graph_geometry

    artifact_dir = Path(
        str(config.get("shape_metric_artifact_dir") or DEFAULT_ARTIFACT_DIR)
    ).expanduser()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = _graph_geometry_fingerprint(G)
    path = artifact_dir / f"area_perimeter_{level.name}_{fingerprint}.pkl"
    meta_path = artifact_dir / f"area_perimeter_{level.name}_{fingerprint}.json"
    if path in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[path].copy()
    if path.exists():
        gdf = pd.read_pickle(path)
    else:
        gdf = _build_node_area_metrics(G, level)
        gdf.to_pickle(path)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "level": level.name,
                    "unit": level.unit,
                    "nodes": int(G.number_of_nodes()),
                    "fingerprint": fingerprint,
                    "projected_crs": PROJECTED_CRS,
                },
                f,
                indent=2,
                sort_keys=True,
            )
    _GEOMETRY_CACHE[path] = gdf
    return gdf.copy()


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


def _build_node_area_metrics(G, level: LevelSpec) -> gpd.GeoDataFrame:
    base = load_census_shapefile(level.unit, False)
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

    geo = base[[level.unit, "geometry"]].dropna(subset=[level.unit, "geometry"]).copy()
    geo[level.unit] = geo[level.unit].astype("int64")
    geo["node"] = geo[level.unit].map(area_to_node)
    geo = geo.dropna(subset=["node"]).copy()
    if geo.empty:
        raise ValueError(
            f"No {level.unit} geometries matched {level.name} graph nodes."
        )
    geo["node"] = geo["node"].astype(int)

    if geo.crs is None:
        geo = geo.set_crs(epsg=4326, allow_override=True)
    dissolved = geo.dissolve(by="node", as_index=False)[["node", "geometry"]]
    projected = dissolved.to_crs(PROJECTED_CRS)
    projected["geometry"] = projected["geometry"].apply(_clean_geometry)
    projected["area"] = projected.geometry.area
    projected["perimeter"] = projected.geometry.length
    return projected


def _node_area_ids(G, node: int) -> list[int]:
    attrs = G.nodes[node]
    if "area_id" in attrs:
        return [int(attrs["area_id"])]
    return [int(area_id) for area_id in attrs.get("block_ids", [])]


def _graph_geometry_fingerprint(G) -> str:
    h = hashlib.sha1()
    for node in sorted(G.nodes()):
        h.update(str(int(node)).encode("utf-8"))
        h.update(b":")
        for area_id in sorted(_node_area_ids(G, node)):
            h.update(str(int(area_id)).encode("utf-8"))
            h.update(b",")
        h.update(b";")
    return h.hexdigest()[:12]


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
