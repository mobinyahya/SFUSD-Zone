"""Versioned geometry-based closer-neighbor artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import pickle
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd

from loaders import DataScenario
from optimization.data import loaders
from optimization.levels import LevelSpec

# Bumped to 4 when Mission Bay's canonical school ID became 1731, and to 5 when
# each variant moved to its own file: one shared file of every variant made a
# Block_0 run unpickle all of them (160 MB on disk, gigabytes in memory) to use
# one. ``split_legacy_cache`` converts a v4 cache.
CLOSER_NEIGHBOR_CACHE_SCHEMA_VERSION = 5
CLOSER_NEIGHBORS_GRAPH_KEY = "closer_neighbors"
SCHOOL_GEOMETRY_DISTANCES_GRAPH_KEY = "school_geometry_distances_miles"
METERS_PER_MILE = 1609.344

# Validated relations by (cache file, variant fingerprint), shared by every
# store in the process: a Dataset per run (metrics regeneration, the Block_0
# graph spatial metrics build) otherwise re-reads and re-validates ~1 s each.
# The fingerprint binds graph membership, adjacency and source contents; the
# file's stat signature, taken when it was read or written, catches a file
# changed since.
_PROCESS_MEMO: "OrderedDict[tuple[Path, str], tuple[tuple, CloserNeighborData]]" = (
    OrderedDict()
)
_PROCESS_MEMO_SIZE = 4

GeometryLoader = Callable[[str], gpd.GeoDataFrame]
SchoolLoader = Callable[[], pd.DataFrame]
CloserNeighborLookup = dict[int, dict[int, frozenset[int]]]
SchoolDistanceLookup = dict[int, dict[int, float]]


@dataclass(frozen=True)
class CloserNeighborData:
    """Geometry distances and closer adjacent nodes for every node-school pair."""

    school_ids: tuple[int, ...]
    closer_neighbors: CloserNeighborLookup
    distances_miles: SchoolDistanceLookup


class CloserNeighborArtifactStore:
    """Build and load source-aware closer-neighbor artifacts by level."""

    def __init__(
        self,
        data: DataScenario,
        geometry_loader: GeometryLoader | None = None,
        school_loader: SchoolLoader | None = None,
    ) -> None:
        if not isinstance(data, DataScenario):
            raise TypeError("CloserNeighborArtifactStore data must be a DataScenario.")
        self.data = data
        self.cache_dir = (
            data.cache_root
            / "closer_neighbors"
            / f"v{CLOSER_NEIGHBOR_CACHE_SCHEMA_VERSION}"
        )
        self.geometry_loader = geometry_loader or self._load_geometry
        self.school_loader = school_loader or (
            lambda: loaders.load_school_coordinates(self.data)
        )
        self._memory_cache: dict[tuple[Path, str], CloserNeighborData] = {}

    def _load_geometry(self, unit: str) -> gpd.GeoDataFrame:
        return loaders.load_census_shapefile(unit, self.data)

    def attach_to_graph(self, level, G: nx.Graph) -> CloserNeighborData:
        """Attach the cached relation to ``G`` and return it."""
        data = self.for_graph(level, G)
        G.graph[CLOSER_NEIGHBORS_GRAPH_KEY] = data.closer_neighbors
        G.graph[SCHOOL_GEOMETRY_DISTANCES_GRAPH_KEY] = data.distances_miles
        return data

    def for_graph(self, level, G: nx.Graph) -> CloserNeighborData:
        """Return the exact geometry relation for one graph variant."""
        level = LevelSpec.parse(level)
        graph_fingerprint = graph_geometry_fingerprint(G)
        source_fingerprint = self._source_fingerprint()
        fingerprint = hashlib.sha256(
            f"{graph_fingerprint}:{source_fingerprint}".encode("ascii")
        ).hexdigest()[:20]
        path = self.cache_path(level, fingerprint)
        memory_key = (path, fingerprint)
        if memory_key in self._memory_cache:
            return self._memory_cache[memory_key]
        memo = _PROCESS_MEMO.get(memory_key)
        if memo is not None and memo[0] == _stat_signature(path):
            _PROCESS_MEMO.move_to_end(memory_key)
            self._memory_cache[memory_key] = memo[1]
            return memo[1]

        payload = self._read_payload(path, level)
        variant = payload["variants"].get(fingerprint)
        data = (
            self._validated_data(
                variant,
                G,
                graph_fingerprint=graph_fingerprint,
                source_fingerprint=source_fingerprint,
            )
            if variant is not None
            else None
        )
        if data is None:
            data = self._build(level, G)
            data = self._save_variant(
                path,
                level,
                fingerprint,
                graph_fingerprint,
                source_fingerprint,
                G,
                data,
            )

        self._memory_cache[memory_key] = data
        signature = _stat_signature(path)
        if signature is not None:
            _PROCESS_MEMO[memory_key] = (signature, data)
            _PROCESS_MEMO.move_to_end(memory_key)
        while len(_PROCESS_MEMO) > _PROCESS_MEMO_SIZE:
            _PROCESS_MEMO.popitem(last=False)
        return data

    def level_variants(self, level) -> dict:
        """Every cached variant of a level, by fingerprint (for inspection)."""
        level = LevelSpec.parse(level)
        variants = {}
        for path in sorted(
            (self.cache_dir / f"closer_neighbors_{level.name}").glob("*.pickle")
        ):
            variants.update(self._read_payload(path, level)["variants"])
        return variants

    def cache_path(self, level, fingerprint: str) -> Path:
        """Return the cache file of one graph variant of a level."""
        level = LevelSpec.parse(level)
        return (
            self.cache_dir / f"closer_neighbors_{level.name}" / f"{fingerprint}.pickle"
        )

    def _source_fingerprint(self) -> str:
        roles = [loaders.SCHOOL_ROLE]
        for role in (
            loaders.CENSUS_ROLE,
            loaders.CROSSWALK_ROLE,
            "optimization.geography.blockgroups",
            "optimization.geography.tracts",
        ):
            try:
                self.data.resolved(role)
            except KeyError:
                continue
            roles.append(role)
        payload = {
            "source_manifest": self.data.source_manifest(roles),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    def _build(self, level: LevelSpec, G: nx.Graph) -> CloserNeighborData:
        node_geometry = self._node_geometry(level, G)
        schools = self._school_geometry()
        school_ids = tuple(sorted(int(school_id) for school_id in schools.index))
        graph_nodes = sorted(int(node) for node in G.nodes())
        closer: CloserNeighborLookup = {node: {} for node in graph_nodes}
        distances: SchoolDistanceLookup = {node: {} for node in graph_nodes}

        for school_id in school_ids:
            point = schools.loc[school_id, "geometry"]
            school_distances = node_geometry.distance(point) / METERS_PER_MILE
            by_node = {
                int(node): float(distance)
                for node, distance in school_distances.items()
            }
            for node in graph_nodes:
                node_distance = by_node[node]
                distances[node][school_id] = node_distance
                closer[node][school_id] = frozenset(
                    int(neighbor)
                    for neighbor in G.neighbors(node)
                    if by_node[int(neighbor)] < node_distance
                )

        return CloserNeighborData(school_ids, closer, distances)

    def _node_geometry(self, level: LevelSpec, G: nx.Graph) -> gpd.GeoSeries:
        base = self.geometry_loader(level.unit)
        missing = {level.unit, "geometry"} - set(base.columns)
        if missing:
            raise ValueError(
                f"Base geometry for {level.name} is missing columns: {sorted(missing)}."
            )
        if base.crs is None:
            base = base.set_crs(loaders.OUTPUT_LATLON_CRS)
        base = base.to_crs(loaders.PROJECTED_CENTROID_CRS)

        area_to_node: dict[int, int] = {}
        for node in G.nodes():
            area_ids = _node_area_ids(G, node)
            if not area_ids:
                raise ValueError(f"Node {node} in {level.name} has no base area IDs.")
            for area_id in area_ids:
                previous = area_to_node.setdefault(int(area_id), int(node))
                if previous != int(node):
                    raise ValueError(
                        f"Base area {area_id} belongs to nodes {previous} and {node}."
                    )

        geometry = base[[level.unit, "geometry"]].dropna().copy()
        geometry[level.unit] = geometry[level.unit].astype("int64")
        geometry["node"] = geometry[level.unit].map(area_to_node)
        geometry = geometry.dropna(subset=["node"]).copy()
        geometry["node"] = geometry["node"].astype(int)
        geometry["geometry"] = geometry.geometry.make_valid()
        dissolved = geometry.dissolve(by="node")["geometry"]

        missing_nodes = set(G.nodes()) - set(dissolved.index)
        if missing_nodes:
            raise ValueError(
                f"Missing {level.name} geometry for graph nodes: {sorted(missing_nodes)}."
            )
        return dissolved.reindex(sorted(G.nodes()))

    def _school_geometry(self) -> gpd.GeoDataFrame:
        schools = self.school_loader().copy()
        required = {"school_id", "lat", "lon"}
        missing = required - set(schools.columns)
        if missing:
            raise ValueError(
                f"School coordinates are missing columns: {sorted(missing)}."
            )
        schools = schools.dropna(subset=sorted(required)).copy()
        schools["school_id"] = schools["school_id"].astype(int)
        if schools["school_id"].duplicated().any():
            duplicates = sorted(
                schools.loc[
                    schools["school_id"].duplicated(False), "school_id"
                ].unique()
            )
            raise ValueError(f"Duplicate school coordinates for IDs: {duplicates}.")
        geometry = gpd.GeoDataFrame(
            schools[["school_id"]],
            geometry=gpd.points_from_xy(schools["lon"], schools["lat"]),
            crs=loaders.OUTPUT_LATLON_CRS,
        ).to_crs(loaders.PROJECTED_CENTROID_CRS)
        return geometry.set_index("school_id").sort_index()

    def _save_variant(
        self,
        path: Path,
        level: LevelSpec,
        fingerprint: str,
        graph_fingerprint: str,
        source_fingerprint: str,
        G: nx.Graph,
        data: CloserNeighborData,
    ) -> CloserNeighborData:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(path.with_suffix(path.suffix + ".lock")):
            payload = self._read_payload(path, level)
            existing = payload["variants"].get(fingerprint)
            validated = (
                self._validated_data(
                    existing,
                    G,
                    graph_fingerprint=graph_fingerprint,
                    source_fingerprint=source_fingerprint,
                )
                if existing
                else None
            )
            if validated is not None:
                return validated
            payload["variants"][fingerprint] = {
                "graph_fingerprint": graph_fingerprint,
                "source_fingerprint": source_fingerprint,
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "school_ids": data.school_ids,
                "closer_neighbors": data.closer_neighbors,
                "distances_miles": data.distances_miles,
            }
            tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            with tmp_path.open("wb") as file:
                pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        return data

    def _read_payload(self, path: Path, level: LevelSpec) -> dict:
        payload = None
        if path.exists():
            try:
                with path.open("rb") as file:
                    payload = pickle.load(file)
            except (EOFError, OSError, pickle.UnpicklingError):
                payload = None
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != CLOSER_NEIGHBOR_CACHE_SCHEMA_VERSION
            or payload.get("level") != level.name
            or not isinstance(payload.get("variants"), dict)
        ):
            return {
                "schema_version": CLOSER_NEIGHBOR_CACHE_SCHEMA_VERSION,
                "level": level.name,
                "unit": level.unit,
                "variants": {},
            }
        return payload

    @staticmethod
    def _validated_data(
        variant,
        G: nx.Graph,
        *,
        graph_fingerprint: str,
        source_fingerprint: str,
    ) -> CloserNeighborData | None:
        if not isinstance(variant, dict):
            return None
        if (
            variant.get("graph_fingerprint") != graph_fingerprint
            or variant.get("source_fingerprint") != source_fingerprint
        ):
            return None
        try:
            school_ids = tuple(int(school_id) for school_id in variant["school_ids"])
            closer = variant["closer_neighbors"]
            distances = variant["distances_miles"]
        except (KeyError, TypeError, ValueError):
            return None

        nodes = {int(node) for node in G.nodes()}
        schools = set(school_ids)
        if not isinstance(closer, dict) or not isinstance(distances, dict):
            return None
        if closer.keys() != nodes or distances.keys() != nodes:
            return None
        for node in nodes:
            by_school = closer[node]
            node_distances = distances[node]
            if by_school.keys() != schools or node_distances.keys() != schools:
                return None
            graph_neighbors = {int(neighbor) for neighbor in G.neighbors(node)}
            if not _supports_valid(by_school.values(), graph_neighbors):
                return None
            if not _distances_valid(node_distances.values()):
                return None
        return CloserNeighborData(school_ids, closer, distances)


def _stat_signature(path: Path) -> tuple | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


_SET_TYPES = frozenset({set, frozenset})


def _supports_valid(supports, graph_neighbors: set[int]) -> bool:
    """Every support is a set of the node's graph neighbors."""
    if set(map(type, supports)) <= _SET_TYPES:
        # One C-level union instead of a subset test per school.
        return graph_neighbors.issuperset(frozenset().union(*supports))
    return all(
        isinstance(support, (set, frozenset)) and set(support) <= graph_neighbors
        for support in supports
    )


def _distances_valid(values) -> bool:
    """Every distance is finite and non-negative."""
    values = list(values)
    if set(map(type, values)) <= {float}:
        # A NaN or infinite term makes the exact sum non-finite.
        try:
            total = math.fsum(values)
        except (OverflowError, ValueError):  # huge terms, or inf + -inf
            total = None
        if total is not None:
            return math.isfinite(total) and (not values or min(values) >= 0)
    return all(
        math.isfinite(float(distance)) and float(distance) >= 0 for distance in values
    )


def graph_geometry_fingerprint(G: nx.Graph) -> str:
    """Hash runtime labels, base memberships, and adjacency for cache variants."""
    digest = hashlib.sha256()
    for node in sorted(G.nodes()):
        digest.update(f"{int(node)}:".encode("ascii"))
        for area_id in sorted(_node_area_ids(G, node)):
            digest.update(f"{int(area_id)},".encode("ascii"))
        digest.update(b";")
    digest.update(b"|edges|")
    for u, v in sorted(tuple(sorted((int(u), int(v)))) for u, v in G.edges()):
        digest.update(f"{u},{v};".encode("ascii"))
    return digest.hexdigest()[:20]


def _node_area_ids(G: nx.Graph, node: int) -> list[int]:
    attrs = G.nodes[node]
    if "area_id" in attrs:
        return [int(attrs["area_id"])]
    return [int(area_id) for area_id in attrs.get("block_ids", [])]


@contextmanager
def _exclusive_lock(path: Path):
    with path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def split_legacy_cache(cache_root: Path) -> list[Path]:
    """Write every variant of a v4 cache (one file per level) to the v5 layout.

    Returns the files written; variants already in v5 are left alone. Run it
    once per cache root (``python -m optimization.data.closer_neighbors
    --split-legacy <cache root>``) so runs do not rebuild them.
    """
    legacy = Path(cache_root) / "closer_neighbors" / "v4"
    target = (
        Path(cache_root)
        / "closer_neighbors"
        / f"v{CLOSER_NEIGHBOR_CACHE_SCHEMA_VERSION}"
    )
    written = []
    for path in sorted(legacy.glob("closer_neighbors_*.pickle")):
        with path.open("rb") as file:
            payload = pickle.load(file)
        level = payload["level"]
        for fingerprint, variant in payload.get("variants", {}).items():
            out = target / f"closer_neighbors_{level}" / f"{fingerprint}.pickle"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            single = {
                **{k: v for k, v in payload.items() if k != "variants"},
                "schema_version": CLOSER_NEIGHBOR_CACHE_SCHEMA_VERSION,
                "variants": {fingerprint: variant},
            }
            tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
            with tmp.open("wb") as file:
                pickle.dump(single, file, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, out)
            written.append(out)
        del payload
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Closer-neighbor cache tools.")
    parser.add_argument(
        "--split-legacy", type=Path, required=True, metavar="CACHE_ROOT"
    )
    args = parser.parse_args()
    files = split_legacy_cache(args.split_legacy)
    print(f"Wrote {len(files)} variant files.")
