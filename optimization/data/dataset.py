"""The :class:`Dataset` -- the data layer's public face.

A ``Dataset`` lazily provides the graph for any requested level through one
validated, content-addressed cache namespace, resolves centroids to node
indices, and mints solver-agnostic :class:`ZoneProblem` instances. Strategies
operate purely against a ``Dataset``; they never read raw files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import networkx as nx

from loaders import CacheStore
from optimization.data import closer_neighbors, graph_builder, loaders
from optimization.data.loaders import IngestConfig
from optimization.levels import LEVEL_NODE_TARGETS, LevelSpec
from optimization.problem import ZoneProblem

if TYPE_CHECKING:
    from optimization.config import OptimizationConfig


class Dataset:
    """Lazy, cached access to graphs, centroids and problems for one config."""

    def __init__(self, config: "OptimizationConfig"):
        self.config = config
        self.data = config.data_scenario
        self.ingest = IngestConfig(unit=config.unit, data=self.data)
        graph_roles = [
            *loaders.student_source_roles(self.ingest),
            loaders.SCHOOL_ROLE,
            *loaders.capacity_source_roles(self.ingest),
            *loaders.census_geometry_roles(self.data, self.ingest.unit),
            loaders.ADJACENCY_ROLE,
            loaders.MANUAL_EDGE_ROLE,
        ]
        self._graph_namespace = CacheStore(self.data).namespace(
            "graphs",
            {
                "unit": self.ingest.unit,
                "optimization_filters": self.ingest.filters,
                "partition_policy": graph_builder.partition_cache_policy(
                    self.ingest.unit
                ),
            },
            schema_version=graph_builder.GRAPH_CACHE_SCHEMA_VERSION,
            roles=graph_roles,
        )
        self.graphs_dir = str(self._graph_namespace.version_dir)
        self.graph_cache_dir = str(self._graph_namespace.path)
        self._graphs: dict[str, nx.Graph] = {}
        self._centroids: dict[tuple[str, tuple[int, ...]], list[int]] = {}
        self._closer_neighbor_store = closer_neighbors.CloserNeighborArtifactStore(
            self.data,
            geometry_loader=lambda unit: loaders.load_census_shapefile(
                unit, self.data
            ),
            school_loader=lambda: loaders.load_school_coordinates(self.data),
        )

    # ------------------------------------------------------------------ #
    # graphs
    # ------------------------------------------------------------------ #
    def graph_for(self, level) -> nx.Graph:
        level = LevelSpec.parse(level)
        key = level.name
        if key in self._graphs:
            return self._graphs[key]

        G = self._graph_namespace.load_pickle(level.filename)
        if not isinstance(G, nx.Graph):
            G = self._generate(level)
            self._save(level, G)

        self._graphs[key] = G
        return G

    def _generate(self, level: LevelSpec) -> nx.Graph:
        if level.is_base:
            return graph_builder.build_base_graph(self.ingest)
        targets = LEVEL_NODE_TARGETS.get(level.unit, {})
        if level.depth not in targets:
            raise ValueError(f"No predefined graph size for level {level.name}.")
        parent = self.graph_for(level.finer())
        return graph_builder.aggregate_level(
            parent,
            targets[level.depth],
            self.ingest.program_population,
        )

    def _save(self, level: LevelSpec, G: nx.Graph) -> None:
        self._graph_namespace.save_pickle(level.filename, G)

    def _graph_path(self, level: LevelSpec) -> str:
        return str(self._graph_namespace.payload_path(level.filename))

    def _graph_cache_namespace(self) -> str:
        """Return the content-addressed graph key for introspection."""
        return self._graph_namespace.key

    def closer_neighbors_for(self, level) -> dict[int, dict[int, frozenset[int]]]:
        """Load and attach the shared geometry relation for ``level``."""
        level = LevelSpec.parse(level)
        data = self._closer_neighbor_store.attach_to_graph(
            level, self.graph_for(level)
        )
        return data.closer_neighbors

    # ------------------------------------------------------------------ #
    # centroids
    # ------------------------------------------------------------------ #
    def school_ids_for(self, level) -> list[int]:
        """Eligible school IDs represented by the graph, in stable order."""
        G = self.graph_for(level)
        school_data = G.graph.get("school_data", {})
        if school_data:
            return sorted(int(sid) for sid in school_data if sid is not None)
        return sorted(
            {
                int(sid)
                for _, attrs in G.nodes(data=True)
                for sid in attrs.get("school_ids", [])
            }
        )

    def centroids_for(self, level, school_ids=None) -> list[int]:
        level = LevelSpec.parse(level)
        if school_ids is None:
            school_ids = loaders.load_centroid_schools(
                self.config.centroids_type, self.data
            )
        school_ids = tuple(int(sid) for sid in school_ids)
        key = (level.name, school_ids)
        if key in self._centroids:
            return self._centroids[key]

        G = self.graph_for(level)
        school_to_node = {}
        for node, attrs in G.nodes(data=True):
            for sid in attrs.get("school_ids", []):
                school_to_node.setdefault(int(sid), node)
        raw_school_to_node = None

        centroids = []
        for sid in school_ids:
            if sid not in school_to_node:
                if raw_school_to_node is None:
                    raw_school_to_node = self._raw_centroid_node_lookup(G)
                if sid not in raw_school_to_node:
                    raise ValueError(
                        f"Centroid school {sid} not found in any node or raw "
                        f"school location at {level.name}."
                    )
                centroids.append(raw_school_to_node[sid])
            else:
                centroids.append(school_to_node[sid])
        self._centroids[key] = centroids
        return centroids

    def _raw_centroid_node_lookup(self, G: nx.Graph) -> dict[int, int]:
        area_to_node = {}
        for node, attrs in G.nodes(data=True):
            if "area_id" in attrs:
                area_to_node.setdefault(int(attrs["area_id"]), node)
            for area_id in attrs.get("block_ids", []):
                area_to_node.setdefault(int(area_id), node)

        school_to_node = {}
        locations = loaders.load_school_locations(self.ingest)
        for row in locations.itertuples(index=False):
            sid = int(row.school_id)
            area_id = int(getattr(row, self.ingest.unit))
            node = area_to_node.get(area_id)
            if node is not None:
                school_to_node.setdefault(sid, node)
        return school_to_node

    # ------------------------------------------------------------------ #
    # problems
    # ------------------------------------------------------------------ #
    def problem_for(
        self,
        level,
        fixed: Optional[dict[int, int]] = None,
        candidates: Optional[dict[int, set[int]]] = None,
        hint: Optional[dict[int, int]] = None,
        choice_objective=None,
        constraint_multiplier: float = 1.0,
        centroid_school_ids=None,
    ) -> ZoneProblem:
        level = LevelSpec.parse(level)
        constraint_multiplier = float(constraint_multiplier)
        if centroid_school_ids is None:
            centroid_school_ids = loaders.load_centroid_schools(
                self.config.centroids_type, self.data
            )
        centroid_school_ids = [int(school_id) for school_id in centroid_school_ids]
        G = self.graph_for(level)
        self._closer_neighbor_store.attach_to_graph(level, G)
        return ZoneProblem(
            G=G,
            level=level,
            centroids=self.centroids_for(level, centroid_school_ids),
            centroid_school_ids=centroid_school_ids,
            program_population=self.config.program_population,
            frl_dev=self.config.frl_dev * constraint_multiplier,
            racial_dev=self.config.racial_dev * constraint_multiplier,
            overage=self.config.overage * constraint_multiplier,
            shortage=self.config.shortage * constraint_multiplier,
            max_distance=self.config.max_distance,
            fixed=fixed,
            candidates=candidates,
            hint=hint,
            choice_objective=choice_objective,
            optimization_config=self.config,
        )
