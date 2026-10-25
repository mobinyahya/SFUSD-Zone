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
from optimization.data import closer_neighbors, graph_builder, loaders, node_areas
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
                    self.ingest.unit,
                    config.graph_weight_edges,
                    config.partition_area_weight,
                ),
            },
            schema_version=graph_builder.GRAPH_CACHE_SCHEMA_VERSION,
            roles=graph_roles,
        )
        self.graphs_dir = str(self._graph_namespace.version_dir)
        self.graph_cache_dir = str(self._graph_namespace.path)
        self._graphs: dict[str, nx.Graph] = {}
        self._centroids: dict[tuple[str, tuple[int, ...]], list[int]] = {}
        self._centroid_schools: dict[str, tuple[int, ...]] = {}
        self._area_hints: dict[tuple[str, str, tuple[int, ...]], dict[int, int]] = {}
        self._area_assignments: dict[str, dict[int, int]] = {}
        self._closer_neighbor_graphs: dict[str, nx.Graph] = {}
        self._land_area_graphs: dict[str, nx.Graph] = {}
        self._closer_neighbor_store = closer_neighbors.CloserNeighborArtifactStore(
            self.data,
            geometry_loader=lambda unit: loaders.load_census_shapefile(unit, self.data),
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
            return graph_builder.build_base_graph(
                self.ingest, weight_edges=self.config.graph_weight_edges
            )
        targets = LEVEL_NODE_TARGETS.get(level.unit, {})
        if level.depth not in targets:
            raise ValueError(f"No predefined graph size for level {level.name}.")
        parent = self.graph_for(level.parent())

        # KaHIP is only reproducible against its own native build, so the first
        # machine to coarsen a level publishes the membership it found and every
        # later reader replays it instead of re-partitioning.
        name = self._partition_payload(level)
        artifact = self._graph_namespace.load_json(name)
        partition = (
            graph_builder.partition_from_portable(parent, artifact)
            if artifact is not None
            else None
        )
        area_weight = self.config.partition_area_weight
        if area_weight and partition is None:
            node_areas.attach_land_areas(parent, level.unit, self.data)
        G = graph_builder.aggregate_level(
            parent,
            targets[level.depth],
            self.ingest.program_population,
            partition=partition,
            area_weight=area_weight,
        )
        # Publishing is a determinism aid, not part of producing the graph, so a
        # coarsening that carries no membership is used as-is rather than failing.
        membership = G.graph.get("partition")
        if artifact is None and membership is not None:
            self._graph_namespace.save_json(
                name, graph_builder.partition_to_portable(parent, membership)
            )
        return G

    def _uses_school_distance(self, config=None) -> bool:
        config = config or self.config
        return (
            config.compactness_objective == "school_distance"
            or config.recom_objective == "school_distance"
        )

    @staticmethod
    def _partition_payload(level: LevelSpec) -> str:
        return f"partition_{level.name}.json"

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
        G = self.graph_for(level)
        self._attach_closer_neighbors(level, G)
        return G.graph[closer_neighbors.CLOSER_NEIGHBORS_GRAPH_KEY]

    def _attach_closer_neighbors(self, level: LevelSpec, G: nx.Graph) -> None:
        if self._closer_neighbor_graphs.get(level.name) is not G:
            self._closer_neighbor_store.attach_to_graph(level, G)
            self._closer_neighbor_graphs[level.name] = G

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

    def area_hint_for(self, level, G: nx.Graph, centroids: list[int], *, path=None):
        """``config.hint_zoning`` projected onto ``level`` as a warm start."""
        from optimization.data import conversion

        level = LevelSpec.parse(level)
        path = path if path is not None else self.config.hint_zoning
        if path is None:
            raise ValueError("An area hint requires a zoning file path.")
        path = str(path)
        key = (path, level.name, tuple(centroids))
        if key not in self._area_hints:
            if path not in self._area_assignments:
                self._area_assignments[path] = conversion.load_area_assignment(path)
            self._area_hints[key] = conversion.hint_from_area_assignment(
                self._area_assignments[path],
                conversion.area_file_level(path, level),
                G,
                level,
                centroids,
                converter=conversion.LevelConverter(data=self.data),
            )
        return dict(self._area_hints[key])

    def centroid_school_ids(self, centroids_type=None) -> list[int]:
        """Read a centroid definition once across all levels and pipeline steps."""
        centroids_type = centroids_type or self.config.centroids_type
        if centroids_type not in self._centroid_schools:
            if centroids_type == loaders.ALL_SCHOOLS_CENTROIDS:
                # Every school the zoning problem holds -- the attendance-area
                # schools, plus citywide ones under include_citywide_zoning --
                # in ascending id order, so zone k is the same school on
                # every level and every machine.
                school_ids = sorted(
                    int(sid) for sid in loaders.load_schools(self.ingest)["school_id"]
                )
            else:
                school_ids = loaders.load_centroid_schools(centroids_type, self.data)
            self._centroid_schools[centroids_type] = tuple(
                int(sid) for sid in school_ids
            )
        return list(self._centroid_schools[centroids_type])

    def centroids_for(self, level, school_ids=None) -> list[int]:
        level = LevelSpec.parse(level)
        if school_ids is None:
            school_ids = self.centroid_school_ids()
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
        config: "OptimizationConfig | None" = None,
    ) -> ZoneProblem:
        """Build a problem using this run's data and an optional step config.

        Steps can change constraints, objectives, and search settings while
        sharing the scenario, graphs, and geometry already loaded by the run.
        Data selectors and graph construction settings belong to the dataset
        and cannot change halfway through a pipeline.
        """
        config = config or self.config
        level = LevelSpec.parse(level)
        if config.data_scenario is not self.data and config.data_scenario != self.data:
            raise ValueError("Pipeline steps must use the dataset's data scenario.")
        if (
            level.unit != self.ingest.unit
            or config.unit != self.ingest.unit
            or (config.weight_edges and not self.config.graph_weight_edges)
        ):
            raise ValueError(
                "Pipeline steps must preserve the dataset's geography unit; weighted objectives require boundary lengths in the dataset."
            )
        constraint_multiplier = float(constraint_multiplier)
        if centroid_school_ids is None:
            centroid_school_ids = self.centroid_school_ids(config.centroids_type)
        centroid_school_ids = [int(school_id) for school_id in centroid_school_ids]
        G = self.graph_for(level)
        self._attach_closer_neighbors(level, G)
        centroids = self.centroids_for(level, centroid_school_ids)
        if hint is None and config.hint_zoning is not None:
            hint = self.area_hint_for(level, G, centroids, path=config.hint_zoning)
        if (
            self._uses_school_distance(config)
            and config.school_distance_area_weight
            and self._land_area_graphs.get(level.name) is not G
        ):
            node_areas.attach_land_areas(G, level.unit, self.data)
            self._land_area_graphs[level.name] = G
        return ZoneProblem(
            G=G,
            level=level,
            centroids=centroids,
            centroid_school_ids=centroid_school_ids,
            program_population=config.program_population,
            frl_dev=config.frl_dev * constraint_multiplier,
            racial_dev=config.racial_dev * constraint_multiplier,
            overage=config.overage * constraint_multiplier,
            shortage=config.shortage * constraint_multiplier,
            max_distance=config.max_distance,
            boundary_prop=config.boundary_prop,
            weight_edges=config.weight_edges,
            compactness_objective=config.compactness_objective,
            school_distance_area_weight=config.school_distance_area_weight,
            school_distance_student_weight=config.school_distance_student_weight,
            fixed=fixed,
            candidates=candidates,
            hint=hint,
            choice_objective=choice_objective,
            optimization_config=config,
        )
