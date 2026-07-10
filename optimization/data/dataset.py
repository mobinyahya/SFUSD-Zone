"""The :class:`Dataset` -- the data layer's public face.

A ``Dataset`` lazily provides the graph for any requested level (loading a
cached pickle or generating and caching it), resolves centroids to node
indices, and mints solver-agnostic :class:`ZoneProblem` instances. Strategies
operate purely against a ``Dataset``; they never read raw files.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import TYPE_CHECKING, Optional

import networkx as nx

from optimization.data import graph_builder, loaders
from optimization.data.loaders import IngestConfig
from optimization.levels import LEVEL_NODE_TARGETS, LevelSpec
from optimization.problem import ZoneProblem

if TYPE_CHECKING:
    from optimization.config import OptimizationConfig


class Dataset:
    """Lazy, cached access to graphs, centroids and problems for one config."""

    def __init__(self, config: "OptimizationConfig"):
        self.config = config
        self.ingest = IngestConfig(
            unit=config.unit,
            years=list(config.years),
            population_type=config.population_type,
            drop_optout=config.drop_optout,
            capacity_scenario=config.capacity_scenario,
            new_schools=config.new_schools,
            include_k8=config.include_k8,
        )
        self.graphs_dir = config.graphs_dir
        self.graph_cache_dir = os.path.join(
            self.graphs_dir,
            self._graph_cache_namespace(),
        )
        self._graphs: dict[str, nx.Graph] = {}
        self._centroids: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ #
    # graphs
    # ------------------------------------------------------------------ #
    def graph_for(self, level) -> nx.Graph:
        level = LevelSpec.parse(level)
        key = level.name
        if key in self._graphs:
            return self._graphs[key]

        path = self._graph_path(level)
        if os.path.exists(path):
            with open(path, "rb") as f:
                G = pickle.load(f)
        else:
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
            self.ingest.population_type,
        )

    def _save(self, level: LevelSpec, G: nx.Graph) -> None:
        os.makedirs(self.graph_cache_dir, exist_ok=True)
        path = self._graph_path(level)
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(G, f)
        os.replace(tmp_path, path)

    def _graph_path(self, level: LevelSpec) -> str:
        return os.path.join(self.graph_cache_dir, level.filename)

    def _graph_cache_namespace(self) -> str:
        payload = {
            "schema_version": graph_builder.GRAPH_CACHE_SCHEMA_VERSION,
            "unit": self.ingest.unit,
            "years": list(self.ingest.years),
            "population_type": self.ingest.population_type,
            "drop_optout": bool(self.ingest.drop_optout),
            "capacity_scenario": self.ingest.capacity_scenario,
            "new_schools": bool(self.ingest.new_schools),
            "include_k8": bool(self.ingest.include_k8),
            "partition_policy": graph_builder.partition_cache_policy(self.ingest.unit),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"{self.ingest.unit}_{digest}"

    # ------------------------------------------------------------------ #
    # centroids
    # ------------------------------------------------------------------ #
    def centroids_for(self, level) -> list[int]:
        level = LevelSpec.parse(level)
        key = level.name
        if key in self._centroids:
            return self._centroids[key]

        G = self.graph_for(level)
        school_to_node = {}
        for node, attrs in G.nodes(data=True):
            for sid in attrs.get("school_ids", []):
                school_to_node.setdefault(int(sid), node)
        raw_school_to_node = None

        centroids = []
        for sid in loaders.load_centroid_schools(self.config.centroids_type):
            if sid not in school_to_node:
                if raw_school_to_node is None:
                    raw_school_to_node = self._raw_centroid_node_lookup(G)
                if sid not in raw_school_to_node:
                    raise ValueError(
                        f"Centroid school {sid} not found in any node or raw "
                        f"school location at {key}."
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
    ) -> ZoneProblem:
        level = LevelSpec.parse(level)
        constraint_multiplier = float(constraint_multiplier)
        return ZoneProblem(
            G=self.graph_for(level),
            level=level,
            centroids=self.centroids_for(level),
            frl_dev=self.config.frl_dev * constraint_multiplier,
            racial_dev=self.config.racial_dev * constraint_multiplier,
            overage=self.config.overage * constraint_multiplier,
            shortage=self.config.shortage * constraint_multiplier,
            max_distance=self.config.max_distance,
            fixed=fixed,
            candidates=candidates,
            hint=hint,
            choice_objective=choice_objective,
        )
