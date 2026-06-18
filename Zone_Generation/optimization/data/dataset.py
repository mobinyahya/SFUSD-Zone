"""The :class:`Dataset` -- the data layer's public face.

A ``Dataset`` lazily provides the graph for any requested level (loading a
cached pickle or generating and caching it), resolves centroids to node
indices, and mints solver-agnostic :class:`ZoneProblem` instances. Strategies
operate purely against a ``Dataset``; they never read raw files.
"""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Optional

import networkx as nx

from Zone_Generation.pipeline.data import graph_builder, loaders
from Zone_Generation.pipeline.data.loaders import IngestConfig
from Zone_Generation.pipeline.levels import LevelSpec
from Zone_Generation.pipeline.problem import ZoneProblem

if TYPE_CHECKING:
    from Zone_Generation.pipeline.config import PipelineConfig


class Dataset:
    """Lazy, cached access to graphs, centroids and problems for one config."""

    def __init__(self, config: "PipelineConfig"):
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
        self.level_to_split = dict(config.level_to_split)
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

        path = os.path.join(self.graphs_dir, level.filename)
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
        base = self.graph_for(level.base())
        if level.depth not in self.level_to_split:
            raise ValueError(
                f"No METIS split depth configured for level depth {level.depth}; "
                f"set level_to_split[{level.depth}] in the config."
            )
        return graph_builder.aggregate_level(
            base, self.level_to_split[level.depth]
        )

    def _save(self, level: LevelSpec, G: nx.Graph) -> None:
        os.makedirs(self.graphs_dir, exist_ok=True)
        with open(os.path.join(self.graphs_dir, level.filename), "wb") as f:
            pickle.dump(G, f)

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
                school_to_node.setdefault(sid, node)

        centroids = []
        for sid in loaders.load_centroid_schools(self.config.centroids_type):
            if sid not in school_to_node:
                raise ValueError(
                    f"Centroid school {sid} not found in any node at {key}."
                )
            centroids.append(school_to_node[sid])
        self._centroids[key] = centroids
        return centroids

    # ------------------------------------------------------------------ #
    # problems
    # ------------------------------------------------------------------ #
    def problem_for(
        self,
        level,
        fixed: Optional[dict[int, int]] = None,
        candidates: Optional[dict[int, set[int]]] = None,
        hint: Optional[dict[int, int]] = None,
    ) -> ZoneProblem:
        level = LevelSpec.parse(level)
        return ZoneProblem(
            G=self.graph_for(level),
            level=level,
            centroids=self.centroids_for(level),
            frl_dev=self.config.frl_dev,
            racial_dev=self.config.racial_dev,
            overage=self.config.overage,
            shortage=self.config.shortage,
            max_distance=self.config.max_distance,
            fixed=fixed,
            candidates=candidates,
            hint=hint,
        )
