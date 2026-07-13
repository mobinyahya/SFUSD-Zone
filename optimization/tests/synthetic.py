"""Synthetic, data-free fixtures for testing the optimization.

These build node-attributed graphs and ZoneProblems by hand so the solver,
contiguity, conversion and strategy logic can be exercised without any SFUSD
source data.
"""

from __future__ import annotations

import math

import networkx as nx
from shapely.geometry import box

from Config.Constants import AREA_ETHNICITIES
from optimization.levels import LevelSpec
from optimization.problem import ZoneProblem


def make_grid_graph(rows: int = 3, cols: int = 3) -> nx.Graph:
    """A ``rows x cols`` grid with uniform, balanced demographics.

    Node ``r*cols + c`` sits at coordinate ``(r, c)``; distances are plain grid
    euclidean so the contiguity support relation is well defined. Corner nodes
    (top-left, bottom-right) carry a school so a 2-zone school-count balance is
    satisfiable.
    """
    G = nx.Graph()
    n = rows * cols
    coords = {}
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            coords[idx] = (r, c)

    for idx, (r, c) in coords.items():
        eth = {e: 0.2 for e in AREA_ETHNICITIES}  # 5 ethnicities, sum to 1.0
        schools = []
        if idx == 0:
            schools = [100]
        elif idx == n - 1:
            schools = [200]
        G.add_node(
            idx,
            area_id=1000 + idx,
            ge_students=1.0,
            ge_capacity=1.0,
            all_prog_students=1.0,
            all_prog_capacity=1.0,
            num_schools=len(schools),
            FRL=0.5,
            school_ids=schools,
            lat=float(r),
            lon=float(c),
            geometry=box(c, r, c + 1, r + 1),
            **eth,
        )

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if c + 1 < cols:
                G.add_edge(idx, idx + 1)
            if r + 1 < rows:
                G.add_edge(idx, idx + cols)

    distance_dict = {}
    for i, (ri, ci) in coords.items():
        distance_dict[i] = {}
        for j, (rj, cj) in coords.items():
            distance_dict[i][j] = math.hypot(ri - rj, ci - cj)
    G.graph["distance_dict"] = distance_dict
    G.graph["F"] = 0.5
    G.graph["R"] = {e: 0.2 for e in AREA_ETHNICITIES}
    G.graph["school_data"] = {100: {}, 200: {}}
    return G


def make_grid_problem(rows: int = 3, cols: int = 3, **overrides) -> ZoneProblem:
    G = make_grid_graph(rows, cols)
    params = dict(
        frl_dev=1.0,
        racial_dev=1.0,
        overage=5.0,
        shortage=0.0,
        max_distance=float("inf"),
    )
    params.update(overrides)
    return ZoneProblem(
        G=G,
        level=LevelSpec("BlockGroup", 0),
        centroids=[0, rows * cols - 1],
        **params,
    )


def make_solver_contract_problem(**overrides) -> ZoneProblem:
    """Four-node path with one easy, uniquely determined two-zone split."""
    G = nx.path_graph(4)
    school_counts = [2, 1, 1, 2]
    school_id = 100
    school_data = {}
    for node in G.nodes:
        schools = list(range(school_id, school_id + school_counts[node]))
        school_id += school_counts[node]
        school_data.update({sid: {} for sid in schools})
        G.nodes[node].update(
            {
                "area_id": 1000 + node,
                "ge_students": 1.0,
                "ge_capacity": 1.0,
                "all_prog_students": 1.0,
                "all_prog_capacity": 1.0,
                "num_schools": school_counts[node],
                "FRL": 0.5,
                "school_ids": schools,
                "lat": 0.0,
                "lon": float(node),
                **{ethnicity: 0.2 for ethnicity in AREA_ETHNICITIES},
            }
        )

    G.graph["distance_dict"] = {
        source: {target: abs(source - target) for target in G.nodes}
        for source in G.nodes
    }
    G.graph["F"] = 0.5
    G.graph["R"] = {ethnicity: 0.2 for ethnicity in AREA_ETHNICITIES}
    G.graph["school_data"] = school_data

    params = {
        "frl_dev": 0.0,
        "racial_dev": 0.0,
        "overage": 0.0,
        "shortage": 0.0,
        "max_distance": 1.0,
        "hint": {0: 0, 1: 0, 2: 1, 3: 1},
    }
    params.update(overrides)
    return ZoneProblem(
        G=G,
        level=LevelSpec("BlockGroup", 0),
        centroids=[0, 3],
        **params,
    )


def make_single_zone_problem(**overrides) -> ZoneProblem:
    """Seven-node path whose optimal one-school zone has three nodes."""
    G = nx.path_graph(7)
    school_nodes = {0: 200, 3: 100, 6: 300}
    for node in G.nodes:
        school_id = school_nodes.get(node)
        G.nodes[node].update(
            {
                "area_id": 2000 + node,
                "ge_students": 1.0,
                "ge_capacity": 3.0 if node == 3 else 0.0,
                "all_prog_students": 1.0,
                "all_prog_capacity": 3.0 if node == 3 else 0.0,
                "num_schools": int(school_id is not None),
                "FRL": 0.5,
                "school_ids": [school_id] if school_id is not None else [],
                "lat": 0.0,
                "lon": float(node),
                "geometry": box(node, 0, node + 1, 1),
                **{ethnicity: 0.2 for ethnicity in AREA_ETHNICITIES},
            }
        )

    G.graph["distance_dict"] = {
        source: {target: abs(source - target) for target in G.nodes}
        for source in G.nodes
    }
    G.graph["F"] = 0.5
    G.graph["R"] = {ethnicity: 0.2 for ethnicity in AREA_ETHNICITIES}
    G.graph["school_data"] = {
        school_id: {"program_types": ["GE"], "ge_capacity": 1.0}
        for school_id in school_nodes.values()
    }

    params = {
        "frl_dev": 0.0,
        "racial_dev": 0.0,
        "overage": 0.0,
        "shortage": 0.0,
        "max_distance": float("inf"),
    }
    params.update(overrides)
    return ZoneProblem(
        G=G,
        level=LevelSpec("Block", 0),
        centroids=[3],
        **params,
    )


def make_path_graphs():
    """A 4-node base path and its 2-node aggregation, for conversion tests."""
    base = nx.Graph()
    for idx in range(4):
        base.add_node(idx, area_id=10 + idx, ge_students=1.0, lat=0.0, lon=float(idx))
    base.add_edges_from([(0, 1), (1, 2), (2, 3)])
    base.graph["distance_dict"] = {
        i: {j: abs(i - j) for j in range(4)} for i in range(4)
    }
    base.graph["F"] = 0.0
    base.graph["R"] = {e: 0.0 for e in AREA_ETHNICITIES}
    base.graph["school_data"] = {}

    coarse = nx.Graph()
    coarse.add_node(0, block_ids=[10, 11])
    coarse.add_node(1, block_ids=[12, 13])
    coarse.add_edge(0, 1)
    coarse.graph["partition"] = {0: 0, 1: 0, 2: 1, 3: 1}
    return base, coarse


class FakeDataset:
    """Minimal Dataset stand-in returning a single synthetic problem/graph."""

    def __init__(self, problem: ZoneProblem):
        self._problem = problem

    def graph_for(self, level):
        return self._problem.G

    def school_ids_for(self, level):
        return sorted(int(sid) for sid in self._problem.G.graph["school_data"])

    def centroids_for(self, level, school_ids=None):
        if school_ids is None:
            return self._problem.centroids
        school_to_node = {
            int(sid): node
            for node, attrs in self._problem.G.nodes(data=True)
            for sid in attrs.get("school_ids", [])
        }
        return [school_to_node[int(sid)] for sid in school_ids]

    def problem_for(
        self,
        level,
        fixed=None,
        candidates=None,
        hint=None,
        choice_objective=None,
        constraint_multiplier=1.0,
        centroid_school_ids=None,
    ):
        constraint_multiplier = float(constraint_multiplier)
        return ZoneProblem(
            G=self._problem.G,
            level=LevelSpec.parse(level),
            centroids=self.centroids_for(level, centroid_school_ids),
            frl_dev=self._problem.frl_dev * constraint_multiplier,
            racial_dev=self._problem.racial_dev * constraint_multiplier,
            overage=self._problem.overage * constraint_multiplier,
            shortage=self._problem.shortage * constraint_multiplier,
            max_distance=self._problem.max_distance,
            fixed=fixed,
            candidates=candidates,
            hint=hint,
            choice_objective=choice_objective,
        )
