"""Synthetic, data-free fixtures for testing the optimization.

These build node-attributed graphs and ZoneProblems by hand so the solver,
contiguity, conversion and strategy logic can be exercised without any SFUSD
source data.
"""

from __future__ import annotations

import math

import networkx as nx
from shapely.geometry import box

from Zone_Generation.Config.Constants import AREA_ETHNICITIES
from Zone_Generation.optimization.levels import LevelSpec
from Zone_Generation.optimization.problem import ZoneProblem


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

    def centroids_for(self, level):
        return self._problem.centroids

    def problem_for(
        self, level, fixed=None, candidates=None, hint=None, choice_objective=None
    ):
        return ZoneProblem(
            G=self._problem.G,
            level=self._problem.level,
            centroids=self._problem.centroids,
            frl_dev=self._problem.frl_dev,
            racial_dev=self._problem.racial_dev,
            overage=self._problem.overage,
            shortage=self._problem.shortage,
            max_distance=self._problem.max_distance,
            fixed=fixed,
            candidates=candidates,
            hint=hint,
            choice_objective=choice_objective,
        )
