import math

import networkx as nx

from Zone_Generation.Config.Constants import AREA_ETHNICITIES
from Zone_Generation.optimization.config import OptimizationConfig
from Zone_Generation.optimization.data import initial_solutions
from Zone_Generation.optimization.levels import LevelSpec
from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution


def test_math_prog_initial_hint_generates_block0_cache(tmp_path, monkeypatch):
    config = OptimizationConfig(
        centroids_type="2-zone-test",
        levels=["Block_0"],
        graphs_dir=str(tmp_path / "graphs"),
        frl_dev=1.0,
        racial_dev=1.0,
        overage=5.0,
        shortage=0.0,
        workers=1,
    )

    monkeypatch.setattr(initial_solutions, "Dataset", _FakeDataset)
    solver_factory = _CountingSolverFactory()
    monkeypatch.setattr(initial_solutions, "get_solver", solver_factory)
    monkeypatch.setattr(
        initial_solutions.LevelConverter,
        "b2bg",
        lambda self: {10: 100, 11: 100, 12: 101, 13: 101},
    )

    block_dataset = _FakeDataset(config)
    problem = block_dataset.problem_for("Block_0")

    hint = initial_solutions.math_prog_initial_hint(
        block_dataset,
        problem,
        {
            "seed": 1,
            "workers": 1,
            "recom_initial_time_limit": 1,
            "recom_initial_constraint_multiplier": 10,
        },
    )

    assert hint == {0: 0, 1: 0, 2: 1, 3: 1}
    assert solver_factory.calls == 1
    assert problem._math_prog_initial_cache["cache_hit"] is False
    assert problem._math_prog_initial_cache["available"] is True
    cache_path = (
        tmp_path
        / "graphs"
        / "Block_cache"
        / "recom_initial_solutions"
        / "2-zone-test"
        / "zone_dict_Block_0.json"
    )
    assert cache_path.exists()

    second_problem = block_dataset.problem_for("Block_0")
    second_hint = initial_solutions.math_prog_initial_hint(
        block_dataset,
        second_problem,
        {
            "seed": 1,
            "workers": 1,
            "recom_initial_time_limit": 1,
            "recom_initial_constraint_multiplier": 10,
        },
    )

    assert second_hint == hint
    assert solver_factory.calls == 1
    assert second_problem._math_prog_initial_cache["cache_hit"] is True


class _FakeCpBoolSolver:
    def solve(self, problem):
        return ZoneSolution(
            problem=problem,
            assignment={0: 0, 1: 1},
            status="FEASIBLE",
            objective=0.0,
            wall_time=0.0,
        )


class _CountingSolverFactory:
    def __init__(self):
        self.calls = 0

    def __call__(self, name, **options):
        self.calls += 1
        assert name == "cp_bool"
        return _FakeCpBoolSolver()


class _FakeDataset:
    def __init__(self, config):
        self.config = config
        self.graph_cache_dir = str(
            (config.graphs_dir and f"{config.graphs_dir}/{config.unit}_cache")
        )
        self._graphs = {
            "Block_0": _block_graph(),
            "BlockGroup_1": _blockgroup_graph(),
        }

    def graph_for(self, level):
        return self._graphs[LevelSpec.parse(level).name]

    def centroids_for(self, level):
        level = LevelSpec.parse(level)
        if level.unit == "Block":
            return [0, 3]
        return [0, 1]

    def problem_for(
        self,
        level,
        fixed=None,
        candidates=None,
        hint=None,
        choice_objective=None,
        constraint_multiplier=1.0,
    ):
        level = LevelSpec.parse(level)
        return ZoneProblem(
            G=self.graph_for(level),
            level=level,
            centroids=self.centroids_for(level),
            frl_dev=1.0,
            racial_dev=1.0,
            overage=5.0,
            shortage=0.0,
            max_distance=float("inf"),
            fixed=fixed,
            candidates=candidates,
            hint=hint,
            choice_objective=choice_objective,
        )


def _block_graph():
    G = nx.path_graph(4)
    area_ids = [10, 11, 12, 13]
    for node, area_id in enumerate(area_ids):
        _set_attrs(
            G,
            node,
            area_id=area_id,
            schools=[100 + node] if node in (0, 3) else [],
        )
    _set_distances(G)
    return G


def _blockgroup_graph():
    G = nx.Graph()
    G.add_edge(0, 1)
    _set_attrs(G, 0, block_ids=[100], schools=[100])
    _set_attrs(G, 1, block_ids=[101], schools=[103])
    _set_distances(G)
    return G


def _set_attrs(G, node, *, area_id=None, block_ids=None, schools=None):
    schools = schools or []
    attrs = {
        "ge_students": 1.0,
        "ge_capacity": 1.0,
        "all_prog_students": 1.0,
        "all_prog_capacity": 1.0,
        "num_schools": len(schools),
        "FRL": 0.5,
        "school_ids": schools,
        "lat": 0.0,
        "lon": float(node),
        **{ethnicity: 0.2 for ethnicity in AREA_ETHNICITIES},
    }
    if area_id is not None:
        attrs["area_id"] = area_id
    if block_ids is not None:
        attrs["block_ids"] = block_ids
    G.nodes[node].update(attrs)


def _set_distances(G):
    G.graph["distance_dict"] = {
        i: {j: math.fabs(i - j) for j in G.nodes()} for i in G.nodes()
    }
    G.graph["F"] = 0.5
    G.graph["R"] = {ethnicity: 0.2 for ethnicity in AREA_ETHNICITIES}
    G.graph["school_data"] = {}
