import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import box

from metrics.spatial import compute_spatial_metrics
from optimization.config import OptimizationConfig
from optimization.data.loaders import CROSSWALK_ROLE
from optimization.levels import LevelSpec
from optimization.problem import ZoneProblem
from optimization.solution import ZoneSolution


def test_spatial_metrics_load_block0_through_solution_config(monkeypatch, tmp_path):
    block0 = nx.Graph()
    block0.add_node(0, area_id=10, geometry=box(0, 0, 1, 1))
    block0.add_node(1, area_id=11, geometry=box(1, 0, 2, 1))
    block0.add_edge(0, 1)

    coarse = nx.Graph()
    coarse.add_node(0, area_id=100)
    crosswalk_path = tmp_path / "crosswalk.csv"
    pd.DataFrame({"Block": [10, 11], "BlockGroup": [100, 100]}).to_csv(
        crosswalk_path, index=False
    )
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        data={
            "scenario": "legacy",
            "overrides": {
                "roots": {"cache": str(tmp_path / "cache")},
                "sources": {CROSSWALK_ROLE: {"path": str(crosswalk_path)}},
            },
        },
    )
    problem = ZoneProblem(
        coarse,
        LevelSpec("BlockGroup", 0),
        [0],
        [100],
        optimization_config=config,
    )
    requested_levels = []

    class StubDataset:
        def graph_for(self, level):
            requested_levels.append(LevelSpec.parse(level).name)
            return block0

    monkeypatch.setattr(
        OptimizationConfig,
        "make_dataset",
        lambda derived: StubDataset(),
    )

    result = compute_spatial_metrics(
        ZoneSolution(problem=problem, assignment={0: 0}, status="FEASIBLE")
    )

    assert requested_levels == ["Block_0"]
    assert result.cut_edges == 0
    assert result.avg_polsby_popper_score > 0


def test_spatial_metrics_do_not_read_legacy_graph_paths(tmp_path):
    legacy_path = tmp_path / "Block_0.pickle"
    legacy_path.write_bytes(b"legacy graph payload")
    coarse = nx.Graph()
    coarse.add_node(0, block_ids=[10])
    solution = ZoneSolution(
        problem=ZoneProblem(coarse, LevelSpec("BlockGroup", 0), [0], [100]),
        assignment={0: 0},
        status="FEASIBLE",
    )

    with pytest.raises(ValueError, match="strict OptimizationConfig"):
        compute_spatial_metrics(
            solution,
            config={"block0_graph_path": str(legacy_path)},
        )


def test_spatial_metrics_include_maximum_zone_shape_scores():
    graph = nx.Graph()
    graph.add_node(0, area_id=10, geometry=box(0, 0, 1, 1))
    graph.add_node(1, area_id=11, geometry=box(2, 0, 6, 1))
    solution = ZoneSolution(
        problem=ZoneProblem(
            graph,
            LevelSpec("Block", 0),
            [0, 1],
            [100, 200],
        ),
        assignment={0: 0, 1: 1},
        status="FEASIBLE",
    )

    result = compute_spatial_metrics(solution)

    assert result.max_reock_score > result.avg_reock_score
    assert result.max_polsby_popper_score > result.avg_polsby_popper_score
    assert result.max_reock_score <= 1
    assert result.max_polsby_popper_score <= 1
