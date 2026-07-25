import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import choice_reserves_pareto_frontier as pareto  # noqa: E402


def test_reshape_choice_metrics_excludes_non_feasible_solutions():
    feasible_path = "/runs/solution_12345678"
    infeasible_path = "/runs/solution_abcdef12"
    summary = pd.DataFrame(
        {
            "path": [feasible_path, infeasible_path],
            "status": ["FEASIBLE", "INFEASIBLE"],
            "num_zones": [6, 8],
        }
    )
    results = pd.DataFrame(
        {
            "metric": list(pareto.SOFT_RESERVES_METRICS),
            feasible_path: [0.1, 1.2, 0.14],
            infeasible_path: [0.2, 1.3, 0.12],
        }
    )

    points = pareto.reshape_choice_metrics(summary, results)

    assert points["path"].tolist() == [feasible_path]
    assert points["run_id"].tolist() == ["12345678"]
    assert points["frl_max_dev"].tolist() == [0.14]


def test_extract_status_quo_points_uses_configured_policies():
    results = pd.DataFrame(
        {
            "metric": list(pareto.PARETO_METRICS),
            "status_quo": [0.21, 1.38],
        }
    )

    points = pareto.extract_status_quo_points(results)

    assert points.to_dict("records") == [
        {
            "policy": "status_quo",
            "label": "Status Quo",
            "frl_dissimilarity": 0.21,
            "avg_student_distance": 1.38,
        },
    ]


def test_extract_special_zone_points_uses_all_three_plans():
    results = pd.DataFrame(
        {
            "metric": list(pareto.PARETO_METRICS),
            "small_zones_1": [0.11, 1.41],
            "small_zones_2": [0.12, 1.32],
            "medium_zones": [0.13, 1.23],
        }
    )

    points = pareto.extract_special_zone_points(results)

    assert points["policy"].tolist() == [
        "small_zones_1",
        "small_zones_2",
        "medium_zones",
    ]
    assert points["label"].tolist() == [
        "Small Zones 1",
        "Small Zones 2",
        "Medium Zones",
    ]


def test_find_latest_status_quo_results_uses_timestamped_filename(tmp_path):
    older = tmp_path / "status_quo_policies_25_eval_assignment_full_20260101Z.csv"
    newer = tmp_path / "status_quo_policies_25_eval_assignment_full_20260201Z.csv"
    older.write_text("metric,status_quo\n", encoding="utf-8")
    newer.write_text("metric,status_quo\n", encoding="utf-8")

    result = pareto.find_latest_status_quo_results(tmp_path)

    assert result == newer


def test_filter_by_frl_max_dev_is_strict():
    points = pd.DataFrame(
        {
            "frl_max_dev": [0.149, 0.15, 0.151],
            "label": ["below", "equal", "above"],
        }
    )

    result = pareto.filter_by_frl_max_dev(points, 0.15)

    assert result["label"].tolist() == ["below"]


def test_export_frontier_solutions_renders_numbered_block0_plot(tmp_path, monkeypatch):
    run_path = tmp_path / "solution_55bd8aa7"
    run_path.mkdir()
    manifest = {
        "task_id": "55bd8aa7e924",
        "config_hash": "55bd8aa7e924889af3e3c7e3e4cebdfb",
        "status": "FEASIBLE",
        "final_stage": "stage_01_Block_0",
    }
    (run_path / "benchmark_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (run_path / "visualization_stage_01_Block_0.png").write_bytes(b"old")
    frontier = pd.DataFrame(
        {
            "frontier_number": [3],
            "path": [str(run_path)],
            "summary_task_id": [pd.NA],
        }
    )
    destination = tmp_path / "pareto"
    solution = object()
    calls = []

    monkeypatch.setattr(pareto, "VisualizationArtifactStore", object)
    monkeypatch.setattr(
        pareto,
        "load_final_block0_solution",
        lambda path, loaded_manifest: (solution, "stage_01_Block_0"),
    )

    def fake_render(loaded_solution, stage, path, artifact_store):
        calls.append((loaded_solution, stage, path, artifact_store))
        path.write_bytes(b"rendered")

    monkeypatch.setattr(pareto, "render_solution_to_path", fake_render)

    exported = pareto.export_frontier_solutions(frontier, destination)

    assert exported["task_id"].tolist() == ["55bd8aa7e924"]
    assert (destination / "3.png").read_bytes() == b"rendered"
    assert calls[0][:3] == (solution, "stage_01_Block_0", destination / "3.png")


def test_load_final_block0_solution_does_not_load_coarser_graphs(tmp_path, monkeypatch):
    run_path = tmp_path / "run"
    stage_path = run_path / "stages/final"
    stage_path.mkdir(parents=True)
    (stage_path / "solution_Block_0.json").write_text(
        json.dumps(
            {
                "status": "OPTIMAL",
                "objective": 10,
                "wall_time": 2,
                "centroids": [7],
            }
        ),
        encoding="utf-8",
    )
    (stage_path / "zone_dict_area_Block_0.json").write_text(
        json.dumps({"100": 0, "200": 0}),
        encoding="utf-8",
    )
    graph = nx.Graph()
    graph.add_nodes_from([(7, {"area_id": 100}), (8, {"area_id": 200})])
    loaded_levels = []

    class FakeDataset:
        def graph_for(self, level):
            loaded_levels.append(level.name)
            return graph

    config_data = []

    class FakeConfig:
        def __init__(self, **kwargs):
            config_data.append(kwargs)

        def make_dataset(self):
            return FakeDataset()

    monkeypatch.setattr(pareto, "OptimizationConfig", FakeConfig)
    manifest = {
        "final_stage": "stage_01_Block_0",
        "stages": [
            {
                "name": "stage_00_Block_1",
                "level": "Block_1",
                "path": "stages/coarse",
            },
            {
                "name": "stage_01_Block_0",
                "level": "Block_0",
                "path": "stages/final",
            },
        ],
        "config": {"graphs_dir": str(tmp_path / "missing-graphs")},
    }

    solution, stage = pareto.load_final_block0_solution(run_path, manifest)

    assert stage == "stage_01_Block_0"
    assert loaded_levels == ["Block_0"]
    assert config_data[0]["graphs_dir"] == ""
    assert solution.assignment == {7: 0, 8: 0}
