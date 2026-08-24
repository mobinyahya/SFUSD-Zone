import json
import os
from dataclasses import replace
from pathlib import Path

import networkx as nx

import pytest
import yaml

from optimization.config import OptimizationConfig
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import FakeDataset, make_grid_problem
from benchmark.config import (
    BenchmarkTask,
    SimulationSweep,
    optimization_config_hash,
    optimization_config_from_dict,
    optimization_config_to_dict,
)
from benchmark.regenerate import regenerate_metrics
from benchmark.results import aggregate_results
from benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    load_solutions,
    manifest_for,
    result_payload_for,
    save_stage_artifacts,
    stage_names_for,
    write_json,
)
from metrics import MetricsCalculator


def test_sweep_yaml_generates_cartesian_optimization_tasks(tmp_path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        f"""
name: unit-test
mode: run
optimization_defaults:
  centroids_type: '5-zone-AF'
  levels: ['BlockGroup_1', 'BlockGroup_0']
  solver: 'cp_int'
  strategy: 'recursive'
  solve_time_limits: [1, 2]
  gap_limits: [0, 0]
  save_solver_logs: true
  save_solver_progress: true
  workers: 4
  data:
    scenario: legacy
    overrides:
      roots:
        cache: '{tmp_path / "graphs"}'
sweep:
  frl_dev: [0.1, 0.2]
  seed: [1, 2]
execution:
  output_dir: '{tmp_path / "out"}'
  task_capacity: 3
  max_workers: 2
metrics:
  strict: true
  compute_stage_metrics: true
""",
        encoding="utf-8",
    )

    sweep = SimulationSweep.from_yaml(str(config_path))
    tasks = sweep.generate_tasks()

    assert len(tasks) == 4
    assert {task.config["frl_dev"] for task in tasks} == {0.1, 0.2}
    assert {task.config["seed"] for task in tasks} == {1, 2}
    assert all(task.capacity_slots == 3 for task in tasks)
    assert all(
        task.config["levels"] == ["BlockGroup_1", "BlockGroup_0"] for task in tasks
    )
    assert all(task.config["save_solver_logs"] is True for task in tasks)
    assert all(task.config["save_solver_progress"] is True for task in tasks)
    assert all(str(tmp_path / "out") in task.output_dir for task in tasks)
    assert sweep.metrics.compute_stage_metrics is True


def test_single_zone_sweep_generates_scalar_optimization_values():
    config_path = Path(__file__).parent / "configs" / "sweep.test-one.yaml"

    tasks = SimulationSweep.from_yaml(str(config_path)).generate_tasks()

    assert len(tasks) == 3
    assert {task.config["seed"] for task in tasks} == {14, 33, 42}
    for task in tasks:
        config = task.optimization_config()
        assert config.solver == "cp_single_zone"
        assert config.strategy == "single"
        assert config.save_solver_progress is False
        assert config.frl_dev * 1.0 == 0.15
        assert config.overage * 1.0 == 0.15
        assert config.shortage * 1.0 == 0.15


def test_sweep_yaml_accepts_secondary_objective_task(tmp_path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        f"""
name: unit-test
mode: run
optimization_defaults:
  levels: ['BlockGroup_0']
  data:
    scenario: legacy
    overrides:
      roots:
        cache: '{tmp_path / "graphs"}'
tasks:
  - solver: 'cp_bool'
    secondary_objective: true
execution:
  output_dir: '{tmp_path / "out"}'
""",
        encoding="utf-8",
    )

    sweep = SimulationSweep.from_yaml(str(config_path))
    tasks = sweep.generate_tasks()

    assert len(tasks) == 1
    assert tasks[0].config["solver"] == "cp_bool"
    assert tasks[0].config["secondary_objective"] is True


def test_sweep_expands_weight_edges_and_changes_task_identity(tmp_path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        f"""
optimization_defaults:
  levels: ['BlockGroup_0']
  data:
    scenario: legacy
    overrides:
      roots:
        cache: '{tmp_path / "graphs"}'
sweep:
  weight_edges: [false, true]
execution:
  output_dir: '{tmp_path / "out"}'
""",
        encoding="utf-8",
    )

    tasks = SimulationSweep.from_yaml(str(config_path)).generate_tasks()

    assert [task.config["weight_edges"] for task in tasks] == [False, True]
    assert tasks[0].config_hash != tasks[1].config_hash


def test_sweep_yaml_rejects_aggregate_only_mode(tmp_path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        """
mode: aggregate
optimization_defaults:
  levels: ['BlockGroup_0']
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run, metrics"):
        SimulationSweep.from_yaml(str(config_path))


@pytest.mark.parametrize(
    "old_key",
    [
        "years",
        "population_type",
        "capacity_scenario",
        "new_schools",
        "include_k8",
        "remove_city_wide",
        "graphs_dir",
    ],
)
def test_sweep_rejects_removed_optimization_data_keys(tmp_path, old_key):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "optimization_defaults": {
                    "levels": ["BlockGroup_0"],
                    old_key: [] if old_key == "years" else True,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown keys in optimization_defaults"):
        SimulationSweep.from_yaml(str(config_path))


def test_bundled_sweeps_use_only_strict_data_maps():
    removed = {
        "years",
        "population_type",
        "capacity_scenario",
        "new_schools",
        "include_k8",
        "remove_city_wide",
        "graphs_dir",
    }
    benchmark_root = Path(__file__).parent
    paths = [benchmark_root / "sweep.example.yaml", benchmark_root / "sweep.test.yaml"]
    paths.extend(sorted((benchmark_root / "configs").glob("*.yaml")))

    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        defaults = raw["optimization_defaults"]
        assert not removed.intersection(defaults), path
        assert set(defaults["data"]) == {"scenario", "overrides"}, path
        assert isinstance(defaults["data"]["scenario"], str), path
        assert isinstance(defaults["data"]["overrides"], dict), path
        SimulationSweep.from_yaml(str(path))


def test_optimization_hash_ignores_cache_root_but_snapshot_retains_it(tmp_path):
    first = OptimizationConfig(
        data={
            "scenario": "legacy",
            "overrides": {"roots": {"cache": str(tmp_path / "cache-a")}},
        }
    )
    second = OptimizationConfig(
        data={
            "scenario": "legacy",
            "overrides": {"roots": {"cache": str(tmp_path / "cache-b")}},
        }
    )

    first_snapshot = optimization_config_to_dict(first)
    second_snapshot = optimization_config_to_dict(second)
    assert first_snapshot["data"] != second_snapshot["data"]
    assert first.data_scenario.cache_root != second.data_scenario.cache_root
    assert optimization_config_hash(first) == optimization_config_hash(second)


def test_stage_artifacts_reconstruct_and_aggregate(tmp_path):
    run_dir, problem = _write_synthetic_run(tmp_path)

    loaded, _, manifest = load_solutions(str(run_dir), dataset=FakeDataset(problem))
    assert manifest["stages"][0]["name"] == "stage_00_Block_0"
    assert loaded[0].assignment == _assignment()

    summary, stages = aggregate_results(
        str(tmp_path),
        summary_csv="summary.csv",
        stages_csv="stages.csv",
    )
    assert len(summary) == 1
    assert len(stages) == 1
    assert summary.loc[0, "status"] == "FEASIBLE"
    assert summary.loc[0, "num_zones"] == 2
    assert stages.loc[0, "stage_name"] == "stage_00_Block_0"
    assert stages.loc[0, "solver_log_path"] == "solver_logs/test.jsonl"
    assert stages.loc[0, "solver_log_format"] == "jsonl"
    assert (
        stages.loc[0, "solver_progress_path"] == "solver_progress/test/progress.jsonl"
    )
    assert stages.loc[0, "solver_progress_format"] == "jsonl"
    assert stages.loc[0, "solver_progress_count"] == 2
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "stages.csv").exists()


def test_saved_area_assignment_reconstructs_on_relabeled_graph(tmp_path):
    run_dir, problem = _write_synthetic_run(tmp_path)
    mapping = {node: node + 100 for node in problem.G}
    relabeled = nx.relabel_nodes(problem.G, mapping)
    changed_problem = replace(
        problem,
        G=relabeled,
        centroids=[mapping[node] for node in problem.centroids],
    )

    loaded, _, _ = load_solutions(
        str(run_dir),
        dataset=FakeDataset(changed_problem),
    )

    assert loaded[0].assignment == {
        mapping[node]: zone for node, zone in _assignment().items()
    }


def test_single_zone_stage_reconstructs_explicit_centroid(tmp_path):
    run_dir = tmp_path / "single-zone"
    run_dir.mkdir()
    base_problem = make_grid_problem(3, 3)
    base_problem.level = LevelSpec("Block", 0)
    dataset = FakeDataset(base_problem)
    child_problem = dataset.problem_for(
        "Block_0",
        centroid_school_ids=[100],
    )
    solutions = [
        ZoneSolution(
            problem=child_problem,
            assignment={0: 0, 1: 0},
            status="FEASIBLE",
            wall_time=0.1,
            metadata={
                "partial_assignment": True,
                "centroid_school_id": 100,
            },
        ),
    ]
    config = OptimizationConfig(
        levels=["Block_0"],
        strategy="single",
        solver="cp_single_zone",
        workers=1,
        data={
            "scenario": "legacy",
            "overrides": {"roots": {"cache": str(tmp_path / "graphs")}},
        },
    )
    config_dict = optimization_config_to_dict(config)
    config_hash = optimization_config_hash(config_dict)
    task = BenchmarkTask(
        task_id=config_hash[:12],
        config_hash=config_hash,
        config=config_dict,
        output_dir=str(run_dir),
        capacity_slots=1,
    )
    stage_names = stage_names_for(solutions, config)
    records = save_stage_artifacts(solutions, str(run_dir), stage_names)
    write_json(
        os.path.join(run_dir, MANIFEST_FILENAME),
        manifest_for(
            task=task,
            config=config,
            status="FEASIBLE",
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            stages=records,
            final_stage=stage_names[-1],
            error_message=None,
        ),
    )

    loaded, _, _ = load_solutions(str(run_dir), dataset=dataset)

    assert loaded[0].problem.centroids == [0]


def test_saved_config_rejects_legacy_level_to_split():
    with pytest.raises(ValueError, match="level_to_split"):
        optimization_config_from_dict(
            {"levels": ["BlockGroup_0"], "level_to_split": {"1": 2, "2": 1}}
        )


def test_regenerate_metrics_rewrites_result_payload(tmp_path):
    run_dir, problem = _write_synthetic_run(tmp_path)
    write_json(
        os.path.join(run_dir, RESULT_FILENAME), {"status": "STALE", "metrics": {}}
    )

    result = regenerate_metrics(
        str(tmp_path),
        dataset_factory=lambda config, manifest: FakeDataset(problem),
    )

    assert result.regenerated == 1
    summary, _ = aggregate_results(str(tmp_path))
    assert summary.loc[0, "status"] == "FEASIBLE"
    assert summary.loc[0, "num_zones"] == 2


def test_aggregation_prefers_evaluated_stage_contiguity(tmp_path):
    run_dir, _ = _write_synthetic_run(tmp_path)
    manifest_path = run_dir / MANIFEST_FILENAME
    result_path = run_dir / RESULT_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest["stages"][0]["contiguous"] = False
    result["run"]["stages"][0]["contiguous"] = True
    write_json(str(manifest_path), manifest)
    write_json(str(result_path), result)

    _, stages = aggregate_results(
        str(tmp_path), summary_csv="summary.csv", stages_csv="stages.csv"
    )

    assert bool(stages.loc[0, "contiguous"]) is True
    assert (tmp_path / ".benchmark-aggregate.lock").exists()


def _write_synthetic_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    problem = make_grid_problem(3, 3)
    problem.level = LevelSpec("Block", 0)
    solution = ZoneSolution(
        problem=problem,
        assignment=_assignment(),
        status="FEASIBLE",
        objective=7.0,
        wall_time=1.25,
        metadata={
            "solver": "test",
            "solver_log_path": "solver_logs/test.jsonl",
            "solver_log_format": "jsonl",
            "solver_progress_path": "solver_progress/test/progress.jsonl",
            "solver_progress_format": "jsonl",
            "solver_progress_count": 2,
        },
    )
    config = OptimizationConfig(
        centroids_type="5-zone-AF",
        levels=["Block_0"],
        solver="cp_int",
        strategy="single",
        frl_dev=1.0,
        racial_dev=1.0,
        overage=5.0,
        shortage=0.0,
        workers=1,
        data={
            "scenario": "legacy",
            "overrides": {"roots": {"cache": str(tmp_path / "graphs")}},
        },
    )
    config_dict = optimization_config_to_dict(config)
    config_hash = optimization_config_hash(config_dict)
    task = BenchmarkTask(
        task_id=config_hash[:12],
        config_hash=config_hash,
        config=config_dict,
        output_dir=str(run_dir),
        capacity_slots=1,
    )
    solutions = [solution]
    stage_records = save_stage_artifacts(
        solutions,
        str(run_dir),
        stage_names_for(solutions, config),
    )
    metrics = MetricsCalculator(solutions, config=config).compute()
    solution.save(str(run_dir))
    write_json(
        os.path.join(run_dir, RESULT_FILENAME),
        result_payload_for(
            metrics=metrics, config=config, solutions=solutions, task=task
        ),
    )
    write_json(
        os.path.join(run_dir, MANIFEST_FILENAME),
        manifest_for(
            task=task,
            config=config,
            status="FEASIBLE",
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            stages=stage_records,
            final_stage="stage_00_Block_0",
            error_message=None,
        ),
    )
    return run_dir, problem


def _assignment():
    return {
        0: 0,
        1: 0,
        3: 0,
        4: 0,
        2: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
    }
