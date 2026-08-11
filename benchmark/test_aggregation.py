import os
from dataclasses import replace
from pathlib import Path

import networkx as nx

import pytest

from optimization.config import OptimizationConfig
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import FakeDataset, make_grid_problem
from benchmark.config import (
    BenchmarkTask,
    SimulationSweep,
    optimization_config_from_dict,
    optimization_config_to_dict,
    stable_hash,
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
  graphs_dir: '{tmp_path / "graphs"}'
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
  graphs_dir: '{tmp_path / "graphs"}'
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


def test_overlapping_stages_reconstruct_stage_specific_centroids(tmp_path):
    run_dir = tmp_path / "overlapping"
    run_dir.mkdir()
    base_problem = make_grid_problem(3, 3)
    base_problem.level = LevelSpec("Block", 0)
    dataset = FakeDataset(base_problem)
    child_problem = dataset.problem_for(
        "Block_0",
        centroid_school_ids=[100],
    )
    final_problem = dataset.problem_for(
        "Block_0",
        centroid_school_ids=[100, 200],
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
        ZoneSolution(
            problem=final_problem,
            assignment=_assignment(),
            status="FEASIBLE",
            wall_time=0.2,
            metadata={"centroid_school_ids": [100, 200]},
        ),
    ]
    config = OptimizationConfig(
        levels=["Block_0"],
        strategy="overlapping",
        workers=1,
        graphs_dir=str(tmp_path / "graphs"),
    )
    config_dict = optimization_config_to_dict(config)
    config_hash = stable_hash(config_dict)
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
    assert loaded[1].problem.centroids == [0, 8]


def test_saved_config_ignores_legacy_level_to_split():
    config = optimization_config_from_dict(
        {"levels": ["BlockGroup_0"], "level_to_split": {"1": 2, "2": 1}}
    )

    assert config.levels == ["BlockGroup_0"]
    assert not hasattr(config, "level_to_split")


def test_saved_config_migrates_strategy_specific_recom_seed_runs():
    config = optimization_config_from_dict(
        {
            "levels": ["BlockGroup_0"],
            "strategy": "zoned_column_generation",
            "solver": "cp_bool",
            "years": [23],
            "population_type": "All",
            "remove_city_wide": True,
            "zoned_cg_recom_seed_runs": 3,
            "zoned_benders_recom_seed_runs": 7,
        }
    )

    assert config.zoned_recom_seed_runs == 3


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
        graphs_dir=str(tmp_path / "graphs"),
    )
    config_dict = optimization_config_to_dict(config)
    config_hash = stable_hash(config_dict)
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
