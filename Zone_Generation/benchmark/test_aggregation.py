import os

import pytest

from Zone_Generation.optimization.config import OptimizationConfig
from Zone_Generation.optimization.levels import LevelSpec
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.tests.synthetic import FakeDataset, make_grid_problem
from Zone_Generation.benchmark.config import (
    BenchmarkTask,
    SimulationSweep,
    optimization_config_to_dict,
    stable_hash,
)
from Zone_Generation.benchmark.regenerate import regenerate_metrics
from Zone_Generation.benchmark.results import aggregate_results
from Zone_Generation.benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    load_solutions,
    manifest_for,
    result_payload_for,
    save_stage_artifacts,
    stage_names_for,
    write_json,
)
from Zone_Generation.metrics import MetricsCalculator


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
    assert all(task.config["levels"] == ["BlockGroup_1", "BlockGroup_0"] for task in tasks)
    assert all(str(tmp_path / "out") in task.output_dir for task in tasks)
    assert sweep.metrics.compute_stage_metrics is True


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
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "stages.csv").exists()


def test_regenerate_metrics_rewrites_result_payload(tmp_path):
    run_dir, problem = _write_synthetic_run(tmp_path)
    write_json(os.path.join(run_dir, RESULT_FILENAME), {"status": "STALE", "metrics": {}})

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
        metadata={"solver": "test"},
    )
    config = OptimizationConfig(
        centroids_type="5-zone-AF",
        levels=["Block_0"],
        solver="local_search",
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
        result_payload_for(metrics=metrics, config=config, solutions=solutions, task=task),
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
