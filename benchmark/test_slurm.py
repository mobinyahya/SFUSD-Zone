"""Dry tests for benchmark Slurm planning and phase workers."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from benchmark.config import (
    BenchmarkTask,
    ChoiceMetricsRunConfig,
    MatchingRunConfig,
    MetricsRunConfig,
    SimulationSweep,
    optimization_config_to_dict,
)
from benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    TaskResult,
    evaluate_optimization_task,
    run_optimization_phase,
)
from benchmark.slurm import (
    SlurmPlan,
    _build_parser,
    create_plan,
    load_plan,
    run_evaluation_worker,
    run_optimization_worker,
    submission_script,
    submit_plan,
    write_plan,
)
from optimization.config import OptimizationConfig
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import FakeDataset, make_grid_problem


@pytest.mark.parametrize("command", ["generate", "submit"])
def test_public_commands_require_config_option(command):
    parser = _build_parser()

    args = parser.parse_args([command, "--config", "sweep.yaml"])

    assert args.command == command
    assert args.config == "sweep.yaml"
    with pytest.raises(SystemExit):
        parser.parse_args([command, "sweep.yaml"])


def test_submission_script_has_required_directives_and_dependency_wiring(tmp_path):
    plan = _plan(tmp_path)
    plan_path = tmp_path / ".slurm" / "plan.json"

    script = submission_script(plan, plan_path)

    assert script.count("sbatch --parsable -A soal -p soal") == 2
    assert script.count("--ntasks=1") == 2
    assert "--cpus-per-task=4" in script
    assert "--cpus-per-task=1" in script
    assert "--export=ALL,OMP_NUM_THREADS=1" in script
    assert "optimization_job_0=$(sbatch" in script
    assert '--dependency="afterany:${optimization_job_0}"' in script
    assert "worker-optimize" in script
    assert "worker-evaluate" in script
    assert "srun" not in script
    assert "--cpus-per-task=99" not in script


def test_submit_uses_parsable_account_partition_and_afterany(tmp_path):
    plan = _plan(tmp_path)
    calls = []
    job_ids = iter(("120;cluster\n", "121\n"))

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=next(job_ids), stderr="")

    submitted = submit_plan(plan, tmp_path / "plan.json", run=fake_run)

    assert len(submitted) == 1
    assert submitted[0].optimization_job_id == "120"
    assert submitted[0].evaluation_job_id == "121"
    assert calls[0][0][:6] == ["sbatch", "--parsable", "-A", "soal", "-p", "soal"]
    assert calls[1][0][:6] == ["sbatch", "--parsable", "-A", "soal", "-p", "soal"]
    assert "--dependency=afterany:120" in calls[1][0]
    assert "--cpus-per-task=4" in calls[0][0]
    assert "--cpus-per-task=1" in calls[1][0]
    assert all(
        call[1] == {"check": True, "capture_output": True, "text": True}
        for call in calls
    )


@pytest.mark.parametrize(
    ("sweep", "message"),
    [
        (SimulationSweep(mode="metrics"), "supports only mode 'run'"),
        (
            SimulationSweep(matching=MatchingRunConfig(enabled=True)),
            "enabled matching",
        ),
        (
            SimulationSweep(
                choice_metrics=ChoiceMetricsRunConfig(enabled=True),
            ),
            "choice_metrics",
        ),
    ],
)
def test_planning_rejects_unsupported_modes_before_generating_tasks(
    tmp_path, monkeypatch, sweep, message
):
    monkeypatch.setattr(
        SimulationSweep,
        "from_yaml",
        classmethod(lambda cls, path: sweep),
    )
    monkeypatch.setattr(
        SimulationSweep,
        "generate_tasks",
        lambda self: pytest.fail("tasks should not be generated"),
    )

    with pytest.raises(ValueError, match=message):
        create_plan(str(tmp_path / "sweep.yaml"))


def test_plan_roundtrip_snapshots_tasks_metrics_and_absolute_paths(tmp_path):
    plan = _plan(tmp_path)

    path = write_plan(plan)
    loaded = load_plan(str(path))

    assert path.is_absolute()
    assert path.is_relative_to(tmp_path)
    assert loaded.to_dict() == plan.to_dict()
    assert loaded.tasks[0].config["workers"] == 4
    assert loaded.metrics.compute_stage_metrics is True
    assert Path(loaded.output_root).is_absolute()
    assert Path(loaded.tasks[0].output_dir).is_absolute()


@pytest.mark.parametrize("duplicate", ["hash", "output"])
def test_submit_rejects_duplicate_tasks_before_sbatch(tmp_path, duplicate):
    plan = _plan(tmp_path)
    first = plan.tasks[0]
    second = replace(
        first,
        task_id="second",
        config_hash=first.config_hash if duplicate == "hash" else "different",
        output_dir=first.output_dir
        if duplicate == "output"
        else str(tmp_path / "different-output"),
    )
    duplicate_plan = replace(plan, tasks=[first, second])
    calls = []

    with pytest.raises(ValueError, match="Duplicate benchmark task"):
        submit_plan(
            duplicate_plan,
            tmp_path / "plan.json",
            run=lambda *args, **kwargs: calls.append(args),
        )
    assert calls == []


def test_workers_enforce_allocations_and_return_nonzero_on_logical_errors(
    tmp_path, monkeypatch
):
    plan = _plan(tmp_path)
    monkeypatch.setattr("benchmark.slurm.load_plan", lambda path: plan)
    optimization_calls = []
    monkeypatch.setattr(
        "benchmark.slurm.run_optimization_phase",
        lambda task: optimization_calls.append(task)
        or TaskResult(task.task_id, task.output_dir, "FEASIBLE"),
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")

    assert run_optimization_worker("plan.json", 0) == 1
    assert optimization_calls == []

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setattr(
        "benchmark.slurm.evaluate_optimization_task",
        lambda *args, **kwargs: TaskResult("task", "output", "ERROR"),
    )
    aggregate_calls = []
    monkeypatch.setattr(
        "benchmark.slurm.aggregate_results",
        lambda *args, **kwargs: aggregate_calls.append((args, kwargs)),
    )

    assert run_evaluation_worker("plan.json", 0) == 1
    assert len(aggregate_calls) == 1


def test_optimization_and_metrics_are_separate_persisted_phases(tmp_path, monkeypatch):
    problem = make_grid_problem(3, 3)
    problem.level = LevelSpec("Block", 0)
    solution = ZoneSolution(
        problem=problem,
        assignment=_assignment(),
        status="FEASIBLE",
        objective=3.0,
        wall_time=1.5,
    )
    config = OptimizationConfig(
        levels=["Block_0"],
        strategy="single",
        solver="cp_int",
        workers=2,
        frl_dev=1.0,
        racial_dev=1.0,
        overage=5.0,
        data={
            "scenario": "legacy",
            "overrides": {"roots": {"cache": str(tmp_path / "graphs")}},
        },
    )
    task = BenchmarkTask(
        task_id="phase-test",
        config_hash="phase-test-hash",
        config=optimization_config_to_dict(config),
        output_dir=str(tmp_path / "run"),
        capacity_slots=17,
    )

    class FakeStrategy:
        def run(self, dataset, solver):
            return [solution]

    monkeypatch.setattr(OptimizationConfig, "make_dataset", lambda self: object())
    monkeypatch.setattr(
        OptimizationConfig, "make_solver", lambda self, output_dir=None: object()
    )
    monkeypatch.setattr(OptimizationConfig, "make_strategy", lambda self: FakeStrategy())

    import benchmark.runner as runner

    calculator_calls = []
    real_calculator = runner.MetricsCalculator

    class CalculatorSpy:
        def __init__(self, *args, **kwargs):
            calculator_calls.append((args, kwargs))
            self.delegate = real_calculator(*args, **kwargs)

        def compute(self):
            return self.delegate.compute()

        @property
        def context(self):
            return self.delegate.context

    monkeypatch.setattr(runner, "MetricsCalculator", CalculatorSpy)

    optimization = run_optimization_phase(task)

    assert optimization.status == "FEASIBLE"
    assert calculator_calls == []
    run_dir = Path(task.output_dir)
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    result = json.loads((run_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
    assert manifest["phase"] == "optimization"
    assert manifest["final_stage"] is None
    assert result["metrics"] == {}
    assert not (run_dir / "zone_dict_Block_0.json").exists()
    assert (run_dir / "stages/stage_00_Block_0/zone_dict_Block_0.json").exists()

    evaluated = evaluate_optimization_task(
        task,
        dataset=FakeDataset(problem),
        compute_stage_metrics=True,
    )

    assert evaluated.status == "FEASIBLE"
    assert len(calculator_calls) == 1
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    result = json.loads((run_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
    assert manifest["phase"] == "complete"
    assert manifest["final_stage"] == "stage_00_Block_0"
    assert result["metrics"]
    assert (run_dir / "zone_dict_Block_0.json").exists()


def test_optimization_failure_persists_error_artifacts(tmp_path, monkeypatch):
    config = OptimizationConfig(
        workers=1,
        data={
            "scenario": "legacy",
            "overrides": {"roots": {"cache": str(tmp_path / "graphs")}},
        },
    )
    task = BenchmarkTask(
        task_id="failure-test",
        config_hash="failure-test-hash",
        config=optimization_config_to_dict(config),
        output_dir=str(tmp_path / "failed-run"),
        capacity_slots=1,
    )

    class FailingStrategy:
        def run(self, dataset, solver):
            raise RuntimeError("synthetic optimization failure")

    monkeypatch.setattr(OptimizationConfig, "make_dataset", lambda self: object())
    monkeypatch.setattr(
        OptimizationConfig, "make_solver", lambda self, output_dir=None: object()
    )
    monkeypatch.setattr(
        OptimizationConfig, "make_strategy", lambda self: FailingStrategy()
    )

    result = run_optimization_phase(task)

    assert result.status == "ERROR"
    run_dir = Path(task.output_dir)
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    payload = json.loads((run_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
    assert manifest["phase"] == "optimization_error"
    assert manifest["status"] == "ERROR"
    assert "synthetic optimization failure" in manifest["error_message"]
    assert "traceback" in manifest
    assert payload["status"] == "ERROR"


def _plan(tmp_path: Path) -> SlurmPlan:
    task = BenchmarkTask(
        task_id="abc123",
        config_hash="unique-config-hash",
        config={"workers": 4},
        output_dir=str((tmp_path / "run").resolve()),
        capacity_slots=99,
    )
    return SlurmPlan(
        name="test-plan",
        created_at="2026-08-17T00:00:00+00:00",
        source_config=str((tmp_path / "sweep.yaml").resolve()),
        project_root=str(tmp_path.resolve()),
        output_root=str(tmp_path.resolve()),
        metrics=MetricsRunConfig(strict=False, compute_stage_metrics=True),
        tasks=[task],
    )


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
