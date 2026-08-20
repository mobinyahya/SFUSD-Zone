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
    MAX_SLURM_JOBS,
    SlurmPlan,
    _build_parser,
    _plan_allocations,
    create_plan,
    load_plan,
    run_benchmark_worker,
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


def test_submission_script_has_required_directives(tmp_path):
    plan = _plan(tmp_path)
    plan_path = tmp_path / ".slurm" / "plan.json"

    script = submission_script(plan, plan_path)

    assert script.count("sbatch --parsable -A soal -p soal") == 1
    assert script.count("--ntasks=1") == 1
    assert "--cpus-per-task=4" in script
    assert "--export=ALL,OMP_NUM_THREADS=1" in script
    assert "benchmark_job_0=$(sbatch" in script
    assert "--dependency" not in script
    assert "worker-allocation" in script
    assert "--allocation-index" in script
    assert "srun" not in script
    assert "--cpus-per-task=99" not in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_submit_uses_parsable_account_partition_without_dependencies(tmp_path):
    plan = _plan(tmp_path)
    calls = []
    job_ids = iter(("120;cluster\n",))

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=next(job_ids), stderr="")

    submitted = submit_plan(plan, tmp_path / "plan.json", run=fake_run)

    assert len(submitted) == 1
    assert submitted[0].phase == "benchmark"
    assert submitted[0].job_id == "120"
    assert calls[0][0][:6] == ["sbatch", "--parsable", "-A", "soal", "-p", "soal"]
    assert "--cpus-per-task=4" in calls[0][0]
    assert not any(value.startswith("--dependency") for value in calls[0][0])
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


def test_worker_enforces_allocation_and_evaluates_after_optimization(
    tmp_path, monkeypatch
):
    plan = _plan(tmp_path)
    second_task = replace(
        plan.tasks[0],
        task_id="second",
        config_hash="second-hash",
        output_dir=str((tmp_path / "second-run").resolve()),
    )
    plan = replace(plan, tasks=[plan.tasks[0], second_task])
    monkeypatch.setattr("benchmark.slurm.load_plan", lambda path: plan)
    events = []
    monkeypatch.setattr(
        "benchmark.slurm.run_optimization_phase",
        lambda task: (
            events.append(("optimize", task.task_id))
            or TaskResult(task.task_id, task.output_dir, "FEASIBLE")
        ),
    )
    aggregate_calls = []
    monkeypatch.setattr(
        "benchmark.slurm.aggregate_results",
        lambda *args, **kwargs: aggregate_calls.append((args, kwargs)),
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")

    assert run_benchmark_worker("plan.json", 0) == 1
    assert events == []
    assert aggregate_calls == []

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    executor_workers = []

    class ImmediateExecutor:
        def __init__(self, max_workers):
            executor_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def submit(self, function, *args):
            from concurrent.futures import Future

            future = Future()
            try:
                future.set_result(function(*args))
            except Exception as exc:
                future.set_exception(exc)
            return future

    monkeypatch.setattr("benchmark.slurm.ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(
        "benchmark.slurm.evaluate_optimization_task",
        lambda task, **kwargs: (
            events.append(("evaluate", task.task_id))
            or TaskResult(task.task_id, task.output_dir, "ERROR")
        ),
    )

    assert run_benchmark_worker("plan.json", 0) == 1
    assert [phase for phase, _task_id in events] == [
        "optimize",
        "optimize",
        "evaluate",
        "evaluate",
    ]
    assert executor_workers == [2, 8]
    assert len(aggregate_calls) == 1


def test_large_benchmark_plan_uses_twelve_allocations_and_usable_cpus(tmp_path):
    tasks = [
        BenchmarkTask(
            task_id=f"task-{index}",
            config_hash=f"hash-{index}",
            config={"workers": 9},
            output_dir=str((tmp_path / f"run-{index}").resolve()),
            capacity_slots=99,
        )
        for index in range(100)
    ]
    plan = replace(_plan(tmp_path), tasks=tasks)

    allocations = _plan_allocations(plan)

    assert len(allocations) == MAX_SLURM_JOBS
    assert {item.phase for item in allocations} == {"benchmark"}
    assert {item.cpus for item in allocations} == {36}
    assert all(item.cpus // item.task_cpus == 4 for item in allocations)
    assert sorted(index for item in allocations for index in item.task_indices) == list(
        range(100)
    )
    assert (
        submission_script(plan, tmp_path / "plan.json").count("sbatch --parsable")
        == MAX_SLURM_JOBS
    )


@pytest.mark.parametrize(
    ("workers", "task_count", "expected_cpus"),
    [(9, 4, 36), (8, 5, 40), (24, 1, 24)],
)
def test_allocation_cpu_requests_have_no_unusable_remainder(
    tmp_path, workers, task_count, expected_cpus
):
    tasks = [
        BenchmarkTask(
            task_id=f"task-{index}",
            config_hash=f"hash-{index}",
            config={"workers": workers},
            output_dir=str((tmp_path / f"run-{index}").resolve()),
            capacity_slots=99,
        )
        for index in range(task_count)
    ]

    allocations = _plan_allocations(replace(_plan(tmp_path), tasks=tasks))

    assert allocations[0].cpus == expected_cpus


def test_submit_rejects_tasks_requiring_more_than_one_node(tmp_path):
    task = replace(_plan(tmp_path).tasks[0], config={"workers": 41})
    calls = []

    with pytest.raises(ValueError, match="cannot exceed 40"):
        submit_plan(
            replace(_plan(tmp_path), tasks=[task]),
            tmp_path / "plan.json",
            run=lambda *args, **kwargs: calls.append(args),
        )

    assert calls == []


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
    monkeypatch.setattr(
        OptimizationConfig, "make_strategy", lambda self: FakeStrategy()
    )

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
