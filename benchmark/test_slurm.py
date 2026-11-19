"""Dry tests for benchmark Slurm planning and phase workers."""

from __future__ import annotations

from dataclasses import replace
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

import pytest

from benchmark.config import (
    BenchmarkTask,
    ExecutionConfig,
    MatchingRunConfig,
    MetricsRunConfig,
    SimulationSweep,
    VisualizationRunConfig,
    optimization_config_to_dict,
)
from benchmark.assignment import assignment_targets
from benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    TaskResult,
    evaluate_optimization_task,
    run_optimization_phase,
)
from benchmark.slurm import (
    MAX_SLURM_JOBS,
    MAX_SUBMITTED_SLURM_JOBS,
    SlurmAllocation,
    SlurmPlan,
    _build_parser,
    _plan_allocations,
    create_plan,
    load_plan,
    run_benchmark_worker,
    submission_script,
    submit_plan,
    write_plan,
    write_submission_script,
)
from optimization import run_archive
from optimization.config import OptimizationConfig
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import FakeDataset, make_grid_problem
from optimization.visualization import RenderResult


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
    assert script.count("--mem-per-cpu=5G") == 1
    assert "--export=ALL,OMP_NUM_THREADS=1" in script
    assert "benchmark_job_0=$(sbatch" in script
    assert "--dependency" not in script
    assert "worker-allocation" in script
    assert "--allocation-index" in script
    assert "GRB_LICENSE_FILE" in script
    assert "srun" not in script
    assert "--cpus-per-task=99" not in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_matching_plan_submits_only_its_allocations(tmp_path):
    assignment_config = (
        Path(__file__).parents[1] / "assignment/configs/8-19-choice.yaml"
    ).resolve()
    plan = replace(
        _plan(tmp_path),
        matching=MatchingRunConfig(enabled=True, config=str(assignment_config)),
    )

    script = submission_script(plan, tmp_path / "benchmark-plan.json")

    # Matching runs inside the allocation: no assignment job graph behind it.
    assert script.count("sbatch --parsable") == 1
    assert "--upstream-job-id" not in script
    assert "--dependency" not in script
    assert "--mem-per-cpu=5G" in script
    assert not (tmp_path / ".slurm" / "assignment").exists()
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_saa_slurm_plan_covers_every_possible_iteration(tmp_path):
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="mip",
        strategy="saa",
        max_iterations=3,
        data={
            "scenario": "legacy",
            "overrides": {"filters": {"optimization": {"program_population": "All"}}},
        },
    )
    task = BenchmarkTask(
        task_id="saa-task",
        config_hash="hash",
        config=optimization_config_to_dict(config),
        output_dir=str(tmp_path / "run"),
        capacity_slots=1,
    )

    targets = assignment_targets(
        [task],
        MatchingRunConfig(
            enabled=True,
            config="assignment.yaml",
            compute_stage_assignments=True,
        ),
    )

    assert [target["id"] for target in targets] == [
        "saa-task-root",
        "saa-task-stage-0",
        "saa-task-stage-1",
        "saa-task-stage-2",
        "saa-task-stage-3",
    ]
    assert "iteration_03_BlockGroup_0" in targets[-1]["zone_file"]


def test_submit_uses_parsable_account_partition_without_dependencies(tmp_path):
    plan = _plan(tmp_path)
    calls = []
    job_ids = iter(("120;cluster\n",))

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=next(job_ids), stderr="")

    submitted = submit_plan(plan, tmp_path / "plan.json", run=fake_run)

    assert len(submitted) == 1
    assert submitted[0].phase == "benchmark"
    assert submitted[0].job_id == "120"
    assert calls[0][0][0] == "squeue"
    assert calls[1][0][:6] == ["sbatch", "--parsable", "-A", "soal", "-p", "soal"]
    assert "--cpus-per-task=4" in calls[1][0]
    assert not any(value.startswith("--dependency") for value in calls[1][0])
    assert all(
        call[1] == {"check": True, "capture_output": True, "text": True}
        for call in calls
    )


def test_complete_matching_sweep_submits_twelve_allocations_and_nothing_else(
    tmp_path, monkeypatch
):
    plan = _large_matching_plan(tmp_path, monkeypatch)
    assert len(plan.allocations) == MAX_SLURM_JOBS == MAX_SUBMITTED_SLURM_JOBS == 12
    assert all(allocation.cpus == 40 for allocation in plan.allocations)
    assert sorted(i for a in plan.allocations for i in a.task_indices) == list(
        range(1000)
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        assert command[0] == "sbatch"
        return subprocess.CompletedProcess(
            command, 0, stdout=f"{100 + len(calls)}\n", stderr=""
        )

    submitted = submit_plan(plan, tmp_path / "benchmark-plan.json", run=fake_run)

    assert len(submitted) == 12
    assert Counter(job.phase for job in submitted) == {"benchmark": 12}
    sbatch_calls = [command for command in calls if command[0] == "sbatch"]
    assert len(sbatch_calls) == 12
    for command in sbatch_calls:
        assert "--cpus-per-task=40" in command
        assert "--mem-per-cpu=5G" in command
        assert not any(value.startswith("--dependency") for value in command)
    subprocess.run(
        ["bash", "-n"],
        input=submission_script(plan, tmp_path / "benchmark-plan.json"),
        text=True,
        check=True,
    )


@pytest.mark.parametrize("entrypoint", ["api", "script"])
def test_oversized_saved_plan_is_rejected_before_any_submission(
    tmp_path, monkeypatch, entrypoint
):
    plan = _large_matching_plan(tmp_path, monkeypatch)
    monkeypatch.setattr("benchmark.slurm.MAX_SUBMITTED_SLURM_JOBS", 10)
    calls = []

    with pytest.raises(ValueError, match="12 jobs.*10-job submission limit"):
        if entrypoint == "api":
            submit_plan(
                plan, tmp_path / "plan.json", run=lambda *a, **kw: calls.append(a)
            )
        else:
            submission_script(plan, tmp_path / "plan.json")

    assert calls == []


@pytest.mark.parametrize("existing", [11, 12])
def test_submit_counts_existing_user_jobs_before_submitting(tmp_path, existing):
    plan = _plan(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(
                command, 0, stdout="\n".join(str(i) for i in range(existing)), stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="123\n", stderr="")

    if existing == 12:
        with pytest.raises(RuntimeError, match="No jobs were submitted"):
            submit_plan(plan, tmp_path / "plan.json", run=fake_run)
        assert len(calls) == 1
    else:
        assert len(submit_plan(plan, tmp_path / "plan.json", run=fake_run)) == 1
        assert calls[-1][0] == "sbatch"


def test_generated_script_checks_existing_jobs_before_submitting(tmp_path):
    directory = tmp_path / "bin"
    directory.mkdir()
    squeue = directory / "squeue"
    squeue.write_text(
        "#!/bin/sh\nprintf '%s\\n' " + " ".join(str(i) for i in range(12)) + "\n"
    )
    squeue.chmod(0o755)
    sbatch = directory / "sbatch"
    marker = tmp_path / "submitted"
    sbatch.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    sbatch.chmod(0o755)

    result = subprocess.run(
        ["bash"],
        input=submission_script(_plan(tmp_path), tmp_path / "plan.json"),
        env={**os.environ, "PATH": f"{directory}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "12 existing jobs plus 1 planned jobs" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    ("sweep", "message"),
    [
        (SimulationSweep(mode="metrics"), "supports only mode 'run'"),
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


def test_matching_plan_builds_no_assignment_graph(tmp_path, monkeypatch):
    assignment_config = tmp_path / "assignment.yaml"
    assignment_config.write_text("{}\n", encoding="utf-8")
    tasks = [
        BenchmarkTask(
            task_id=f"task-{index}",
            config_hash=f"hash-{index}",
            config={"workers": 1},
            output_dir=str((tmp_path / f"run-{index}").resolve()),
            capacity_slots=1,
        )
        for index in range(3)
    ]
    sweep = SimulationSweep(
        execution=ExecutionConfig(output_dir=str(tmp_path)),
        matching=MatchingRunConfig(enabled=True, config=str(assignment_config)),
    )
    monkeypatch.setattr(
        SimulationSweep,
        "from_yaml",
        classmethod(lambda cls, path: sweep),
    )
    monkeypatch.setattr(SimulationSweep, "generate_tasks", lambda self: tasks)

    def no_graph(*args, **kwargs):
        pytest.fail("matching must not build an assignment job graph")

    monkeypatch.setattr("assignment.slurm.build_generated_zone_slurm_plan", no_graph)
    # Task 0 is optimized and evaluated, so it is planned for its matching;
    # task 1 is matched too, so it is done; task 2 has not run.
    _write_complete_run(tasks[0])
    _write_complete_run(tasks[1])
    _write_matching_marker(tasks[1], sweep.matching)

    plan = create_plan(str(tmp_path / "sweep.yaml"))

    assert [task.task_id for task in plan.tasks] == ["task-0", "task-2"]
    assert "assignment_plan_path" not in plan.to_dict()
    assert len(plan.allocations) == 1
    assert plan.allocations[0].cpus == 2


def test_task_phases_skip_only_what_is_complete(tmp_path):
    from benchmark.slurm import task_phases

    assignment_config = tmp_path / "assignment.yaml"
    assignment_config.write_text("{}\n", encoding="utf-8")
    matching = MatchingRunConfig(enabled=True, config=str(assignment_config))
    task = BenchmarkTask(
        task_id="task",
        config_hash="hash",
        config={"workers": 1},
        output_dir=str((tmp_path / "run").resolve()),
        capacity_slots=1,
    )
    skip = ExecutionConfig(skip_existing=True, rerun_failed=True)

    assert task_phases(task, skip, matching) == ("optimize", "evaluate", "match")
    _write_complete_run(task, phase="optimization")
    assert task_phases(task, skip, matching) == ("evaluate", "match")
    _write_complete_run(task)
    # Optimized and evaluated, but not matched: matching alone.
    assert task_phases(task, skip, matching) == ("match",)
    assert task_phases(task, skip, MatchingRunConfig()) == ()
    _write_matching_marker(task, matching)
    assert task_phases(task, skip, matching) == ()
    # Another matching config means matching again.
    assignment_config.write_text("{changed: true}\n", encoding="utf-8")
    assert task_phases(task, skip, matching) == ("match",)
    _write_matching_marker(task, matching)
    assert task_phases(task, skip, matching) == ()
    assert task_phases(task, replace(skip, skip_existing=False), matching) == (
        "optimize",
        "evaluate",
        "match",
    )
    # A failed run under rerun_failed starts over, matching included.
    _write_complete_run(task, status="ERROR")
    assert task_phases(task, skip, matching) == ("optimize", "evaluate", "match")
    assert task_phases(task, replace(skip, rerun_failed=False), matching) == ()


def _write_complete_run(task, *, phase="complete", status="FEASIBLE"):
    run_dir = Path(task.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "config_hash": task.config_hash,
        "phase": phase,
        "status": status,
    }
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / RESULT_FILENAME).write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )


def _write_matching_marker(task, matching, status="complete"):
    from benchmark.assignment import (
        MATCHING_COMPLETE_FILENAME,
        _matching_identity,
        matching_dir,
    )
    from benchmark.runner import write_json

    write_json(
        str(matching_dir(task.output_dir) / MATCHING_COMPLETE_FILENAME),
        {**_matching_identity(task, matching), "status": status},
    )


def test_plan_roundtrip_snapshots_tasks_metrics_and_absolute_paths(tmp_path):
    plan = _plan(tmp_path)

    path = write_plan(plan)
    loaded = load_plan(str(path))

    assert path.is_absolute()
    assert path.is_relative_to(tmp_path)
    assert loaded.to_dict() == plan.to_dict()
    assert loaded.tasks[0].config["workers"] == 4
    assert loaded.metrics.compute_stage_metrics is True
    assert loaded.visualization.enabled is True
    assert loaded.visualization.stages == "all"
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


class _ImmediateExecutor:
    """Runs each submitted call at once; records how it was built."""

    created: list = []

    def __init__(self, max_workers, **kwargs):
        type(self).created.append({"max_workers": max_workers, **kwargs})

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
    _ImmediateExecutor.created = []
    evaluation_kwargs = []
    monkeypatch.setattr("benchmark.slurm.ProcessPoolExecutor", _ImmediateExecutor)

    def fake_evaluate(task, **kwargs):
        evaluation_kwargs.append(kwargs)
        events.append(("evaluate", task.task_id))
        return TaskResult(task.task_id, task.output_dir, "ERROR")

    monkeypatch.setattr(
        "benchmark.slurm.evaluate_optimization_task",
        fake_evaluate,
    )

    # A saved task ERROR must not fail the allocation.
    assert run_benchmark_worker("plan.json", 0) == 0
    assert [phase for phase, _task_id in events] == [
        "optimize",
        "optimize",
        "evaluate",
        "evaluate",
    ]
    # One pool: 8 CPUs / 4 per task, a fresh process for every phase.
    assert _ImmediateExecutor.created == [{"max_workers": 2, "max_tasks_per_child": 1}]
    assert all(
        kwargs["visualization"] == plan.visualization for kwargs in evaluation_kwargs
    )
    assert len(aggregate_calls) == 1
    assert aggregate_calls[0][1]["matching_dir"] is None

    def crashing_evaluate(task, **kwargs):
        raise RuntimeError("worker crash")

    monkeypatch.setattr(
        "benchmark.slurm.evaluate_optimization_task", crashing_evaluate
    )
    assert run_benchmark_worker("plan.json", 0) == 1
    assert len(aggregate_calls) == 2


def test_worker_matches_each_task_right_after_its_own_evaluation(
    tmp_path, monkeypatch
):
    assignment_config = tmp_path / "assignment.yaml"
    assignment_config.write_text("{}\n", encoding="utf-8")
    base = _plan(tmp_path)
    tasks = [
        replace(
            base.tasks[0],
            task_id=f"task-{index}",
            config_hash=f"hash-{index}",
            config={"workers": 1},
            output_dir=str((tmp_path / f"run-{index}").resolve()),
        )
        for index in range(3)
    ]
    matching = MatchingRunConfig(enabled=True, config=str(assignment_config))
    plan = replace(base, tasks=tasks, matching=matching)
    # Task 1 is optimized and evaluated already: it is only matched.
    _write_complete_run(tasks[1])
    monkeypatch.setattr("benchmark.slurm.load_plan", lambda path: plan)
    monkeypatch.setattr("benchmark.slurm.ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    events = []
    monkeypatch.setattr(
        "benchmark.slurm.run_optimization_phase",
        lambda task: (
            events.append(("optimize", task.task_id))
            or TaskResult(task.task_id, task.output_dir, "FEASIBLE")
        ),
    )

    def fake_evaluate(task, **kwargs):
        assert kwargs["matching"] == matching
        events.append(("evaluate", task.task_id))
        return TaskResult(task.task_id, task.output_dir, "FEASIBLE")

    def fake_match(task, matching_config):
        assert matching_config == matching
        events.append(("match", task.task_id))
        return "complete"

    monkeypatch.setattr("benchmark.slurm.evaluate_optimization_task", fake_evaluate)
    monkeypatch.setattr("benchmark.slurm.run_task_matching", fake_match)
    aggregate_calls = []
    monkeypatch.setattr(
        "benchmark.slurm.aggregate_results",
        lambda *args, **kwargs: aggregate_calls.append((args, kwargs)),
    )

    # Three CPUs: every task holds a slot through all its phases.
    assert run_benchmark_worker("plan.json", 0) == 0
    by_task = {}
    for phase, task_id in events:
        by_task.setdefault(task_id, []).append(phase)
    assert by_task == {
        "task-0": ["optimize", "evaluate", "match"],
        "task-1": ["match"],
        "task-2": ["optimize", "evaluate", "match"],
    }
    assert aggregate_calls[-1][1]["matching_dir"] == "assignments"

    # One CPU: a task's next phase runs before the next task starts.
    events.clear()
    for task in tasks:
        _write_complete_run(task, status="ERROR")
    allocation = SlurmAllocation("benchmark", (0, 1, 2), 1, 1)
    monkeypatch.setattr(
        "benchmark.slurm._allocation_at", lambda plan, index: allocation
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert run_benchmark_worker("plan.json", 0) == 0
    assert events == [
        (phase, f"task-{index}")
        for index in range(3)
        for phase in ("optimize", "evaluate", "match")
    ]


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


def test_optimization_and_metrics_are_separate_persisted_phases(
    tmp_path, monkeypatch, shared_cache_root
):
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
            "overrides": {"roots": {"cache": str(shared_cache_root)}},
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
    visualization_calls = []

    def fake_visualize(solutions, output_dir, **kwargs):
        visualization_calls.append((solutions, output_dir, kwargs))
        path = Path(output_dir) / "visualization_stage_00_Block_0.png"
        path.write_bytes(b"png")
        return [
            RenderResult(
                stage="stage_00_Block_0",
                figure_paths=[path],
                geometry_artifact=tmp_path / "cached-geometry.pkl",
            )
        ]

    monkeypatch.setattr("benchmark.visualize.visualize_solutions", fake_visualize)

    optimization = run_optimization_phase(task)

    assert optimization.status == "FEASIBLE"
    assert calculator_calls == []
    run_dir = Path(task.output_dir)
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    result = json.loads((run_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
    assert manifest["phase"] == "optimization"
    assert manifest["final_stage"] is None
    assert result["metrics"] == {}
    archive = run_archive.read_zonings(run_dir)
    assert [stage["name"] for stage in archive["stages"]] == ["stage_00_Block_0"]
    assert archive["final_stage"] is None
    assert archive["config"] == manifest["config"]
    assert (run_dir / "zoning_config.yaml").exists()
    assert not (run_dir / "stages").exists()

    evaluated = evaluate_optimization_task(
        task,
        dataset=FakeDataset(problem),
        compute_stage_metrics=True,
        visualization=VisualizationRunConfig(enabled=True),
    )

    assert evaluated.status == "FEASIBLE"
    assert len(calculator_calls) == 1
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    result = json.loads((run_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
    assert manifest["phase"] == "complete"
    assert manifest["final_stage"] == "stage_00_Block_0"
    assert result["metrics"]
    assert manifest["visualization"]["stages"] == "final"
    assert visualization_calls[0][2]["artifact_dir"] is None
    assert run_archive.read_zonings(run_dir)["final_stage"] == "stage_00_Block_0"
    assert not (run_dir / "zone_dict_Block_0.json").exists()


def test_optimization_failure_persists_error_artifacts(
    tmp_path, monkeypatch, shared_cache_root
):
    config = OptimizationConfig(
        workers=1,
        data={
            "scenario": "legacy",
            "overrides": {"roots": {"cache": str(shared_cache_root)}},
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


def test_optimization_error_marks_its_assignment_targets_skipped(
    tmp_path, monkeypatch, shared_cache_root
):
    from benchmark.assignment import planned_stage_names

    config = OptimizationConfig(
        workers=1,
        strategy="recursive",
        levels=["BlockGroup_1", "BlockGroup_0"],
        data={
            "scenario": "legacy",
            "overrides": {"roots": {"cache": str(shared_cache_root)}},
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
    run_optimization_phase(task)
    run_dir = Path(task.output_dir)
    stages = planned_stage_names(task.config)
    assert len(stages) == 2
    # A target that already has a zone file keeps it and is still matched.
    matched = run_dir / "stages" / stages[0]
    matched.mkdir(parents=True)
    (matched / "assignment_zones.csv").write_text("zone\n", encoding="utf-8")

    result = evaluate_optimization_task(
        task,
        matching=MatchingRunConfig(
            enabled=True, config="unused.yaml", compute_stage_assignments=True
        ),
    )

    assert result.status == "ERROR"
    for folder in (run_dir, run_dir / "stages" / stages[1]):
        assert (folder / ".assignment-skipped").read_text() == "optimization error\n"
    assert not (matched / ".assignment-skipped").exists()
    assert (matched / "assignment_zones.csv").exists()


def _large_matching_plan(tmp_path, monkeypatch):
    assignment_config = (
        Path(__file__).parents[1]
        / "assignment/configs/attendance_areas/restrictive_aa_2627.yaml"
    ).resolve()
    tasks = [
        BenchmarkTask(
            task_id=f"task-{i}",
            config_hash=f"hash-{i}",
            config={"workers": 1},
            output_dir=str(tmp_path / f"run-{i}"),
            capacity_slots=1,
        )
        for i in range(1000)
    ]
    sweep = SimulationSweep(
        execution=ExecutionConfig(output_dir=str(tmp_path)),
        matching=MatchingRunConfig(enabled=True, config=str(assignment_config)),
    )
    monkeypatch.setattr(
        SimulationSweep, "from_yaml", classmethod(lambda cls, path: sweep)
    )
    monkeypatch.setattr(SimulationSweep, "generate_tasks", lambda self: tasks)
    return create_plan(str(tmp_path / "sweep.yaml"))


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
        visualization=VisualizationRunConfig(
            enabled=True,
            stages="all",
            artifact_dir=str((tmp_path / "visualization-cache").resolve()),
        ),
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


def test_create_plan_skips_existing_results(tmp_path, monkeypatch):
    task1 = BenchmarkTask(
        task_id="task-1",
        config_hash="hash1",
        config={"workers": 4},
        output_dir=str(tmp_path / "task1"),
        capacity_slots=4,
    )
    task2 = BenchmarkTask(
        task_id="task-2",
        config_hash="hash2",
        config={"workers": 4},
        output_dir=str(tmp_path / "task2"),
        capacity_slots=4,
    )

    (tmp_path / "task1").mkdir(parents=True)
    manifest1 = {
        "schema_version": 1,
        "config_hash": "hash1",
        "phase": "complete",
        "status": "FEASIBLE",
    }
    with open(tmp_path / "task1" / "benchmark_manifest.json", "w") as f:
        json.dump(manifest1, f)
    with open(tmp_path / "task1" / "result.json", "w") as f:
        json.dump({"status": "FEASIBLE"}, f)

    sweep = SimulationSweep(
        name="test-skip",
        mode="run",
        execution=ExecutionConfig(output_dir=str(tmp_path), skip_existing=True),
    )
    monkeypatch.setattr(
        SimulationSweep, "from_yaml", classmethod(lambda cls, path: sweep)
    )
    monkeypatch.setattr(SimulationSweep, "generate_tasks", lambda self: [task1, task2])

    plan = create_plan(str(tmp_path / "sweep.yaml"))
    assert len(plan.tasks) == 1
    assert plan.tasks[0].task_id == "task-2"


def test_create_plan_retains_all_tasks_when_skip_existing_false(tmp_path, monkeypatch):
    task1 = BenchmarkTask(
        task_id="task-1",
        config_hash="hash1",
        config={"workers": 4},
        output_dir=str(tmp_path / "task1"),
        capacity_slots=4,
    )

    (tmp_path / "task1").mkdir(parents=True)
    manifest1 = {
        "schema_version": 1,
        "config_hash": "hash1",
        "phase": "complete",
        "status": "FEASIBLE",
    }
    with open(tmp_path / "task1" / "benchmark_manifest.json", "w") as f:
        json.dump(manifest1, f)
    with open(tmp_path / "task1" / "result.json", "w") as f:
        json.dump({"status": "FEASIBLE"}, f)

    sweep = SimulationSweep(
        name="test-no-skip",
        mode="run",
        execution=ExecutionConfig(output_dir=str(tmp_path), skip_existing=False),
    )
    monkeypatch.setattr(
        SimulationSweep, "from_yaml", classmethod(lambda cls, path: sweep)
    )
    monkeypatch.setattr(SimulationSweep, "generate_tasks", lambda self: [task1])

    plan = create_plan(str(tmp_path / "sweep.yaml"))
    assert len(plan.tasks) == 1
    assert plan.tasks[0].task_id == "task-1"


def test_create_plan_all_skipped_handles_zero_allocations(tmp_path, monkeypatch):
    task1 = BenchmarkTask(
        task_id="task-1",
        config_hash="hash1",
        config={"workers": 4},
        output_dir=str(tmp_path / "task1"),
        capacity_slots=4,
    )

    (tmp_path / "task1").mkdir(parents=True)
    manifest1 = {
        "schema_version": 1,
        "config_hash": "hash1",
        "phase": "complete",
        "status": "FEASIBLE",
    }
    with open(tmp_path / "task1" / "benchmark_manifest.json", "w") as f:
        json.dump(manifest1, f)
    with open(tmp_path / "task1" / "result.json", "w") as f:
        json.dump({"status": "FEASIBLE"}, f)

    sweep = SimulationSweep(
        name="test-all-skipped",
        mode="run",
        execution=ExecutionConfig(output_dir=str(tmp_path), skip_existing=True),
    )
    monkeypatch.setattr(
        SimulationSweep, "from_yaml", classmethod(lambda cls, path: sweep)
    )
    monkeypatch.setattr(SimulationSweep, "generate_tasks", lambda self: [task1])

    plan = create_plan(str(tmp_path / "sweep.yaml"))
    assert len(plan.tasks) == 0
    plan_path = write_plan(plan)
    script_path = write_submission_script(plan, plan_path)
    assert script_path.is_file()
    assert "no jobs to submit" in script_path.read_text(encoding="utf-8")


def test_run_benchmark_worker_skips_when_valid_existing_result(tmp_path, monkeypatch):
    task = BenchmarkTask(
        task_id="task-1",
        config_hash="hash1",
        config={"workers": 4},
        output_dir=str(tmp_path / "task1"),
        capacity_slots=4,
    )
    (tmp_path / "task1").mkdir(parents=True)
    manifest1 = {
        "schema_version": 1,
        "config_hash": "hash1",
        "phase": "complete",
        "status": "FEASIBLE",
    }
    with open(tmp_path / "task1" / "benchmark_manifest.json", "w") as f:
        json.dump(manifest1, f)
    with open(tmp_path / "task1" / "result.json", "w") as f:
        json.dump({"status": "FEASIBLE"}, f)

    plan = SlurmPlan(
        name="test-worker-skip",
        created_at="2026-08-17T00:00:00+00:00",
        source_config=str(tmp_path / "sweep.yaml"),
        project_root=str(tmp_path),
        output_root=str(tmp_path),
        metrics=MetricsRunConfig(),
        tasks=[task],
        execution=ExecutionConfig(skip_existing=True),
    )
    plan_path = write_plan(plan)

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")

    def fail_opt(t, execution=None):
        pytest.fail("Optimization phase should have been skipped!")

    monkeypatch.setattr("benchmark.slurm.run_optimization_phase", fail_opt)

    exit_code = run_benchmark_worker(str(plan_path), allocation_index=0)
    assert exit_code == 0
