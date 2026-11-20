"""Dry tests for benchmark Slurm planning and phase workers."""

from __future__ import annotations

from dataclasses import replace
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time

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
    PHASE_TIMINGS_FILENAME,
    RESULT_FILENAME,
    TaskResult,
    evaluate_optimization_task,
    run_optimization_phase,
)
from benchmark.slurm import (
    CLAIMED,
    DONE,
    HELD,
    MAX_SLURM_JOBS,
    PLAN_SCHEMA_VERSION,
    STALE_CLAIM_SECONDS,
    ClaimStore,
    TaskQueue,
    claim_order,
    execution_units,
    expected_task_seconds,
    MAX_SUBMITTED_SLURM_JOBS,
    topup_plan,
    SlurmAllocation,
    SlurmPlan,
    _build_parser,
    _plan_allocations,
    claims_dir,
    create_plan,
    init_worker_process,
    live_slurm_jobs as _real_live_slurm_jobs,
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


@pytest.fixture(autouse=True)
def _no_cluster(monkeypatch):
    """No test reads squeue or warms real data unless it asks to."""
    monkeypatch.setenv("BENCHMARK_SLURM_WARM", "0")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("benchmark.slurm.live_slurm_jobs", lambda run=None: None)
    monkeypatch.setattr("benchmark.slurm.WORKER_START_STAGGER_SECONDS", 0.0)


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

    assert script.count("--parsable -A soal -p soal") == 1
    assert script.count("--ntasks=1") == 1
    assert "--cpus-per-task=4" in script
    assert script.count("--mem-per-cpu=4G") == 1
    assert "--export=ALL,OMP_NUM_THREADS=1" in script
    assert "if job=$(sbatch" in script
    assert f"--comment=sfusd-plan:{plan.plan_id}" in script
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
    assert script.count("--parsable -A soal") == 1
    assert "--upstream-job-id" not in script
    assert "--dependency" not in script
    assert "--mem-per-cpu=4G" in script
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
    assert calls[0][0][:6] == ["sbatch", "--parsable", "-A", "soal", "-p", "soal"]
    assert "--cpus-per-task=4" in calls[0][0]
    assert not any(value.startswith("--dependency") for value in calls[0][0])
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
    # Every allocation claims from the one shared queue.
    assert {a.task_indices for a in plan.allocations} == {tuple(range(1000))}
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
        assert "--mem-per-cpu=4G" in command
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


def _limit_refusal(command):
    return subprocess.CalledProcessError(
        1,
        command,
        output="",
        stderr="sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
        "sbatch: error: Batch job submission failed: Job violates accounting/QOS "
        "policy (job submit limit, user's size and/or time limits)\n",
    )


def test_submit_stops_at_a_job_limit_and_keeps_what_was_accepted(
    tmp_path, monkeypatch, capsys
):
    plan = _large_matching_plan(tmp_path, monkeypatch)
    accepted = iter(range(100, 108))

    def fake_run(command, **kwargs):
        assert command[0] == "sbatch"
        job = next(accepted, None)
        if job is None:
            raise _limit_refusal(command)
        return subprocess.CompletedProcess(command, 0, stdout=f"{job}\n", stderr="")

    submitted = submit_plan(plan, tmp_path / "plan.json", run=fake_run)

    assert [job.job_id for job in submitted] == [str(i) for i in range(100, 108)]
    assert [job.allocation_index for job in submitted] == list(range(8))
    assert "4 allocation(s) not submitted" in capsys.readouterr().err


def test_submit_raises_other_sbatch_errors(tmp_path):
    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, output="", stderr="boom")

    with pytest.raises(subprocess.CalledProcessError):
        submit_plan(_plan(tmp_path), tmp_path / "plan.json", run=fake_run)


def test_topup_submits_only_the_plan_allocations_not_live(tmp_path, monkeypatch):
    plan = _large_matching_plan(tmp_path, monkeypatch)
    tag = f"sfusd-plan:{plan.plan_id}"
    squeue = "\n".join(
        [
            f"500|benchmark-0|{tag}",
            f"501|benchmark-3|{tag}",
            "502|benchmark-1|sfusd-plan:someotherplan",
            "503|unrelated|",
        ]
    )
    sbatched = []

    def fake_run(command, **kwargs):
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout=squeue, stderr="")
        name = next(value for value in command if value.startswith("--job-name="))
        sbatched.append(int(name.rsplit("-", 1)[1]))
        return subprocess.CompletedProcess(
            command, 0, stdout=f"{600 + len(sbatched)}\n", stderr=""
        )

    live, submitted = topup_plan(plan, tmp_path / "plan.json", run=fake_run)

    assert live == {0: "500", 3: "501"}
    assert sbatched == [i for i in range(12) if i not in (0, 3)]
    assert [job.allocation_index for job in submitted] == sbatched


def test_generated_script_submits_past_a_refused_allocation(tmp_path, monkeypatch):
    directory = tmp_path / "bin"
    directory.mkdir()
    counter = tmp_path / "count"
    sbatch = directory / "sbatch"
    # Accept the first two jobs, refuse the third, accept the rest.
    sbatch.write_text(
        "#!/bin/sh\n"
        f"n=$(cat '{counter}' 2>/dev/null || echo 0); n=$((n + 1)); echo $n > '{counter}'\n"
        'if [ "$n" = 3 ]; then echo "sbatch: error: QOSMaxSubmitJobPerUserLimit" >&2; exit 1; fi\n'
        'echo "$((700 + n))"\n'
    )
    sbatch.chmod(0o755)
    plan = _large_matching_plan(tmp_path, monkeypatch)

    result = subprocess.run(
        ["bash"],
        input=submission_script(plan, tmp_path / "plan.json"),
        env={**os.environ, "PATH": f"{directory}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.count("allocation ") == 11
    assert "allocation 2 not submitted" in result.stderr
    assert "Run this script again" in result.stderr


def test_rerunning_the_script_submits_only_allocations_not_live(tmp_path, monkeypatch):
    plan = _large_matching_plan(tmp_path, monkeypatch)
    directory = tmp_path / "bin"
    directory.mkdir()
    log = tmp_path / "sbatch.log"
    tag = f"sfusd-plan:{plan.plan_id}"
    squeue = directory / "squeue"
    squeue.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' 'benchmark-0|{tag}' 'benchmark-5|{tag}' "
        "'benchmark-1|sfusd-plan:other' 'benchmark-2|'\n"
    )
    squeue.chmod(0o755)
    sbatch = directory / "sbatch"
    sbatch.write_text(
        "#!/bin/sh\n"
        f"for a in \"$@\"; do case $a in --job-name=*) echo $a >> '{log}';; esac; done\n"
        "echo 900\n"
    )
    sbatch.chmod(0o755)

    result = subprocess.run(
        ["bash"],
        input=submission_script(plan, tmp_path / "plan.json"),
        env={**os.environ, "PATH": f"{directory}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "allocation 0 already live" in result.stdout
    assert "allocation 5 already live" in result.stdout
    submitted = log.read_text().split()
    assert submitted == [
        f"--job-name=benchmark-{i}" for i in range(12) if i not in (0, 5)
    ]


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


def _fresh_claims(plan):
    """Forget earlier runs' claims, as a regenerated plan would."""
    import shutil

    shutil.rmtree(claims_dir(plan), ignore_errors=True)


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
    fused_kwargs = []

    def fake_fused(task, **kwargs):
        fused_kwargs.append(kwargs)
        events.append(("optimize+evaluate", task.task_id))
        return TaskResult(task.task_id, task.output_dir, "ERROR")

    monkeypatch.setattr("benchmark.slurm.run_optimize_and_evaluate", fake_fused)
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
    monkeypatch.setattr("benchmark.slurm.ProcessPoolExecutor", _ImmediateExecutor)

    # A saved task ERROR must not fail the allocation.
    assert run_benchmark_worker("plan.json", 0) == 0
    # Optimize and evaluate share one process per task.
    assert sorted(events) == [
        ("optimize+evaluate", "abc123"),
        ("optimize+evaluate", "second"),
    ]
    # One pool: 8 CPUs / 4 per task, a fresh process for every unit, each
    # process filtering loader warnings.
    created = _ImmediateExecutor.created
    assert len(created) == 1
    context = created[0].pop("mp_context")
    assert context.get_start_method() == "forkserver"
    assert created == [
        {"max_workers": 2, "max_tasks_per_child": 1, "initializer": init_worker_process}
    ]
    assert all(kwargs["visualization"] == plan.visualization for kwargs in fused_kwargs)
    assert len(aggregate_calls) == 1
    assert aggregate_calls[0][1]["matching_dir"] is None

    # The same plan's done claims are final: a second run finds no work, and
    # with nothing new to report it leaves the sweep's aggregation alone.
    events.clear()
    assert run_benchmark_worker("plan.json", 0) == 0
    assert events == []
    assert len(aggregate_calls) == 1

    def crashing(task, **kwargs):
        raise RuntimeError("worker crash")

    _fresh_claims(plan)
    monkeypatch.setattr("benchmark.slurm.run_optimize_and_evaluate", crashing)
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
        for index in range(4)
    ]
    matching = MatchingRunConfig(enabled=True, config=str(assignment_config))
    plan = replace(
        base, tasks=tasks, matching=matching, execution=ExecutionConfig()
    )
    # Task 1 is optimized and evaluated already: it is only matched.
    _write_complete_run(tasks[1])
    # Task 2 is optimized but not evaluated (a crash between phases): it is
    # evaluated standalone, then matched.
    _write_complete_run(tasks[2], phase="optimization")
    # Task 3 is complete and matched: nothing is left.
    _write_complete_run(tasks[3])
    _write_matching_marker(tasks[3], matching)
    monkeypatch.setattr("benchmark.slurm.load_plan", lambda path: plan)
    monkeypatch.setattr("benchmark.slurm.ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    events = []

    def fake_fused(task, **kwargs):
        assert kwargs["matching"] == matching
        events.append(("optimize+evaluate", task.task_id))
        return TaskResult(task.task_id, task.output_dir, "FEASIBLE")

    def fake_evaluate(task, **kwargs):
        assert kwargs["matching"] == matching
        # A standalone evaluate reloads what optimization saved.
        assert "solutions" not in kwargs and "dataset" not in kwargs
        events.append(("evaluate", task.task_id))
        return TaskResult(task.task_id, task.output_dir, "FEASIBLE")

    def fake_match(task, matching_config):
        assert matching_config == matching
        events.append(("match", task.task_id))
        return "complete"

    monkeypatch.setattr("benchmark.slurm.run_optimize_and_evaluate", fake_fused)
    monkeypatch.setattr("benchmark.slurm.evaluate_optimization_task", fake_evaluate)
    monkeypatch.setattr("benchmark.slurm.run_task_matching", fake_match)
    aggregate_calls = []
    monkeypatch.setattr(
        "benchmark.slurm.aggregate_results",
        lambda *args, **kwargs: aggregate_calls.append((args, kwargs)),
    )

    assert run_benchmark_worker("plan.json", 0) == 0
    by_task = {}
    for phase, task_id in events:
        by_task.setdefault(task_id, []).append(phase)
    assert by_task == {
        "task-0": ["optimize+evaluate", "match"],
        "task-1": ["match"],
        "task-2": ["evaluate", "match"],
    }
    assert aggregate_calls[-1][1]["matching_dir"] == "assignments"
    # Each phase is timed in the run's phase_timings.json.
    timings = json.loads(
        (Path(tasks[1].output_dir) / PHASE_TIMINGS_FILENAME).read_text()
    )
    assert set(timings["phases"]) == {"match"}

    # One CPU: a task's next phase runs before the next task starts.
    events.clear()
    _fresh_claims(plan)
    for task in tasks:
        _write_complete_run(task, status="ERROR")
    allocation = SlurmAllocation("benchmark", (0, 1, 2, 3), 1, 1)
    monkeypatch.setattr(
        "benchmark.slurm._allocation_at", lambda plan, index: allocation
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert run_benchmark_worker("plan.json", 0) == 0
    assert events == [
        (phase, f"task-{index}")
        for index in range(4)
        for phase in ("optimize+evaluate", "match")
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
    assert {item.task_indices for item in allocations} == {tuple(range(100))}
    assert (
        submission_script(plan, tmp_path / "plan.json").count("--parsable -A soal")
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


# ---------------------------------------------------------------------------
# The shared claim queue.


def _queue_plan(tmp_path, count=6, **config):
    tasks = [
        BenchmarkTask(
            task_id=f"task-{index}",
            config_hash=f"hash-{index}",
            config={"workers": 1, **config},
            output_dir=str((tmp_path / f"run-{index}").resolve()),
            capacity_slots=1,
        )
        for index in range(count)
    ]
    return replace(_plan(tmp_path), tasks=tasks)


def _store(plan, job_id, *, live_jobs=None, allocation_index=0, **kwargs):
    return ClaimStore(
        claims_dir(plan),
        plan_id=plan.plan_id,
        allocation_index=allocation_index,
        live_jobs=live_jobs,
        job_id=job_id,
        **kwargs,
    )


def _drain_queue(plan_path, job_id, out_path):
    """One simulated allocation: claim until the queue is empty."""
    plan = load_plan(plan_path)
    store = _store(plan, job_id, live_jobs=frozenset({"1001", "1002"}))
    queue = TaskQueue(plan, range(len(plan.tasks)), store)
    claimed = []
    while (index := queue.claim_next()) is not None:
        claimed.append(plan.tasks[index].task_id)
        time.sleep(0.001)
        store.mark_done(plan.tasks[index].task_id, "complete")
    Path(out_path).write_text(json.dumps(claimed), encoding="utf-8")


def test_two_allocations_racing_claim_every_task_exactly_once(tmp_path):
    import multiprocessing

    plan = _queue_plan(tmp_path, count=150)
    plan_path = write_plan(plan)
    context = multiprocessing.get_context("spawn")
    outputs = [tmp_path / "a.json", tmp_path / "b.json"]
    processes = [
        context.Process(target=_drain_queue, args=(str(plan_path), job, str(out)))
        for job, out in zip(("1001", "1002"), outputs)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(120)
        assert process.exitcode == 0
    first, second = (json.loads(path.read_text()) for path in outputs)
    assert not set(first) & set(second)
    assert sorted(first + second) == sorted(task.task_id for task in plan.tasks)


def test_claims_are_atomic_and_record_their_owner(tmp_path):
    plan = _queue_plan(tmp_path)
    first = _store(plan, "1001", live_jobs=frozenset({"1001", "1002"}))
    second = _store(plan, "1002", live_jobs=frozenset({"1001", "1002"}))

    assert first.try_claim("task-0") == CLAIMED
    assert second.try_claim("task-0") == HELD
    claim = json.loads((claims_dir(plan) / "task-0.0").read_text())
    assert claim["job_id"] == "1001"
    assert claim["status"] == "running"
    assert claim["plan_id"] == plan.plan_id
    assert {"host", "pid", "claimed_at", "allocation_index"} <= set(claim)

    first.mark_done("task-0", "complete")
    assert second.try_claim("task-0") == DONE
    # A regenerated plan reclaims what an earlier plan finished.
    regenerated = replace(plan, created_at="2026-11-19T00:00:00+00:00")
    third = _store(regenerated, "1003", live_jobs=frozenset({"1003"}))
    assert third.try_claim("task-0") == CLAIMED
    assert (claims_dir(plan) / "task-0.1").exists()


def test_stale_claims_are_recovered(tmp_path, monkeypatch):
    plan = _queue_plan(tmp_path)
    dead = _store(plan, "1001", live_jobs=frozenset({"1001"}))
    for index in range(4):
        assert dead.try_claim(f"task-{index}") == CLAIMED

    # A job squeue no longer lists, its claims older than the squeue read.
    past = time.time() - 10
    for index in range(4):
        os.utime(claims_dir(plan) / f"task-{index}.0", (past, past))
    resubmitted = _store(plan, "2002", live_jobs=frozenset({"2002"}))
    assert resubmitted.try_claim("task-0") == CLAIMED
    assert json.loads((claims_dir(plan) / "task-0.1").read_text())["job_id"] == "2002"
    # Taken over once: the next claimant sees the new owner, which is live.
    third = _store(plan, "3003", live_jobs=frozenset({"2002", "3003"}))
    assert third.try_claim("task-0") == HELD

    # A live job's claim is held, until its heartbeat is too old.
    alive = _store(plan, "2002", live_jobs=frozenset({"1001", "2002"}))
    assert alive.try_claim("task-1") == HELD
    old = time.time() - STALE_CLAIM_SECONDS - 60
    os.utime(claims_dir(plan) / "task-1.0", (old, old))
    assert alive.try_claim("task-1") == CLAIMED

    # Without squeue only the heartbeat counts, and a heartbeat keeps a
    # claim fresh.
    blind = _store(plan, "4004", live_jobs=None)
    assert blind.try_claim("task-2") == HELD
    dead.heartbeat()
    assert blind.try_claim("task-3") == HELD

    # A requeued job's earlier run (same job id, another process) is stale.
    with monkeypatch.context() as patch:
        patch.setattr(os, "getpid", lambda: 999999)
        requeued = _store(plan, "1001", live_jobs=frozenset({"1001"}))
    assert requeued.try_claim("task-2") == CLAIMED

    # A local run's claim on this host whose process is gone is stale.
    local = _store(plan, "local-x-1", live_jobs=None)
    assert local.try_claim("task-4") == CLAIMED
    path = claims_dir(plan) / "task-4.0"
    record = json.loads(path.read_text())
    record["pid"] = 2**22 + 12345  # No such process.
    path.write_text(json.dumps(record))
    assert _store(plan, "local-y-2", live_jobs=None).try_claim("task-4") == CLAIMED


def test_queue_recovers_a_dead_allocations_tasks_once_dry(tmp_path):
    plan = _queue_plan(tmp_path, count=3)
    other = _store(plan, "1001", live_jobs=None)
    assert other.try_claim("task-1") == CLAIMED
    now = [0.0]
    store = _store(plan, "2002", live_jobs=None)
    queue = TaskQueue(
        plan, range(3), store, recovery_seconds=100, clock=lambda: now[0]
    )

    assert [queue.claim_next(), queue.claim_next(), queue.claim_next()] == [
        0,
        2,
        None,
    ]
    # The other allocation dies; its heartbeat ages.
    old = time.time() - STALE_CLAIM_SECONDS - 60
    os.utime(claims_dir(plan) / "task-1.0", (old, old))
    now[0] = 50.0
    assert queue.claim_next() is None  # Throttled.
    assert queue.claim_next(force_recovery=True) == 1
    assert queue.claim_next(force_recovery=True) is None


def _keyed_task(tmp_path, name, seconds, data, policy):
    return BenchmarkTask(
        task_id=name,
        config_hash=f"{name}-hash",
        config={
            "workers": 1,
            "data": {"scenario": data},
            "matching_policy": policy,
            "steps": [{"level": "Block_0", "time_limit": seconds}],
        },
        output_dir=str(tmp_path / name),
        capacity_slots=1,
    )


def test_claim_order_is_longest_first_with_markets_adjacent(tmp_path):
    tasks = [
        _keyed_task(tmp_path, "aa-1", 60, "d1", "p1"),
        _keyed_task(tmp_path, "map-d1p1-a", 1800, "d1", "p1"),
        _keyed_task(tmp_path, "map-d1p2-a", 1800, "d1", "p2"),
        _keyed_task(tmp_path, "map-d1p1-b", 1800, "d1", "p1"),
        _keyed_task(tmp_path, "aa-2", 60, "d1", "p2"),
        _keyed_task(tmp_path, "map-d2p1-a", 1800, "d2", "p1"),
        _keyed_task(tmp_path, "map-d1p2-b", 1800, "d1", "p2"),
    ]
    ordered = [item.task_id for item in claim_order(tasks)]
    assert ordered == [
        "map-d1p1-a",
        "map-d1p1-b",
        "map-d1p2-a",
        "map-d1p2-b",
        "map-d2p1-a",
        "aa-1",
        "aa-2",
    ]
    assert expected_task_seconds({"solve_time_limits": [30, 60]}) == 60
    assert (
        expected_task_seconds({"strategy": "recursive", "solve_time_limits": [30, 60]})
        == 90
    )

    plan = replace(_plan(tmp_path), tasks=claim_order(tasks))
    # Another allocation already took the first of d1/p1.
    assert _store(plan, "1001").try_claim("map-d1p1-a") == CLAIMED
    queue = TaskQueue(plan, range(len(plan.tasks)), _store(plan, "2002"))
    claimed = []
    while (index := queue.claim_next()) is not None:
        claimed.append(plan.tasks[index].task_id)
    assert claimed == [
        "map-d1p1-b",
        "map-d1p2-a",
        "map-d1p2-b",
        "map-d2p1-a",
        "aa-1",
        "aa-2",
    ]


def test_queue_prefers_a_loaded_market_over_plan_order(tmp_path):
    tasks = [
        _keyed_task(tmp_path, name, 60, data, "p")
        for name, data in (("a1", "A"), ("b1", "B"), ("a2", "A"), ("b2", "B"))
    ]
    # Plan order interleaves the markets (claim_order would not), so the
    # preference is what keeps an allocation on its market.
    plan = replace(_plan(tmp_path), tasks=tasks)
    assert _store(plan, "1001").try_claim("a1") == CLAIMED
    queue = TaskQueue(plan, range(4), _store(plan, "2002"))
    assert [plan.tasks[queue.claim_next()].task_id for _ in range(3)] == [
        "b1",
        "b2",
        "a2",
    ]


def test_execution_units_fuse_optimize_and_evaluate():
    assert execution_units(("optimize", "evaluate", "match")) == (
        "optimize+evaluate",
        "match",
    )
    assert execution_units(("optimize", "evaluate")) == ("optimize+evaluate",)
    assert execution_units(("evaluate", "match")) == ("evaluate", "match")
    assert execution_units(("match",)) == ("match",)
    assert execution_units(()) == ()


def test_plan_schema_is_versioned_and_keeps_claim_order(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    payload = plan.to_dict()
    assert payload["schema_version"] == PLAN_SCHEMA_VERSION == 7
    with pytest.raises(ValueError, match="Unsupported Slurm plan schema 6"):
        SlurmPlan.from_dict({**payload, "schema_version": 6})

    def task(name, seconds):
        return BenchmarkTask(
            task_id=name,
            config_hash=f"{name}-hash",
            config={"workers": 1, "solve_time_limits": [seconds]},
            output_dir=str(tmp_path / name),
            capacity_slots=1,
        )

    sweep = SimulationSweep(
        execution=ExecutionConfig(output_dir=str(tmp_path), skip_existing=False)
    )
    monkeypatch.setattr(
        SimulationSweep, "from_yaml", classmethod(lambda cls, path: sweep)
    )
    monkeypatch.setattr(
        SimulationSweep,
        "generate_tasks",
        lambda self: [task("short", 60), task("long", 600)],
    )
    created = create_plan(str(tmp_path / "sweep.yaml"))
    assert [item.task_id for item in created.tasks] == ["long", "short"]
    loaded = load_plan(str(write_plan(created)))
    assert [item.task_id for item in loaded.tasks] == ["long", "short"]
    assert loaded.plan_id == created.plan_id


def test_allocations_share_one_queue_and_finish_it_together(tmp_path, monkeypatch):
    """A second allocation starting mid-run takes the rest of the queue."""
    plan = _queue_plan(tmp_path, count=5)
    monkeypatch.setattr("benchmark.slurm.load_plan", lambda path: plan)
    monkeypatch.setattr("benchmark.slurm.ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr("benchmark.slurm.aggregate_results", lambda *a, **k: None)
    allocation = SlurmAllocation("benchmark", tuple(range(5)), 1, 1)
    monkeypatch.setattr("benchmark.slurm._allocation_at", lambda plan, i: allocation)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    live = frozenset({"1001", "1002"})
    ran = []

    def second_fused(task, **kwargs):
        ran.append(("second", task.task_id))
        return TaskResult(task.task_id, task.output_dir, "FEASIBLE")

    def first_fused(task, **kwargs):
        ran.append(("first", task.task_id))
        if len(ran) == 1:
            monkeypatch.setattr(
                "benchmark.slurm.run_optimize_and_evaluate", second_fused
            )
            second = _store(plan, "1002", live_jobs=live, allocation_index=1)
            assert run_benchmark_worker("plan.json", 1, claims=second) == 0
        return TaskResult(task.task_id, task.output_dir, "FEASIBLE")

    monkeypatch.setattr("benchmark.slurm.run_optimize_and_evaluate", first_fused)
    first = _store(plan, "1001", live_jobs=live)
    assert run_benchmark_worker("plan.json", 0, claims=first) == 0
    assert ran == [("first", "task-0")] + [
        ("second", f"task-{index}") for index in range(1, 5)
    ]
    for index in range(5):
        claim = json.loads((claims_dir(plan) / f"task-{index}.0").read_text())
        assert claim["status"] == "done"


def test_first_wave_waits_for_its_dataset_warm_up(tmp_path, monkeypatch):
    plan = _queue_plan(tmp_path, count=3, data={"scenario": "x"}, levels=["Block_0"])
    monkeypatch.setenv("BENCHMARK_SLURM_WARM", "1")
    monkeypatch.setattr("benchmark.slurm.load_plan", lambda path: plan)
    monkeypatch.setattr("benchmark.slurm.ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr("benchmark.slurm.aggregate_results", lambda *a, **k: None)
    allocation = SlurmAllocation("benchmark", (0, 1, 2), 2, 1)
    monkeypatch.setattr("benchmark.slurm._allocation_at", lambda plan, i: allocation)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    events = []

    def fake_warm(spec):
        # The warm-up peeks at the tasks it warms for; it claims none.
        assert not list(claims_dir(plan).glob("task-*"))
        events.append(("warm", tuple(spec.levels), tuple(spec.policies)))
        return {"key": spec.key, "seconds": 0.0, "levels": {}, "markets": {}}

    monkeypatch.setattr("benchmark.slurm.warm_dataset", fake_warm)
    monkeypatch.setattr(
        "benchmark.slurm.run_optimize_and_evaluate",
        lambda task, **kwargs: (
            events.append(("task", task.task_id))
            or TaskResult(task.task_id, task.output_dir, "FEASIBLE")
        ),
    )
    assert run_benchmark_worker("plan.json", 0) == 0
    # One warm-up for the first wave's one dataset, before any of its tasks.
    assert events[0] == ("warm", ("Block_0",), (None,))
    assert sorted(events[1:]) == [("task", f"task-{i}") for i in range(3)]


def test_warm_specs_group_tasks_by_dataset():
    from benchmark.warmup import market_key, warm_specs

    configs = [
        {
            "data": {"s": 1},
            "matching_policy": "p1",
            "steps": [{"level": "Block_1"}, {"level": "Block_0"}],
        },
        {"data": {"s": 1}, "matching_policy": "p2", "levels": ["Block_0"]},
        {"data": {"s": 2}, "matching_policy": "p1", "levels": ["Block_0"]},
    ]
    specs = warm_specs(configs)
    assert [(spec.levels, spec.policies) for spec in specs] == [
        (["Block_1", "Block_0"], ["p1", "p2"]),
        (["Block_0"], ["p1"]),
    ]
    assert market_key(configs[0]) != market_key(configs[1])
    assert market_key(configs[0]) == market_key({**configs[0], "seed": 3})


def test_worker_processes_show_each_loader_warning_once():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.simplefilter("always")
        init_worker_process()
        namespace = {"__name__": "loaders.tables"}
        exec("import warnings\ndef f():\n    warnings.warn('w', UserWarning)", namespace)
        for _ in range(3):
            namespace["f"]()
    assert len(caught) == 1


def test_fused_optimize_evaluate_matches_separate_phases_and_is_timed(
    tmp_path, monkeypatch, shared_cache_root
):
    from benchmark.runner import (
        load_phase_timings,
        record_phase_timing,
        run_optimize_and_evaluate,
    )

    problem = make_grid_problem(3, 3)
    problem.level = LevelSpec("Block", 0)

    class FakeStrategy:
        def run(self, dataset, solver):
            return [
                ZoneSolution(
                    problem=problem,
                    assignment=_assignment(),
                    status="FEASIBLE",
                    objective=3.0,
                    wall_time=1.5,
                )
            ]

    datasets = []

    def make_dataset(self):
        datasets.append(self)
        return FakeDataset(problem)

    monkeypatch.setattr(OptimizationConfig, "make_dataset", make_dataset)
    monkeypatch.setattr(
        OptimizationConfig, "make_solver", lambda self, output_dir=None: object()
    )
    monkeypatch.setattr(
        OptimizationConfig, "make_strategy", lambda self: FakeStrategy()
    )
    config = OptimizationConfig(
        levels=["Block_0"],
        strategy="single",
        solver="cp_int",
        workers=1,
        frl_dev=1.0,
        racial_dev=1.0,
        overage=5.0,
        data={
            "scenario": "legacy",
            "overrides": {"roots": {"cache": str(shared_cache_root)}},
        },
    )

    def task(name):
        return BenchmarkTask(
            task_id="same-id",
            config_hash="same-hash",
            config=optimization_config_to_dict(config),
            output_dir=str(tmp_path / name),
            capacity_slots=1,
        )

    fused, separate = task("fused"), task("separate")
    result = run_optimize_and_evaluate(fused, compute_stage_metrics=True)
    assert result.status == "FEASIBLE"
    # The fused evaluation reuses the optimization's dataset.
    assert len(datasets) == 1
    run_optimization_phase(separate)
    evaluate_optimization_task(separate, compute_stage_metrics=True)
    assert len(datasets) == 3  # A standalone evaluate builds its own.

    def comparable(name):
        text = (tmp_path / name / RESULT_FILENAME).read_text()
        return json.loads(text.replace(str(tmp_path / name), "<run>"))

    assert comparable("fused") == comparable("separate")
    manifest = json.loads((tmp_path / "fused" / MANIFEST_FILENAME).read_text())
    assert manifest["phase"] == "complete"

    timings = load_phase_timings(fused.output_dir)
    assert timings["schema_version"] == 1
    assert set(timings["phases"]) == {"optimize", "evaluate"}
    assert timings["phases"]["optimize"]["fused_with"] is None
    evaluate = timings["phases"]["evaluate"]
    assert evaluate["fused_with"] == "optimize"
    assert evaluate["status"] == "FEASIBLE"
    for record in timings["phases"].values():
        assert record["wall_seconds"] >= 0
        assert record["cpu_seconds"] >= 0
        assert record["peak_rss_mb"] > 0
        assert record["started_at"] <= record["completed_at"]
        assert record["pid"] == os.getpid()
    # A new optimization starts a new record.
    record_phase_timing(fused.output_dir, dict(timings["phases"]["optimize"]))
    assert list(load_phase_timings(fused.output_dir)["phases"]) == ["optimize"]


def test_phase_timing_prints_one_line(tmp_path, capsys):
    from benchmark.runner import timed_phase

    task = replace(_plan(tmp_path).tasks[0], output_dir=str(tmp_path / "run"))
    with timed_phase(task, "match") as timing:
        timing["status"] = "COMPLETE"
    with pytest.raises(RuntimeError):
        with timed_phase(task, "evaluate"):
            raise RuntimeError("boom")
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 2
    assert lines[0].startswith("phase task=abc123 phase=match status=COMPLETE wall=")
    assert "status=CRASHED" in lines[1]
    phases = json.loads((tmp_path / "run" / PHASE_TIMINGS_FILENAME).read_text())
    assert set(phases["phases"]) == {"match", "evaluate"}


# ---------------------------------------------------------------------------
# Lazy claiming, the not-started guard, squeue refresh, and the forkserver.


def _thread_executor(slots):
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=slots)


def _fast_loop(monkeypatch):
    monkeypatch.setattr("benchmark.slurm.POLL_SECONDS", 0.02)
    monkeypatch.setattr("benchmark.slurm.IDLE_POLL_SECONDS", 0.02)


def _one_allocation(monkeypatch, plan, cpus):
    monkeypatch.setattr("benchmark.slurm.load_plan", lambda path: plan)
    monkeypatch.setattr("benchmark.slurm.aggregate_results", lambda *a, **k: None)
    allocation = SlurmAllocation("benchmark", tuple(range(len(plan.tasks))), cpus, 1)
    monkeypatch.setattr("benchmark.slurm._allocation_at", lambda plan, i: allocation)


def test_lazy_claiming_never_holds_more_claims_than_running_slots(
    tmp_path, monkeypatch
):
    import threading

    plan = _queue_plan(tmp_path, count=9)
    _fast_loop(monkeypatch)
    _one_allocation(monkeypatch, plan, 2)
    store = _store(plan, "1001")
    observed, lock = [], threading.Lock()
    original = store.try_claim

    def counting_claim(task_id):
        state = original(task_id)
        observed.append(len(store.held))
        return state

    store.try_claim = counting_claim

    def fused(task, **kwargs):
        with lock:
            observed.append(len(store.held))
        time.sleep(0.02)
        return TaskResult(task.task_id, task.output_dir, "FEASIBLE")

    monkeypatch.setattr("benchmark.slurm.run_optimize_and_evaluate", fused)
    assert (
        run_benchmark_worker(
            "plan.json", 0, claims=store, make_executor=_thread_executor
        )
        == 0
    )
    assert observed and max(observed) <= 2  # Two slots, never a whole wave.
    assert len(list(claims_dir(plan).glob("task-*"))) == 9
    for index in range(9):
        claim = json.loads((claims_dir(plan) / f"task-{index}.0").read_text())
        assert claim["status"] == "done" and claim["started"] is True


def _worker_process(plan_path, job_id, out_path):
    """One allocation in its own process, its tasks faked, on threads."""
    import benchmark.slurm as slurm

    os.environ["BENCHMARK_SLURM_WARM"] = "0"
    plan = slurm.load_plan(plan_path)
    slurm.POLL_SECONDS = 0.01
    slurm.aggregate_results = lambda *a, **k: None
    allocation = slurm.SlurmAllocation(
        "benchmark", tuple(range(len(plan.tasks))), 3, 1
    )
    slurm._allocation_at = lambda plan, index: allocation
    store = _store(plan, job_id, live_jobs=frozenset({"1001", "1002"}))
    ran, most = [], [0]

    def fused(task, **kwargs):
        most[0] = max(most[0], len(store.held))
        ran.append(task.task_id)
        time.sleep(0.002)
        return TaskResult(task.task_id, task.output_dir, "FEASIBLE")

    slurm.run_optimize_and_evaluate = fused
    code = slurm.run_benchmark_worker(
        plan_path, 0, claims=store, make_executor=_thread_executor
    )
    Path(out_path).write_text(
        json.dumps({"code": code, "ran": ran, "most_held": most[0]}),
        encoding="utf-8",
    )


def test_two_racing_worker_processes_cover_the_plan_without_overlap(tmp_path):
    import multiprocessing

    plan = _queue_plan(tmp_path, count=90)
    plan_path = write_plan(plan)
    context = multiprocessing.get_context("spawn")
    outputs = [tmp_path / "a.json", tmp_path / "b.json"]
    processes = [
        context.Process(target=_worker_process, args=(str(plan_path), job, str(out)))
        for job, out in zip(("1001", "1002"), outputs)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(180)
        assert process.exitcode == 0
    first, second = (json.loads(path.read_text()) for path in outputs)
    assert first["code"] == second["code"] == 0
    assert not set(first["ran"]) & set(second["ran"])
    assert sorted(first["ran"] + second["ran"]) == sorted(
        task.task_id for task in plan.tasks
    )
    assert first["most_held"] <= 3 and second["most_held"] <= 3


class _StalledExecutor:
    """A node whose new processes do not start until ``recovered`` is set."""

    def __init__(self, slots, recovered):
        from concurrent.futures import ThreadPoolExecutor

        self.pool = ThreadPoolExecutor(max_workers=slots)
        self.recovered = recovered

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.pool.shutdown(wait=True)

    def submit(self, function, *args):
        def stalled():
            self.recovered.wait(30)
            return function(*args)

        return self.pool.submit(stalled)


def test_unstarted_tasks_are_released_and_run_once_elsewhere(tmp_path, monkeypatch):
    import threading

    plan = _queue_plan(tmp_path, count=4)
    _fast_loop(monkeypatch)
    _one_allocation(monkeypatch, plan, 2)
    ran, lock = [], threading.Lock()

    def fused(task, **kwargs):
        with lock:
            ran.append(task.task_id)
        return TaskResult(task.task_id, task.output_dir, "FEASIBLE")

    monkeypatch.setattr("benchmark.slurm.run_optimize_and_evaluate", fused)
    recovered = threading.Event()
    stuck = _store(plan, "1001")
    codes = {}

    def run_stuck():
        codes["stuck"] = run_benchmark_worker(
            "plan.json",
            0,
            claims=stuck,
            make_executor=lambda slots: _StalledExecutor(slots, recovered),
            start_timeout=0.2,
        )

    worker = threading.Thread(target=run_stuck)
    worker.start()
    try:
        deadline = time.time() + 20
        released = []
        while time.time() < deadline and len(released) < 2:
            released = [
                path
                for path in claims_dir(plan).glob("task-*.0")
                if json.loads(path.read_text()).get("status") == "released"
            ]
            time.sleep(0.02)
        assert len(released) == 2
        # Its two stalled slots stay occupied: it holds nothing and claims
        # nothing new while its processes cannot start.
        time.sleep(0.3)
        assert stuck.held == {}
        assert len(list(claims_dir(plan).glob("task-*"))) == 2
        record = json.loads(released[0].read_text())
        assert record["started"] is False and "did not start" in record["reason"]

        # A healthy allocation takes the released tasks and the rest.
        healthy = _store(plan, "1002")
        assert (
            run_benchmark_worker(
                "plan.json", 0, claims=healthy, make_executor=_ImmediateExecutor
            )
            == 0
        )
        assert sorted(ran) == [f"task-{i}" for i in range(4)]
    finally:
        recovered.set()  # The node recovers: its late processes see the release.
        worker.join(30)
    assert codes == {"stuck": 0}
    # Every task ran exactly once: the stalled node's late processes did not.
    assert sorted(ran) == [f"task-{i}" for i in range(4)]
    for path in released:
        successor = path.with_suffix(".1")
        assert json.loads(successor.read_text())["status"] == "done"


def test_unit_entry_runs_only_if_it_takes_the_start_token(tmp_path):
    from benchmark.slurm import ABANDONED, _take_token, _unit_entry

    calls = []
    token = tmp_path / "1.start"
    assert _take_token(token, {"abandoned": True})
    assert (
        _unit_entry(str(token), None, None, "match", 0.0, calls.append, 1)
        == ABANDONED
    )
    assert calls == []
    other = tmp_path / "2.start"
    assert _unit_entry(str(other), None, None, "match", 0.0, calls.append, 2) is None
    assert calls == [2]
    assert not _take_token(other, {"abandoned": True})  # Too late to abandon.


def _squeue(monkeypatch, jobs):
    """Fake squeue: ``jobs`` (a list the test may change) are live."""
    calls = []

    def fake(run=None):
        calls.append(time.time())
        return frozenset(jobs)

    monkeypatch.setattr("benchmark.slurm.live_slurm_jobs", fake)
    return calls


def test_a_job_cancelled_mid_run_frees_its_claims_at_the_next_recheck(
    tmp_path, monkeypatch
):
    plan = _queue_plan(tmp_path, count=2)
    jobs = ["1001", "2002"]
    calls = _squeue(monkeypatch, jobs)
    cancelled = _store(plan, "1001", live_jobs=frozenset(jobs))
    assert cancelled.try_claim("task-0") == CLAIMED
    assert cancelled.try_claim("task-1") == CLAIMED
    clock = [0.0]
    survivor = _store(plan, "2002", live_jobs=frozenset(jobs), clock=lambda: clock[0])
    queue = TaskQueue(plan, range(2), survivor, clock=lambda: clock[0])
    assert queue.claim_next() is None  # Both held by a live job.
    assert calls == []  # The start-of-allocation read is under a minute old.

    jobs.remove("1001")  # scancel: its heartbeats stop.
    past = time.time() - 5
    for index in range(2):
        os.utime(claims_dir(plan) / f"task-{index}.0", (past, past))
    clock[0] = 61.0
    assert queue.claim_next(force_recovery=True) == 0
    assert queue.claim_next(force_recovery=True) == 1
    # One squeue read served both rechecks: at most one a minute.
    assert len(calls) == 1
    assert queue.claim_next(force_recovery=True) is None
    assert len(calls) == 1


def test_squeue_reads_only_live_job_states():
    output = (
        "101 101 RUNNING\n102 102 COMPLETING\n103 99_3 PENDING\n104 104 CANCELLED\n"
    )

    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=output)

    jobs = _real_live_slurm_jobs(run)
    assert {"101", "103", "99_3", "99"} <= jobs
    assert "102" not in jobs and "104" not in jobs


def test_idle_allocation_waits_for_a_dead_jobs_claims_before_exiting(
    tmp_path, monkeypatch
):
    plan = _queue_plan(tmp_path, count=2)
    _fast_loop(monkeypatch)
    _one_allocation(monkeypatch, plan, 1)
    ran = []
    monkeypatch.setattr(
        "benchmark.slurm.run_optimize_and_evaluate",
        lambda task, **kwargs: (
            ran.append(task.task_id)
            or TaskResult(task.task_id, task.output_dir, "FEASIBLE")
        ),
    )
    jobs = ["2002", "3003"]
    _squeue(monkeypatch, jobs)
    assert _store(plan, "1001").try_claim("task-0") == CLAIMED
    assert _store(plan, "3003").try_claim("task-1") == CLAIMED
    # task-1: a live job that started it; not worth waiting for.
    mark = claims_dir(plan) / "task-1.0"
    mark.write_text(json.dumps({**json.loads(mark.read_text()), "started": True}))
    # task-0: job 1001 is gone from squeue, but its last heartbeat came after
    # the read, so it is not stale yet; it is at a later read.
    soon = time.time() + 0.5
    os.utime(claims_dir(plan) / "task-0.0", (soon, soon))
    survivor = _store(plan, "2002", live_jobs=frozenset(jobs), refresh_seconds=0.0)
    started = time.time()
    code = run_benchmark_worker(
        "plan.json", 0, claims=survivor, make_executor=_ImmediateExecutor
    )
    assert code == 0
    assert ran == ["task-0"]  # Waited, then recovered it; task-1 untouched.
    assert time.time() - started < 10
    assert json.loads(mark.read_text())["job_id"] == "3003"


def test_idle_wait_is_capped(tmp_path, monkeypatch):
    plan = _queue_plan(tmp_path, count=1)
    _fast_loop(monkeypatch)
    monkeypatch.setattr("benchmark.slurm.EXIT_WAIT_LIMIT_SECONDS", 0.3)
    _one_allocation(monkeypatch, plan, 1)
    _squeue(monkeypatch, ["2002"])
    assert _store(plan, "1001").try_claim("task-0") == CLAIMED
    future = time.time() + 3600  # A heartbeat that never ages: never stale.
    os.utime(claims_dir(plan) / "task-0.0", (future, future))
    survivor = _store(
        plan, "2002", live_jobs=frozenset({"2002"}), refresh_seconds=0.0
    )
    started = time.time()
    code = run_benchmark_worker(
        "plan.json", 0, claims=survivor, make_executor=_ImmediateExecutor
    )
    assert code == 0
    assert 0.3 <= time.time() - started < 10


def _pool_probe(seed):
    """What a phase process computes and sees; compared across start methods."""
    import warnings as warnings_module

    import networkx as nx
    import numpy as np
    from ortools.sat.python import cp_model
    from pyproj import Transformer

    rng = np.random.default_rng(seed)
    values = rng.integers(1, 50, size=12).tolist()
    model = cp_model.CpModel()
    picks = [model.NewBoolVar(f"x{i}") for i in range(len(values))]
    model.Add(sum(v * x for v, x in zip(values, picks)) <= 100)
    model.Maximize(sum(v * x for v, x in zip(values, picks)))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = seed
    solver.Solve(model)
    graph = nx.grid_2d_graph(5, 5)
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(-122.4, 37.7)
    once = [f for f in warnings_module.filters if f[0] == "once"]
    try:
        parent_threads = len(os.listdir(f"/proc/{os.getppid()}/task"))
    except OSError:
        parent_threads = None
    observed = {
        "knapsack": solver.ObjectiveValue(),
        "path": nx.shortest_path_length(graph, (0, 0), (4, 4)),
        "xy": (round(x, 3), round(y, 3)),
        "loader_warning_once": bool(once),
        "omp": os.environ.get("OMP_NUM_THREADS"),
    }
    return observed, parent_threads, os.getpid()


def test_forkserver_pool_matches_spawn_and_forks_from_a_quiet_server(monkeypatch):
    from benchmark.slurm import _make_executor

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    results = {}
    for method in ("spawn", "forkserver"):
        monkeypatch.setenv("BENCHMARK_SLURM_START_METHOD", method)
        with _make_executor(2) as executor:
            futures = [executor.submit(_pool_probe, seed) for seed in (1, 2, 3)]
            results[method] = [future.result() for future in futures]
    assert [r[0] for r in results["spawn"]] == [r[0] for r in results["forkserver"]]
    assert all(r[0]["loader_warning_once"] for r in results["forkserver"])
    assert all(r[0]["omp"] == "1" for r in results["forkserver"])
    # A fresh process per call, each forked from one single-threaded server.
    assert len({r[2] for r in results["forkserver"]}) == 3
    assert {r[1] for r in results["forkserver"]} == {1}
