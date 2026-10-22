from benchmark.config import (
    BenchmarkTask,
    ExecutionConfig,
    MatchingRunConfig,
    MetricsRunConfig,
)
from benchmark.parallel import BatchResult, _print_progress, run_tasks
from benchmark.runner import TaskResult


def _task_result(status: str, *, skipped: bool = False) -> TaskResult:
    return TaskResult(
        task_id=status,
        output_dir="",
        status=status,
        skipped=skipped,
    )


def test_batch_result_counts_solver_statuses() -> None:
    batch = BatchResult(total=7)
    for status in (
        "FEASIBLE",
        "OPTIMAL",
        "UNKNOWN",
        "INFEASIBLE",
        "TIMED_OUT",
        "ERROR",
    ):
        batch.add(_task_result(status))
    batch.add(_task_result("FEASIBLE", skipped=True))

    assert batch.feasible == 1
    assert batch.optimal == 1
    assert batch.unknown == 2
    assert batch.infeasible == 1
    assert batch.failed == 1
    assert batch.skipped == 1
    assert batch.successful == 5
    assert batch.completed == 7
    assert batch.status_count_summary(separator=", ") == (
        "num_feasible=1, num_optimal=1, num_unknown=2, "
        "num_infeasible=1, num_failed=1, num_skipped=1"
    )


def test_print_progress_reports_status_counts(capsys) -> None:
    batch = BatchResult(total=2)
    batch.add(_task_result("OPTIMAL"))
    batch.add(_task_result("INFEASIBLE"))

    _print_progress(batch)

    output = capsys.readouterr().out
    assert output == (
        "[2/2] num_feasible=0 num_optimal=1 num_unknown=0 "
        "num_infeasible=1 num_failed=0 num_skipped=0\n"
    )
    assert "ok=" not in output


def test_local_matching_evaluates_and_runs_only_the_sweep_targets(
    tmp_path, monkeypatch
):
    events = []
    task = BenchmarkTask(
        task_id="task",
        config_hash="hash",
        config={"workers": 2},
        output_dir=str(tmp_path / "run"),
        capacity_slots=1,
    )

    def prepare(tasks, pending, matching):
        events.append(("prepare", tasks, pending))

    def targets(tasks, matching, *, planned_stages):
        events.append(("targets", tasks, planned_stages))
        return [{"id": "task-root"}]

    def run_task(task, **kwargs):
        events.append(("benchmark", task.task_id, kwargs["matching"]))
        return TaskResult(task.task_id, task.output_dir, "FEASIBLE")

    def run_ready(targets, matching, output_root, *, workers):
        events.append(("matching", targets, output_root, workers))
        return len(targets)

    monkeypatch.setattr("benchmark.parallel.run_optimization_task", run_task)
    monkeypatch.setattr("benchmark.assignment.prepare_assignment_targets", prepare)
    monkeypatch.setattr("benchmark.assignment.assignment_targets", targets)
    monkeypatch.setattr("benchmark.assignment.run_ready_assignments", run_ready)
    matching = MatchingRunConfig(enabled=True, config="assignment.yaml")

    run_tasks(
        [task],
        execution=ExecutionConfig(
            output_dir=str(tmp_path),
            skip_existing=False,
            sequential=True,
        ),
        metrics=MetricsRunConfig(),
        matching=matching,
    )

    assert events == [
        ("prepare", [task], [task]),
        ("benchmark", "task", matching),
        ("targets", [task], False),
        ("matching", [{"id": "task-root"}], str(tmp_path), 2),
    ]
