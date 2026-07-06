from Zone_Generation.benchmark.parallel import BatchResult, _print_progress
from Zone_Generation.benchmark.runner import TaskResult


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
