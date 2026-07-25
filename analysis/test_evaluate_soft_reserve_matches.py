import sys
from concurrent.futures import Future
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import evaluate_soft_reserve_matches as soft_reserves  # noqa: E402


class ImmediateExecutor:
    instances = []

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_matching_run_task_keeps_iterations_serial(monkeypatch, tmp_path):
    task = soft_reserves.MatchingRunTask(
        label="run",
        source_run=str(tmp_path / "source"),
        destination=str(tmp_path / "destination"),
    )
    solution = object()
    manifest = {"task_id": "task"}
    calls = []

    monkeypatch.setattr(
        soft_reserves,
        "load_saved_area_solution",
        lambda source_run: (solution, manifest),
    )

    def fake_ensure_matching_output(**kwargs):
        calls.append(kwargs)
        return Path(task.destination)

    monkeypatch.setattr(
        soft_reserves,
        "ensure_matching_output",
        fake_ensure_matching_output,
    )

    output = soft_reserves.run_matching_task(task, {"policy": "value"})

    assert output == task.destination
    assert calls[0]["workers"] == 1
    assert calls[0]["solution"] is solution
    assert calls[0]["manifest"] is manifest


def test_matching_tasks_parallelize_runs_and_preserve_order(monkeypatch, tmp_path):
    ImmediateExecutor.instances = []
    tasks = [
        soft_reserves.MatchingRunTask(
            label=label,
            source_run=str(tmp_path / label),
            destination=str(tmp_path / "matches" / label),
        )
        for label in ["first", "bad", "last"]
    ]

    def fake_run_matching_task(task, policy):
        assert policy == {"policy": "value"}
        if task.label == "bad":
            raise RuntimeError("failed")
        return task.destination

    monkeypatch.setattr(soft_reserves, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(
        soft_reserves,
        "run_matching_task",
        fake_run_matching_task,
    )

    outputs, failures = soft_reserves.run_matching_tasks(
        tasks,
        {"policy": "value"},
        workers=8,
    )

    assert ImmediateExecutor.instances[0].max_workers == 3
    assert list(outputs) == ["first", "last"]
    assert outputs["first"] == Path(tasks[0].destination)
    assert failures == 1


def test_evaluation_tasks_parallelize_runs_and_preserve_order(monkeypatch, tmp_path):
    ImmediateExecutor.instances = []
    matching_outputs = {
        "first": tmp_path / "first",
        "bad": tmp_path / "bad",
        "last": tmp_path / "last",
    }

    def fake_evaluate_run(output_root, new_ctip_path):
        assert new_ctip_path == tmp_path / "ctip.npy"
        if output_root.name == "bad":
            raise RuntimeError("failed")
        return pd.Series({"metric": len(output_root.name)})

    monkeypatch.setattr(soft_reserves, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(soft_reserves, "evaluate_run", fake_evaluate_run)

    metrics, failures = soft_reserves.run_evaluation_tasks(
        matching_outputs,
        tmp_path / "ctip.npy",
        workers=8,
    )

    assert ImmediateExecutor.instances[0].max_workers == 3
    assert list(metrics) == ["first", "last"]
    assert metrics["first"]["metric"] == 5
    assert failures == 1


def test_load_base_zone_solution_reads_row_per_zone_csv(tmp_path):
    zone_path = tmp_path / "zones.csv"
    zone_path.write_text("100,101\n200,201,202\n", encoding="utf-8")

    solution = soft_reserves.load_base_zone_solution(zone_path)

    assert solution.level.name == "BlockGroup_0"
    assert solution.assignment == {100: 0, 101: 0, 200: 1, 201: 1, 202: 1}
    assert solution.feasible


def test_parse_base_zones_preserves_labels_and_paths(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.touch()
    second.touch()

    parsed = soft_reserves.parse_base_zones(
        [f"small_zones_1={first}", f"medium_zones={second}"]
    )

    assert parsed == [
        ("small_zones_1", first.resolve()),
        ("medium_zones", second.resolve()),
    ]
