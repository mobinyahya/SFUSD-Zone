"""The sweep's end-of-run aggregation: run once, and fast, without changing output."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from benchmark import slurm
from benchmark.assignment import _concatenate_metric_csvs
from benchmark.results import discover_run_dirs
from benchmark.test_slurm import _queue_plan, _store


def _row_by_row(paths, output):
    """The concatenation as it was before the byte-copy path, for reference."""
    sources = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            first = next(reader, None)
        key = "" if first is None else first[header.index("config_name")]
        sources.append((key, str(path), header))
    sources.sort()
    columns = list(dict.fromkeys(c for _k, _p, header in sources for c in header))
    with output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(columns)
        for _key, path, header in sources:
            positions = [header.index(c) if c in header else None for c in columns]
            with open(path, newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                next(reader)
                for row in reader:
                    writer.writerow(["" if i is None else row[i] for i in positions])


def _write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


def test_byte_copy_gives_the_row_by_row_output_exactly(tmp_path):
    paths = [
        # Plain files take the byte-copy path ...
        _write(tmp_path / "b.csv", "config_name,x,y\nb:1,0.1,2\nb:2,1e-20,\n"),
        _write(tmp_path / "a.csv", "config_name,x,y\na:1,3.25,NaN\na:2,-0.0,7\n"),
        # ... a quoted field and a different header take the row path.
        _write(tmp_path / "c.csv", 'config_name,x,y\nc:1,"1,5",4\n'),
        _write(tmp_path / "d.csv", "config_name,y,z\nd:1,5,6\n"),
        _write(tmp_path / "e.csv", "config_name,x,y\n"),
    ]
    fast, reference = tmp_path / "fast.csv", tmp_path / "reference.csv"

    rows = _concatenate_metric_csvs(paths, fast)
    _row_by_row(paths, reference)

    assert fast.read_bytes() == reference.read_bytes()
    assert rows == 6


def test_a_quoted_header_still_takes_the_byte_copy(tmp_path, monkeypatch):
    # Metric names such as "Income below 50000, CTIP" are quoted in the
    # header; the rows below them are plain and are copied as bytes.
    header = 'config_name,"Income below 50000, CTIP",x\n'
    paths = [
        _write(tmp_path / "a.csv", header + "a:1,0.5,1\n"),
        _write(tmp_path / "b.csv", header + "b:1,0.25,2\nb:2,,3\n"),
        _write(tmp_path / "c.csv", header + 'c:1,"1,2",3\n'),  # row path
    ]
    copied = []
    import benchmark.assignment as assignment

    original = assignment._copy_csv_body

    def spy(*args):
        result = original(*args)
        copied.append(result is not None)
        return result

    monkeypatch.setattr(assignment, "_copy_csv_body", spy)
    fast, reference = tmp_path / "fast.csv", tmp_path / "reference.csv"
    _concatenate_metric_csvs(paths, fast)
    _row_by_row(paths, reference)

    assert fast.read_bytes() == reference.read_bytes()
    assert copied == [True, True, False]


def test_byte_copy_still_rejects_interleaved_runs(tmp_path):
    import pytest

    paths = [
        _write(tmp_path / "a.csv", "config_name,x\na:1,1\nc:1,2\n"),
        _write(tmp_path / "b.csv", "config_name,x\nb:1,3\n"),
    ]
    with pytest.raises(ValueError, match="interleave"):
        _concatenate_metric_csvs(paths, tmp_path / "out.csv")


def test_run_discovery_does_not_descend_into_runs(tmp_path):
    run = tmp_path / "a" / "run"
    (run / "matching" / "deep").mkdir(parents=True)
    (run / "benchmark_manifest.json").write_text("{}")
    (run / "matching" / "deep" / "benchmark_manifest.json").write_text("{}")
    other = tmp_path / "b" / "run2"
    other.mkdir(parents=True)
    (other / "benchmark_manifest.json").write_text("{}")

    assert discover_run_dirs(str(tmp_path)) == [str(run), str(other)]


def test_only_the_last_allocation_aggregates(tmp_path, monkeypatch):
    plan = _queue_plan(tmp_path, count=2)
    calls = []
    monkeypatch.setattr(slurm, "_aggregate", lambda plan: calls.append(plan.name))
    live = frozenset({"1001", "1002"})
    first = _store(plan, "1001", live_jobs=live)
    second = _store(plan, "1002", live_jobs=live)
    monkeypatch.setattr(first, "refresh_live_jobs", lambda **_: True)
    monkeypatch.setattr(second, "refresh_live_jobs", lambda **_: True)
    assert first.try_claim("task-0") == slurm.CLAIMED
    assert second.try_claim("task-1") == slurm.CLAIMED
    first.mark_done("task-0", "complete")

    # task-1 is still running in the other allocation: leave the aggregation.
    assert slurm._aggregate_if_last(plan, first) is False
    assert calls == []

    second.mark_done("task-1", "complete")
    assert slurm._aggregate_if_last(plan, second) is True
    assert calls == [plan.name]


def test_an_allocation_with_new_results_aggregates_again(tmp_path, monkeypatch):
    plan = _queue_plan(tmp_path, count=2)
    calls = []
    monkeypatch.setattr(slurm, "_aggregate", lambda plan: calls.append(plan.name))
    live = frozenset({"1001", "1002"})
    first = _store(plan, "1001", live_jobs=live)
    monkeypatch.setattr(first, "refresh_live_jobs", lambda **_: True)
    first.try_claim("task-0")
    first.mark_done("task-0", "complete")
    assert slurm._aggregate_if_last(plan, first) is True

    idle = _store(plan, "1002", live_jobs=live)
    monkeypatch.setattr(idle, "refresh_live_jobs", lambda **_: True)
    assert slurm._aggregate_if_last(plan, idle) is False  # nothing new

    busy = _store(plan, "1002", live_jobs=live)
    monkeypatch.setattr(busy, "refresh_live_jobs", lambda **_: True)
    busy.try_claim("task-1")
    busy.mark_done("task-1", "complete")
    assert slurm._aggregate_if_last(plan, busy) is True
    assert calls == [plan.name, plan.name]


def test_a_failed_aggregation_can_be_retried(tmp_path, monkeypatch):
    import pytest

    plan = _queue_plan(tmp_path, count=1)

    def boom(plan):
        raise RuntimeError("disk full")

    monkeypatch.setattr(slurm, "_aggregate", boom)
    store = _store(plan, "1001", live_jobs=frozenset({"1001"}))
    monkeypatch.setattr(store, "refresh_live_jobs", lambda **_: True)
    with pytest.raises(RuntimeError):
        slurm._aggregate_if_last(plan, store)

    monkeypatch.setattr(slurm, "_aggregate", lambda plan: None)
    retry = _store(plan, "1002", live_jobs=frozenset({"1002"}))
    monkeypatch.setattr(retry, "refresh_live_jobs", lambda **_: True)
    assert slurm._aggregate_if_last(plan, retry) is True


def test_loader_warnings_show_once_across_processes_sharing_a_dir(tmp_path):
    script = textwrap.dedent(
        """
        import warnings
        warnings.simplefilter("default")
        from loaders.table_cache import loader_warn
        loader_warn("13 students rank no school")
        loader_warn("13 students rank no school")
        """
    )
    env = {**os.environ, "SFUSD_WARN_ONCE_DIR": str(tmp_path / "warn")}
    root = Path(__file__).resolve().parents[1]
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stderr
        for _ in range(3)
    ]

    assert sum(out.count("13 students rank no school") for out in outputs) == 1
