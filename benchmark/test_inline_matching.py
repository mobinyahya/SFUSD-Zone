"""Per-run matching inside the optimization allocation, and its sweep reports."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from assignment.generated_zones import (
    GENERATED_ZONE_FILENAME,
    SKIP_MARKER_FILENAME,
    write_generated_zones,
)
from assignment.run_custom_config import load_custom_config
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)
from benchmark.assignment import (
    MATCHING_COMPLETE_FILENAME,
    aggregate_matching_metrics,
    clear_assignment_targets,
    matching_complete,
    matching_dir,
    run_task_matching,
)
from benchmark.config import BenchmarkTask, MatchingRunConfig
from benchmark.results import aggregate_results
from benchmark.runner import MANIFEST_FILENAME, RESULT_FILENAME


ASSIGNMENT_CONFIG = Path(__file__).parents[1] / "assignment/configs/8-19-choice.yaml"
MATCHING_DATA = load_custom_config(ASSIGNMENT_CONFIG)["data"]


def _task(root: Path, name: str, *, policy: str | None = None) -> BenchmarkTask:
    config = {
        "workers": 1,
        "strategy": "single",
        "levels": ["BlockGroup_0"],
        "data": MATCHING_DATA,
    }
    if policy is not None:
        config["matching_policy"] = policy
    return BenchmarkTask(
        task_id=name,
        config_hash=f"{name}-hash",
        config=config,
        output_dir=str((root / name).resolve()),
        capacity_slots=1,
    )


def _write_evaluated_run(task: BenchmarkTask, *, eligible: bool = True) -> Path:
    run_dir = Path(task.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": task.task_id,
                "config_hash": task.config_hash,
                "phase": "complete",
                "status": "FEASIBLE",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / RESULT_FILENAME).write_text('{"status": "FEASIBLE"}', encoding="utf-8")
    if eligible:
        write_generated_zones({10: 0, 11: 1}, run_dir / GENERATED_ZONE_FILENAME)
    else:
        (run_dir / SKIP_MARKER_FILENAME).write_text("ineligible\n", encoding="utf-8")
    return run_dir


@pytest.fixture
def optimization_scenario(monkeypatch):
    """The matching config's data has no optimization group to read."""
    monkeypatch.setattr(
        "loaders.load_scenario",
        lambda data: SimpleNamespace(filter=lambda group, key: "2020"),
    )


def _matching(tmp_path: Path) -> MatchingRunConfig:
    config = tmp_path / "matching.yaml"
    config.write_text(ASSIGNMENT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    return MatchingRunConfig(enabled=True, config=str(config))


def test_a_run_is_matched_into_its_own_folder(
    tmp_path, monkeypatch, optimization_scenario
):
    matching = _matching(tmp_path)
    task = _task(tmp_path, "abc", policy="ctip")
    run_dir = _write_evaluated_run(task)
    stale = matching_dir(run_dir) / "stale.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("old\n", encoding="utf-8")
    calls = []

    def fake_run(config, targets, *, assignment_folder, workers):
        calls.append((config, targets, assignment_folder, workers))
        # The folder is cleared before matching starts.
        assert not stale.exists()

    monkeypatch.setattr("benchmark.assignment.run_generated_zone_assignments", fake_run)

    assert not matching_complete(task, matching)
    assert run_task_matching(task, matching) == "complete"

    [(config, targets, folder, workers)] = calls
    assert config == matching.config
    assert folder == run_dir / "matching"
    assert workers == 1
    assert [target["id"] for target in targets] == ["abc-root"]
    # A run optimized for one policy is matched under it alone.
    assert targets[0]["policies"] == ["ctip"]
    marker = json.loads((folder / MATCHING_COMPLETE_FILENAME).read_text())
    assert marker["status"] == "complete"
    assert marker["targets"] == ["abc-root"]
    assert matching_complete(task, matching)

    # Another matching config, or another zoning of the same folder, is not.
    Path(matching.config).write_text("{changed: true}\n", encoding="utf-8")
    assert not matching_complete(task, matching)
    other = BenchmarkTask(**{**task.__dict__, "config_hash": "other-hash"})
    assert not matching_complete(other, _matching(tmp_path))


def test_an_ineligible_run_is_recorded_as_skipped(
    tmp_path, monkeypatch, optimization_scenario
):
    matching = _matching(tmp_path)
    task = _task(tmp_path, "abc")
    _write_evaluated_run(task, eligible=False)
    monkeypatch.setattr(
        "benchmark.assignment.run_generated_zone_assignments",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("matched")),
    )

    assert run_task_matching(task, matching) == "skipped"
    assert matching_complete(task, matching)


def test_a_run_optimized_again_loses_its_matching(tmp_path):
    folder = matching_dir(tmp_path / "run")
    folder.mkdir(parents=True)
    (folder / MATCHING_COMPLETE_FILENAME).write_text("{}", encoding="utf-8")
    (folder / "aggregate_metrics").mkdir()

    clear_assignment_targets(tmp_path / "run")

    assert not folder.exists()


def _citywide(config_name: str, value: float, extra: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "config_name": [config_name],
            "Dissimilarity (FRL)": [value],
            "Label": ["a, quoted \"label\""],
        }
    )
    if extra:
        frame["Alternative # of GE programs above +10% district FRL"] = [3]
    return frame


def _program(config_name: str, base: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "config_name": [config_name] * 3,
            "program_id": ["GE-1", "GE-2", "SB-3"],
            "school_id": [1, 2, 3],
            "school_name": ["A", "B", "C"],
            "school_category": ["Attendance"] * 3,
            "program_type": ["GE", "GE", "SB"],
            "seats": [base, base + 0.1, None],
        }
    )


def _write_run_reports(root: Path, folder: str, config_name: str, frames, status):
    run_dir = root / folder
    run_dir.mkdir(parents=True)
    (run_dir / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    reports = MarketGenerator.combine_aggregate_metric_reports([frames])
    MarketGenerator.write_aggregate_metric_reports(matching_dir(run_dir), reports)
    if status is not None:
        (matching_dir(run_dir) / MATCHING_COMPLETE_FILENAME).write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )
    return reports


def test_sweep_reports_are_the_runs_reports_rebuilt_from_their_folders(tmp_path):
    runs = {
        # Folder order differs from config_name order.
        "a-run": ("ccc-root:pol/variant", 0.3),
        "b-run": ("aaa-root:pol/variant", 1 / 3),
        "c-run": ("bbb-root:pol/variant", 2.5e-17),
    }
    expected = []
    for folder, (config_name, value) in runs.items():
        frames = {
            "citywide": _citywide(config_name, value, extra=folder == "c-run"),
            "program": _program(config_name, value),
        }
        expected.append(
            _write_run_reports(tmp_path, folder, config_name, frames, "complete")
        )
    # Unmatched, skipped, and interrupted runs are left out.
    _write_run_reports(
        tmp_path,
        "d-run",
        "ddd-root:pol/variant",
        {"citywide": _citywide("ddd-root:pol/variant", 9.0)},
        None,
    )
    (tmp_path / "e-run").mkdir()
    (tmp_path / "e-run" / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")

    counts = aggregate_matching_metrics(tmp_path, "assignments")

    assert counts == {"program": 9, "citywide": 3}
    output = tmp_path / "assignments" / "aggregate_metrics"
    # Exactly what writing every run's reports together at once would write.
    reference = tmp_path / "reference"
    MarketGenerator.write_aggregate_metric_reports(
        reference, MarketGenerator.combine_aggregate_metric_reports(expected)
    )
    for filename in ("metrics_citywide.csv", "metrics_by_program.csv"):
        assert (output / filename).read_bytes() == (
            reference / "aggregate_metrics" / filename
        ).read_bytes()
    citywide = pd.read_csv(output / "metrics_citywide.csv")
    assert citywide["config_name"].tolist() == [
        "aaa-root:pol/variant",
        "bbb-root:pol/variant",
        "ccc-root:pol/variant",
    ]
    assert not (output / "metrics_by_zone.csv").exists()

    # Rebuilding is idempotent, and follows the run folders.
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    aggregate_matching_metrics(tmp_path, "assignments")
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first
    (matching_dir(tmp_path / "a-run") / MATCHING_COMPLETE_FILENAME).unlink()
    assert aggregate_matching_metrics(tmp_path, "assignments") == {
        "program": 6,
        "citywide": 2,
    }
    assert "ccc-root" not in (output / "metrics_citywide.csv").read_text()
    assert sorted(path.name for path in (tmp_path / "assignments").iterdir()) == [
        "aggregate_metrics"
    ]


def test_aggregation_rebuilds_matching_reports_only_when_asked(tmp_path):
    _write_run_reports(
        tmp_path,
        "run",
        "aaa-root:pol/variant",
        {"citywide": _citywide("aaa-root:pol/variant", 1.0)},
        "complete",
    )

    aggregate_results(str(tmp_path), summary_csv="summary.csv")
    assert not (tmp_path / "assignments").exists()

    aggregate_results(str(tmp_path), summary_csv="summary.csv", matching_dir="assignments")
    assert (tmp_path / "assignments/aggregate_metrics/metrics_citywide.csv").is_file()
    assert (tmp_path / "summary.csv").is_file()
