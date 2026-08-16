import sys
from pathlib import Path

import pandas as pd
import pytest
from loaders import load_scenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import evaluate_status_quo_matches as status_quo  # noqa: E402


def _write_assignment(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "studentno": [1],
            "programno": [1],
            "programcodes": ["GE"],
            "rank": [1],
        }
    ).to_csv(path, index=False)


def test_validate_assignment_output_returns_iteration_order(tmp_path):
    output = tmp_path / "status_quo" / "run"
    for iteration in reversed(range(status_quo.ITERATION_COUNT)):
        _write_assignment(output / f"assignment_iteration{iteration}.csv")

    paths = status_quo.validate_assignment_output(tmp_path, "status_quo")

    assert len(paths) == status_quo.ITERATION_COUNT
    assert paths[0].name.endswith("iteration0.csv")
    assert paths[-1].name.endswith("iteration24.csv")


def test_validate_assignment_output_rejects_incomplete_run(tmp_path):
    _write_assignment(tmp_path / "status_quo" / "run" / "assignment_iteration0.csv")

    with pytest.raises(ValueError, match="output is incomplete"):
        status_quo.validate_assignment_output(tmp_path, "status_quo")


def test_build_simulation_config_sets_requested_run(tmp_path):
    base = {
        "output_dir": "unused",
        "paths": {"assignment-folder": "old"},
        "subconfigs": ["old"],
        "iterations": {"start": 2, "end": 3},
    }

    result = status_quo.build_simulation_config(
        base, ["reserves", "baseline"], tmp_path
    )

    assert "output_dir" not in result
    assert result["paths"]["assignment-folder"] == str(tmp_path)
    assert result["subconfigs"] == ["reserves", "baseline"]
    assert result["iterations"] == {"start": 0, "end": 25}
    assert result["save-assignment"] is True
    assert base["paths"]["assignment-folder"] == "old"


def test_run_policy_injects_configurator_at_construction(tmp_path, monkeypatch):
    captured = {}
    label = "status_quo"

    class FakeMarketGenerator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def simulate(self):
            loader = captured["configurator"]
            assert loader.load_next_subconfig() is True
            for iteration in range(status_quo.ITERATION_COUNT):
                _write_assignment(
                    tmp_path
                    / label
                    / "run"
                    / f"assignment_iteration{iteration}.csv"
                )

    monkeypatch.setattr(status_quo, "MarketGenerator", FakeMarketGenerator)

    assignments = status_quo.run_policy(
        {"subconfigs": ["unused"], "save-assignment": True},
        label,
        {"policies": [label]},
        tmp_path,
    )

    assert len(assignments) == status_quo.ITERATION_COUNT
    assert captured["assignment_path"] == str(tmp_path)
    assert captured["write_config"] is False
    assert "config" not in captured
    assert not (tmp_path / "config.json").exists()


def test_write_metrics_csv_matches_reference_layout(tmp_path):
    output = status_quo.write_metrics_csv(
        {
            "status_quo+reserves_06frl": pd.Series({"metric one": 1.5}),
            "status_quo": pd.Series({"metric one": 2.5}),
        },
        tmp_path,
    )

    frame = pd.read_csv(output, index_col="metric")
    assert list(frame.columns) == ["status_quo+reserves_06frl", "status_quo"]
    assert frame.loc["metric one"].tolist() == [1.5, 2.5]


def test_evaluation_tasks_can_include_all_rounds(tmp_path):
    assignments = [tmp_path / "assignment.csv"]
    sources = {
        role: tmp_path / f"{role.rsplit('.', 1)[-1]}.csv"
        for role in (
            "assignment.students",
            "assignment.programs",
            "assignment.school_coordinates",
        )
    }
    for path in sources.values():
        path.touch()

    tasks = status_quo.evaluation_tasks(
        assignments,
        {
            "data": {
                "scenario": "legacy",
                "overrides": {
                    "sources": {
                        role: {
                            "path": str(path),
                            "classification": "internal",
                        }
                        for role, path in sources.items()
                    },
                    "filters": {
                        "assignment": {
                            "year": "2324",
                            "grades": ["KG"],
                            "student_population": "applicant",
                            "rounds": [1],
                            "special_programs": "exclude_any_special",
                            "capacity_profile": "status_quo",
                            "include_mission_bay": True,
                        }
                    },
                },
            }
        },
        None,
        first_round=False,
    )

    assert len(tasks) == 1
    assert load_scenario(tasks[0].data).filter("assignment", "rounds") == "all"
