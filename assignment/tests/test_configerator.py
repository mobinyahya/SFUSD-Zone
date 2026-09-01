import copy
from pathlib import Path

import pytest
import yaml

from assignment.student_assignment.configerator import Configerator
from assignment.student_assignment.definitions import CONFIGS_DIR


def _valid_config():
    config_root = Path(CONFIGS_DIR)
    config = yaml.safe_load((config_root / "base_config.yaml").read_text())
    config.update(yaml.safe_load((config_root / "local_path_config.yaml").read_text()))
    return config


def test_aggregate_metrics_export_is_disabled_by_default():
    assert _valid_config()["export-aggregate-metrics"] is False


def test_local_metrics_export_is_disabled_by_default():
    assert _valid_config()["export-local-metrics"] is False


def test_heatmap_export_is_disabled_by_default():
    assert _valid_config()["export_heatmaps"] is False


def test_assignment_reuse_is_enabled_by_default():
    assert _valid_config()["reuse_assignments"] is True


def test_aggregate_metrics_export_must_be_boolean():
    config = _valid_config()
    config["export-aggregate-metrics"] = "true"

    with pytest.raises(ValueError):
        Configerator.from_config(config)


def test_local_metrics_export_must_be_boolean():
    config = _valid_config()
    config["export-local-metrics"] = "true"

    with pytest.raises(ValueError):
        Configerator.from_config(config)


def test_heatmap_export_must_be_boolean():
    config = _valid_config()
    config["export_heatmaps"] = "true"

    with pytest.raises(ValueError):
        Configerator.from_config(config)


def test_local_metrics_require_aggregate_metrics():
    config = _valid_config()
    config["export-local-metrics"] = True

    with pytest.raises(
        ValueError,
        match="export-local-metrics requires export-aggregate-metrics",
    ):
        Configerator.from_config(config)


def test_assignment_reuse_must_be_boolean():
    config = _valid_config()
    config["reuse_assignments"] = "true"

    with pytest.raises(ValueError):
        Configerator.from_config(config)


def test_duplicate_subconfigs_are_rejected():
    config = _valid_config()
    config["subconfigs"] = ["status_quo", "status_quo"]

    with pytest.raises(ValueError, match="duplicate subconfigs"):
        Configerator.from_config(config)


@pytest.mark.parametrize(
    "iterations",
    [
        {"start": -1, "end": 1},
        {"start": 1, "end": 1},
        {"start": 2, "end": 1},
    ],
)
def test_invalid_iteration_ranges_are_rejected(iterations):
    config = _valid_config()
    config["iterations"] = iterations

    with pytest.raises(ValueError, match="0 <= start < end"):
        Configerator.from_config(config)


def test_from_config_creates_an_isolated_policy_loader(monkeypatch):
    config = _valid_config()
    config.update(
        {
            "subconfigs": ["status_quo"],
            "nested": {"value": "base"},
        }
    )
    configurator = Configerator.from_config(config)
    monkeypatch.setattr(configurator, "_validate_schema", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        configurator,
        "_load_yaml",
        lambda path: {"nested": {"value": "policy"}},
    )

    config["nested"]["value"] = "caller"
    assert configurator.config["nested"]["value"] == "base"
    assert configurator.original_config["nested"]["value"] == "base"

    assert configurator.load_next_subconfig() is True
    assert configurator.config["nested"]["value"] == "policy"
    assert configurator.config["data"] == configurator.original_config["data"]
    assert configurator.config["subconfig-name"] == "status_quo"
    assert configurator.load_next_subconfig() is False


@pytest.mark.parametrize("legacy_key", ["year", "grade", "remove-special-lps"])
def test_old_top_level_data_fields_are_rejected(legacy_key):
    config = _valid_config()
    config[legacy_key] = 23 if legacy_key == "year" else "KG"

    with pytest.raises(ValueError, match="forbidden top-level keys"):
        Configerator.from_config(config)


@pytest.mark.parametrize(
    "legacy_key",
    [
        "sfusd",
        "student-save",
        "student-data",
        "program-data",
        "school-data",
        "estimate-path",
        "zone-files",
        "citywide-or-lp-zones",
        "new-ctip-path",
        "new-ctip-blockgroup-path",
        "lotteries-path",
    ],
)
def test_old_input_path_fields_are_rejected(legacy_key):
    config = _valid_config()
    config["paths"][legacy_key] = {} if legacy_key.endswith("zones") else "old"

    with pytest.raises(ValueError, match="forbidden paths keys"):
        Configerator.from_config(config)


def test_file_and_in_memory_configs_receive_the_same_data_validation(
    monkeypatch,
):
    config = _valid_config()
    config["year"] = 23
    user = "_pytest_invalid_data"
    file_path = Path(CONFIGS_DIR) / f"{user}.config.yaml"
    file_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("SFUSD_ASSIGNMENT_CONFIG_USER", user)

    try:
        with pytest.raises(ValueError) as memory_error:
            Configerator.from_config(copy.deepcopy(config))
        Configerator.instance = None
        with pytest.raises(ValueError) as file_error:
            Configerator()
    finally:
        Configerator.instance = None
        file_path.unlink(missing_ok=True)

    assert str(file_error.value) == str(memory_error.value)


def test_custom_scenario_yaml_is_accepted(tmp_path):
    scenario_path = tmp_path / "custom.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "id": "custom-assignment",
                "sources": {},
                "filters": {
                    "assignment": {
                        "year": "2324",
                        "grades": ["KG"],
                        "student_population": "applicant",
                        "rounds": "all",
                        "special_programs": "include",
                        "capacity_profile": "default",
                        "include_mission_bay": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = _valid_config()
    config["data"] = {"scenario": str(scenario_path), "overrides": {}}

    configurator = Configerator.from_config(config)

    assert configurator.config["data"]["scenario"] == str(scenario_path)


def _write_custom_scenario(path):
    path.write_text(
        yaml.safe_dump(
            {
                "id": "relative-assignment",
                "sources": {},
                "filters": {
                    "assignment": {
                        "year": "2324",
                        "grades": ["KG"],
                        "student_population": "applicant",
                        "rounds": "all",
                        "special_programs": "include",
                        "capacity_profile": "default",
                        "include_mission_bay": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_file_config_anchors_all_relative_data_paths_to_yaml(tmp_path, monkeypatch):
    declared = tmp_path / "declared"
    declared.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _write_custom_scenario(declared / "scenario.yaml")
    config = _valid_config()
    config["data"] = {
        "scenario": "scenario.yaml",
        "overrides": {
            "roots": {"data": "data", "cache": "cache"},
            "sources": {
                "assignment.students": {
                    "path": "inputs/students.csv",
                    "classification": "restricted",
                }
            },
        },
    }
    config_path = declared / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.chdir(elsewhere)

    configurator = Configerator.from_path(config_path)

    data = configurator.config["data"]
    assert data["scenario"] == str((declared / "scenario.yaml").resolve())
    assert data["overrides"]["roots"] == {
        "data": str((declared / "data").resolve()),
        "cache": str((declared / "cache").resolve()),
    }
    assert data["overrides"]["sources"]["assignment.students"]["path"] == str(
        (declared / "inputs/students.csv").resolve()
    )


def test_programmatic_config_uses_cwd_when_no_declaring_path(tmp_path, monkeypatch):
    _write_custom_scenario(tmp_path / "scenario.yaml")
    config = _valid_config()
    config["data"] = {
        "scenario": "scenario.yaml",
        "overrides": {
            "roots": {"cache": "cache"},
            "sources": {
                "assignment.students": {"path": "students.csv"},
            },
        },
    }
    monkeypatch.chdir(tmp_path)

    configurator = Configerator.from_config(config)

    data = configurator.config["data"]
    assert data["scenario"] == str((tmp_path / "scenario.yaml").resolve())
    assert data["overrides"]["roots"]["cache"] == str((tmp_path / "cache").resolve())
    assert data["overrides"]["sources"]["assignment.students"]["path"] == str(
        (tmp_path / "students.csv").resolve()
    )


def test_assignment_execution_requires_exactly_one_grade():
    config = _valid_config()
    config["data"]["overrides"] = {"filters": {"assignment": {"grades": ["KG", "01"]}}}

    with pytest.raises(ValueError, match="exactly one grade"):
        Configerator.from_config(config)


def test_checked_in_executable_configs_keep_data_external():
    forbidden_paths = {
        "sfusd",
        "student-save",
        "student-data",
        "program-data",
        "school-data",
        "estimate-path",
        "zone-files",
        "citywide-or-lp-zones",
        "new-ctip-path",
        "new-ctip-blockgroup-path",
        "lotteries-path",
    }
    for path in Path(CONFIGS_DIR).rglob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, dict) or config.get("save-assignment") is not True:
            continue
        assert isinstance(config["data"]["scenario"], str), path
        assert set(config["data"]) == {"scenario", "overrides"}, path
        assert not {"year", "grade", "remove-special-lps"}.intersection(config), path
        assert not forbidden_paths.intersection(config.get("paths", {})), path
