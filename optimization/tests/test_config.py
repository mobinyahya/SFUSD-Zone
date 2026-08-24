from dataclasses import asdict, fields
from pathlib import Path

import pytest
import yaml

from optimization.config import OptimizationConfig


def test_weight_edges_defaults_false_and_requires_boolean():
    assert OptimizationConfig(levels=["BlockGroup_0"]).weight_edges is False
    assert (
        OptimizationConfig(levels=["BlockGroup_0"], weight_edges=True).weight_edges
        is True
    )

    with pytest.raises(ValueError, match="weight_edges must be a Boolean"):
        OptimizationConfig(levels=["BlockGroup_0"], weight_edges=1)


def test_enumerated_solutions_defaults_disabled_and_is_passed_to_single_strategy():
    default = OptimizationConfig(levels=["BlockGroup_0"])
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="cp_bool",
        strategy="single",
        enumerated_solutions=7,
        seed=13,
    )

    assert default.enumerated_solutions == -1
    strategy = config.make_strategy()
    assert strategy.options["enumerated_solutions"] == 7
    assert strategy.options["seed"] == 13


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_enumerated_solutions_requires_an_integer(value):
    with pytest.raises(ValueError, match="enumerated_solutions must be an integer"):
        OptimizationConfig(levels=["BlockGroup_0"], enumerated_solutions=value)


def test_enumerated_solutions_rejects_incompatible_solver_and_strategy():
    with pytest.raises(ValueError, match="solver='cp_bool' or 'cp_int'"):
        OptimizationConfig(
            levels=["BlockGroup_0"],
            solver="mip",
            enumerated_solutions=2,
        )
    with pytest.raises(ValueError, match="strategy='single'"):
        OptimizationConfig(
            levels=["BlockGroup_0"],
            solver="cp_int",
            strategy="recursive",
            enumerated_solutions=2,
        )


@pytest.mark.parametrize(
    ("old_key", "value"),
    [
        ("years", [21]),
        ("population_type", "All"),
        ("drop_optout", False),
        ("capacity_scenario", "B"),
        ("new_schools", False),
        ("include_k8", True),
        ("remove_city_wide", True),
        ("graphs_dir", "/tmp/graphs"),
    ],
)
def test_old_optimization_data_keys_are_rejected(tmp_path, old_key, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({old_key: value}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config keys"):
        OptimizationConfig.from_yaml(str(path))


@pytest.mark.parametrize(
    "old_filter",
    [
        "grade",
        "participation",
        "population",
        "capacity",
        "mission_bay",
        "school_id_aliases",
        "drop_optout",
        "source_family",
    ],
)
def test_old_optimization_filter_names_are_rejected(old_filter):
    with pytest.raises(ValueError, match=rf"optimization filter.*{old_filter}"):
        OptimizationConfig(
            levels=["BlockGroup_0"],
            data={
                "scenario": "legacy",
                "overrides": {"filters": {"optimization": {old_filter: "unused"}}},
            },
        )


def test_data_field_is_strict_and_private_scenario_is_not_serialized():
    config = OptimizationConfig(levels=["BlockGroup_0"])

    assert [field.name for field in fields(config)].count("data") == 1
    snapshot = asdict(config)
    assert snapshot["data"] == {"scenario": "legacy", "overrides": {}}
    assert "_data_scenario" not in snapshot
    assert config.data_scenario is config.data_scenario

    with pytest.raises(ValueError, match="Unknown run configuration keys"):
        OptimizationConfig(
            levels=["BlockGroup_0"],
            data={"scenario": "legacy", "overrides": {}, "extra": True},
        )


def test_scenario_backed_properties_are_read_only():
    config = OptimizationConfig(levels=["BlockGroup_0"])

    assert config.years == ("1415", "1516", "1617", "1718", "1819", "2122", "2223")
    assert config.grades == ("KG",)
    assert config.student_population == "enrolled"
    assert config.rounds == "all"
    assert config.special_programs == "include"
    assert config.program_population == "GE"
    assert config.capacity_scenario == "programs"
    assert config.include_k8 is False
    assert config.include_citywide is False
    assert config.include_mission_bay is True
    assert config.frl_estimate is None
    assert config.outside_district_students == "ignore"

    with pytest.raises(AttributeError):
        config.years = ("2122",)


def test_example_uses_central_2021_through_2024_selectors():
    path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert not {
        "years",
        "population_type",
        "drop_optout",
        "capacity_scenario",
        "new_schools",
        "include_k8",
        "remove_city_wide",
        "graphs_dir",
    } & set(raw)
    overrides = raw["data"]["overrides"]
    assert "sources" not in overrides
    assert overrides["filters"]["optimization"] == {
        "years": ["2122", "2223", "2324"],
        "grades": ["KG"],
        "student_population": "enrolled",
        "rounds": "all",
        "special_programs": "include",
        "program_population": "GE",
        "capacity_scenario": "programs",
        "include_k8": False,
        "include_citywide": False,
        "include_mission_bay": True,
        "geography_vintage": "2020",
        "frl_estimate": "updated_2526",
        "outside_district_students": "ignore",
    }


@pytest.mark.parametrize("student_population", ["applicant", "enrolled"])
def test_config_resolves_annual_student_population_from_registry(
    student_population,
):
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        data={
            "scenario": "legacy",
            "overrides": {
                "filters": {
                    "optimization": {
                        "years": ["2324", "2122"],
                        "student_population": student_population,
                    }
                }
            },
        },
    )

    assert config.years == ("2324", "2122")
    assert [
        source.catalog_id
        for source in config.data_scenario.sources("optimization.students")
    ] == [
        f"optimization.students.{student_population}.2324",
        f"optimization.students.{student_population}.2122",
    ]
    assert config.data_scenario.source("optimization.schools").catalog_id == (
        "optimization.schools.current"
    )
    assert config.data_scenario.source("optimization.programs").catalog_id == (
        "assignment.programs.2324"
    )
    assert config.data_scenario.source("optimization.capacity").catalog_id == (
        "capacity.stanford.scenarios_abcd"
    )


def test_from_yaml_anchors_data_paths_and_preserves_them_in_snapshot(tmp_path):
    config_dir = tmp_path / "configs"
    scenario_dir = config_dir / "scenarios"
    scenario_dir.mkdir(parents=True)
    scenario_path = scenario_dir / "custom.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "id": "custom",
                "sources": {"optimization.students": {"path": "scenario-students.csv"}},
                "filters": {
                    "optimization": {
                        "years": ["2324"],
                        "grades": ["KG"],
                        "student_population": "enrolled",
                        "rounds": "all",
                        "special_programs": "include",
                        "program_population": "GE",
                        "capacity_scenario": "A",
                        "include_k8": False,
                        "include_citywide": False,
                        "include_mission_bay": True,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = config_dir / "optimization.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "levels": ["BlockGroup_0"],
                "data": {
                    "scenario": "scenarios/custom.yaml",
                    "overrides": {
                        "roots": {"data": "data", "cache": "cache"},
                        "sources": {
                            "optimization.students": {
                                "path": "inputs/students.csv",
                                "companions": ["inputs/students.meta"],
                            }
                        },
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = OptimizationConfig.from_yaml(str(config_path))
    snapshot = asdict(config)

    assert snapshot["data"]["scenario"] == str(scenario_path.resolve())
    assert snapshot["data"]["overrides"]["roots"] == {
        "data": str((config_dir / "data").resolve()),
        "cache": str((config_dir / "cache").resolve()),
    }
    source = snapshot["data"]["overrides"]["sources"]["optimization.students"]
    assert source == {
        "path": str((config_dir / "inputs/students.csv").resolve()),
        "companions": [str((config_dir / "inputs/students.meta").resolve())],
    }
    assert config.data_scenario.source("optimization.students").path == Path(
        source["path"]
    )
