from __future__ import annotations

import json
import tomllib
from importlib.resources import files
from pathlib import Path

import pytest
import yaml

from loaders.config import (
    BASE_CONFIG_PATH,
    PACKAGE_CONFIG_ROOT,
    REPOSITORY_ROOT,
    anchor_data_config,
    load_scenario,
)


def _write_scenario(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_config_deep_merge_and_root_override_precedence(tmp_path):
    environment_data = tmp_path / "environment-data"
    explicit_data = tmp_path / "explicit-data"
    environment_cache = tmp_path / "environment-cache"
    explicit_cache = tmp_path / "explicit-cache"
    explicit_data.mkdir()
    (explicit_data / "replacement.csv").write_text("value\n1\n", encoding="utf-8")

    scenario_path = tmp_path / "scenario.yaml"
    _write_scenario(
        scenario_path,
        {
            "id": "merge",
            "sources": {
                "test.input": {
                    "path": "original.csv",
                    "root": "data",
                    "classification": "restricted",
                }
            },
            "filters": {
                "assignment": {
                    "year": "1819",
                    "grades": ["KG"],
                    "student_population": "applicant",
                    "rounds": [1, 2],
                    "special_programs": "include",
                    "capacity_profile": "default",
                    "include_mission_bay": False,
                    "geography_vintage": "2010",
                }
            },
        },
    )
    scenario = load_scenario(
        {
            "scenario": str(scenario_path),
            "overrides": {
                "roots": {
                    "data": str(explicit_data),
                    "cache": str(explicit_cache),
                },
                "sources": {"test.input": {"path": "replacement.csv"}},
                "filters": {
                    "assignment": {
                        "rounds": [2],
                    }
                },
            },
        },
        environ={
            "SFUSD_DATA_ROOT": str(environment_data),
            "SFUSD_CACHE_ROOT": str(environment_cache),
        },
    )

    source = scenario.source("test.input")
    assert source.path == (explicit_data / "replacement.csv").resolve()
    assert source.classification == "restricted"
    assert scenario.cache_root == explicit_cache.resolve()
    assert scenario.filter("assignment", "rounds") == (2,)
    assert scenario.filter("assignment", "grades") == ("KG",)


def test_environment_roots_override_base_when_no_explicit_override(tmp_path):
    scenario_path = tmp_path / "scenario.yaml"
    _write_scenario(
        scenario_path,
        {
            "id": "environment",
            "sources": {"test.input": {"path": "input.csv", "root": "data"}},
            "filters": {},
        },
    )
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    scenario = load_scenario(
        {"scenario": str(scenario_path), "overrides": {}},
        environ={
            "SFUSD_DATA_ROOT": str(data_root),
            "SFUSD_CACHE_ROOT": str(cache_root),
        },
    )

    assert scenario.source("test.input").path == (data_root / "input.csv").resolve()
    assert scenario.cache_root == cache_root.resolve()


def test_unknown_root_override_has_no_typo_fallback(tmp_path):
    with pytest.raises(ValueError, match="Unknown root override.*student_assignmnt"):
        load_scenario(
            {
                "scenario": "assignment-generated-zones-2324",
                "overrides": {"roots": {"student_assignmnt": str(tmp_path)}},
            },
            environ={},
        )


def test_bundled_scenarios_are_declared_as_package_data():
    expected = {
        "assignment-generated-zones-2324.yaml",
        "historical-2324.yaml",
        "legacy.yaml",
        "mission-bay-2324.yaml",
        "summer-26-zoning.yaml",
    }
    scenario_resources = files("loaders").joinpath("configs", "scenarios")
    assert {resource.name for resource in scenario_resources.iterdir()} == expected

    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    patterns = pyproject["tool"]["setuptools"]["package-data"]["loaders"]
    assert "configs/**/*.yaml" in patterns


def test_paths_are_independent_of_current_working_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "configuration"
    elsewhere = tmp_path / "elsewhere"
    config_dir.mkdir()
    elsewhere.mkdir()
    (config_dir / "input.csv").write_text("value\n1\n", encoding="utf-8")
    _write_scenario(
        config_dir / "scenario.yaml",
        {
            "id": "cwd-independent",
            "sources": {
                "test.local": {"path": "input.csv"},
                "test.special": {
                    "package": {
                        "path": "manual_block_edges.yaml",
                        "root": "package",
                    },
                    "repository": {
                        "path": "Config/centroids.yaml",
                        "root": "repository",
                    },
                },
            },
            "filters": {},
        },
    )
    run_path = config_dir / "run.yaml"
    run_path.write_text(
        yaml.safe_dump({"scenario": "scenario.yaml", "overrides": {}}),
        encoding="utf-8",
    )

    monkeypatch.chdir(elsewhere)
    scenario = load_scenario(run_path, environ={})

    assert scenario.source("test.local").path == (config_dir / "input.csv").resolve()
    source_map = scenario.source_map("test.special")
    assert source_map["package"].path == PACKAGE_CONFIG_ROOT / "manual_block_edges.yaml"
    assert source_map["repository"].path == REPOSITORY_ROOT / "Config/centroids.yaml"


def test_anchor_data_config_resolves_only_declared_relative_paths(tmp_path):
    config_dir = tmp_path / "run"
    config_dir.mkdir()
    original = {
        "scenario": "scenarios/custom.yaml",
        "overrides": {
            "roots": {"data": "data", "cache": "cache"},
            "sources": {
                "test.direct": {
                    "path": "inputs/table.csv",
                    "companions": ["inputs/table.meta"],
                },
                "test.rooted": {
                    "path": "catalog/table.csv",
                    "root": "data",
                    "companions": ["catalog/table.meta"],
                },
                "test.catalog": "optimization.students.enrolled.2324",
            },
            "filters": {"assignment": {"include_mission_bay": True}},
        },
    }

    anchored = anchor_data_config(original, config_dir)

    assert original["scenario"] == "scenarios/custom.yaml"
    assert anchored["scenario"] == str((config_dir / "scenarios/custom.yaml").resolve())
    assert anchored["overrides"]["roots"] == {
        "data": str((config_dir / "data").resolve()),
        "cache": str((config_dir / "cache").resolve()),
    }
    direct = anchored["overrides"]["sources"]["test.direct"]
    assert direct["path"] == str((config_dir / "inputs/table.csv").resolve())
    assert direct["companions"] == [str((config_dir / "inputs/table.meta").resolve())]
    rooted = anchored["overrides"]["sources"]["test.rooted"]
    assert rooted["path"] == "catalog/table.csv"
    assert rooted["companions"] == ["catalog/table.meta"]
    assert anchored["overrides"]["sources"]["test.catalog"] == (
        "optimization.students.enrolled.2324"
    )
    assert json.loads(json.dumps(anchored))["scenario"] == anchored["scenario"]
    assert (
        anchor_data_config({"scenario": "legacy", "overrides": {}}, config_dir)[
            "scenario"
        ]
        == "legacy"
    )


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"scenario": "legacy", "overrides": {}, "extra": True}, "run.*extra"),
        ({"scenario": "legacy"}, "missing.*overrides"),
        (
            {"scenario": "legacy", "overrides": {"unknown": {}}},
            "override.*unknown",
        ),
    ],
)
def test_run_config_rejects_unknown_or_missing_keys(payload, error):
    with pytest.raises(ValueError, match=error):
        load_scenario(payload, environ={})


@pytest.mark.parametrize(
    "filter_name",
    [
        "grade",
        "participation",
        "population",
        "capacity",
        "mission_bay",
        "school_id_aliases",
    ],
)
def test_legacy_optimization_filter_names_are_rejected(filter_name):
    with pytest.raises(ValueError, match=rf"optimization filter.*{filter_name}"):
        load_scenario(
            {
                "scenario": "legacy",
                "overrides": {"filters": {"optimization": {filter_name: "unused"}}},
            },
            environ={},
        )


@pytest.mark.parametrize(
    ("scenario_payload", "error"),
    [
        (
            {
                "id": "bad",
                "sources": {"test.input": {"path": "x", "typo": True}},
                "filters": {},
            },
            "source-ref.*typo",
        ),
        (
            {
                "id": "bad",
                "sources": {"test.input": {"path": "x"}},
                "filters": {"assignment": {"typo": True}},
            },
            "assignment filter.*typo",
        ),
        (
            {
                "id": "bad",
                "sources": {"not_dotted": {"path": "x"}},
                "filters": {},
            },
            "flat dotted",
        ),
        (
            {
                "id": "bad",
                "sources": {"test.input": {"path": "x"}},
                "filters": {"optimization": {"school_id_aliases": [909, 999]}},
            },
            "optimization filter.*school_id_aliases",
        ),
    ],
)
def test_scenario_rejects_unknown_source_and_filter_keys(
    tmp_path, scenario_payload, error
):
    scenario_path = tmp_path / "bad.yaml"
    _write_scenario(scenario_path, scenario_payload)
    with pytest.raises(ValueError, match=error):
        load_scenario({"scenario": str(scenario_path), "overrides": {}}, environ={})


def test_absolute_string_source_override_is_allowed(tmp_path):
    direct = tmp_path / "direct.csv"
    direct.write_text("value\n1\n", encoding="utf-8")
    scenario = load_scenario(
        {
            "scenario": "legacy",
            "overrides": {"sources": {"assignment.students": str(direct)}},
        },
        environ={},
    )

    assert scenario.source("assignment.students").path == direct.resolve()


def test_legacy_exposes_assignment_integration_roles_and_default_cache_root():
    scenario = load_scenario({"scenario": "legacy", "overrides": {}}, environ={})

    assert scenario.cache_root == Path("/soalnas/share/data/school_choice/Data/caches")
    assert scenario.source("assignment.estimate").path == Path(
        "/soalnas/share/data/school_choice/simulation-files/choice-model/Oct1estimates.npy"
    )
    assert scenario.source("assignment.block_data").path.name == (
        "SF 2010 blks 022119 with field descriptions (1).xlsx"
    )
    assert scenario.source("assignment.new_ctip").path.name == "ETB_2024.npy"
    assert scenario.source("assignment.new_ctip_blockgroup").path.name == (
        "ETB_2024_BlockGroup.npy"
    )
    assert scenario.source("assignment.school_coordinates") == scenario.source(
        "assignment.schools"
    )
    assert set(scenario.source_map("assignment.zones")) == {
        "Medium1",
        "Z1",
        "optGE780873",
        "Large1",
        "Con0",
        "Con1",
        "Con2",
        "Con3",
        "59-zone-1_B",
        "6-zone-1_BG",
        "10-zone-11_BG",
        "13-zone-7_BG",
        "18-zone-1_1_BG",
        "18-zone-1_2_BG",
    }
    assert set(scenario.source_map("assignment.citywide_zones")) == {
        "Medium1-citywide",
        "18zone-BG-0point3miles",
    }
    assert scenario.source("assignment.students").path.name == "student_1819.csv"
    assert scenario.source("choice.estimate").path.name == (
        "estimates_2324_exp8_0514.csv"
    )


def test_2324_scenarios_are_coherent_and_preserve_optimization_inputs():
    legacy = load_scenario({"scenario": "legacy", "overrides": {}}, environ={})
    historical = load_scenario(
        {"scenario": "historical-2324", "overrides": {}}, environ={}
    )
    mission_bay = load_scenario(
        {"scenario": "mission-bay-2324", "overrides": {}}, environ={}
    )

    optimization_roles = [
        role
        for role in legacy.source_manifest()["sources"]
        if role.startswith("optimization.")
    ]
    for scenario in (historical, mission_bay):
        assert scenario.filters["optimization"] == legacy.filters["optimization"]
        assert (
            scenario.source_manifest(optimization_roles)["sources"]
            == (legacy.source_manifest(optimization_roles)["sources"])
        )

    assert historical.source("assignment.students").catalog_id == (
        "assignment.students.2324"
    )
    assert historical.source("assignment.programs").catalog_id == (
        "assignment.programs.2324"
    )
    assert historical.source("assignment.capacity").catalog_id == (
        "capacity.stanford.scenarios_abcd"
    )
    assert historical.source("assignment.schools").catalog_id == (
        "assignment.schools.2324"
    )
    assert historical.filter("assignment", "include_mission_bay") is False

    assert mission_bay.source("assignment.students").catalog_id == (
        "assignment.students.2324"
    )
    assert mission_bay.source("assignment.programs").catalog_id == (
        "assignment.programs.2324.status_quo"
    )
    assert mission_bay.source("assignment.programs.catalog").catalog_id == (
        "assignment.programs.2324.mission_bay"
    )
    assert mission_bay.source("assignment.schools").catalog_id == (
        "assignment.schools.current_mission_bay"
    )
    assert mission_bay.source("assignment.estimate").catalog_id == ("utility.2324.exp8")
    assert mission_bay.filter("assignment", "include_mission_bay") is True

    required_assignment_roles = {
        "assignment.students",
        "assignment.programs",
        "assignment.capacity",
        "assignment.schools",
        "assignment.school_coordinates",
        "assignment.program_codes",
        "assignment.estimate",
        "assignment.block_data",
        "assignment.new_ctip",
        "assignment.new_ctip_blockgroup",
        "assignment.zones",
        "assignment.citywide_zones",
        "assignment.ctip",
    }
    for scenario in (historical, mission_bay):
        assert required_assignment_roles <= set(scenario.source_manifest()["sources"])
        assert scenario.filter("assignment", "year") == "2324"
        assert scenario.filter("assignment", "grades") == ("KG",)
        assert scenario.filter("assignment", "rounds") == (1,)
    assert historical.filter("assignment", "special_programs") == (
        "exclude_any_special"
    )
    assert mission_bay.filter("assignment", "special_programs") == "include"


def test_generated_zones_scenario_uses_supported_registry_bundle():
    scenario = load_scenario(
        {
            "scenario": "assignment-generated-zones-2324",
            "overrides": {},
        },
        environ={},
    )

    assert scenario.filter("assignment", "capacity_profile") == "status_quo"
    assert scenario.filter("assignment", "special_programs") == "include"
    assert scenario.source("assignment.students").catalog_id == (
        "assignment.students.2324"
    )
    assert scenario.source("assignment.programs").catalog_id == (
        "assignment.programs.2324.status_quo"
    )
    assert scenario.source("assignment.programs.catalog").catalog_id == (
        "assignment.programs.2324.mission_bay"
    )
    assert scenario.source("assignment.schools").catalog_id == (
        "assignment.schools.current_mission_bay"
    )
    assert scenario.filter("assignment", "geography_vintage") == "2020"
    assert scenario.source("assignment.geography.blockgroups").catalog_id == (
        "census.blockgroups.2020"
    )
    zones = scenario.source_map("assignment.zones")
    assert len(zones) == 292
    assert "Con1" in zones
    assert all(
        source.geography_vintage == "2020"
        for alias, source in zones.items()
        if alias != "Con1"
    )
    assert zones["18zone_2"].path == Path(
        "/soalnas/share/data/school_choice/Data/assignment/zones_2020/"
        "18-zone-1_2_BG_2020.csv"
    )
    assert zones[
        "Zones_4_FRL_Dev_0.10_Objective_1060.0_4-zone-5_2020"
    ].path == (
        Path(
            "/soalnas/share/data/school_choice/Data/assignment/zones_2020/"
            "Zones_4_FRL_Dev_0.10_Objective_1060.0_4-zone-5_2020.csv"
        )
    )


def test_school_aliases_are_not_user_configurable():
    with pytest.raises(ValueError, match="school_id_aliases"):
        load_scenario(
            {
                "scenario": "mission-bay-2324",
                "overrides": {
                    "filters": {"assignment": {"school_id_aliases": {909: 999}}}
                },
            },
            environ={},
        )


def test_expanded_assignment_catalog_resolves_concrete_paths():
    scenario = load_scenario(
        {
            "scenario": "legacy",
            "overrides": {
                "sources": {
                    "catalog.choice_inputs": {
                        "students": "choice.inputs.2324.students",
                        "programs": "choice.inputs.2324.programs",
                        "students_no_special": (
                            "choice.inputs.2324.students.no_special"
                        ),
                        "programs_no_special": (
                            "choice.inputs.2324.programs.no_special"
                        ),
                    },
                    "catalog.t14": [
                        "utility.t14.2122",
                        "utility.t14.2223",
                        "utility.t14.2324",
                    ],
                    "catalog.alternatives": {
                        "status_quo": "assignment.programs.2324.status_quo",
                        "mission_bay": "assignment.programs.2324.mission_bay",
                        "schools": "assignment.schools.current_mission_bay",
                    },
                }
            },
        },
        environ={},
    )

    choice_inputs = scenario.source_map("catalog.choice_inputs")
    assert choice_inputs["students"].path.name == "student_2324_kg_r1.csv"
    assert choice_inputs["programs"].path.name == "programs_2324_kg_r1.csv"
    assert [source.path.name for source in scenario.sources("catalog.t14")] == [
        "estimates_2122.csv",
        "estimates_2223.csv",
        "estimates_2324.csv",
    ]
    alternatives = scenario.source_map("catalog.alternatives")
    assert alternatives["status_quo"].path.name == "programs_statusQuo_2324.csv"
    assert alternatives["mission_bay"].path.name == ("programs_withMissionBay_2324.csv")
    assert alternatives["schools"].path.name == (
        "schools_rehauled_withMissionBay_2324.csv"
    )


def test_base_schema_two_registry_is_strict_and_catalog_backed(tmp_path):
    base = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))

    assert base["schema_version"] == 2
    assert set(base["school_years"]) == {
        "1415",
        "1516",
        "1617",
        "1718",
        "1819",
        "1920",
        "2021",
        "2122",
        "2223",
        "2324",
    }
    catalog_ids = set(base["files"])
    for entry in base["school_years"].values():
        for source in entry["optimization"]["students"].values():
            assert source in catalog_ids

    invalid = dict(base)
    invalid["schema_version"] = 1
    invalid_path = tmp_path / "base.yaml"
    invalid_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version must be 2"):
        load_scenario(
            {"scenario": "legacy", "overrides": {}},
            base_path=invalid_path,
            environ={},
        )


def test_geography_vintage_selects_complete_2020_bundles():
    scenario = load_scenario(
        {
            "scenario": "legacy",
            "overrides": {
                "filters": {
                    "optimization": {"geography_vintage": "2020"},
                    "assignment": {"geography_vintage": "2020"},
                }
            },
        },
        environ={},
    )

    assert scenario.source("optimization.census").catalog_id == "census.blocks.2020"
    assert scenario.source("optimization.crosswalk").catalog_id == (
        "census.block_crosswalk.2020"
    )
    assert set(scenario.source_map("optimization.adjacency")) == {
        "block",
        "blockgroup",
        "tract",
    }
    assert scenario.source("assignment.geography.blocks").catalog_id == (
        "census.blocks.2020"
    )
    assert scenario.source("assignment.geography.blockgroups").catalog_id == (
        "census.blockgroups.2020"
    )
    assert scenario.source("assignment.geography.tracts").catalog_id == (
        "census.tracts.2020"
    )
    assert scenario.filter("optimization", "outside_district_students") == "ignore"
    assert scenario.filter("assignment", "outside_district_students") == "ignore"

    with pytest.raises(ValueError, match="vintage '1990' is unavailable"):
        load_scenario(
            {
                "scenario": "legacy",
                "overrides": {
                    "filters": {"optimization": {"geography_vintage": "1990"}}
                },
            },
            environ={},
        )


def test_named_frl_estimate_requires_matching_census_vintage():
    scenario = load_scenario(
        {
            "scenario": "legacy",
            "overrides": {
                "filters": {
                    "optimization": {
                        "geography_vintage": "2020",
                        "frl_estimate": "updated_2526",
                    },
                    "assignment": {
                        "geography_vintage": "2020",
                        "frl_estimate": "updated_2526",
                    },
                }
            },
        },
        environ={},
    )

    assert scenario.source("optimization.frl_estimate").catalog_id == (
        "student_frl.updated_2526"
    )
    assert scenario.source("assignment.frl_estimate").path == (
        scenario.roots["data"] / "Data/student_frl"
        / "updated_frl_blocks_2526.csv"
    )

    default = load_scenario({"scenario": "legacy", "overrides": {}}, environ={})
    assert default.filter("optimization", "frl_estimate") is None
    with pytest.raises(KeyError, match="optimization.frl_estimate"):
        default.source("optimization.frl_estimate")

    with pytest.raises(ValueError, match="uses Census geography '2020'.*'2010'"):
        load_scenario(
            {
                "scenario": "legacy",
                "overrides": {
                    "filters": {"optimization": {"frl_estimate": "updated_2526"}}
                },
            },
            environ={},
        )

    with pytest.raises(ValueError, match="FRL estimate 'unknown'.*unavailable"):
        load_scenario(
            {
                "scenario": "legacy",
                "overrides": {"filters": {"optimization": {"frl_estimate": "unknown"}}},
            },
            environ={},
        )


@pytest.mark.parametrize("year", [23, "23", "2324-25", " 2324"])
def test_noncanonical_school_year_values_are_rejected(year):
    with pytest.raises(ValueError, match="four-character school-year string"):
        load_scenario(
            {
                "scenario": "legacy",
                "overrides": {"filters": {"assignment": {"year": year}}},
            },
            environ={},
        )


@pytest.mark.parametrize(
    ("group", "values", "error"),
    [
        ("optimization", {"years": []}, "years must be a non-empty list"),
        ("optimization", {"grades": ["K"]}, "must be a canonical grade"),
        (
            "optimization",
            {"student_population": "all"},
            "must be applicant or enrolled",
        ),
        ("optimization", {"rounds": 1}, "rounds must be all or"),
        ("optimization", {"program_population": " "}, "non-empty string"),
        ("optimization", {"include_k8": 1}, "include_k8 must be a boolean"),
        ("optimization", {"frl_estimate": 1}, "must be a non-empty string"),
        (
            "optimization",
            {"outside_district_students": "drop"},
            "must be ignore or include",
        ),
        ("assignment", {"grades": []}, "grades must be a non-empty list"),
        (
            "assignment",
            {"special_programs": "remove"},
            "exclude_only_special",
        ),
        ("assignment", {"capacity_profile": ""}, "non-empty string"),
        (
            "assignment",
            {"include_mission_bay": "include"},
            "include_mission_bay must be a boolean",
        ),
    ],
)
def test_filter_values_are_strict(group, values, error):
    with pytest.raises(ValueError, match=error):
        load_scenario(
            {
                "scenario": "legacy",
                "overrides": {"filters": {group: values}},
            },
            environ={},
        )


def test_scenario_filter_groups_must_define_every_selector(tmp_path):
    scenario_path = tmp_path / "scenario.yaml"
    _write_scenario(
        scenario_path,
        {
            "id": "incomplete",
            "sources": {"test.input": {"path": "input.csv"}},
            "filters": {"assignment": {"year": "2324"}},
        },
    )

    with pytest.raises(ValueError, match="is missing filters"):
        load_scenario(
            {"scenario": str(scenario_path), "overrides": {}}, environ={}
        )


@pytest.mark.parametrize("population", ["applicant", "enrolled"])
def test_optimization_registry_resolves_new_1920_2021_sources_in_order(population):
    scenario = load_scenario(
        {
            "scenario": "legacy",
            "overrides": {
                "filters": {
                    "optimization": {
                        "years": ["2021", "1920"],
                        "student_population": population,
                    }
                }
            },
        },
        environ={},
    )

    assert [
        source.catalog_id for source in scenario.sources("optimization.students")
    ] == [
        f"optimization.students.{population}.2021",
        f"optimization.students.{population}.1920",
    ]


@pytest.mark.parametrize(
    "year", [f"{start:02d}{start + 1:02d}" for start in range(15, 24)]
)
@pytest.mark.parametrize("population", ["applicant", "enrolled"])
def test_assignment_kg_registry_resolves_all_available_years(year, population):
    scenario = load_scenario(
        {
            "scenario": "legacy",
            "overrides": {
                "filters": {
                    "assignment": {
                        "year": year,
                        "student_population": population,
                    }
                }
            },
        },
        environ={},
    )

    expected_students = (
        f"assignment.students.{year}"
        if population == "applicant"
        else f"optimization.students.enrolled.{year}"
    )
    assert scenario.source("assignment.students").catalog_id == expected_students
    assert scenario.source("assignment.programs").catalog_id == (
        f"assignment.programs.{year}"
    )
    assert scenario.source("assignment.schools").catalog_id == (
        f"assignment.schools.{year}"
    )
    assert scenario.source("assignment.school_coordinates") == scenario.source(
        "assignment.schools"
    )


@pytest.mark.parametrize("grade", ["06", "09"])
@pytest.mark.parametrize(
    "year", [f"{start:02d}{start + 1:02d}" for start in range(15, 23)]
)
def test_assignment_secondary_bundles_resolve_only_when_both_sources_exist(year, grade):
    scenario = load_scenario(
        {
            "scenario": "legacy",
            "overrides": {"filters": {"assignment": {"year": year, "grades": [grade]}}},
        },
        environ={},
    )

    assert scenario.source("assignment.programs").catalog_id == (
        f"assignment.programs.{grade}.{year}"
    )
    assert scenario.source("assignment.schools").catalog_id == (
        f"assignment.schools.{grade}.{year}"
    )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"year": "1415"}, "no program/school bundle"),
        ({"year": "2324", "grades": ["06"]}, "available grades.*KG"),
        ({"grades": ["KG", "06"]}, "requires exactly one grade"),
        (
            {"include_mission_bay": True},
            "capacity profile 'default'.*Mission Bay included",
        ),
        (
            {
                "year": "2324",
                "capacity_profile": "status_quo",
                "include_mission_bay": False,
            },
            "capacity profile 'status_quo'.*Mission Bay excluded",
        ),
        (
            {"year": "2223", "include_mission_bay": True},
            "Mission Bay included",
        ),
    ],
)
def test_assignment_registry_rejects_unsupported_combinations(overrides, error):
    with pytest.raises(ValueError, match=error):
        load_scenario(
            {
                "scenario": "legacy",
                "overrides": {"filters": {"assignment": overrides}},
            },
            environ={},
        )


def test_registry_has_no_nearby_year_fallback():
    with pytest.raises(ValueError, match="school year '2425'.*available years"):
        load_scenario(
            {
                "scenario": "legacy",
                "overrides": {"filters": {"optimization": {"years": ["2425"]}}},
            },
            environ={},
        )


def test_explicit_source_overrides_registry_generated_roles(tmp_path):
    students = tmp_path / "students.csv"
    programs = tmp_path / "programs.csv"
    students.write_text("studentno,grade\n1,KG\n", encoding="utf-8")
    programs.write_text(
        "program_id,school_id,program_type,capacity\n10-GE-KG,10,GE,10\n",
        encoding="utf-8",
    )
    scenario = load_scenario(
        {
            "scenario": "historical-2324",
            "overrides": {
                "sources": {
                    "assignment.students": str(students),
                    "assignment.programs": str(programs),
                }
            },
        },
        environ={},
    )

    assert scenario.source("assignment.students").path == students
    assert scenario.source("assignment.students").catalog_id is None
    assert scenario.source("assignment.programs").path == programs
    assert scenario.source("assignment.schools").catalog_id == (
        "assignment.schools.2324"
    )


def test_registry_generated_role_overrides_scenario_invariant_role(tmp_path):
    scenario_path = tmp_path / "scenario.yaml"
    _write_scenario(
        scenario_path,
        {
            "id": "precedence",
            "sources": {"optimization.students": "optimization.students.enrolled.1415"},
            "filters": {
                "optimization": {
                    "years": ["2324"],
                    "grades": ["KG"],
                    "student_population": "applicant",
                    "rounds": "all",
                    "special_programs": "include",
                    "program_population": "GE",
                    "capacity_scenario": "A",
                    "include_k8": False,
                    "include_citywide": False,
                    "include_mission_bay": False,
                    "geography_vintage": "2010",
                }
            },
        },
    )

    scenario = load_scenario(
        {"scenario": str(scenario_path), "overrides": {}}, environ={}
    )

    assert scenario.source("optimization.students").catalog_id == (
        "optimization.students.applicant.2324"
    )
