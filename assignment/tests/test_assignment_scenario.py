import copy
import hashlib
import json
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from loaders import (
    identity_fingerprint,
    load_program_records,
    load_scenario,
    load_student_records,
)

from assignment.student_assignment.configerator import Configerator
from assignment.student_assignment.evaluation.match_evaluator import MatchEvaluator
from assignment.student_assignment.market_generator.school_choice_market import (
    SchoolChoiceMarket,
)
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)
from assignment.student_assignment.market_generator.preference_generator import (
    PreferenceGenerator,
)
from assignment.student_assignment.market_generator.priority_generator import (
    PriorityGenerator,
)


_CONSOLIDATED_RUN_ESTIMATES = {
    "all_zones.yaml": (
        "/share/data/school_choice/simulation-files/choice-model/"
        "estimates_2324_exp8_0514.csv"
    ),
    "all_zones_gesplit.yaml": (
        "/tmp/sfusd-choice/local_outputs/models/t7_2223_k1_base_gesplit/"
        "estimates_2324.csv"
    ),
    "all_zones_selected_iter10.yaml": (
        "/tmp/sfusd-choice/local_outputs/models/selected_2223_k1_prog_gesplit/"
        "estimates_2324.csv"
    ),
    "new_run_03_11.yaml": (
        "/tmp/sfusd-choice/local_outputs/models/t7_2223_k1_base/"
        "estimates_2324.csv"
    ),
    "new_run_ge_split.yaml": (
        "/tmp/sfusd-choice/local_outputs/models/t7_2223_k1_base_gesplit/"
        "estimates_2324.csv"
    ),
    "prog_ge_split_2324.yaml": (
        "/tmp/sfusd-choice/local_outputs/models/t7_2223_k1_prog_gesplit/"
        "estimates_2324.csv"
    ),
}
_SHARED_SOURCE_MAP_SHA256 = (
    "37b8b3c8b43991c0f93345f567ffd6d50b55e1e65e5accad31f5af6519fc5f49"
)


@pytest.mark.parametrize(
    ("config_name", "expected_estimate"), _CONSOLIDATED_RUN_ESTIMATES.items()
)
def test_consolidated_run_source_maps_match_shared_data_semantics(
    config_name, expected_estimate
):
    config_path = Path(__file__).parents[1] / "configs" / config_name
    substituted = (
        config_path.read_text(encoding="utf-8")
        .replace("<STUDENT_ASSIGNMENT_PATH>", "/tmp/student-assignment")
        .replace("<SFUSD_CHOICE_PATH>", "/tmp/sfusd-choice")
    )
    config = yaml.safe_load(substituted)
    data = config["data"]

    assert data["scenario"] == "assignment-generated-zones-2324"
    assert data["overrides"].get("roots", {}) == {}
    assert set(data["overrides"].get("sources", {})) <= {
        "assignment.estimate"
    }

    scenario = load_scenario(data, environ={})
    semantic_maps = {
        role: {
            alias: {
                "path": str(source.path),
                "classification": source.classification,
            }
            for alias, source in scenario.source_map(role).items()
        }
        for role in ("assignment.zones", "assignment.citywide_zones")
    }
    encoded = json.dumps(
        semantic_maps, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    assert len(semantic_maps["assignment.zones"]) == 256
    assert len(semantic_maps["assignment.citywide_zones"]) == 1
    assert hashlib.sha256(encoded).hexdigest() == (
        _SHARED_SOURCE_MAP_SHA256
    )
    assert scenario.source("assignment.estimate").path == Path(expected_estimate)
    assert scenario.source("assignment.students").catalog_id == (
        "assignment.students.2324"
    )
    assert scenario.source("assignment.programs").catalog_id == (
        "assignment.programs.2324.status_quo"
    )
    assert scenario.source("assignment.schools").catalog_id == (
        "assignment.schools.current_mission_bay"
    )
    assert scenario.filter("assignment", "include_mission_bay") is True
    assert scenario.filter("assignment", "special_programs") == (
        "exclude_any_special"
    )


def _write_sources(tmp_path, *, school_id=101, student_ids=(1, 2, 3)):
    students = tmp_path / "students.csv"
    programs = tmp_path / "programs.csv"
    schools = tmp_path / "schools.csv"
    pd.DataFrame(
        [
            {
                "studentno": student_ids[0],
                "grade": "KG",
                "r1_ranked_idschool": f"[{school_id}]",
                "r1_programs": "['GE']",
                "r2_ranked_idschool": "[]",
                "r2_programs": "[]",
                "latitude": 37.75,
                "longitude": -122.45,
                "HOCidx1": 0.2,
            },
            {
                "studentno": student_ids[1],
                "grade": "KG",
                "r1_ranked_idschool": "[]",
                "r1_programs": "[]",
                "r2_ranked_idschool": f"[{school_id}]",
                "r2_programs": "['GE']",
                "latitude": 37.76,
                "longitude": -122.44,
                "HOCidx1": 0.4,
            },
            {
                "studentno": student_ids[2],
                "grade": "06",
                "r1_ranked_idschool": f"[{school_id}]",
                "r1_programs": "['GE']",
                "r2_ranked_idschool": "[]",
                "r2_programs": "[]",
                "latitude": 37.77,
                "longitude": -122.43,
                "HOCidx1": 0.6,
            },
        ]
    ).to_csv(students, index=False)
    pd.DataFrame(
        {
            "programno": [1],
            "program_id": [f"{school_id}-GE-KG"],
            "school_id": [school_id],
            "program_type": ["GE"],
            "capacity": [10],
            "r1_assigned": [0],
            "r2_capacity": [10],
        }
    ).to_csv(programs, index=False)
    pd.DataFrame(
        {
            "school_id": [school_id],
            "category": ["Attendance"],
            "lat": [37.75],
            "lon": [-122.45],
        }
    ).to_csv(schools, index=False)
    return students, programs, schools


def _config(
    tmp_path,
    sources,
    *,
    cache_root=None,
    rounds="all",
    special_programs="include",
    include_mission_bay=False,
):
    students, programs, schools = sources
    return {
        "data": {
            "scenario": "legacy",
            "overrides": {
                "roots": {"cache": str(cache_root or tmp_path / "cache")},
                "sources": {
                    "assignment.students": {
                        "path": str(students),
                        "classification": "restricted",
                    },
                    "assignment.programs": {
                        "path": str(programs),
                        "classification": "internal",
                    },
                    "assignment.programs.catalog": {
                        "path": str(programs),
                        "classification": "internal",
                    },
                    "assignment.schools": {
                        "path": str(schools),
                        "classification": "internal",
                    },
                    "assignment.school_coordinates": {
                        "path": str(schools),
                        "classification": "internal",
                    },
                },
                "filters": {
                    "assignment": {
                        "year": "2324" if include_mission_bay else "2122",
                        "grades": ["KG"],
                        "student_population": "applicant",
                        "rounds": rounds,
                        "special_programs": special_programs,
                        "capacity_profile": (
                            "status_quo" if include_mission_bay else "default"
                        ),
                        "include_mission_bay": include_mission_bay,
                    }
                },
            },
        },
        "iterations": {"start": 0, "end": 1},
        "paths": {"assignment-folder": str(tmp_path / "assignments")},
        "random-seed": 7,
        "rounds-merged-options": [0],
        "save-assignment": True,
        "subconfigs": [],
        "utility-model": {"enable": False, "list-length": "7"},
    }


def test_scenario_grade_and_round_filters_are_materialized(tmp_path):
    sources = _write_sources(tmp_path)
    market = SchoolChoiceMarket(
        config=_config(
            tmp_path,
            sources,
            rounds=[1],
        )
    )

    assert market.config["year"] == 21
    assert market.config["grade"] == "KG"
    assert market.config["special_programs"] == "include"
    assert market.students.student_data.index.tolist() == [1]
    assert market.students.rounds == 1
    assert market.students.only_keep_rows.tolist() == [0]
    assert "r2_ranked_idschool" not in market.students.student_data.columns
    assert market.config["paths"]["student-data"] == str(sources[0])
    assert market.config["data-provenance"]["scenario"] == "legacy"
    assert "data-provenance" not in market.resolved_config
    assert "year" not in market.resolved_config
    assert "student-data" not in market.resolved_config["paths"]


def test_mission_bay_alias_is_derived_from_the_scenario_filter(tmp_path):
    sources = _write_mission_bay_sources(tmp_path)
    excluded = SchoolChoiceMarket(config=_config(tmp_path, sources))
    included = SchoolChoiceMarket(
        config=_config(tmp_path, sources, include_mission_bay=True)
    )

    assert excluded.programs.program_df["school_id"].unique().tolist() == [101]
    assert excluded.students.student_data.loc[1, "selected_ranked_idschool"] == [101]
    assert set(included.programs.program_df["school_id"]) == {999, 101}
    assert included.schools.school_df.index.tolist() == [999, 101]
    assert included.students.student_data.loc[1, "selected_ranked_idschool"] == [
        999,
        101,
    ]


def test_distance_cache_identity_tracks_sources_but_not_cache_root(tmp_path):
    sources = _write_sources(tmp_path)
    first_config = _config(tmp_path, sources, cache_root=tmp_path / "cache-a")
    first = SchoolChoiceMarket(config=first_config)
    relocated = SchoolChoiceMarket(
        config=_config(tmp_path, sources, cache_root=tmp_path / "cache-b")
    )

    first_key = first.students.distance_cache.key
    reference = first.students.distance_cache_reference
    assert reference["artifact"] == "student_program_distances"
    assert reference["schema_version"] == 4
    assert reference["classification"] == "restricted-derived"
    assert reference["key"] == first_key
    assert reference["payload"] == "distances.pkl"
    assert set(reference["roles"]) == {
        "assignment.students",
        "assignment.programs",
        "assignment.programs.catalog",
        "assignment.school_coordinates",
    }
    assert not any(key.endswith("path") for key in reference)
    assert relocated.students.distance_cache.key == first_key
    assert relocated.students.distance_cache.root != first.students.distance_cache.root

    student_frame = pd.read_csv(sources[0])
    student_frame["source_revision"] = "changed"
    student_frame.to_csv(sources[0], index=False)
    changed_students = SchoolChoiceMarket(config=copy.deepcopy(first_config))
    assert changed_students.students.distance_cache.key != first_key

    student_key = changed_students.students.distance_cache.key
    school_frame = pd.read_csv(sources[2])
    school_frame.loc[0, "lat"] = 37.751
    school_frame.to_csv(sources[2], index=False)
    changed_coordinates = SchoolChoiceMarket(config=copy.deepcopy(first_config))
    assert changed_coordinates.students.distance_cache.key != student_key


def test_distance_cache_metadata_never_serializes_student_identities(tmp_path):
    student_ids = (
        "private-student-alpha",
        "private-student-beta",
        "private-student-gamma",
    )
    sources = _write_sources(tmp_path, student_ids=student_ids)
    market = SchoolChoiceMarket(config=_config(tmp_path, sources))
    namespace = market.students.distance_cache
    reference = market.students.distance_cache_reference
    manifest = json.loads(namespace.manifest_path.read_text(encoding="utf-8"))

    parameters = reference["parameters"]
    assert parameters["student_count"] == 2
    assert parameters["student_identity_fingerprint"] == identity_fingerprint(
        student_ids[:2]
    )
    assert "students" not in parameters
    assert manifest["classification"] == "restricted-derived"
    assert reference["classification"] == "restricted-derived"

    public_metadata = json.dumps({"manifest": manifest, "reference": reference})
    cache_names = "/".join(namespace.path.relative_to(namespace.root).parts)
    for student_id in student_ids:
        assert student_id not in public_metadata
        assert student_id not in cache_names

    assert stat.S_IMODE(namespace.path.stat().st_mode) == 0o770
    assert stat.S_IMODE(namespace.manifest_path.stat().st_mode) == 0o660
    assert stat.S_IMODE(
        namespace.payload_path(market.students.DISTANCE_CACHE_PAYLOAD).stat().st_mode
    ) == 0o660


def test_saved_public_config_round_trips_after_cwd_change(tmp_path, monkeypatch):
    config_dir = tmp_path / "declared"
    config_dir.mkdir()
    sources = _write_sources(config_dir)
    config = _config(config_dir, sources)
    config["data"]["overrides"]["roots"]["cache"] = "cache"
    for source in config["data"]["overrides"]["sources"].values():
        source["path"] = str(Path(source["path"]).relative_to(config_dir))
    declaring_path = config_dir / "run.yaml"
    declaring_path.write_text("# declaring path\n", encoding="utf-8")

    MarketGenerator(
        configurator=Configerator.from_config(
            config, declaring_path=declaring_path
        ),
        assignment_path=str(tmp_path / "output"),
    )
    snapshot_path = tmp_path / "output" / "config.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    reloaded = Configerator.from_config(snapshot)
    replay = SchoolChoiceMarket(config=snapshot)

    assert reloaded.config == snapshot
    assert replay.students.student_data.index.tolist() == [1, 2]
    assert "data-provenance" not in snapshot
    assert not {"year", "grade", "remove-special-lps"}.intersection(snapshot)
    assert set(snapshot["paths"]) == {"assignment-folder"}
    assert snapshot["data"]["overrides"]["roots"]["cache"] == str(
        (config_dir / "cache").resolve()
    )


def test_market_reconfigure_reuses_tables_until_source_identity_changes(
    tmp_path, monkeypatch
):
    from assignment.student_assignment.market_generator import school_choice_market

    sources = _write_sources(tmp_path)
    config = _config(tmp_path, sources)
    market = MarketGenerator(
        config=config,
        assignment_path=str(tmp_path / "first-output"),
        write_config=False,
    )
    original_students = market.students
    calls = {"students": 0, "programs": 0, "schools": 0}
    original_load_students = school_choice_market.load_student_records
    original_load_programs = school_choice_market.load_program_records
    original_load_schools = school_choice_market.load_school_records

    def load_students(*args, **kwargs):
        calls["students"] += 1
        return original_load_students(*args, **kwargs)

    def load_programs(*args, **kwargs):
        calls["programs"] += 1
        return original_load_programs(*args, **kwargs)

    def load_schools(*args, **kwargs):
        calls["schools"] += 1
        return original_load_schools(*args, **kwargs)

    monkeypatch.setattr(school_choice_market, "load_student_records", load_students)
    monkeypatch.setattr(school_choice_market, "load_program_records", load_programs)
    monkeypatch.setattr(school_choice_market, "load_school_records", load_schools)

    parameter_change = copy.deepcopy(config)
    parameter_change["random-seed"] = 99
    market.reconfigure(parameter_change, str(tmp_path / "second-output"))

    assert calls == {"students": 0, "programs": 0, "schools": 0}
    assert market.students is original_students

    student_frame = pd.read_csv(sources[0])
    student_frame["source_revision"] = "changed"
    student_frame.to_csv(sources[0], index=False)
    market.reconfigure(parameter_change, str(tmp_path / "third-output"))

    assert calls == {"students": 1, "programs": 1, "schools": 2}
    assert market.students is not original_students


def _write_round_gap_sources(tmp_path):
    students = tmp_path / "round-gap-students.csv"
    programs = tmp_path / "round-gap-programs.csv"
    schools = tmp_path / "round-gap-schools.csv"
    rows = []
    for studentno, choices in (
        (1, {1: [], 2: [102], 4: [104]}),
        (2, {1: [], 2: [], 4: [104]}),
        (3, {1: [], 2: [], 4: []}),
    ):
        row = {
            "studentno": studentno,
            "grade": "KG",
            "latitude": 37.74 + studentno / 100,
            "longitude": -122.45,
            "HOCidx1": 0.2,
        }
        for round_label, ranked_schools in choices.items():
            row[f"r{round_label}_ranked_idschool"] = str(ranked_schools)
            row[f"r{round_label}_programs"] = str(
                ["GE"] * len(ranked_schools)
            )
            row[f"r{round_label}_randomnumber"] = str(
                [round_label / 10] * len(ranked_schools)
            )
            row[f"r{round_label}_designation_randomnumber"] = round_label / 100
            row[f"r{round_label}_cohortstring"] = str(
                [f"round-{round_label}"] * len(ranked_schools)
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(students, index=False)
    pd.DataFrame(
        {
            "programno": [1, 2],
            "program_id": ["102-GE-KG", "104-GE-KG"],
            "school_id": [102, 104],
            "program_type": ["GE", "GE"],
            "capacity": [10, 10],
            "r1_assigned": [0, 0],
            "r2_capacity": [10, 10],
        }
    ).to_csv(programs, index=False)
    pd.DataFrame(
        {
            "school_id": [102, 104],
            "category": ["Attendance", "Attendance"],
            "lat": [37.75, 37.76],
            "lon": [-122.45, -122.44],
        }
    ).to_csv(schools, index=False)
    return students, programs, schools


def test_nonconsecutive_rounds_use_selected_preferences_and_lotteries(tmp_path):
    sources = _write_round_gap_sources(tmp_path)
    market = SchoolChoiceMarket(
        config=_config(tmp_path, sources, rounds=[1, 2, 4])
    )

    assert market.students.student_data.index.tolist() == [1, 2]
    assert market.students.student_data.index.is_unique
    assert market.students.round_labels == (1, 2, 4)
    assert market.students.first_round.tolist() == [1, 2]
    assert market.students.first_participating_round.tolist() == [2, 4]
    assert market.students.student_data["selected_ranked_idschool"].tolist() == [
        [102],
        [104],
    ]

    preference_generator = PreferenceGenerator(market)
    preferences = preference_generator.initialize_real_preferences(designate=False)
    np.testing.assert_array_equal(preferences, [[1, 0], [2, 0]])

    priority_generator = PriorityGenerator(market)
    mtb = priority_generator._mtb_real()
    np.testing.assert_allclose(mtb, [[0.2, 0.02], [0.04, 0.4]])
    with pytest.warns(UserWarning, match="selected choice"):
        stb = priority_generator._stb_real()
    np.testing.assert_allclose(stb, [[0.2, 0.2], [0.4, 0.4]])


def _write_special_round_sources(tmp_path):
    students = tmp_path / "special-round-students.csv"
    programs = tmp_path / "special-round-programs.csv"
    schools = tmp_path / "special-round-schools.csv"
    pd.DataFrame(
        [
            {
                "studentno": 1,
                "grade": "KG",
                "r1_ranked_idschool": "[101]",
                "r1_programs": "['GE']",
                "r4_ranked_idschool": "[202]",
                "r4_programs": "['SA']",
                "latitude": 37.75,
                "longitude": -122.45,
                "HOCidx1": 0.2,
            },
            {
                "studentno": 2,
                "grade": "KG",
                "r1_ranked_idschool": "[]",
                "r1_programs": "[]",
                "r4_ranked_idschool": "[202]",
                "r4_programs": "['SA']",
                "latitude": 37.76,
                "longitude": -122.44,
                "HOCidx1": 0.3,
            },
            {
                "studentno": 3,
                "grade": "KG",
                "r1_ranked_idschool": "[101]",
                "r1_programs": "['GE']",
                "r4_ranked_idschool": "[]",
                "r4_programs": "[]",
                "latitude": 37.77,
                "longitude": -122.43,
                "HOCidx1": 0.4,
            },
        ]
    ).to_csv(students, index=False)
    pd.DataFrame(
        {
            "programno": [1, 2],
            "program_id": ["101-GE-KG", "202-SA-KG"],
            "school_id": [101, 202],
            "program_type": ["GE", "SA"],
            "capacity": [10, 10],
            "r1_assigned": [0, 0],
            "r2_capacity": [10, 10],
        }
    ).to_csv(programs, index=False)
    pd.DataFrame(
        {
            "school_id": [101, 202],
            "category": ["Attendance", "Citywide"],
            "lat": [37.75, 37.76],
            "lon": [-122.45, -122.44],
        }
    ).to_csv(schools, index=False)
    return students, programs, schools


@pytest.mark.parametrize(
    ("special_programs", "expected_students", "expected_programs"),
    [
        ("include", [1, 2, 3], ["101-GE-KG", "202-SA-KG"]),
        ("exclude_only_special", [1, 3], ["101-GE-KG"]),
        ("exclude_any_special", [3], ["101-GE-KG"]),
    ],
)
def test_market_does_not_duplicate_loader_special_or_population_filters(
    tmp_path, special_programs, expected_students, expected_programs
):
    sources = _write_special_round_sources(tmp_path)
    config = _config(
        tmp_path,
        sources,
        rounds=[1, 4],
        special_programs=special_programs,
    )
    scenario = load_scenario(config["data"])
    expected_student_frame = load_student_records(
        scenario, "assignment.students", filter_group="assignment"
    )
    expected_program_frame = load_program_records(
        scenario, "assignment.programs", filter_group="assignment"
    )

    market = SchoolChoiceMarket(config=config)

    assert expected_student_frame["studentno"].tolist() == expected_students
    assert market.students.student_data.index.tolist() == expected_students
    assert expected_program_frame["program_id"].tolist() == expected_programs
    assert market.programs.program_df["program_id"].tolist() == expected_programs


def test_utility_npy_uses_loader_source_identity_alignment(tmp_path):
    students = tmp_path / "utility-students.csv"
    programs = tmp_path / "utility-programs.csv"
    schools = tmp_path / "utility-schools.csv"
    estimates = tmp_path / "utilities.npy"
    pd.DataFrame(
        [
            {
                "studentno": 90,
                "grade": "06",
                "r1_ranked_idschool": "[999]",
                "r1_programs": "['GE']",
                "latitude": 37.7,
                "longitude": -122.4,
                "HOCidx1": 0.1,
            },
            {
                "studentno": 11,
                "grade": "KG",
                "r1_ranked_idschool": "[101]",
                "r1_programs": "['GE']",
                "latitude": 37.71,
                "longitude": -122.41,
                "HOCidx1": 0.2,
            },
            {
                "studentno": 12,
                "grade": "KG",
                "r1_ranked_idschool": "[]",
                "r1_programs": "[]",
                "latitude": 37.72,
                "longitude": -122.42,
                "HOCidx1": 0.3,
            },
            {
                "studentno": 13,
                "grade": "KG",
                "r1_ranked_idschool": "[101]",
                "r1_programs": "['GE']",
                "latitude": 37.73,
                "longitude": -122.43,
                "HOCidx1": 0.4,
            },
        ]
    ).to_csv(students, index=False)
    pd.DataFrame(
        {
            "programno": [1, 2],
            "program_id": ["999-GE-06", "101-GE-KG"],
            "school_id": [999, 101],
            "program_type": ["GE", "GE"],
            "capacity": [10, 10],
            "r1_assigned": [0, 0],
            "r2_capacity": [10, 10],
        }
    ).to_csv(programs, index=False)
    pd.DataFrame(
        {
            "school_id": [999, 101],
            "category": ["Attendance", "Attendance"],
            "lat": [37.7, 37.75],
            "lon": [-122.4, -122.45],
        }
    ).to_csv(schools, index=False)
    np.save(estimates, np.array([[1, 2], [11, 12], [21, 22], [31, 32]]))
    config = _config(tmp_path, (students, programs, schools), rounds=[1])
    config["utility-model"] = {"enable": True, "list-length": "1"}
    config["data"]["overrides"]["sources"]["assignment.estimate"] = {
        "path": str(estimates),
        "classification": "restricted",
    }

    market = SchoolChoiceMarket(config=config)
    market.umodel.draw_utility_model_randomness(
        rows_to_keep=market.students.only_keep_rows,
        cols_to_keep=market.programs.only_keep_cols,
        gumbel_scale=0,
    )

    assert market.students.student_data.index.tolist() == [11, 13]
    np.testing.assert_array_equal(market.students.only_keep_rows, [1, 3])
    np.testing.assert_array_equal(market.programs.only_keep_cols, [1])
    np.testing.assert_array_equal(market.umodel.original_utilities, [[12], [32]])


def _write_mission_bay_sources(tmp_path):
    students = tmp_path / "mission-students.csv"
    programs = tmp_path / "mission-programs.csv"
    schools = tmp_path / "mission-schools.csv"
    pd.DataFrame(
        [
            {
                "studentno": 1,
                "grade": "KG",
                "r1_ranked_idschool": "[909, 101]",
                "r1_programs": "['GE', 'GE']",
                "r1_cohortstring": "['CL;mission', 'regular']",
                "sibling": "[909, 101]",
                "aaprek": "[909]",
                "prek": "[101]",
                "currentlpsibling": "['909-CN-KG', '101-CN-KG']",
                "latitude": 37.75,
                "longitude": -122.45,
                "census_block": 1,
                "freelunch_prob": 0.5,
                "reducedlunch_prob": 0.0,
                "resolved_ethnicity": "Asian",
                "HOCidx1": 0.2,
            }
        ]
    ).to_csv(students, index=False)
    pd.DataFrame(
        {
            "programno": [1, 2, 3, 4],
            "program_id": [
                "909-GE-KG",
                "101-GE-KG",
                "909-CN-KG",
                "101-CN-KG",
            ],
            "school_id": [909, 101, 909, 101],
            "program_type": ["GE", "GE", "CN", "CN"],
            "capacity": [10, 10, 10, 10],
            "r1_assigned": [0, 0, 0, 0],
            "r2_capacity": [10, 10, 10, 10],
        }
    ).to_csv(programs, index=False)
    pd.DataFrame(
        {
            "school_id": [909, 101],
            "category": ["Attendance", "Attendance"],
            "lat": [37.76, 37.75],
            "lon": [-122.39, -122.45],
        }
    ).to_csv(schools, index=False)
    return students, programs, schools


@pytest.mark.parametrize(
    ("include_mission_bay", "expected_schools", "expected_lp_siblings"),
    [
        (False, [101], ["101-CN-KG"]),
        (True, [999, 101], ["999-CN-KG", "101-CN-KG"]),
    ],
)
def test_evaluator_uses_mission_bay_normalized_student_tables(
    tmp_path,
    include_mission_bay,
    expected_schools,
    expected_lp_siblings,
):
    sources = _write_mission_bay_sources(tmp_path)
    config = _config(
        tmp_path,
        sources,
        include_mission_bay=include_mission_bay,
    )
    assignments = pd.DataFrame(
        {
            "studentno": [1],
            "programno": [0],
            "programcodes": [""],
            "rank": [1],
            "designation": [0],
            "In-Zone Rank": [1],
        }
    )

    evaluator = MatchEvaluator.from_scenario(
        config["data"], assignments
    )
    row = evaluator.student_data.iloc[0]

    assert row["r1_ranked_idschool"] == expected_schools
    assert row["sibling"] == expected_schools
    assert row["aaprek"] + row["prek"] == expected_schools
    assert row["currentlpsibling"] == expected_lp_siblings
    expected_cohorts = ["regular"] if not include_mission_bay else [
        "CL;mission",
        "regular",
    ]
    assert row["r1_cohortstring"] == expected_cohorts

    market = SchoolChoiceMarket(config=config)
    sibling = market.students.sibling(market.programs)[0]
    expected_sibling_programs = {
        program_id
        for program_id in market.programs.indices
        if int(program_id.split("-", 1)[0]) in expected_schools
    }
    actual_sibling_programs = {
        market.programs.codes[index + 1]
        for index in sibling.nonzero()[0]
    }
    assert actual_sibling_programs == expected_sibling_programs

    prek = market.students.prek()[0]
    assert market.programs.codes[prek.nonzero()[0][0] + 1] == (
        f"{expected_schools[0]}-GE-KG"
    )
    language_sibling = market.students.language_pathway_sibling(
        market.programs.indices
    )[0]
    assert {
        market.programs.codes[index + 1]
        for index in language_sibling.nonzero()[0]
    } == set(expected_lp_siblings)

    language_pathway = market.students.language_pathway_priority_kg(
        market.programs.indices
    )[0]
    expected_pathway = (
        set()
        if not include_mission_bay
        else {f"{expected_schools[0]}-GE-KG"}
    )
    assert {
        market.programs.codes[index + 1]
        for index in language_pathway.nonzero()[0]
    } == expected_pathway
