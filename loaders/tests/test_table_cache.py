"""The normalised-table cache: exactness, invalidation, locking, warnings."""

from __future__ import annotations

import ast
import json
import os
import pickle
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pandas as pd
import pytest
import yaml

import loaders.table_cache as table_cache
from loaders import (
    LoaderWarning,
    clear_table_memo,
    load_census_geometry,
    load_geography_crosswalk,
    load_program_records,
    load_scenario,
    load_school_records,
    load_student_records,
)
from loaders.geography import _normalize_geoid, _normalize_geoid_int64
from loaders.table_cache import TABLE_CACHE_ENV, TableKey, cached_table
from loaders.tables import _literal_list, _simple_literal_list

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    monkeypatch.delenv(TABLE_CACHE_ENV, raising=False)
    clear_table_memo()
    yield
    clear_table_memo()


def _deep(value):
    if isinstance(value, list | tuple):
        return (type(value).__name__, tuple(_deep(item) for item in value))
    return (type(value).__name__, repr(value))


def assert_identical(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Equal frames down to dtypes, index, attrs, and the types inside cells."""
    assert type(left) is type(right)
    pd.testing.assert_frame_equal(
        left,
        right,
        check_exact=True,
        check_index_type=True,
        check_column_type=True,
        check_frame_type=True,
        check_flags=True,
    )
    assert list(left.dtypes.astype(str)) == list(right.dtypes.astype(str))
    assert left.attrs == right.attrs
    assert getattr(left, "crs", None) == getattr(right, "crs", None)
    for column in left.columns:
        if left[column].dtype == object:
            assert [_deep(value) for value in left[column]] == [
                _deep(value) for value in right[column]
            ], column


def _uncached(monkeypatch, load):
    monkeypatch.setenv(TABLE_CACHE_ENV, "0")
    try:
        return load()
    finally:
        monkeypatch.delenv(TABLE_CACHE_ENV)


def _messages(record) -> list[str]:
    """Distinct loader-warning messages, in order of first appearance."""
    return list(
        dict.fromkeys(
            str(item.message) for item in record if item.category is LoaderWarning
        )
    )


# --------------------------------------------------------------------------- #
# Synthetic scenarios
# --------------------------------------------------------------------------- #
def _students() -> pd.DataFrame:
    def rounds(schools, programs):
        return {
            "r1_ranked_idschool": str(schools),
            "r1_programs": str(programs),
            "r1_listed_ranks": str(list(range(1, len(schools) + 1))),
            "r1_randomnumber": str([0.25 + index / 7 for index in range(len(schools))]),
            "r1_cohortstring": str(["CT1;" for _ in schools]),
            "r1_designation_randomnumber": 0.5,
        }

    return pd.DataFrame(
        [
            {
                "studentno": 1,
                "grade": "K",
                "sibling": "[20]",
                **rounds([20, 30], ["GE", "AF"]),
            },
            {"studentno": 2, "grade": "KG", "sibling": "[]", **rounds([30], ["GE"])},
            # Ranks nothing: dropped with a LoaderWarning.
            {"studentno": 3, "grade": "KG", "sibling": "[]", **rounds([], [])},
            {"studentno": 4, "grade": "01", "sibling": "[]", **rounds([20], ["GE"])},
        ]
    )


def _run_config(tmp_path: Path, students: pd.DataFrame) -> dict:
    students_path = tmp_path / "students.csv"
    students.to_csv(students_path, index=False)
    scenario_path = tmp_path / "synthetic.yaml"
    filters = {
        "assignment": {
            "year": "1819",
            "grade": "KG",
            "student_population": "applicant",
            "rounds": "all",
            "special_programs": "include",
            "capacity_profile": "default",
            "capacity_scenario": "programs",
            "include_mission_bay": False,
            "geography_vintage": "2010",
        }
    }
    sources = {"assignment.students": {"path": str(students_path)}}
    scenario_path.write_text(
        yaml.safe_dump({"id": "synthetic", "sources": sources, "filters": filters}),
        encoding="utf-8",
    )
    return {
        "scenario": str(scenario_path),
        "overrides": {"roots": {"cache": str(tmp_path / "cache")}, "sources": sources},
    }


@pytest.fixture
def synthetic(tmp_path):
    config = _run_config(tmp_path, _students())
    return config, load_scenario(config, environ={})


def _load(scenario):
    return load_student_records(scenario, "assignment.students")


def test_cold_disk_and_memo_tables_equal_the_uncached_table(synthetic, monkeypatch):
    _, scenario = synthetic
    with pytest.warns(LoaderWarning):
        expected = _uncached(monkeypatch, lambda: _load(scenario))
    with pytest.warns(LoaderWarning):
        cold = _load(scenario)
    with pytest.warns(LoaderWarning):
        memo = _load(scenario)
    clear_table_memo()
    with pytest.warns(LoaderWarning):
        disk = _load(scenario)
    for frame in (cold, memo, disk):
        assert_identical(frame, expected)
    assert frame.attrs["source_rows"] == [0, 1]
    entries = list((scenario.cache_root / "loader_students").glob("v*/*/table.pkl"))
    assert len(entries) == 1


def test_returned_frames_can_be_mutated_without_touching_the_cache(synthetic):
    _, scenario = synthetic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = _load(scenario)
        expected = first.copy(deep=True)
        expected_lists = [list(value) for value in first["r1_ranked_idschool"]]
        first["r1_ranked_idschool"].iloc[0].append(999)
        first.loc[0, "grade"] = "changed"
        first.attrs["source_rows"].append(7)
        again = _load(scenario)
    assert again["grade"].tolist() == expected["grade"].tolist()
    assert [list(value) for value in again["r1_ranked_idschool"]] == expected_lists
    assert again.attrs["source_rows"] == [0, 1]


def test_a_changed_source_file_is_a_new_key(tmp_path):
    students = _students()
    config = _run_config(tmp_path, students)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = _load(load_scenario(config, environ={}))
        students.loc[1, "r1_ranked_idschool"] = "[20]"
        students.to_csv(tmp_path / "students.csv", index=False)
        after = _load(load_scenario(config, environ={}))
    assert before["r1_ranked_idschool"].tolist() == [[20, 30], [30]]
    assert after["r1_ranked_idschool"].tolist() == [[20, 30], [20]]
    entries = list((tmp_path / "cache" / "loader_students").glob("v*/*/table.pkl"))
    assert len(entries) == 2


def test_a_selector_change_is_a_new_key(tmp_path):
    config = _run_config(tmp_path, _students())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _load(load_scenario(config, environ={}))
        config["overrides"]["filters"] = {
            "assignment": {"special_programs": "exclude_any_special"}
        }
        general = _load(load_scenario(config, environ={}))
    assert general["studentno"].tolist() == [2]


def test_cached_tables_replay_their_warnings(synthetic):
    _, scenario = synthetic
    with warnings.catch_warnings(record=True) as cold:
        warnings.simplefilter("always")
        _load(scenario)
    with warnings.catch_warnings(record=True) as memo:
        warnings.simplefilter("always")
        _load(scenario)
    clear_table_memo()
    with warnings.catch_warnings(record=True) as disk:
        warnings.simplefilter("always")
        _load(scenario)
    assert _messages(cold) == [
        "1 of 3 students in the applicant population rank no school in round(s) "
        "[1] and are excluded from this table, so no metric computed over it "
        "counts them."
    ]
    assert _messages(memo) == _messages(cold)
    assert _messages(disk) == _messages(cold)


def test_an_inner_table_s_warnings_and_inputs_join_the_outer_entry(synthetic):
    _, scenario = synthetic
    inner = TableKey("inner_test", frozenset({"assignment.students"}), frozenset(), {})
    outer = TableKey("outer_test", frozenset({"assignment.students"}), frozenset(), {})

    def build_inner():
        table_cache.loader_warn("inner says hello")
        scenario.resolved("assignment.students")
        return 1

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        # The inner entry exists before the outer build, so the outer build
        # only sees it replayed.
        cached_table(scenario, inner, build_inner)
        assert (
            cached_table(
                scenario, outer, lambda: cached_table(scenario, inner, build_inner) + 1
            )
            == 2
        )
    clear_table_memo()
    with warnings.catch_warnings(record=True) as replayed:
        warnings.simplefilter("always")
        assert cached_table(scenario, outer, lambda: pytest.fail("rebuilt")) == 2
    assert _messages(replayed) == ["inner says hello"]
    payload = next((scenario.cache_root / "outer_test").glob("v*/*/table.pkl"))
    assert pickle.loads(payload.read_bytes())["roles"] == ["assignment.students"]


def test_a_build_reading_an_unkeyed_input_is_not_cached(synthetic):
    _, scenario = synthetic
    key = TableKey("narrow_test", frozenset(), frozenset(), {})
    calls = []

    def build():
        calls.append(1)
        scenario.resolved("assignment.students")
        scenario.filter("assignment", "grade")
        return "value"

    with pytest.warns(RuntimeWarning, match="does not name every input"):
        assert cached_table(scenario, key, build) == "value"
    with pytest.warns(RuntimeWarning):
        assert cached_table(scenario, key, build) == "value"
    assert len(calls) == 2
    assert not (scenario.cache_root / "narrow_test").exists() or not list(
        (scenario.cache_root / "narrow_test").glob("v*/*/table.pkl")
    )


def test_the_cache_can_be_disabled_or_kept_in_memory(synthetic, monkeypatch):
    _, scenario = synthetic
    monkeypatch.setenv(TABLE_CACHE_ENV, "0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _load(scenario)
    assert table_cache.table_memo_bytes() == 0
    assert not (scenario.cache_root / "loader_students").exists()

    monkeypatch.setenv(TABLE_CACHE_ENV, "memory")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _load(scenario)
    assert table_cache.table_memo_bytes() > 0
    assert not (scenario.cache_root / "loader_students").exists()


def test_an_unusable_cache_root_falls_back_to_computing(tmp_path, monkeypatch):
    config = _run_config(tmp_path, _students())
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    config["overrides"]["roots"]["cache"] = str(blocker)
    scenario = load_scenario(config, environ={})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = _uncached(monkeypatch, lambda: _load(scenario))
        loaded = _load(scenario)
        clear_table_memo()
        again = _load(scenario)
    assert_identical(loaded, expected)
    assert_identical(again, expected)


def test_a_corrupt_entry_is_rebuilt(synthetic):
    _, scenario = synthetic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = _load(scenario)
        payload = next((scenario.cache_root / "loader_students").glob("v*/*/table.pkl"))
        payload.write_bytes(b"garbage")
        clear_table_memo()
        assert_identical(_load(scenario), expected)


def test_loader_errors_propagate_unchanged(tmp_path):
    students = _students()
    students.loc[0, "r1_programs"] = "['GE']"
    config = _run_config(tmp_path, students)
    scenario = load_scenario(config, environ={})
    with pytest.raises(ValueError, match="2 ranked schools but 1 ranked programs"):
        _load(scenario)


_RACE_SCRIPT = textwrap.dedent(
    """
    import json, os, pickle, sys, time, warnings
    sys.path.insert(0, sys.argv[1])
    config, marker, output, start = json.loads(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5])
    import loaders.tables as tables
    from loaders import load_scenario, load_student_records

    original = tables._build_student_records

    def slow_build(*args, **kwargs):
        with open(marker, "a") as stream:
            stream.write(f"{os.getpid()}\\n")
        time.sleep(1.5)
        return original(*args, **kwargs)

    tables._build_student_records = slow_build
    scenario = load_scenario(config, environ={})
    while time.time() < start:
        time.sleep(0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = load_student_records(scenario, "assignment.students")
    with open(output, "wb") as stream:
        pickle.dump({"frame": frame, "warnings": [str(w.message) for w in caught]}, stream)
    """
)


def test_racing_processes_build_one_entry_and_share_it(tmp_path, monkeypatch):
    config = _run_config(tmp_path, _students())
    script = tmp_path / "race.py"
    script.write_text(_RACE_SCRIPT, encoding="utf-8")
    marker = tmp_path / "builds.txt"
    import time

    start = time.time() + 3.0
    environment = {**os.environ, "OMP_NUM_THREADS": "1"}
    environment.pop(TABLE_CACHE_ENV, None)
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(REPOSITORY),
                json.dumps(config),
                str(marker),
                str(tmp_path / f"out{index}.pkl"),
                str(start),
            ],
            env=environment,
        )
        for index in range(2)
    ]
    assert [process.wait(timeout=120) for process in processes] == [0, 0]
    assert len(marker.read_text().split()) == 1
    scenario = load_scenario(config, environ={})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = _uncached(monkeypatch, lambda: _load(scenario))
    results = [
        pickle.loads((tmp_path / f"out{index}.pkl").read_bytes()) for index in range(2)
    ]
    for result in results:
        assert_identical(result["frame"], expected)
        assert any("rank no school" in message for message in result["warnings"])


def test_each_loader_warning_is_shown_once_per_process(tmp_path):
    config = _run_config(tmp_path, _students())
    code = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(REPOSITORY)!r})
        from loaders import clear_table_memo, load_scenario, load_student_records
        scenario = load_scenario(json.loads({json.dumps(json.dumps(config))}), environ={{}})
        for _ in range(3):
            load_student_records(scenario, "assignment.students")
            clear_table_memo()
        """
    )
    environment = {**os.environ}
    environment.pop(TABLE_CACHE_ENV, None)
    environment.pop("PYTHONWARNINGS", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stderr.count("rank no school") == 1, result.stderr


def test_show_loader_warnings_once_overrides_an_always_filter():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table_cache.show_loader_warnings_once()
        for _ in range(3):
            table_cache.loader_warn("a once-only loader notice for this test")
    assert _messages(caught) == ["a once-only loader notice for this test"]


# --------------------------------------------------------------------------- #
# Vectorised parsing is exact
# --------------------------------------------------------------------------- #
_LITERALS = [
    "[]",
    "[ ]",
    "[1, 2, 3]",
    "[-1, 0, -0, 10]",
    "[01]",
    "[1_000]",
    "[0.485735298, 0.735859228, 1e-05, 1., .5, -.5, 2E3, 00.5]",
    "['GE', 'GE', \"SB\"]",
    "['', '', 'CT1;']",
    "['a, b', 'c']",
    "['it''s']",
    "['a\\\\b']",
    "['tab\\there']",
    "['é', 'ü']",
    "[1, 'x', 2.5]",
    "[1,]",
    "[,]",
    "[1,\n2]",
    "[None, True]",
    "[nan]",
    "[[1], 2]",
    "(1, 2)",
    " [1, 2]",
    "[1, 2] ",
    "['GE' 'SB']",
    "[b'x']",
    "[+1]",
    "[- 1]",
    "[1 # comment\n]",
]


@pytest.mark.parametrize("text", _LITERALS)
def test_simple_literal_lists_match_ast(text):
    simple = _simple_literal_list(text)
    try:
        expected = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        assert simple is None
        with pytest.raises(ValueError):
            _literal_list(text, "test")
        return
    if simple is not None:
        assert _deep(simple) == _deep(list(expected))
    if isinstance(expected, list | tuple):
        assert _deep(_literal_list(text, "test")) == _deep(list(expected))


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([60750101001000, 60750101001001], dtype="int64"),
        pd.Series([1, 2], dtype="uint32"),
        pd.Series(["060750101001000", "60750101001001.0"]),
        pd.Series([60750101001000.0, 60750101001001.0]),
        pd.Series([-5, 1], dtype="int64"),
        pd.Series([10**15, 1], dtype="int64"),
    ],
)
def test_integer_geoid_normalization_matches_the_string_path(values):
    try:
        expected = _normalize_geoid(values, length=15, label="test").astype("int64")
    except ValueError as exc:
        with pytest.raises(ValueError, match=re.escape(str(exc))):
            _normalize_geoid_int64(values, length=15, label="test")
        return
    pd.testing.assert_series_equal(
        _normalize_geoid_int64(values, length=15, label="test"), expected
    )


# --------------------------------------------------------------------------- #
# Real data: attendance-areas-2627
# --------------------------------------------------------------------------- #
_DATA_ROOT = Path("/soalnas/share/data/school_choice/Data/Cleaned")
_COMBOS = {
    "applicant": {"student_population": "applicant"},
    "enrolled-nov13": {
        "student_population": "enrolled",
        "capacity_scenario": "ideal_baseline_nov_13_2026",
    },
    "enrolled_all-webster": {
        "student_population": "enrolled_all",
        "capacity_scenario": "ideal_webster_w2w_10_23_26",
    },
    "applied_or_enrolled-nov13-mandarin": {
        "student_population": "applied_or_enrolled",
        "capacity_scenario": "ideal_baseline_nov_13_2026",
        "include_mandarin_immersion": True,
    },
    "applicant-webster-mandarin": {
        "student_population": "applicant",
        "capacity_scenario": "ideal_webster_w2w_10_23_26",
        "include_mandarin_immersion": True,
    },
}
_REAL_LOADS = {
    "students-assignment": lambda s: load_student_records(
        s, "assignment.students", filter_group="assignment", low_memory=False
    ),
    "students-optimization": lambda s: load_student_records(
        s, "optimization.students", filter_group="optimization", low_memory=False
    ),
    "schools-assignment": lambda s: load_school_records(
        s, "assignment.schools", filter_group="assignment"
    ),
    "school-coordinates-assignment": lambda s: load_school_records(
        s, "assignment.school_coordinates", filter_group="assignment"
    ),
    "schools-optimization": lambda s: load_school_records(
        s, "optimization.schools", filter_group="optimization", low_memory=False
    ),
    "programs-assignment": lambda s: load_program_records(
        s, "assignment.programs", filter_group="assignment"
    ),
    "programs-optimization": lambda s: load_program_records(
        s, "optimization.programs", filter_group="optimization", low_memory=False
    ),
}
_GEOGRAPHY_LOADS = {
    "blocks-optimization": lambda s: load_census_geometry(s, "optimization", "Block"),
    "blockgroups-optimization": lambda s: load_census_geometry(
        s, "optimization", "BlockGroup"
    ),
    "tracts-assignment": lambda s: load_census_geometry(s, "assignment", "Tract"),
    "blocks-assignment-2010": lambda s: load_census_geometry(
        s, "assignment", "Block", "2010"
    ),
    "crosswalk-optimization": lambda s: load_geography_crosswalk(s, "optimization"),
    "crosswalk-assignment-2010": lambda s: load_geography_crosswalk(
        s, "assignment", "2010"
    ),
}


def _real_scenario(cache_root: Path, selectors: dict):
    if not _DATA_ROOT.is_dir():
        pytest.skip("shared SFUSD data is not mounted")
    filters = {group: dict(selectors) for group in ("optimization", "assignment")}
    return load_scenario(
        {
            "scenario": "attendance-areas-2627",
            "overrides": {"roots": {"cache": str(cache_root)}, "filters": filters},
        }
    )


def _recorded(load):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = load()
    unkeyed = [str(item.message) for item in caught if item.category is RuntimeWarning]
    assert not unkeyed
    return frame, _messages(caught)


@pytest.mark.real_data
@pytest.mark.parametrize("combo", list(_COMBOS))
def test_real_tables_are_identical_cold_warm_and_uncached(combo, tmp_path, monkeypatch):
    scenario = _real_scenario(tmp_path / "cache", _COMBOS[combo])
    loads = dict(_REAL_LOADS)
    if combo == "applicant":
        loads.update(_GEOGRAPHY_LOADS)
    for name, load in loads.items():
        monkeypatch.setenv(TABLE_CACHE_ENV, "0")
        expected, expected_warnings = _recorded(lambda: load(scenario))
        monkeypatch.delenv(TABLE_CACHE_ENV)
        cold, cold_warnings = _recorded(lambda: load(scenario))
        memo, memo_warnings = _recorded(lambda: load(scenario))
        clear_table_memo()
        disk, disk_warnings = _recorded(lambda: load(scenario))
        for frame in (cold, memo, disk):
            assert_identical(frame, expected)
        assert cold_warnings == memo_warnings == disk_warnings == expected_warnings, (
            name
        )


def test_an_error_filter_raises_a_replayed_warning_like_a_live_one(synthetic):
    _, scenario = synthetic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _load(scenario)
    for _ in range(2):  # from the memo, then from disk
        with warnings.catch_warnings():
            warnings.simplefilter("error", LoaderWarning)
            with pytest.raises(LoaderWarning, match="rank no school"):
                _load(scenario)
        clear_table_memo()
