from __future__ import annotations

import os
from pathlib import Path

from loaders import identity_fingerprint
from loaders.cache import CacheStore


def test_source_hash_changes_with_content_and_missing_is_deterministic(
    tmp_path, scenario_factory
):
    source_path = tmp_path / "source.csv"
    source_path.write_text("value\nfirst\n", encoding="utf-8")
    missing_path = tmp_path / "missing.csv"
    scenario = scenario_factory(
        {
            "test.inputs": [
                {"path": str(source_path), "classification": "restricted"},
                {"path": str(missing_path)},
            ]
        }
    )

    first_manifest = scenario.source_manifest()
    first_fingerprint = scenario.source_fingerprint
    assert first_manifest["sources"]["test.inputs"][1]["status"] == "missing"
    assert scenario.source_fingerprint == first_fingerprint

    source_path.write_text("value\nsecond\n", encoding="utf-8")
    os.utime(source_path, None)

    assert scenario.source_fingerprint != first_fingerprint


def test_same_size_same_mtime_replacement_changes_source_and_cache_fingerprints(
    tmp_path, scenario_factory
):
    source_path = tmp_path / "source.csv"
    source_path.write_bytes(b"value\nfirst\n")
    scenario = scenario_factory({"test.input": {"path": str(source_path)}})
    initial_stat = source_path.stat()
    first_source_fingerprint = scenario.source_fingerprint
    first_cache_key = (
        CacheStore(scenario)
        .namespace("table", schema_version=1, roles="test.input")
        .key
    )

    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(b"value\nother\n")
    os.utime(
        replacement,
        ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
    )
    os.replace(replacement, source_path)
    os.utime(
        source_path,
        ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
    )

    assert source_path.stat().st_size == initial_stat.st_size
    assert source_path.stat().st_mtime_ns == initial_stat.st_mtime_ns
    assert scenario.source_fingerprint != first_source_fingerprint
    assert (
        CacheStore(scenario)
        .namespace("table", schema_version=1, roles="test.input")
        .key
        != first_cache_key
    )


def test_identity_fingerprint_is_full_ordered_and_content_sensitive():
    baseline = identity_fingerprint([101, " student-2 ", 303.0])

    assert len(baseline) == 64
    assert baseline == identity_fingerprint((101.0, "student-2", 303))
    assert baseline != identity_fingerprint([303, "student-2", 101])
    assert baseline != identity_fingerprint([101, "student-9", 303])


def test_shapefile_companions_are_hashed(tmp_path, scenario_factory):
    shape = tmp_path / "areas.shp"
    database = tmp_path / "areas.dbf"
    shape.write_bytes(b"shape")
    database.write_bytes(b"database")
    scenario = scenario_factory({"test.shape": {"path": str(shape)}})

    manifest = scenario.source_manifest("test.shape")["sources"]["test.shape"]
    companions = {Path(item["path"]).suffix: item for item in manifest["companions"]}

    assert companions[".dbf"]["status"] == "present"
    assert companions[".shx"]["status"] == "missing"
    assert companions[".prj"]["status"] == "missing"


def test_cache_and_semantic_keys_ignore_only_cache_root(tmp_path, scenario_factory):
    source_path = tmp_path / "source.csv"
    source_path.write_text("value\n1\n", encoding="utf-8")
    sources = {"test.input": {"path": str(source_path)}}
    first = scenario_factory(
        sources, scenario_id="first", cache_root=tmp_path / "cache-one"
    )
    second = scenario_factory(
        sources, scenario_id="first", cache_root=tmp_path / "cache-two"
    )

    first_namespace = CacheStore(first).namespace(
        "graph",
        {"unit": "Block", "levels": [0, 1]},
        schema_version=4,
        roles="test.input",
    )
    second_namespace = CacheStore(second).namespace(
        "graph",
        {"levels": [0, 1], "unit": "Block"},
        schema_version=4,
        roles="test.input",
    )

    assert first.semantic_fingerprint == second.semantic_fingerprint
    assert first.source_fingerprint == second.source_fingerprint
    assert first_namespace.key == second_namespace.key
    assert first_namespace.path != second_namespace.path


def test_normalized_selectors_change_manifests_semantics_and_cache_identity(
    tmp_path, scenario_factory
):
    source_path = tmp_path / "source.csv"
    source_path.write_text("value\n1\n", encoding="utf-8")
    sources = {"test.input": {"path": str(source_path)}}
    first = scenario_factory(
        sources,
        {"assignment": {"rounds": [2, 1], "special_programs": "include"}},
        scenario_id="selectors",
    )
    second = scenario_factory(
        sources,
        {
            "assignment": {
                "rounds": [1, 2],
                "special_programs": "exclude_only_special",
            }
        },
        scenario_id="selectors",
    )

    assert first.filters["assignment"]["rounds"] == (1, 2)
    assert first.source_manifest("test.input")["filters"] == {
        "assignment": {
            "year": "1819",
            "grades": ["KG"],
            "student_population": "applicant",
            "rounds": [1, 2],
            "special_programs": "include",
            "capacity_profile": "default",
            "capacity_scenario": "programs",
            "include_mission_bay": False,
            "geography_vintage": "2010",
            "frl_estimate": None,
            "outside_district_students": "ignore",
        }
    }
    assert first.source_fingerprint != second.source_fingerprint
    assert first.semantic_fingerprint != second.semantic_fingerprint
    first_cache = CacheStore(first).namespace(
        "selectors", schema_version=1, roles="test.input"
    )
    second_cache = CacheStore(second).namespace(
        "selectors", schema_version=1, roles="test.input"
    )
    assert first_cache.key != second_cache.key
