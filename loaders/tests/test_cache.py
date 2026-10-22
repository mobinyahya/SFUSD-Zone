from __future__ import annotations

import json
import os
import stat

import pandas as pd
import pytest

from loaders.cache import CacheStore


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_cache_supports_multiple_validated_payloads(tmp_path, scenario_factory):
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    scenario = scenario_factory({"test.input": {"path": str(source)}})
    namespace = CacheStore(scenario).namespace(
        "graphs",
        {"unit": "Block"},
        schema_version=3,
        roles=["test.input"],
    )

    first_path = namespace.save_pickle("Block_0.pickle", {"nodes": 3})
    namespace.save_pickle("Block_1.pickle", [0, 1])
    namespace.save_dataframe(
        "areas.csv", pd.DataFrame({"area": [10, 20], "zone": [0, 1]})
    )

    assert first_path == namespace.path / "Block_0.pickle"
    assert namespace.load_pickle("Block_0.pickle") == {"nodes": 3}
    assert namespace.load_pickle("Block_1.pickle") == [0, 1]
    pd.testing.assert_frame_equal(
        namespace.load_dataframe("areas.csv"),
        pd.DataFrame({"area": [10, 20], "zone": [0, 1]}),
    )
    manifest = namespace.manifest()
    assert manifest is not None
    assert manifest["schema_version"] == 3
    assert manifest["sources"]["schema_version"] == scenario.schema_version
    assert set(manifest["payloads"]) == {
        "Block_0.pickle",
        "Block_1.pickle",
        "areas.csv",
    }
    assert manifest["sources"] == scenario.source_manifest(["test.input"])
    assert namespace.reference("Block_0.pickle") == {
        "artifact": "graphs",
        "schema_version": 3,
        "key": namespace.key,
        "classification": "derived",
        "parameters": {"unit": "Block"},
        "roles": ["test.input"],
        "payload": "Block_0.pickle",
    }
    assert "cache" not in str(namespace.reference("Block_0.pickle"))


def test_corrupt_payload_and_manifest_are_cache_misses(tmp_path, scenario_factory):
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    scenario = scenario_factory({"test.input": {"path": str(source)}})
    namespace = CacheStore(scenario).namespace(
        "tables", schema_version=2, roles="test.input"
    )
    namespace.save_pickle("first.pickle", {"first": True})
    namespace.save_pickle("second.pickle", {"second": True})

    namespace.payload_path("first.pickle").write_bytes(b"corrupt")

    assert namespace.load_pickle("first.pickle") is None
    assert namespace.load_pickle("second.pickle") == {"second": True}

    manifest = json.loads(namespace.manifest_path.read_text(encoding="utf-8"))
    manifest["parameters"] = {"tampered": True}
    namespace.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert namespace.load_pickle("second.pickle") is None
    assert namespace.manifest() is None


def test_parameter_or_source_changes_create_a_new_key(tmp_path, scenario_factory):
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    scenario = scenario_factory({"test.input": {"path": str(source)}})
    store = CacheStore(scenario)
    baseline = store.namespace(
        "artifact", {"mode": "a"}, schema_version=1, roles="test.input"
    )
    changed_parameter = store.namespace(
        "artifact", {"mode": "b"}, schema_version=1, roles="test.input"
    )
    source.write_text("value\n2\n", encoding="utf-8")
    changed_source = store.namespace(
        "artifact", {"mode": "a"}, schema_version=1, roles="test.input"
    )

    assert baseline.key != changed_parameter.key
    assert baseline.key != changed_source.key


def test_cache_requires_an_explicit_positive_artifact_schema(
    tmp_path, scenario_factory
):
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    scenario = scenario_factory({"test.input": {"path": str(source)}})
    store = CacheStore(scenario)

    with pytest.raises(TypeError, match="schema_version"):
        store.namespace("artifact", roles="test.input")
    with pytest.raises(ValueError, match="positive integer"):
        store.namespace("artifact", schema_version=0, roles="test.input")

    first = store.namespace("artifact", schema_version=2, roles="test.input")
    second = store.namespace("artifact", schema_version=3, roles="test.input")

    assert first.key != second.key
    assert first.path.parent.name == "v2"
    assert second.path.parent.name == "v3"


def test_restricted_cache_modes_are_exact_and_do_not_chmod_root(
    tmp_path, scenario_factory
):
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    cache_root = tmp_path / "cache"
    cache_root.mkdir(mode=0o755)
    cache_root.chmod(0o755)
    scenario = scenario_factory(
        {"test.input": {"path": str(source), "classification": "restricted"}},
        cache_root=cache_root,
    )
    namespace = CacheStore(scenario).namespace(
        "restricted-table",
        schema_version=1,
        roles="test.input",
        classification="restricted-derived",
    )

    previous_umask = os.umask(0o077)
    try:
        payload_path = namespace.save_pickle("payload.pkl", {"private": True})
    finally:
        os.umask(previous_umask)

    assert _mode(cache_root) == 0o755
    assert _mode(namespace.path) == 0o770
    assert _mode(namespace.manifest_path) == 0o660
    assert _mode(payload_path) == 0o660
    assert _mode(namespace.lock_path) == 0o660
