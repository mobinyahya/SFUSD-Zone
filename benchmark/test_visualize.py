from pathlib import Path

from benchmark.config import BenchmarkTask, VisualizationRunConfig
from benchmark.visualize import (
    VISUALIZATION_MANIFEST_SCHEMA_VERSION,
    ensure_task_visualizations,
    visualization_is_current,
)


def test_completed_visualization_artifacts_are_reused(tmp_path, monkeypatch):
    figure = tmp_path / "visualization_stage_00_BlockGroup_0.png"
    figure.write_bytes(b"png")
    manifest = {
        "config_hash": "config-hash",
        "visualization": {
            "schema_version": VISUALIZATION_MANIFEST_SCHEMA_VERSION,
            "stages": "final",
            "artifacts": [
                {
                    "stage": "stage_00_BlockGroup_0",
                    "figures": [figure.name],
                    "geometry_artifact": "/shared/cache/geometry.pkl",
                    "skipped": None,
                }
            ],
        },
    }
    task = BenchmarkTask(
        task_id="task",
        config_hash="config-hash",
        config={},
        output_dir=str(tmp_path),
        capacity_slots=1,
    )
    settings = VisualizationRunConfig(enabled=True)
    monkeypatch.setattr("benchmark.visualize.load_manifest", lambda path: manifest)
    monkeypatch.setattr(
        "benchmark.visualize.load_solutions",
        lambda path: (_ for _ in ()).throw(
            AssertionError("cached visualizations should not reload solutions")
        ),
    )

    results, cached = ensure_task_visualizations(task, settings)

    assert cached is True
    assert results == []


def test_missing_or_empty_visualization_artifact_is_not_current(tmp_path):
    settings = VisualizationRunConfig(enabled=True)
    manifest = {
        "visualization": {
            "schema_version": VISUALIZATION_MANIFEST_SCHEMA_VERSION,
            "stages": "final",
            "artifacts": [
                {
                    "stage": "stage_00_BlockGroup_0",
                    "figures": ["map.png"],
                    "skipped": None,
                }
            ],
        }
    }

    assert not visualization_is_current(manifest, tmp_path, settings)
    Path(tmp_path / "map.png").touch()
    assert not visualization_is_current(manifest, tmp_path, settings)
