from pathlib import Path

import pytest
import yaml

from benchmark.config import (
    SimulationSweep,
    _benchmark_source_manifest,
    config_snapshot,
    optimization_config_from_dict,
    optimization_config_hash,
)
from optimization.config import OptimizationConfig


def _write_custom_sweep(tmp_path: Path, location: str = "optimization_defaults"):
    config_dir = tmp_path / "sweep-config"
    scenario_dir = config_dir / "scenarios"
    inputs_dir = config_dir / "inputs"
    scenario_dir.mkdir(parents=True)
    inputs_dir.mkdir()

    (scenario_dir / "scenario-source.csv").write_text(
        "scenario source", encoding="utf-8"
    )
    source_path = inputs_dir / "students.csv"
    assignment_path = inputs_dir / "assignment-students.csv"
    assignment_programs_path = inputs_dir / "assignment-programs.csv"
    companion_path = inputs_dir / "students.meta"
    source_path.write_text("students-v1", encoding="utf-8")
    assignment_path.write_text("assignment-students-v1", encoding="utf-8")
    assignment_programs_path.write_text("assignment-programs-v1", encoding="utf-8")
    companion_path.write_text("metadata-v1", encoding="utf-8")
    (scenario_dir / "choice-estimate.csv").write_text(
        "choice-estimate-v1", encoding="utf-8"
    )
    scenario_path = scenario_dir / "custom.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "id": "benchmark-custom",
                "sources": {
                    "optimization.students": {"path": "scenario-source.csv"},
                    "choice.estimate": {"path": "choice-estimate.csv"},
                },
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
                        "include_citywide_zoning": False,
                        "include_citywide_choice_opt": False,
                        "include_mission_bay": True,
                    },
                    "assignment": {
                        "year": "2324",
                        "grade": "KG",
                        "student_population": "applicant",
                        "rounds": [1],
                        "special_programs": "include",
                        "capacity_profile": "status_quo",
                        "include_mission_bay": True,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    data = {
        "scenario": "scenarios/custom.yaml",
        "overrides": {
            "roots": {"data": "relative-data", "cache": "relative-cache"},
            "sources": {
                "optimization.students": {
                    "path": "inputs/students.csv",
                    "companions": ["inputs/students.meta"],
                },
                "assignment.students": {"path": "inputs/assignment-students.csv"},
                "assignment.programs": {"path": "inputs/assignment-programs.csv"},
            },
        },
    }
    raw = {"optimization_defaults": {"levels": ["BlockGroup_0"]}}
    if location == "optimization_defaults":
        raw["optimization_defaults"]["data"] = data
    elif location == "tasks":
        raw["tasks"] = [{"data": data}]
    elif location == "sweep":
        raw["sweep"] = {"data": [data]}
    else:  # pragma: no cover - test helper contract
        raise ValueError(location)

    sweep_path = config_dir / "sweep.yaml"
    sweep_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    return sweep_path, source_path, assignment_path, companion_path


@pytest.mark.parametrize("location", ["optimization_defaults", "tasks", "sweep"])
def test_sweep_anchors_every_supported_data_shape_before_cwd_change(
    tmp_path, monkeypatch, location
):
    sweep_path, source_path, assignment_path, companion_path = _write_custom_sweep(
        tmp_path, location
    )
    sweep = SimulationSweep.from_yaml(str(sweep_path))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    task = sweep.generate_tasks()[0]

    data = task.config["data"]
    assert data["scenario"] == str(
        (sweep_path.parent / "scenarios/custom.yaml").resolve()
    )
    assert data["overrides"]["roots"] == {
        "data": str((sweep_path.parent / "relative-data").resolve()),
        "cache": str((sweep_path.parent / "relative-cache").resolve()),
    }
    assert data["overrides"]["sources"]["optimization.students"] == {
        "path": str(source_path.resolve()),
        "companions": [str(companion_path.resolve())],
    }
    assert data["overrides"]["sources"]["assignment.students"] == {
        "path": str(assignment_path.resolve())
    }

    run_snapshot = config_snapshot(task.config)
    restored = optimization_config_from_dict(run_snapshot)
    assert restored.data_scenario.source("optimization.students").path == source_path
    assert optimization_config_hash(run_snapshot) == task.config_hash


def test_task_identity_changes_when_source_bytes_change_at_same_path(tmp_path):
    sweep_path, source_path, _, _ = _write_custom_sweep(tmp_path)
    sweep = SimulationSweep.from_yaml(str(sweep_path))

    first = sweep.generate_tasks()[0]
    source_path.write_text("students-version-two", encoding="utf-8")
    second = sweep.generate_tasks()[0]

    assert first.config["data"] == second.config["data"]
    assert first.config_hash != second.config_hash
    assert first.task_id != second.task_id
    assert first.output_dir != second.output_dir


def test_task_identity_changes_when_central_assignment_year_source_changes(tmp_path):
    sweep_path, _, assignment_path, _ = _write_custom_sweep(tmp_path)
    sweep = SimulationSweep.from_yaml(str(sweep_path))

    first = sweep.generate_tasks()[0]
    assignment_path.write_text("assignment-students-v2", encoding="utf-8")
    second = sweep.generate_tasks()[0]

    assert first.config_hash != second.config_hash


def test_task_identity_ignores_unconsumed_assignment_sources(tmp_path):
    sweep_path, _, _, _ = _write_custom_sweep(tmp_path)
    sweep = SimulationSweep.from_yaml(str(sweep_path))

    first = sweep.generate_tasks()[0]
    (sweep_path.parent / "inputs/assignment-programs.csv").write_text(
        "assignment-programs-v2", encoding="utf-8"
    )
    second = sweep.generate_tasks()[0]

    assert first.config_hash == second.config_hash


def test_saa_manifest_hashes_matching_policy_files():
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="mip",
        strategy="saa",
        data={
            "scenario": "legacy",
            "overrides": {"filters": {"optimization": {"program_population": "All"}}},
        },
    )

    manifest = _benchmark_source_manifest(config)

    assert set(manifest["matching_policy"]) == {
        "assignment/configs/base_config.yaml",
        "assignment/configs/policy_configs/status_quo.yaml",
    }


@pytest.mark.parametrize("strategy, reads_market", [("saa", True), ("single", False)])
def test_market_manifest_tracks_other_vintage_blocks(strategy, reads_market):
    vintage = {"geography_vintage": "2020"}
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="mip",
        strategy=strategy,
        data={
            "scenario": "legacy",
            "overrides": {
                "filters": {
                    "optimization": {"program_population": "All", **vintage},
                    "assignment": vintage,
                }
            },
        },
    )

    sources = _benchmark_source_manifest(config)["sources"]

    other_vintage = {
        "assignment.geography.2010.blocks",
        "assignment.geography.2010.crosswalk",
    }
    assert (other_vintage <= set(sources)) is reads_market


@pytest.mark.parametrize("objective", ["mid", "stable_matching", "boundary"])
def test_dw_manifest_tracks_matching_inputs_only_for_welfare(objective):
    config = OptimizationConfig(
        strategy="dantzig_wolfe",
        solver="cp_bool",
        dw_objective=objective,
        data={
            "scenario": "legacy",
            "overrides": {"filters": {"optimization": {"program_population": "All"}}},
        },
    )
    manifest = _benchmark_source_manifest(config)
    assert ("matching_policy" in manifest) == (objective != "boundary")


@pytest.mark.parametrize(
    "strategy, reads_market",
    [
        ("single", False),
        ("recursive", False),
        ("iterative_choice", False),
        ("mid", True),
        ("mid_decomp", True),
        ("saa", True),
        ("direct_samples", True),
        ("short_bursts_choice", True),
        ("priced_access", True),
        ("stable_cutoff", True),
        ("dantzig_wolfe", True),
    ],
)
def test_manifest_hashes_the_assignment_market_for_strategies_that_read_it(
    strategy, reads_market
):
    solver = {"stable_cutoff": "mip", "short_bursts_choice": "recom"}
    config = OptimizationConfig(
        strategy=strategy,
        solver=solver.get(strategy, "cp_bool"),
        data={
            "scenario": "legacy",
            "overrides": {"filters": {"optimization": {"program_population": "All"}}},
        },
    )
    assert config.make_strategy().reads_assignment_market is reads_market
    manifest = _benchmark_source_manifest(config)
    assert ("matching_policy" in manifest) is reads_market
    assert "assignment" in manifest["filters"]


def test_visualization_config_anchors_shared_artifact_dir(tmp_path):
    config_path = tmp_path / "configs" / "sweep.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "visualization": {
                    "enabled": True,
                    "stages": "all",
                    "artifact_dir": "../shared-viz-cache",
                }
            }
        ),
        encoding="utf-8",
    )

    sweep = SimulationSweep.from_yaml(str(config_path))

    assert sweep.visualization.enabled is True
    assert sweep.visualization.stages == "all"
    assert sweep.visualization.artifact_dir == str(
        (tmp_path / "shared-viz-cache").resolve()
    )


def test_visualization_config_rejects_unknown_stage_mode(tmp_path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        yaml.safe_dump({"visualization": {"stages": "coarse"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="visualization.stages"):
        SimulationSweep.from_yaml(str(config_path))


def test_simulation_sweep_supports_auto_max_distance(tmp_path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "optimization_defaults": {
                    "levels": ["BlockGroup_0"],
                    "max_distance": "auto",
                },
                "sweep": {"solver": ["cp_int", "cp_bool"]},
            }
        ),
        encoding="utf-8",
    )

    sweep = SimulationSweep.from_yaml(str(config_path))
    assert sweep.optimization_defaults["max_distance"] == "auto"
    tasks = sweep.generate_tasks()
    assert len(tasks) == 2
    for task in tasks:
        assert task.config["max_distance"] == "auto"
        opt_cfg = task.optimization_config()
        assert opt_cfg.max_distance == "auto"


def _with_optimization(sweep_path: Path, **values) -> None:
    raw = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    raw["optimization_defaults"].update(values)
    sweep_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def test_task_identity_follows_hint_zoning_bytes_not_its_path(tmp_path):
    sweep_path, _, _, _ = _write_custom_sweep(tmp_path)
    first_hint = tmp_path / "a" / "zone_dict_area_Block_0.json"
    moved_hint = tmp_path / "b" / "zone_dict_area_Block_0.json"
    for path in (first_hint, moved_hint):
        path.parent.mkdir()
        path.write_text('{"1": 0}', encoding="utf-8")

    _with_optimization(sweep_path, hint_zoning=str(first_hint))
    first = SimulationSweep.from_yaml(str(sweep_path)).generate_tasks()[0]
    _with_optimization(sweep_path, hint_zoning=str(moved_hint))
    moved = SimulationSweep.from_yaml(str(sweep_path)).generate_tasks()[0]
    moved_hint.write_text('{"1": 1}', encoding="utf-8")
    rewritten = SimulationSweep.from_yaml(str(sweep_path)).generate_tasks()[0]

    assert first.config_hash == moved.config_hash
    assert rewritten.config_hash != moved.config_hash


def test_adaptive_choice_objectives_hash_the_assignment_market(tmp_path):
    """Their MID market reads assignment programs, as the matching strategies do."""

    sweep_path, _, _, _ = _write_custom_sweep(tmp_path)
    programs = sweep_path.parent / "inputs/assignment-programs.csv"
    _with_optimization(
        sweep_path,
        solver="recom",
        recom_objective="capacity_match",
    )
    sweep = SimulationSweep.from_yaml(str(sweep_path))

    first = sweep.generate_tasks()[0]
    programs.write_text("assignment-programs-v2", encoding="utf-8")
    second = sweep.generate_tasks()[0]

    assert first.config_hash != second.config_hash


@pytest.mark.parametrize(
    "objective, method",
    [
        ("matching_welfare", "cp_bool"),
        ("saa_welfare", "mip"),
        ("choice_welfare", "mip"),
    ],
)
def test_pipeline_hashes_assignment_sources_used_only_by_later_steps(
    tmp_path, objective, method
):
    sweep_path, _, _, _ = _write_custom_sweep(tmp_path)
    raw = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    raw["optimization_defaults"]["data"]["overrides"].setdefault("filters", {})[
        "optimization"
    ] = {"program_population": "All"}
    raw["optimization_defaults"]["steps"] = [
        {
            "name": "shape",
            "level": "BlockGroup_0",
            "method": "cp_int",
            "objective": "cut_edges",
        },
        {
            "name": "access",
            "level": "BlockGroup_0",
            "method": method,
            "objective": objective,
        },
    ]
    steps = raw["optimization_defaults"]["steps"]
    raw["sweep"] = {"steps": [steps[:1], steps]}
    sweep_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    sweep = SimulationSweep.from_yaml(str(sweep_path))

    first = sweep.generate_tasks()
    (sweep_path.parent / "inputs/assignment-programs.csv").write_text(
        "assignment-programs-v2", encoding="utf-8"
    )
    second = sweep.generate_tasks()

    assert first[0].config_hash == second[0].config_hash
    assert first[1].config_hash != second[1].config_hash
    assert len(first[1].config["steps"]) == 2
