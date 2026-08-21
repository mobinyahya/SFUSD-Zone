from pathlib import Path

import pytest

from assignment.generated_zones import (
    GENERATED_ZONE_POLICY,
    resolve_generated_zone_configs,
    write_generated_zones,
)
from assignment.slurm import build_generated_zone_slurm_plan
from benchmark.assignment import process_solution_assignments
from benchmark.config import MatchingRunConfig, SimulationSweep
from optimization.config import OptimizationConfig
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import make_grid_problem


ASSIGNMENT_CONFIG = Path(__file__).parents[1] / "assignment/configs/8-19-choice.yaml"


def test_sweep_uses_one_anchored_assignment_base_config(tmp_path):
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        """
matching:
  enabled: true
  config: assignment.yaml
  compute_stage_assignments: true
""",
        encoding="utf-8",
    )

    sweep = SimulationSweep.from_yaml(str(sweep_path))

    assert sweep.matching == MatchingRunConfig(
        enabled=True,
        config=str((tmp_path / "assignment.yaml").resolve()),
        compute_stage_assignments=True,
    )

    for invalid in (
        "matching:\n  enabled: true\n  configs: []\n",
        "choice_metrics:\n  enabled: true\n",
    ):
        sweep_path.write_text(invalid, encoding="utf-8")
        with pytest.raises(ValueError):
            SimulationSweep.from_yaml(str(sweep_path))


def test_resolver_injects_generated_zones_after_every_policy(tmp_path):
    zone_file = tmp_path / "assignment_zones.csv"
    base, resolved = resolve_generated_zone_configs(
        ASSIGNMENT_CONFIG,
        zone_file=zone_file,
        assignment_folder=tmp_path,
        zone_building_blocks="block_group",
        geography_vintage="2020",
    )

    assert len(resolved) == len(base["subconfigs"]) == 19
    assert base["export-aggregate-metrics"] is True
    assert base["export-local-metrics"] is True
    for entry in resolved:
        config = entry["config"]
        assert config["subconfig-name"] == entry["name"]
        assert config["subconfigs"] == []
        assert config["policies"] == [GENERATED_ZONE_POLICY]
        assert config["zone-building-blocks"] == "block_group"
        assert config["reuse_assignments"] is False
        assert config["data"]["overrides"]["sources"]["assignment.zones"][
            GENERATED_ZONE_POLICY
        ] == str(zone_file.resolve())
        assert config["export-aggregate-metrics"] is True
        assert config["export-local-metrics"] is True


def test_solution_processing_runs_root_and_stage_assignment(tmp_path, monkeypatch):
    problem = make_grid_problem(2, 2)
    problem.level = LevelSpec("BlockGroup", 0)
    solution = ZoneSolution(
        problem=problem,
        assignment={0: 0, 1: 0, 2: 1, 3: 1},
        status="FEASIBLE",
    )
    calls = []
    monkeypatch.setattr(
        "benchmark.assignment.run_generated_zone_assignment",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        workers=3,
        data={
            "scenario": "legacy",
            "overrides": {"filters": {"optimization": {"geography_vintage": "2010"}}},
        },
    )

    process_solution_assignments(
        [solution],
        solution,
        [{"path": "stages/stage_00_BlockGroup_0"}],
        str(tmp_path),
        config,
        MatchingRunConfig(
            enabled=True,
            config=str(ASSIGNMENT_CONFIG),
            compute_stage_assignments=True,
        ),
    )

    assert [call[1]["assignment_folder"] for call in calls] == [
        tmp_path,
        tmp_path / "stages/stage_00_BlockGroup_0",
    ]
    assert all(call[1]["workers"] == 3 for call in calls)


def test_generated_assignment_slurm_plan_uses_eight_jobs(tmp_path):
    target = tmp_path / "run"
    plan, _ = build_generated_zone_slurm_plan(
        ASSIGNMENT_CONFIG,
        [
            {
                "id": "run-root",
                "assignment_folder": str(target),
                "zone_file": str(target / "assignment_zones.csv"),
                "skip_marker": str(target / ".assignment-skipped"),
                "zone_building_blocks": "block_group",
                "geography_vintage": "2020",
            }
        ],
        plan_dir=tmp_path / "plan",
        max_assignment_jobs=6,
        max_metrics_jobs=2,
    )

    assert len(plan["jobs"]) == 8
    assert len([job for job in plan["jobs"] if job["kind"] == "assignment"]) == 6
    assert len([job for job in plan["jobs"] if job["kind"].startswith("metrics")]) == 2
    assert plan["job_limits"] == {"assignment": 6, "metrics": 2}


def test_write_generated_zones_uses_stable_zone_rows(tmp_path):
    path = tmp_path / "zones.csv"

    mapping = write_generated_zones({1002: 8, 1000: 4, 1001: 4}, path)

    assert mapping == {4: 0, 8: 1}
    assert path.read_text(encoding="utf-8") == "1000,1001\n1002\n"
