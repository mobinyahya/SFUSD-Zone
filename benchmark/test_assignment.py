from pathlib import Path
from types import SimpleNamespace

import pytest

from assignment.generated_zones import (
    GENERATED_ZONE_POLICY,
    resolve_generated_zone_batch_configs,
    resolve_generated_zone_configs,
    run_generated_zone_assignments,
    write_generated_zones,
)
from assignment.slurm import build_generated_zone_slurm_plan
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)
from benchmark.assignment import (
    process_solution_assignments,
    run_assignments_for_existing_runs,
)
from benchmark.config import MatchingRunConfig, SimulationSweep
from optimization.config import OptimizationConfig
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import make_grid_problem


ASSIGNMENT_CONFIG = Path(__file__).parents[1] / "assignment/configs/8-19-choice.yaml"
ZONE_ONLY_ASSIGNMENT_CONFIG = (
    Path(__file__).parents[1] / "assignment/configs/8-18-real-pref-zone-only.yaml"
)


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

    assert [entry["name"] for entry in resolved] == base["subconfigs"]
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


def test_batch_resolver_expands_zones_and_policies_under_one_root(tmp_path):
    targets = [
        {
            "id": f"run-{index}-root",
            "zone_file": tmp_path / f"run-{index}/assignment_zones.csv",
            "zone_building_blocks": "block_group",
            "geography_vintage": "2020",
        }
        for index in range(2)
    ]

    base, resolved = resolve_generated_zone_batch_configs(
        ASSIGNMENT_CONFIG,
        targets,
        assignment_folder=tmp_path,
    )

    policy_count = len({entry["policy"] for entry in resolved})
    assert len(resolved) == policy_count * len(targets)
    assert base["subconfigs"] == [entry["name"] for entry in resolved]
    assert len(base["subconfigs"]) == len(set(base["subconfigs"]))
    assert {entry["target"] for entry in resolved} == {
        "run-0-root",
        "run-1-root",
    }
    assert all(
        entry["config"]["paths"]["assignment-folder"] == str(tmp_path.resolve())
        for entry in resolved
    )
    assert {
        entry["config"]["data"]["overrides"]["sources"]["assignment.zones"][
            GENERATED_ZONE_POLICY
        ]
        for entry in resolved
    } == {str(target["zone_file"].resolve()) for target in targets}


def test_generated_zone_batch_publishes_metrics_once_at_root(tmp_path, monkeypatch):
    targets = [
        {
            "id": f"run-{index}-root",
            "zone_file": tmp_path / f"run-{index}/assignment_zones.csv",
            "zone_building_blocks": "block_group",
            "geography_vintage": "2020",
        }
        for index in range(2)
    ]
    resolved_batches = []
    published = []
    monkeypatch.setattr(
        "assignment.generated_zones._write_provenance_config", lambda config: None
    )
    monkeypatch.setattr(
        "assignment.generated_zones._run_resolved_configs",
        lambda resolved, workers: (
            resolved_batches.append((resolved, workers)) or [{"citywide": "report"}]
        ),
    )
    monkeypatch.setattr(
        MarketGenerator,
        "combine_aggregate_metric_reports",
        lambda reports: {"citywide": reports},
    )
    monkeypatch.setattr(
        MarketGenerator,
        "write_aggregate_metric_reports",
        lambda path, reports: published.append((path, reports)),
    )

    run_generated_zone_assignments(
        ASSIGNMENT_CONFIG,
        targets,
        assignment_folder=tmp_path,
        workers=3,
    )

    assert len(resolved_batches) == 1
    resolved = resolved_batches[0][0]
    assert len(resolved) == 2 * len({entry["policy"] for entry in resolved})
    assert resolved_batches[0][1] == 3
    assert published == [(tmp_path.resolve(), {"citywide": [{"citywide": "report"}]})]


def test_existing_run_batch_uses_assignments_subdirectory(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    config = SimpleNamespace(workers=3)
    calls = []
    monkeypatch.setattr(
        "benchmark.assignment.discover_run_dirs", lambda root: [str(run_dir)]
    )
    monkeypatch.setattr(
        "benchmark.assignment.load_solutions",
        lambda path, dataset=None: ([object()], config, {"task_id": "task"}),
    )
    monkeypatch.setattr(
        "benchmark.assignment.MetricsContext",
        lambda solutions, config: SimpleNamespace(solution=object()),
    )
    monkeypatch.setattr(
        "benchmark.assignment.process_solution_assignments",
        lambda *args, **kwargs: [{"id": "task-root"}],
    )
    monkeypatch.setattr(
        "benchmark.assignment.run_generated_zone_assignments",
        lambda assignment_config, targets, **kwargs: calls.append(
            (assignment_config, targets, kwargs)
        ),
    )

    result = run_assignments_for_existing_runs(
        str(tmp_path), MatchingRunConfig(enabled=True, config="assignment.yaml")
    )

    assert result.successful == 1
    assert calls == [
        (
            "assignment.yaml",
            [{"id": "task-root"}],
            {"assignment_folder": tmp_path / "assignments", "workers": 3},
        )
    ]


def test_solution_processing_prepares_root_and_stage_targets(tmp_path):
    problem = make_grid_problem(2, 2)
    problem.level = LevelSpec("BlockGroup", 0)
    solution = ZoneSolution(
        problem=problem,
        assignment={0: 0, 1: 0, 2: 1, 3: 1},
        status="FEASIBLE",
    )
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        workers=3,
        data={
            "scenario": "legacy",
            "overrides": {"filters": {"optimization": {"geography_vintage": "2010"}}},
        },
    )

    targets = process_solution_assignments(
        [solution],
        solution,
        [{"path": "stages/stage_00_BlockGroup_0", "index": 0}],
        str(tmp_path),
        config,
        MatchingRunConfig(
            enabled=True,
            config=str(ASSIGNMENT_CONFIG),
            compute_stage_assignments=True,
        ),
    )

    assert [target["id"] for target in targets] == [
        f"{tmp_path.name}-root",
        f"{tmp_path.name}-stage-0",
    ]
    assert (tmp_path / "assignment_zones.csv").is_file()
    assert (tmp_path / "stages/stage_00_BlockGroup_0/assignment_zones.csv").is_file()


def test_generated_assignment_slurm_plan_uses_eight_jobs(tmp_path):
    target = tmp_path / "run"
    plan, _ = build_generated_zone_slurm_plan(
        ASSIGNMENT_CONFIG,
        [
            {
                "id": "run-root",
                "zone_file": str(target / "assignment_zones.csv"),
                "skip_marker": str(target / ".assignment-skipped"),
                "zone_building_blocks": "block_group",
                "geography_vintage": "2020",
            }
        ],
        assignment_folder=tmp_path,
        plan_dir=tmp_path / "plan",
        max_assignment_jobs=6,
        max_metrics_jobs=2,
    )

    assert len(plan["jobs"]) == 8
    assert len(plan["assignment_tasks"]) == len(plan["subconfigs"])
    assert all("iteration" not in task for task in plan["assignment_tasks"])
    assert plan["assignment_folder"] == str(tmp_path.resolve())
    assert {tuple(task["report_names"]) for task in plan["metrics_tasks"]} == {
        tuple(MarketGenerator.AGGREGATE_METRIC_FILES)
    }
    assert len([job for job in plan["jobs"] if job["kind"] == "assignment"]) == 6
    assert len([job for job in plan["jobs"] if job["kind"].startswith("metrics")]) == 2
    assert plan["job_limits"] == {"assignment": 6, "metrics": 2}


def test_generated_assignment_plan_has_one_task_per_benchmark_run(tmp_path):
    targets = []
    for index in range(2):
        target = tmp_path / f"run-{index}"
        targets.append(
            {
                "id": f"run-{index}-root",
                "zone_file": str(target / "assignment_zones.csv"),
                "skip_marker": str(target / ".assignment-skipped"),
                "zone_building_blocks": "block_group",
                "geography_vintage": "2020",
            }
        )

    plan, _ = build_generated_zone_slurm_plan(
        ZONE_ONLY_ASSIGNMENT_CONFIG,
        targets,
        assignment_folder=tmp_path,
        plan_dir=tmp_path / "plan",
        max_assignment_jobs=2,
        max_metrics_jobs=1,
    )

    assert len(plan["assignment_tasks"]) == len(targets)
    assert len(plan["metrics_tasks"]) == len(targets)
    assert [task["subconfig"] for task in plan["assignment_tasks"]] == [
        "run-0-root:small_zones+no_reserves",
        "run-1-root:small_zones+no_reserves",
    ]
    assert all(
        entry["config"]["paths"]["assignment-folder"] == str(tmp_path.resolve())
        for entry in plan["subconfigs"]
    )
    assert all("metrics_fragment_dir" not in entry for entry in plan["subconfigs"])
    assignment_jobs = [job for job in plan["jobs"] if job["kind"] == "assignment"]
    assert len(assignment_jobs) == 2
    assert {job["cpus"] for job in assignment_jobs} == {25}


def test_write_generated_zones_uses_stable_zone_rows(tmp_path):
    path = tmp_path / "zones.csv"

    mapping = write_generated_zones({1002: 8, 1000: 4, 1001: 4}, path)

    assert mapping == {4: 0, 8: 1}
    assert path.read_text(encoding="utf-8") == "1000,1001\n1002\n"
