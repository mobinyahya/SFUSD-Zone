import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from assignment.generated_zones import (
    GENERATED_ZONE_POLICY,
    SKIP_MARKER_FILENAME,
    matching_data,
    resolve_generated_zone_batch_config,
    resolve_generated_zone_batch_configs,
    resolve_generated_zone_configs,
    run_generated_zone_assignments,
    write_generated_zones,
)
from assignment.run_custom_config import load_custom_config
from assignment.slurm import build_generated_zone_slurm_plan
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)
from benchmark.assignment import (
    assignment_targets,
    planned_stage_names,
    prepare_assignment_targets,
    process_solution_assignments,
    run_assignments_for_existing_runs,
    run_ready_assignments,
)
from benchmark.runner import MANIFEST_FILENAME, run_optimization_phase, write_json
from benchmark.config import (
    BenchmarkTask,
    MatchingRunConfig,
    SimulationSweep,
    optimization_config_to_dict,
)
from optimization.config import OptimizationConfig
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import make_grid_problem


ASSIGNMENT_CONFIG = Path(__file__).parents[1] / "assignment/configs/8-19-choice.yaml"
#: The data a generated target carries from its optimization run; the matching
#: config's own, so these tests keep their market.
MATCHING_DATA = load_custom_config(ASSIGNMENT_CONFIG)["data"]
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
        assert config["reuse_assignments"] is True
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
            "data": MATCHING_DATA,
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
            "data": MATCHING_DATA,
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


def test_saa_assignment_targets_mark_unproduced_iterations(tmp_path):
    problem = make_grid_problem(2, 2, program_population="All")
    solution = ZoneSolution(
        problem=problem,
        assignment={0: 0, 1: 0, 2: 1, 3: 1},
        status="OPTIMAL",
    )
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="mip",
        strategy="saa",
        max_iterations=3,
        data={
            "scenario": "legacy",
            "overrides": {"filters": {"optimization": {"program_population": "All"}}},
        },
    )

    process_solution_assignments(
        [solution],
        solution,
        [{"path": "stages/iteration_00_BlockGroup_0", "index": 0}],
        str(tmp_path),
        config,
        MatchingRunConfig(
            enabled=True,
            config=str(ASSIGNMENT_CONFIG),
            compute_stage_assignments=True,
        ),
    )

    for index in range(1, 4):
        assert (
            tmp_path / f"stages/iteration_{index:02d}_BlockGroup_0/.assignment-skipped"
        ).is_file()


def test_planned_stage_names_follow_each_strategy():
    levels = ["BlockGroup_1", "BlockGroup_0"]

    assert planned_stage_names({"strategy": "recursive", "levels": levels}) == [
        "stage_00_BlockGroup_1",
        "stage_01_BlockGroup_0",
    ]
    assert planned_stage_names({"strategy": "single", "levels": levels}) == [
        "stage_00_BlockGroup_0"
    ]
    assert planned_stage_names(
        {"strategy": "saa", "levels": levels, "max_iterations": 1}
    ) == ["iteration_00_BlockGroup_0", "iteration_01_BlockGroup_0"]
    assert (
        planned_stage_names(
            {"strategy": "single", "levels": levels, "enumerated_solutions": 3}
        )
        is None
    )
    assert (
        planned_stage_names({"strategy": "iterative_choice", "levels": levels}) is None
    )


def test_stage_targets_reject_unplannable_strategies(tmp_path):
    config = OptimizationConfig(levels=["BlockGroup_0"], strategy="iterative_choice")
    task = BenchmarkTask(
        task_id="task",
        config_hash="hash",
        config=optimization_config_to_dict(config),
        output_dir=str(tmp_path),
        capacity_slots=1,
    )
    matching = MatchingRunConfig(
        enabled=True, config="assignment.yaml", compute_stage_assignments=True
    )

    with pytest.raises(ValueError, match="iterative_choice"):
        assignment_targets([task], matching)
    assert [
        target["id"] for target in assignment_targets([task], MatchingRunConfig())
    ] == ["task-root"]


def test_local_stage_targets_follow_the_saved_stages(tmp_path):
    config = OptimizationConfig(levels=["BlockGroup_0"], strategy="iterative_choice")
    task = BenchmarkTask(
        task_id="task",
        config_hash="hash",
        config=optimization_config_to_dict(config),
        output_dir=str(tmp_path),
        capacity_slots=1,
    )
    matching = MatchingRunConfig(
        enabled=True, config="assignment.yaml", compute_stage_assignments=True
    )
    prepare_assignment_targets([task], [task], matching)
    write_json(
        str(tmp_path / MANIFEST_FILENAME),
        {
            "stages": [
                {"index": index, "path": f"stages/iteration_{index:02d}_BlockGroup_0"}
                for index in range(2)
            ]
        },
    )

    targets = assignment_targets([task], matching, planned_stages=False)

    assert [target["id"] for target in targets] == [
        "task-root",
        "task-stage-0",
        "task-stage-1",
    ]
    assert targets[2]["zone_file"] == str(
        (tmp_path / "stages/iteration_01_BlockGroup_0/assignment_zones.csv").resolve()
    )


def _recursive_task(tmp_path):
    config = OptimizationConfig(levels=["BlockGroup_0"], strategy="recursive")
    return BenchmarkTask(
        task_id="task",
        config_hash="hash",
        config=optimization_config_to_dict(config),
        output_dir=str(tmp_path),
        capacity_slots=1,
    )


def test_a_rerun_drops_zone_files_of_the_previous_run_when_it_starts(
    tmp_path, monkeypatch
):
    task = _recursive_task(tmp_path)
    matching = MatchingRunConfig(
        enabled=True, config="assignment.yaml", compute_stage_assignments=True
    )
    stale = [
        tmp_path / "assignment_zones.csv",
        tmp_path / "stages/stage_00_BlockGroup_0/assignment_zones.csv",
    ]
    for zone_file in stale:
        write_generated_zones({1: 0}, zone_file)
    targets = assignment_targets([task], matching)

    # Planning only reads: a plan that is never submitted deletes nothing.
    prepare_assignment_targets([task], [task], matching)
    assert all(zone_file.exists() for zone_file in stale)

    def fail(self):
        raise RuntimeError("optimization failed")

    monkeypatch.setattr(OptimizationConfig, "make_dataset", fail)
    result = run_optimization_phase(task)

    assert result.status == "ERROR"
    assert not any(zone_file.exists() for zone_file in stale)
    assert run_ready_assignments(targets, matching, tmp_path, workers=1) == 0


def test_skipped_tasks_missing_stage_targets_are_rejected(tmp_path):
    task = _recursive_task(tmp_path)
    matching = MatchingRunConfig(
        enabled=True, config="assignment.yaml", compute_stage_assignments=True
    )
    stage = tmp_path / "stages/stage_00_BlockGroup_0"
    write_json(
        str(tmp_path / MANIFEST_FILENAME),
        {"stages": [{"index": 0, "name": stage.name, "path": f"stages/{stage.name}"}]},
    )
    write_generated_zones({1: 0}, tmp_path / "assignment_zones.csv")

    with pytest.raises(ValueError, match=stage.name):
        prepare_assignment_targets([task], [], matching)

    stage.mkdir(parents=True)
    (stage / SKIP_MARKER_FILENAME).write_text("ineligible\n", encoding="utf-8")
    prepare_assignment_targets([task], [], matching)


def test_failed_runs_kept_as_complete_have_nothing_to_prepare(tmp_path):
    task = _recursive_task(tmp_path)
    write_json(str(tmp_path / MANIFEST_FILENAME), {"status": "ERROR", "stages": []})

    prepare_assignment_targets(
        [task], [], MatchingRunConfig(enabled=True, config="assignment.yaml")
    )

    assert not (tmp_path / "assignment_zones.csv").exists()


def test_reusable_assignments_are_keyed_on_the_zone_file_contents(tmp_path):
    from loaders.config import ResolvedSource

    zone_file = tmp_path / "assignment_zones.csv"
    market = SimpleNamespace(
        data_scenario=SimpleNamespace(
            source_map=lambda role: {GENERATED_ZONE_POLICY: ResolvedSource(zone_file)}
        )
    )
    write_generated_zones({1: 0, 2: 1}, zone_file)
    before = MarketGenerator._zone_signature(market, GENERATED_ZONE_POLICY)
    write_generated_zones({1: 1, 2: 0}, zone_file)
    after = MarketGenerator._zone_signature(market, GENERATED_ZONE_POLICY)

    assert before != after
    assert MarketGenerator._zone_signature(market, "missing") is None


def test_generated_assignment_slurm_plan_uses_nine_jobs(tmp_path):
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
                "data": MATCHING_DATA,
            }
        ],
        assignment_folder=tmp_path,
        plan_dir=tmp_path / "plan",
        max_assignment_jobs=6,
        max_metrics_jobs=2,
    )

    # 6 assignment + 2 metrics + the separate one-CPU finalizer.
    assert len(plan["jobs"]) == 9
    assert len(plan["assignment_tasks"]) == len(plan["subconfigs"])
    assert all("iteration" not in task for task in plan["assignment_tasks"])
    assert plan["assignment_folder"] == str(tmp_path.resolve())
    assert {tuple(task["report_names"]) for task in plan["metrics_tasks"]} == {
        tuple(MarketGenerator.AGGREGATE_METRIC_FILES)
    }
    assert len([job for job in plan["jobs"] if job["kind"] == "assignment"]) == 6
    assert len([job for job in plan["jobs"] if job["kind"] == "metrics"]) == 2
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
                "data": MATCHING_DATA,
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


def test_completed_run_target_is_rebuilt_from_saved_files(tmp_path):
    import json

    from assignment.generated_zones import (
        GENERATED_ZONE_FILENAME,
        SKIP_MARKER_FILENAME,
    )
    from benchmark.assignment import prepare_completed_run_target

    feasible = tmp_path / "feasible"
    feasible.mkdir()
    (feasible / "solution_Block_0.json").write_text(
        json.dumps({"status": "FEASIBLE", "metadata": {}}), encoding="utf-8"
    )
    (feasible / "zone_dict_area_Block_0.json").write_text(
        json.dumps({"11": 1, "10": 0, "12": 1}), encoding="utf-8"
    )
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "solution_Block_0.json").write_text(
        json.dumps({"status": "UNKNOWN", "metadata": {}}), encoding="utf-8"
    )

    assert prepare_completed_run_target(feasible, "Block_0") is True
    rows = (feasible / GENERATED_ZONE_FILENAME).read_text().splitlines()
    assert rows == ["10", "11,12"]
    assert not (feasible / SKIP_MARKER_FILENAME).exists()

    assert prepare_completed_run_target(unknown, "Block_0") is False
    assert (unknown / SKIP_MARKER_FILENAME).is_file()
    assert not (unknown / GENERATED_ZONE_FILENAME).exists()


def _variant_data(**assignment_filters):
    data = copy.deepcopy(MATCHING_DATA)
    data["overrides"]["filters"]["assignment"].update(assignment_filters)
    return data


def test_each_target_is_matched_in_its_own_optimization_data(tmp_path):
    variant = _variant_data(special_programs="include")
    targets = [
        {
            "id": f"run-{index}-root",
            "zone_file": tmp_path / f"run-{index}/assignment_zones.csv",
            "zone_building_blocks": "block_group",
            "geography_vintage": "2020",
            "data": data,
        }
        for index, data in enumerate([variant, MATCHING_DATA])
    ]
    base_config = load_custom_config(ASSIGNMENT_CONFIG)
    base_config["data"] = _variant_data(special_programs="exclude_only_special")

    with pytest.warns(UserWarning, match="matching config's data is ignored"):
        base, resolved = resolve_generated_zone_batch_config(
            base_config, targets, assignment_folder=tmp_path
        )

    assert base["data"] == variant
    for entry in resolved:
        filters = entry["config"]["data"]["overrides"]["filters"]["assignment"]
        expected = (
            "include" if entry["target"] == "run-0-root" else ("exclude_any_special")
        )
        assert filters["special_programs"] == expected


def test_matching_requires_target_data_and_agreeing_selectors(tmp_path):
    target = {
        "id": "run-root",
        "zone_file": tmp_path / "assignment_zones.csv",
        "zone_building_blocks": "block",
        "geography_vintage": "2020",
    }
    base_config = load_custom_config(ASSIGNMENT_CONFIG)
    with pytest.raises(ValueError, match="must carry their optimization run's data"):
        resolve_generated_zone_batch_config(
            base_config, [target], assignment_folder=tmp_path
        )

    # Capacities set for the zoning only would match it in other seats.
    data = {
        "scenario": "attendance-areas-2627",
        "overrides": {
            "filters": {"optimization": {"capacity_scenario": "ideal_baseline_2627"}}
        },
    }
    with pytest.raises(
        ValueError, match="capacity_scenario: 'ideal_baseline_2627' != 'programs'"
    ):
        matching_data(data)
    data["overrides"]["filters"]["assignment"] = {"capacity_scenario": "ideal_baseline_2627"}
    assert matching_data(data) == data
