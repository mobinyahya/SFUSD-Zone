"""SFUSD source-data constraint tests through the metrics package.

These tests intentionally keep constraint validation private to the test suite:
the metrics package is used as the source of computed zone outcomes, but no
public metric columns or result schema are added.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from Zone_Generation.Config.Constants import AREA_ETHNICITIES
from Zone_Generation.pipeline.config import PipelineConfig
from Zone_Generation.pipeline.data.dataset import Dataset
from Zone_Generation.pipeline.problem import ZoneProblem
from Zone_Generation.pipeline.solution import ZoneSolution
from Zone_Generation.pipeline.solvers import get_solver
from Zone_Generation.Running_Analysis.benchmark.config import (
    BenchmarkTask,
    pipeline_config_to_dict,
    stable_hash,
)
from Zone_Generation.Running_Analysis.benchmark.regenerate import regenerate_metrics
from Zone_Generation.Running_Analysis.benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    load_solutions,
    manifest_for,
    result_payload_for,
    save_stage_artifacts,
    stage_names_for,
    write_json,
)
from Zone_Generation.Running_Analysis.metrics import MetricsCalculator


# BlockGroup_1 with 5-zone-AF is currently infeasible; this source-backed
# centroid set resolves and solves at the same aggregation level.
SOURCE_LEVEL = "BlockGroup_1"
SOURCE_CENTROIDS = "7-zone-14"
SOLVE_SECONDS = 20
TOL = 1e-6


@dataclass(frozen=True)
class SourceRun:
    config: PipelineConfig
    problem: ZoneProblem
    solution: ZoneSolution


@pytest.fixture(scope="module")
def sfusd_source_run() -> SourceRun:
    config = PipelineConfig(
        centroids_type=SOURCE_CENTROIDS,
        levels=[SOURCE_LEVEL],
        solver="cp_int",
        strategy="single",
        frl_dev=0.3,
        racial_dev=0.3,
        overage=0.8,
        shortage=0.2,
        max_distance=5,
        solve_time_limits=[SOLVE_SECONDS],
        gap_limits=[0.0],
        workers=1,
    )

    try:
        dataset = Dataset(config)
        problem = dataset.problem_for(SOURCE_LEVEL)
    except (FileNotFoundError, OSError, ImportError) as exc:
        pytest.skip(f"SFUSD source data unavailable: {exc}")

    solution = get_solver(
        "cp_int",
        solve_time_limit=SOLVE_SECONDS,
        workers=1,
        seed=config.seed,
    ).solve(problem)

    assert solution.status in {"OPTIMAL", "FEASIBLE"}, (
        f"source-backed fixture did not solve: status={solution.status}"
    )
    assert solution.assignment, "source-backed fixture produced no assignment"
    return SourceRun(config=config, problem=problem, solution=solution)


def test_source_solution_constraints_are_followed_through_metrics(sfusd_source_run):
    _assert_constraints_followed(
        sfusd_source_run.problem,
        sfusd_source_run.solution,
        sfusd_source_run.config,
    )


def test_source_constraint_helper_flags_mutated_solution(sfusd_source_run):
    bad_assignment = dict(sfusd_source_run.solution.assignment)
    bad_assignment[sfusd_source_run.problem.centroids[0]] = 1
    bad_solution = ZoneSolution(
        problem=sfusd_source_run.problem,
        assignment=bad_assignment,
        status="FEASIBLE",
        objective=sfusd_source_run.solution.objective,
        wall_time=sfusd_source_run.solution.wall_time,
        metadata={"solver": "mutated-test"},
    )

    with pytest.raises(AssertionError):
        _assert_constraints_followed(
            sfusd_source_run.problem,
            bad_solution,
            sfusd_source_run.config,
        )


def test_source_stage_regeneration_preserves_constraint_compliance(
    tmp_path,
    sfusd_source_run,
):
    run_dir = tmp_path / "source_run"
    run_dir.mkdir()
    solutions = [sfusd_source_run.solution]

    config_dict = pipeline_config_to_dict(sfusd_source_run.config)
    config_hash = stable_hash(config_dict)
    task = BenchmarkTask(
        task_id=config_hash[:12],
        config_hash=config_hash,
        config=config_dict,
        output_dir=str(run_dir),
        capacity_slots=1,
    )
    stage_records = save_stage_artifacts(
        solutions,
        str(run_dir),
        stage_names_for(solutions, sfusd_source_run.config),
    )
    metrics = MetricsCalculator(solutions, config=sfusd_source_run.config).compute()

    write_json(
        str(run_dir / RESULT_FILENAME),
        result_payload_for(
            metrics=metrics,
            config=sfusd_source_run.config,
            solutions=solutions,
            task=task,
        ),
    )
    write_json(
        str(run_dir / MANIFEST_FILENAME),
        manifest_for(
            task=task,
            config=sfusd_source_run.config,
            status=sfusd_source_run.solution.status,
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            stages=stage_records,
            final_stage="stage_00_BlockGroup_1",
            error_message=None,
        ),
    )

    regen = regenerate_metrics(
        str(run_dir),
        dataset_factory=lambda config, manifest: Dataset(config),
    )
    assert regen.regenerated == 1

    loaded, loaded_config, _ = load_solutions(
        str(run_dir),
        dataset=Dataset(sfusd_source_run.config),
    )
    assert len(loaded) == 1
    _assert_constraints_followed(loaded[0].problem, loaded[0], loaded_config)


def _assert_constraints_followed(
    problem: ZoneProblem,
    solution: ZoneSolution,
    config: PipelineConfig,
) -> None:
    result = MetricsCalculator(solution, config=config).compute()
    assignment = solution.assignment

    assert set(assignment) == set(problem.nodes), "not every source graph node is assigned"
    for node, zone in assignment.items():
        assert zone in problem.candidate_zones(node), (
            f"node {node} assigned to non-candidate zone {zone}"
        )
    for zone, centroid in enumerate(problem.centroids):
        assert assignment[centroid] == zone, (
            f"centroid {centroid} does not anchor zone {zone}"
        )

    assert result.metrics["contiguous"] == 1
    assert set(result.zone_data) == set(range(problem.Z))

    for zone, zone_data in result.zone_data.items():
        _assert_between(
            zone_data["seat_disparity"],
            -problem.shortage,
            problem.overage,
            f"zone {zone} capacity disparity",
        )
        _assert_between(
            zone_data["frl_pct"],
            problem.district_frl - problem.frl_dev,
            problem.district_frl + problem.frl_dev,
            f"zone {zone} FRL pct",
        )
        for ethnicity in AREA_ETHNICITIES:
            _assert_between(
                zone_data["ethnicity_pcts"][ethnicity],
                problem.district_racial[ethnicity] - problem.racial_dev,
                problem.district_racial[ethnicity] + problem.racial_dev,
                f"zone {zone} {ethnicity} pct",
            )

    _assert_school_count_balance(problem, assignment)


def _assert_school_count_balance(problem, assignment: dict[int, int]) -> None:
    total_schools = sum(problem.num_schools(node) for node in problem.nodes)
    if total_schools == 0:
        return
    avg_schools = total_schools / problem.Z
    school_counts = {zone: 0 for zone in range(problem.Z)}
    for node, zone in assignment.items():
        school_counts[zone] += problem.num_schools(node)

    for zone, count in school_counts.items():
        _assert_between(
            count,
            max(0.0, avg_schools - 1.0),
            avg_schools + 1.0,
            f"zone {zone} school count",
        )


def _assert_between(value, lower: float, upper: float, label: str) -> None:
    assert value is not None, f"{label} is missing"
    assert lower - TOL <= value <= upper + TOL, (
        f"{label}={value} outside [{lower}, {upper}]"
    )
