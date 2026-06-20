import os

import pandas as pd

from Zone_Generation.benchmark.config import (
    BenchmarkTask,
    ChoiceMetricsRunConfig,
    MatchingRunConfig,
    SimulationSweep,
    optimization_config_to_dict,
    stable_hash,
)
from Zone_Generation.benchmark.choice_metrics import (
    CHOICE_AVG_STUDENT_DISTANCE,
    CHOICE_FRL_DISSIMILARITY,
    CHOICE_PERCENT_DESIGNATED,
    CHOICE_PERCENT_TOP_1,
    CHOICE_PERCENT_TOP_3,
    CHOICE_PERCENT_UNASSIGNED,
    CHOICE_SCHOOLS_ABOVE_10PCT_DISTRICT_FRL,
    CHOICE_TOTAL_MNL_UTILITY,
    compute_choice_metrics_for_run,
    preserve_choice_metrics_payload,
    run_choice_metrics_for_existing_runs,
)
from Zone_Generation.benchmark.matching import (
    preserve_matching_payload,
    run_matching_for_existing_runs,
    run_matching_for_solution,
)
from Zone_Generation.benchmark.matching import runner as matching_runner
from Zone_Generation.benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    manifest_for,
    result_payload_for,
    save_stage_artifacts,
    stage_names_for,
    write_json,
)
from Zone_Generation.metrics import MetricsCalculator
from Zone_Generation.optimization.config import OptimizationConfig
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.tests.synthetic import FakeDataset, make_grid_problem


def test_sweep_yaml_accepts_matching_config(tmp_path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        f"""
mode: matching
optimization_defaults:
  levels: ['BlockGroup_0']
  graphs_dir: '{tmp_path / "graphs"}'
matching:
  enabled: true
  config: Zone_Generation/benchmark/matching/medium_zones_no_reserves_no_sib.yaml
  compute_stage_assignments: true
""",
        encoding="utf-8",
    )

    sweep = SimulationSweep.from_yaml(str(config_path))

    assert sweep.mode == "matching"
    assert sweep.matching.enabled is True
    assert sweep.matching.config.endswith("medium_zones_no_reserves_no_sib.yaml")
    assert sweep.matching.compute_stage_assignments is True


def test_sweep_yaml_accepts_choice_metrics_config(tmp_path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        f"""
mode: choice-metrics
optimization_defaults:
  levels: ['BlockGroup_0']
  graphs_dir: '{tmp_path / "graphs"}'
choice_metrics:
  enabled: true
  compute_stage_metrics: true
""",
        encoding="utf-8",
    )

    sweep = SimulationSweep.from_yaml(str(config_path))

    assert sweep.mode == "choice_metrics"
    assert sweep.choice_metrics.enabled is True
    assert sweep.choice_metrics.compute_stage_metrics is True


def test_run_matching_for_solution_writes_mapping_and_populations(tmp_path, monkeypatch):
    captured_config = _stub_student_assignment(monkeypatch)
    solution = _solution()

    result = run_matching_for_solution(
        solution,
        str(tmp_path),
        MatchingRunConfig(enabled=True),
    )

    assert result.status == "OK"
    assert result.metrics["matching_assignment_files"] == 1
    assert result.metrics["matching_students_total"] == 3
    assert result.metrics["matching_students_assigned_mean"] == 2
    assert result.metrics["matching_unassigned_rate_mean"] == 1 / 3
    assert captured_config["value"]["zone-building-blocks"] == "block_group"
    assert captured_config["value"]["policies"] == ["generated_zones"]

    zones_text = (tmp_path / "matching" / "zones.csv").read_text(encoding="utf-8")
    assert "1000,1001" in zones_text
    assert "1002,1003" in zones_text

    assignments = pd.read_csv(
        tmp_path / "matching" / "student_school_assignments.csv"
    )
    assert assignments["school_id"].dropna().astype(int).tolist() == [664, 665]

    school_populations = pd.read_csv(tmp_path / "matching" / "school_populations.csv")
    assert school_populations["assigned_count"].tolist() == [1, 1]


def test_choice_metrics_noop_without_assignments(tmp_path):
    result = compute_choice_metrics_for_run(
        str(tmp_path),
        ChoiceMetricsRunConfig(enabled=True),
    )

    assert result is None


def test_choice_metrics_compute_assignment_outcomes(tmp_path):
    assignments_dir = tmp_path / "matching" / "assignments_raw" / "policy"
    assignments_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "studentno": [1, 2, 3, 4],
            "programno": [11, 12, 13, 0],
            "programcodes": ["101-GE-KG", "101-GE-KG", "202-GE-KG", ""],
            "rank": [1, 4, 3, 5],
            "designation": [0, 1, 0, 0],
            "assignment_dist": [1.0, 2.0, 3.0, None],
            "assigned_utility": [10.0, 20.0, 30.0, None],
            "freelunch_prob": [0.8, 0.8, 0.1, 0.1],
            "reducedlunch_prob": [0.1, 0.1, 0.0, 0.0],
        }
    ).to_csv(assignments_dir / "assignment.csv", index=False)

    result = compute_choice_metrics_for_run(
        str(tmp_path),
        ChoiceMetricsRunConfig(enabled=True),
    )

    metrics = result.metrics
    assert result.status == "OK"
    assert metrics[CHOICE_AVG_STUDENT_DISTANCE] == 2.0
    assert metrics[CHOICE_SCHOOLS_ABOVE_10PCT_DISTRICT_FRL] == 0.5
    assert metrics[CHOICE_PERCENT_UNASSIGNED] == 0.25
    assert metrics[CHOICE_PERCENT_DESIGNATED] == 1 / 3
    assert metrics[CHOICE_PERCENT_TOP_1] == 1 / 3
    assert metrics[CHOICE_PERCENT_TOP_3] == 2 / 3
    assert metrics[CHOICE_TOTAL_MNL_UTILITY] == 60.0
    assert round(metrics[CHOICE_FRL_DISSIMILARITY], 6) == round(0.7655502392344498, 6)
    assert (tmp_path / "matching" / "choice_metrics_by_assignment.csv").exists()


def test_choice_metrics_average_total_mnl_utility_across_assignments(tmp_path):
    assignments_dir = tmp_path / "matching" / "assignments_raw" / "policy"
    assignments_dir.mkdir(parents=True)
    base = {
        "studentno": [1, 2],
        "programno": [11, 12],
        "programcodes": ["101-GE-KG", "202-GE-KG"],
        "rank": [1, 2],
        "designation": [0, 0],
        "freelunch_prob": [0.3, 0.7],
        "reducedlunch_prob": [0.0, 0.0],
    }
    pd.DataFrame({**base, "assigned_utility": [1.0, 2.0]}).to_csv(
        assignments_dir / "assignment_a.csv",
        index=False,
    )
    pd.DataFrame({**base, "assigned_utility": [3.0, 4.0]}).to_csv(
        assignments_dir / "assignment_b.csv",
        index=False,
    )

    result = compute_choice_metrics_for_run(
        str(tmp_path),
        ChoiceMetricsRunConfig(enabled=True),
    )

    assert result.metrics[CHOICE_TOTAL_MNL_UTILITY] == 5.0


def test_choice_metrics_mode_updates_existing_result(tmp_path):
    run_dir, _ = _write_synthetic_run(tmp_path)
    assignments_dir = run_dir / "matching" / "assignments_raw" / "policy"
    assignments_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "studentno": [1, 2],
            "programno": [11, 0],
            "programcodes": ["101-GE-KG", ""],
            "rank": [1, 2],
            "designation": [0, 0],
            "assignment_dist": [1.25, None],
            "freelunch_prob": [0.3, 0.7],
            "reducedlunch_prob": [0.0, 0.0],
        }
    ).to_csv(assignments_dir / "assignment.csv", index=False)

    batch = run_choice_metrics_for_existing_runs(
        str(tmp_path),
        ChoiceMetricsRunConfig(enabled=True),
    )

    assert batch.successful == 1
    result = matching_runner._load_json(os.path.join(run_dir, RESULT_FILENAME))
    assert result["metrics"]["num_zones"] == 2
    assert result["metrics"][CHOICE_AVG_STUDENT_DISTANCE] == 1.25
    assert result["choice_metrics"]["status"] == "OK"


def test_matching_mode_updates_existing_result(tmp_path, monkeypatch):
    _stub_student_assignment(monkeypatch)
    run_dir, problem = _write_synthetic_run(tmp_path)

    batch = run_matching_for_existing_runs(
        str(tmp_path),
        MatchingRunConfig(enabled=True),
        dataset_factory=lambda config, manifest: FakeDataset(problem),
    )

    assert batch.successful == 1
    result = matching_runner._load_json(os.path.join(run_dir, RESULT_FILENAME))
    assert result["matching"]["status"] == "OK"
    assert result["metrics"]["matching_assignment_files"] == 1


def test_stage_matching_and_choice_metrics_are_opt_in(tmp_path, monkeypatch):
    _stub_student_assignment(monkeypatch)
    run_dir, problem = _write_synthetic_run(tmp_path)

    batch = run_matching_for_existing_runs(
        str(tmp_path),
        MatchingRunConfig(enabled=True, compute_stage_assignments=True),
        choice_metrics=ChoiceMetricsRunConfig(enabled=True, compute_stage_metrics=True),
        dataset_factory=lambda config, manifest: FakeDataset(problem),
    )

    assert batch.successful == 1
    result = matching_runner._load_json(os.path.join(run_dir, RESULT_FILENAME))
    stage_name = "stage_00_BlockGroup_0"
    stage = result["run"]["stages"][0]
    assert stage["matching"]["status"] == "OK"
    assert stage["choice_metrics"]["status"] == "OK"
    assert stage["choice_metrics_metrics"][CHOICE_PERCENT_UNASSIGNED] == 1 / 3
    assert result["stage_matching"]["stages"][stage_name]["matching"]["status"] == "OK"
    assert (
        run_dir
        / "stages"
        / stage_name
        / "matching"
        / "choice_metrics_by_assignment.csv"
    ).exists()


def test_preserve_matching_payload_keeps_existing_matching_metrics():
    new_payload = {"metrics": {"num_zones": 2}}
    previous_payload = {
        "matching": {"status": "OK"},
        "metrics": {"matching_assignment_files": 1, "num_zones": 99},
    }

    preserve_matching_payload(new_payload, previous_payload)

    assert new_payload["matching"] == {"status": "OK"}
    assert new_payload["metrics"]["num_zones"] == 2
    assert new_payload["metrics"]["matching_assignment_files"] == 1


def test_preserve_choice_metrics_payload_keeps_existing_choice_metrics():
    new_payload = {"metrics": {"num_zones": 2}}
    previous_payload = {
        "choice_metrics": {"status": "OK"},
        "metrics": {CHOICE_AVG_STUDENT_DISTANCE: 1.5, "num_zones": 99},
    }

    preserve_choice_metrics_payload(new_payload, previous_payload)

    assert new_payload["choice_metrics"] == {"status": "OK"}
    assert new_payload["metrics"]["num_zones"] == 2
    assert new_payload["metrics"][CHOICE_AVG_STUDENT_DISTANCE] == 1.5


def _stub_student_assignment(monkeypatch):
    captured = {}

    def fake_run(config, assignments_dir):
        captured["value"] = config
        output = assignments_dir / config["subconfig-name"] / "assignment.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "studentno": [1, 2, 3],
                "programno": [11, 12, 0],
                "programcodes": ["664-GE-KG", "665-GE-KG", ""],
                "rank": [1, 2, 0],
                "designation": [0, 0, 0],
                "In-Zone Rank": [1, 2, 0],
            }
        ).to_csv(output, index=False)

    monkeypatch.setattr(matching_runner, "_run_student_assignment", fake_run)
    return captured


def _solution():
    problem = make_grid_problem(2, 2)
    return ZoneSolution(
        problem=problem,
        assignment={0: 0, 1: 0, 2: 1, 3: 1},
        status="FEASIBLE",
        objective=1.0,
        wall_time=0.5,
    )


def _write_synthetic_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    solution = _solution()
    problem = solution.problem
    config = OptimizationConfig(
        centroids_type="5-zone-AF",
        levels=["BlockGroup_0"],
        solver="local_search",
        strategy="single",
        frl_dev=1.0,
        racial_dev=1.0,
        overage=5.0,
        shortage=0.0,
        workers=1,
        graphs_dir=str(tmp_path / "graphs"),
    )
    config_dict = optimization_config_to_dict(config)
    config_hash = stable_hash(config_dict)
    task = BenchmarkTask(
        task_id=config_hash[:12],
        config_hash=config_hash,
        config=config_dict,
        output_dir=str(run_dir),
        capacity_slots=1,
    )
    solutions = [solution]
    stage_records = save_stage_artifacts(
        solutions,
        str(run_dir),
        stage_names_for(solutions, config),
    )
    metrics = MetricsCalculator(solutions, config=config).compute()
    solution.save(str(run_dir))
    write_json(
        os.path.join(run_dir, RESULT_FILENAME),
        result_payload_for(metrics=metrics, config=config, solutions=solutions, task=task),
    )
    write_json(
        os.path.join(run_dir, MANIFEST_FILENAME),
        manifest_for(
            task=task,
            config=config,
            status="FEASIBLE",
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            stages=stage_records,
            final_stage="stage_00_BlockGroup_0",
            error_message=None,
        ),
    )
    return run_dir, problem
