import os

import pandas as pd

from Zone_Generation.benchmark.config import (
    BenchmarkTask,
    MatchingRunConfig,
    SimulationSweep,
    optimization_config_to_dict,
    stable_hash,
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
""",
        encoding="utf-8",
    )

    sweep = SimulationSweep.from_yaml(str(config_path))

    assert sweep.mode == "matching"
    assert sweep.matching.enabled is True
    assert sweep.matching.config.endswith("medium_zones_no_reserves_no_sib.yaml")


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
