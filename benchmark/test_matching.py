import os

import pandas as pd
import yaml

from benchmark.config import (
    BenchmarkTask,
    ChoiceMetricsRunConfig,
    MatchingConfigSpec,
    MatchingRunConfig,
    SimulationSweep,
    optimization_config_to_dict,
    stable_hash,
)
from benchmark.choice_metrics import (
    CHOICE_AVG_MNL_UTILITY,
    CHOICE_AVG_STUDENT_DISTANCE,
    CHOICE_FRL_DISSIMILARITY,
    CHOICE_METRIC_COLUMNS,
    CHOICE_PERCENT_DESIGNATED,
    CHOICE_PERCENT_TOP_1,
    CHOICE_PERCENT_TOP_3,
    CHOICE_PERCENT_UNASSIGNED,
    CHOICE_SES3_DISSIMILARITY,
    CHOICE_SCHOOLS_ABOVE_10PCT_DISTRICT_FRL,
    CHOICE_TOTAL_MNL_UTILITY,
    compute_choice_metrics_for_run,
    preserve_choice_metrics_payload,
    run_choice_metrics_for_existing_runs,
)
from benchmark.matching import (
    preserve_matching_payload,
    run_matching_for_existing_runs,
    run_matching_for_solution,
)
from benchmark.matching import runner as matching_runner
from benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    manifest_for,
    result_payload_for,
    save_stage_artifacts,
    stage_names_for,
    write_json,
)
from metrics import MetricsCalculator
from optimization.config import OptimizationConfig
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import FakeDataset, make_grid_problem


MATCHING_CONFIG = "benchmark/matching/zones+hard_reserves_06frl.yaml"


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
  config: {MATCHING_CONFIG}
  compute_stage_assignments: true
""",
        encoding="utf-8",
    )

    sweep = SimulationSweep.from_yaml(str(config_path))

    assert sweep.mode == "matching"
    assert sweep.matching.enabled is True
    assert sweep.matching.config.endswith("zones+hard_reserves_06frl.yaml")
    assert sweep.matching.compute_stage_assignments is True


def test_default_matching_template_uses_bundled_path():
    assert str(matching_runner.DEFAULT_MATCHING_TEMPLATE) == MATCHING_CONFIG


def test_sweep_yaml_accepts_multiple_matching_configs(tmp_path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        f"""
mode: matching
optimization_defaults:
  levels: ['BlockGroup_0']
  graphs_dir: '{tmp_path / "graphs"}'
matching:
  enabled: true
  configs:
    - name: no_reserves
      config: {MATCHING_CONFIG}
    - name: sd
      config: benchmark/matching/sd.yaml
""",
        encoding="utf-8",
    )

    sweep = SimulationSweep.from_yaml(str(config_path))

    assert [config.name for config in sweep.matching.config_specs()] == [
        "no_reserves",
        "sd",
    ]
    assert sweep.matching.config_specs()[1].config.endswith("sd.yaml")


def test_sweep_yaml_accepts_choice_metrics_config(tmp_path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        f"""
mode: choice_metrics
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


def test_run_matching_for_solution_writes_mapping_and_populations(
    tmp_path, monkeypatch
):
    captured_config = _stub_student_assignment(monkeypatch)
    solution = _solution()

    result = run_matching_for_solution(
        solution,
        str(tmp_path),
        MatchingRunConfig(enabled=True, config=MATCHING_CONFIG),
        workers=3,
    )

    assert result.status == "OK"
    assert result.metrics["matching_assignment_files"] == 1
    assert result.metrics["matching_students_total"] == 3
    assert result.metrics["matching_students_assigned_mean"] == 2
    assert result.metrics["matching_unassigned_rate_mean"] == 1 / 3
    assert captured_config["value"]["zone-building-blocks"] == "block_group"
    assert captured_config["value"]["policies"] == ["generated_zones"]
    assert captured_config["value"]["workers"] == 3
    assert result.run["workers"] == 3
    expected_precomputed = str((tmp_path / "matching" / "precomputed").resolve())
    assert captured_config["value"]["paths"]["student-save"] == expected_precomputed
    assert captured_config["value"]["utility-model"]["save-path"] == str(
        (tmp_path / "matching" / "precomputed" / "utility_matrix.npy").resolve()
    )

    zones_text = (tmp_path / "matching" / "zones.csv").read_text(encoding="utf-8")
    assert "1000,1001" in zones_text
    assert "1002,1003" in zones_text

    assignments = pd.read_csv(tmp_path / "matching" / "student_school_assignments.csv")
    assert assignments["school_id"].dropna().astype(int).tolist() == [664, 665]

    school_populations = pd.read_csv(tmp_path / "matching" / "school_populations.csv")
    assert school_populations["assigned_count"].tolist() == [1, 1]


def test_run_matching_for_solution_supports_multiple_configs(tmp_path, monkeypatch):
    captured = _stub_student_assignment(monkeypatch)
    solution = _solution()

    result = run_matching_for_solution(
        solution,
        str(tmp_path),
        MatchingRunConfig(
            enabled=True,
            configs=[
                MatchingConfigSpec(name="first", config=MATCHING_CONFIG),
                MatchingConfigSpec(name="second", config=MATCHING_CONFIG),
            ],
        ),
        workers=1,
    )

    assert result.status == "OK"
    assert set(result.run["runs"]) == {"first", "second"}
    assert result.metrics["matching_first_assignment_files"] == 1
    assert result.metrics["matching_second_assignment_files"] == 1
    assert (tmp_path / "matching" / "first" / "student_school_assignments.csv").exists()
    assert (
        tmp_path / "matching" / "second" / "student_school_assignments.csv"
    ).exists()
    assert (tmp_path / "matching" / "summary.json").exists()
    assert len(captured["calls"]) == 2


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
    assert metrics[CHOICE_AVG_MNL_UTILITY] == 20.0
    assert metrics[CHOICE_TOTAL_MNL_UTILITY] == 60.0
    assert round(metrics[CHOICE_SES3_DISSIMILARITY], 6) == round(1 / 6, 6)
    assert round(metrics[CHOICE_FRL_DISSIMILARITY], 6) == round(0.7655502392344498, 6)
    by_assignment_path = tmp_path / "matching" / "choice_metrics_by_assignment.csv"
    assert by_assignment_path.exists()
    by_assignment = pd.read_csv(by_assignment_path)
    assert all(column in by_assignment.columns for column in CHOICE_METRIC_COLUMNS)


def test_choice_metrics_support_multiple_matching_configs(tmp_path):
    for name, utility in [("first", 10.0), ("second", 20.0)]:
        assignments_dir = tmp_path / "matching" / name / "assignments_raw" / "policy"
        assignments_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "studentno": [1, 2],
                "programno": [11, 0],
                "programcodes": ["101-GE-KG", ""],
                "rank": [1, 2],
                "designation": [0, 0],
                "assigned_utility": [utility, None],
                "freelunch_prob": [0.3, 0.7],
                "reducedlunch_prob": [0.0, 0.0],
            }
        ).to_csv(assignments_dir / "assignment.csv", index=False)

    result = compute_choice_metrics_for_run(
        str(tmp_path),
        ChoiceMetricsRunConfig(enabled=True),
    )

    assert set(result.run["runs"]) == {"first", "second"}
    assert result.metrics["choice_first_avg_mnl_utility"] == 10.0
    assert result.metrics["choice_second_avg_mnl_utility"] == 20.0
    assert (tmp_path / "matching" / "choice_metrics_summary.json").exists()
    assert (
        tmp_path / "matching" / "first" / "choice_metrics_by_assignment.csv"
    ).exists()


def test_choice_metrics_average_mnl_utility_across_assignments(tmp_path):
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

    assert result.metrics[CHOICE_AVG_MNL_UTILITY] == 2.5
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


def test_choice_metrics_mode_skips_infeasible_final_solution(tmp_path):
    run_dir, _ = _write_recursive_infeasible_run(tmp_path)
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
    payload = matching_runner._load_json(os.path.join(run_dir, RESULT_FILENAME))
    payload["choice_metrics"] = {"status": "OK"}
    payload.setdefault("metrics", {})[CHOICE_AVG_STUDENT_DISTANCE] = 99
    write_json(os.path.join(run_dir, RESULT_FILENAME), payload)

    batch = run_choice_metrics_for_existing_runs(
        str(tmp_path),
        ChoiceMetricsRunConfig(enabled=True),
    )

    assert batch.skipped == 1
    result = matching_runner._load_json(os.path.join(run_dir, RESULT_FILENAME))
    assert result["status"] == "INFEASIBLE"
    assert "choice_metrics" not in result
    assert CHOICE_AVG_STUDENT_DISTANCE not in result["metrics"]


def test_matching_mode_updates_existing_result(tmp_path, monkeypatch):
    captured = _stub_student_assignment(monkeypatch)
    run_dir, problem = _write_synthetic_run(tmp_path)

    batch = run_matching_for_existing_runs(
        str(tmp_path),
        MatchingRunConfig(enabled=True, config=MATCHING_CONFIG),
        dataset_factory=lambda config, manifest: FakeDataset(problem),
    )

    assert batch.successful == 1
    result = matching_runner._load_json(os.path.join(run_dir, RESULT_FILENAME))
    assert result["matching"]["status"] == "OK"
    assert result["metrics"]["matching_assignment_files"] == 1
    assert len(captured["sessions"]) == 1
    assert len(captured["sessions"][0].calls) == 1
    assert captured["sessions"][0].calls[0]["config"]["paths"]["student-save"] == str(
        (run_dir / "matching" / "precomputed").resolve()
    )


def test_matching_mode_skips_infeasible_final_solution(tmp_path, monkeypatch):
    captured = _stub_student_assignment(monkeypatch)
    run_dir, problem = _write_recursive_infeasible_run(tmp_path)
    payload = matching_runner._load_json(os.path.join(run_dir, RESULT_FILENAME))
    payload["matching"] = {"status": "OK"}
    payload.setdefault("metrics", {})["matching_assignment_files"] = 99
    write_json(os.path.join(run_dir, RESULT_FILENAME), payload)

    batch = run_matching_for_existing_runs(
        str(tmp_path),
        MatchingRunConfig(enabled=True, config=MATCHING_CONFIG),
        dataset_factory=lambda config, manifest: FakeDataset(problem),
    )

    assert batch.successful == 1
    result = matching_runner._load_json(os.path.join(run_dir, RESULT_FILENAME))
    assert result["status"] == "INFEASIBLE"
    assert "matching" not in result
    assert "matching_assignment_files" not in result["metrics"]
    assert sum(len(session.calls) for session in captured["sessions"]) == 0


def test_stage_matching_and_choice_metrics_are_opt_in(tmp_path, monkeypatch):
    captured = _stub_student_assignment(monkeypatch)
    run_dir, problem = _write_synthetic_run(tmp_path)

    batch = run_matching_for_existing_runs(
        str(tmp_path),
        MatchingRunConfig(
            enabled=True,
            config=MATCHING_CONFIG,
            compute_stage_assignments=True,
        ),
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
    assert len(captured["sessions"]) == 1
    assert len(captured["sessions"][0].calls) == 2
    stage_config = yaml.safe_load(
        (
            run_dir / "stages" / stage_name / "matching" / "config.generated.yaml"
        ).read_text(encoding="utf-8")
    )
    assert stage_config["paths"]["student-save"] == str(
        (run_dir / "matching" / "precomputed").resolve()
    )


def test_stage_matching_skips_infeasible_stages(tmp_path, monkeypatch):
    captured = _stub_student_assignment(monkeypatch)
    run_dir, problem = _write_recursive_infeasible_run(tmp_path)

    batch = run_matching_for_existing_runs(
        str(tmp_path),
        MatchingRunConfig(
            enabled=True,
            config=MATCHING_CONFIG,
            compute_stage_assignments=True,
        ),
        choice_metrics=ChoiceMetricsRunConfig(enabled=True, compute_stage_metrics=True),
        dataset_factory=lambda config, manifest: FakeDataset(problem),
    )

    assert batch.successful == 1
    result = matching_runner._load_json(os.path.join(run_dir, RESULT_FILENAME))
    feasible_stage = "stage_00_BlockGroup_0"
    infeasible_stage = "stage_01_BlockGroup_0"
    assert "matching" not in result
    assert set(result["stage_matching"]["stages"]) == {feasible_stage}
    assert (
        result["stage_matching"]["stages"][feasible_stage]["matching"]["status"] == "OK"
    )
    assert result["run"]["stages"][0]["matching"]["status"] == "OK"
    assert result["run"]["stages"][0]["choice_metrics"]["status"] == "OK"
    assert "matching" not in result["run"]["stages"][1]
    assert infeasible_stage not in result["stage_matching"]["stages"]
    assert sum(len(session.calls) for session in captured["sessions"]) == 1


def test_run_student_assignment_uses_market_constructor_shape(tmp_path, monkeypatch):
    fake_market, _, _ = _install_fake_market_generator(monkeypatch)
    precomputed_dir = tmp_path / "matching" / "precomputed"
    zone_csv = tmp_path / "zones.csv"
    assignments_dir = tmp_path / "assignments_raw"

    matching_runner._run_student_assignment(
        _session_config(zone_csv, assignments_dir, precomputed_dir),
        assignments_dir,
        workers=2,
    )

    assert len(fake_market.instances) == 1
    assert fake_market.instances[0].config["workers"] == 2
    assert fake_market.seen == [
        {
            "zone_file": str(zone_csv),
            "assignment_path": str(assignments_dir),
        }
    ]


def test_student_assignment_guardrail_patch_allows_fractional_zone_counts():
    matching_runner._patch_student_assignment_guardrail_pandas_compat()
    from student_assignment.da.guardrail_setup import GuardrailSetup

    fake_guardrail = type("FakeGuardrail", (), {})()
    fake_guardrail.students = type(
        "FakeStudents",
        (),
        {
            "student_data": pd.DataFrame(
                {"diversity_category": [0, 1, 1, 0]},
                index=[101, 102, 103, 104],
            )
        },
    )()
    fake_guardrail.student2zone = {101: 0, 102: 0, 103: 1, 104: 1}

    zone_frac = GuardrailSetup._calculate_zone_fractions(fake_guardrail)

    assert zone_frac.loc[0, 0] == 0.5
    assert zone_frac.loc[0, 1] == 0.5
    assert zone_frac.loc[1, 0] == 0.5
    assert zone_frac.loc[1, 1] == 0.5


def test_student_assignment_empty_excess_match_patch_ignores_empty_heap():
    matching_runner._patch_student_assignment_empty_excess_match_compat()
    from student_assignment.da.da import School

    school = School(index=-1, capacity=-1)

    assert school.has_excess_matches() is False


def test_student_assignment_session_reuses_market_for_dynamic_paths(
    tmp_path, monkeypatch
):
    fake_market, fake_priority, fake_preference = _install_fake_market_generator(
        monkeypatch
    )
    session = matching_runner.StudentAssignmentSession()
    precomputed_dir = tmp_path / "matching" / "precomputed"
    zone_a = tmp_path / "a" / "zones.csv"
    zone_b = tmp_path / "b" / "zones.csv"
    assignments_a = tmp_path / "a" / "assignments_raw"
    assignments_b = tmp_path / "b" / "assignments_raw"

    session.run(_session_config(zone_a, assignments_a, precomputed_dir), assignments_a)
    session.run(_session_config(zone_b, assignments_b, precomputed_dir), assignments_b)

    assert len(fake_market.instances) == 1
    assert fake_market.executions == 2
    assert fake_market.seen == [
        {
            "zone_file": str(zone_a),
            "assignment_path": str(assignments_a),
        },
        {
            "zone_file": str(zone_b),
            "assignment_path": str(assignments_b),
        },
    ]
    assert fake_priority.instances == 2
    assert fake_preference.instances == 2


def test_student_assignment_session_rebuilds_for_static_config_change(
    tmp_path, monkeypatch
):
    fake_market, _, _ = _install_fake_market_generator(monkeypatch)
    session = matching_runner.StudentAssignmentSession()
    precomputed_dir = tmp_path / "matching" / "precomputed"
    assignments_a = tmp_path / "a" / "assignments_raw"
    assignments_b = tmp_path / "b" / "assignments_raw"

    session.run(
        _session_config(tmp_path / "a" / "zones.csv", assignments_a, precomputed_dir),
        assignments_a,
    )
    session.run(
        _session_config(
            tmp_path / "b" / "zones.csv",
            assignments_b,
            precomputed_dir,
            grade="01",
        ),
        assignments_b,
    )

    assert len(fake_market.instances) == 2


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
        "choice_metrics": {
            "status": "OK",
            "metrics": {
                CHOICE_AVG_STUDENT_DISTANCE: 1.5,
                CHOICE_TOTAL_MNL_UTILITY: 99,
            },
        },
        "metrics": {
            CHOICE_AVG_STUDENT_DISTANCE: 1.5,
            CHOICE_TOTAL_MNL_UTILITY: 99,
            "num_zones": 99,
        },
    }

    preserve_choice_metrics_payload(new_payload, previous_payload)

    assert new_payload["choice_metrics"]["status"] == "OK"
    assert new_payload["choice_metrics"]["metrics"][CHOICE_TOTAL_MNL_UTILITY] == 99
    assert new_payload["metrics"]["num_zones"] == 2
    assert new_payload["metrics"][CHOICE_AVG_STUDENT_DISTANCE] == 1.5
    assert new_payload["metrics"][CHOICE_TOTAL_MNL_UTILITY] == 99


def _stub_student_assignment(monkeypatch):
    captured = {"calls": [], "sessions": []}

    def fake_run(config, assignments_dir, *, workers=1):
        captured["value"] = config
        captured["calls"].append(
            {"config": config, "assignments_dir": assignments_dir, "workers": workers}
        )
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

    class FakeSession:
        def __init__(self):
            self.calls = []
            captured["sessions"].append(self)

        def run(self, config, assignments_dir, *, workers=1):
            self.calls.append(
                {
                    "config": config,
                    "assignments_dir": assignments_dir,
                    "workers": workers,
                }
            )
            fake_run(config, assignments_dir, workers=workers)

    monkeypatch.setattr(matching_runner, "_run_student_assignment", fake_run)
    monkeypatch.setattr(
        matching_runner,
        "_new_student_assignment_session",
        FakeSession,
    )
    return captured


def _install_fake_market_generator(monkeypatch):
    class FakePriorityGenerator:
        instances = 0

        def __init__(self, market):
            self.market = market
            FakePriorityGenerator.instances += 1

    class FakePreferenceGenerator:
        instances = 0

        def __init__(self, market):
            self.market = market
            FakePreferenceGenerator.instances += 1

    class FakeMarketGenerator:
        instances = []
        executions = 0
        seen = []
        active_configurator = None

        def __init__(self, estimate_path=None, assignment_path=None):
            self.estimate_path = estimate_path
            self.configurator = FakeMarketGenerator.active_configurator
            assert self.configurator is not None
            self.config = self.configurator.config
            self.priority_generator = FakePriorityGenerator(self)
            self.preference_generator = FakePreferenceGenerator(self)
            self._set_up_save_folder(assignment_path)
            FakeMarketGenerator.instances.append(self)

        def _set_up_save_folder(self, assignment_path):
            self.output_assignment_path = assignment_path

        def create_iterations_generator(self):
            FakeMarketGenerator.seen.append(
                {
                    "zone_file": self.config["paths"]["zone-files"][
                        matching_runner.GENERATED_POLICY_NAME
                    ],
                    "assignment_path": str(self.output_assignment_path),
                }
            )
            return iter([[iter([None])]])

        @staticmethod
        def execute_generator(iterations_generator):
            FakeMarketGenerator.executions += 1
            for policy_suboptions_generator in iterations_generator:
                for priority_suboptions_generator in policy_suboptions_generator:
                    for _ in priority_suboptions_generator:
                        pass

    monkeypatch.setattr(
        matching_runner,
        "_market_generator_class",
        lambda: FakeMarketGenerator,
    )

    def fake_install_student_assignment_config(config):
        configurator = matching_runner._StaticConfigurator(config)
        FakeMarketGenerator.active_configurator = configurator
        return configurator

    monkeypatch.setattr(
        matching_runner,
        "_install_student_assignment_config",
        fake_install_student_assignment_config,
    )
    return FakeMarketGenerator, FakePriorityGenerator, FakePreferenceGenerator


def _session_config(zone_csv, assignments_dir, precomputed_dir, *, grade="KG"):
    return {
        "grade": grade,
        "paths": {
            "assignment-folder": str(assignments_dir),
            "student-save": str(precomputed_dir),
            "zone-files": {
                matching_runner.GENERATED_POLICY_NAME: str(zone_csv),
            },
        },
        "policies": [matching_runner.GENERATED_POLICY_NAME],
        "random-seed": 2023,
        "save-assignment": True,
        "subconfig-name": "generated_zones",
        "subconfigs": ["generated_zones"],
        "year": 23,
        "zone-building-blocks": "block_group",
    }


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
        result_payload_for(
            metrics=metrics, config=config, solutions=solutions, task=task
        ),
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


def _write_recursive_infeasible_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = _solution()
    problem = first.problem
    final = ZoneSolution(
        problem=problem,
        assignment={},
        status="INFEASIBLE",
        objective=None,
        wall_time=0.25,
    )
    config = OptimizationConfig(
        centroids_type="5-zone-AF",
        levels=["BlockGroup_0", "BlockGroup_0"],
        solver="local_search",
        strategy="recursive",
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
    solutions = [first, final]
    stage_records = save_stage_artifacts(
        solutions,
        str(run_dir),
        stage_names_for(solutions, config),
    )
    calculator = MetricsCalculator(solutions, config=config)
    metrics = calculator.compute()
    calculator.context.solution.save(str(run_dir))
    payload = result_payload_for(
        metrics=metrics,
        config=config,
        solutions=solutions,
        task=task,
    )
    write_json(os.path.join(run_dir, RESULT_FILENAME), payload)
    write_json(
        os.path.join(run_dir, MANIFEST_FILENAME),
        manifest_for(
            task=task,
            config=config,
            status=payload["status"],
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            stages=stage_records,
            final_stage=metrics.run.get("final_stage"),
            error_message=None,
        ),
    )
    return run_dir, problem
