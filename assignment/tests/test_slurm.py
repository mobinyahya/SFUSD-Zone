import json
import pathlib
import subprocess
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from assignment.slurm import build_slurm_plan, write_slurm_scripts
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


def test_kumar_plan_has_one_job_per_subconfig_iteration(tmp_path):
    config_path = pathlib.Path(__file__).parents[1] / "configs/kumar.config.yaml"

    plan, plan_path = build_slurm_plan(
        config_path,
        assignment_folder=tmp_path / "assignments",
        plan_dir=tmp_path / "plan",
    )

    assert len(plan["assignment_tasks"]) == 475
    assert len(plan["metrics_tasks"]) == 19
    assert pathlib.Path(plan["assignment_folder"]).is_absolute()
    assert plan_path.is_absolute()
    assert pathlib.Path(plan["provenance_path"]).is_file()
    assert (tmp_path / "assignments/config.json").is_file()
    utility_owners = [
        task for task in plan["assignment_tasks"] if task["write_utility_output"]
    ]
    assert utility_owners == [
        {
            "subconfig": "small_zones+no_reserves",
            "iteration": 0,
            "write_utility_output": True,
        }
    ]


def test_generated_scripts_quote_names_and_wire_dependencies(tmp_path):
    plan_path = (tmp_path / "plan.json").resolve()
    special_name = "policy+reserves_#3"
    plan = {
        "schema_version": 1,
        "workspace_root": str(pathlib.Path(__file__).parents[2]),
        "plan_path": str(plan_path),
        "assignment_folder": str((tmp_path / "assignments").resolve()),
        "subconfigs": [{"name": special_name, "config": {}}],
        "assignment_tasks": [
            {
                "subconfig": special_name,
                "iteration": iteration,
                "write_utility_output": iteration == 0,
            }
            for iteration in range(2)
        ],
        "metrics_tasks": [{"subconfig": special_name}],
    }
    plan_path.write_text(json.dumps(plan))

    submit_path = write_slurm_scripts(plan_path)
    submit_script = submit_path.read_text()
    worker_scripts = sorted(submit_path.parent.glob("assignment-*.sh"))

    assert len(worker_scripts) == 2
    worker_script = worker_scripts[0].read_text()
    assert "#SBATCH -A soal" in worker_script
    assert "#SBATCH -p soal" in worker_script
    assert "#SBATCH --ntasks=1" in worker_script
    assert "#SBATCH --cpus-per-task=1" in worker_script
    assert "export OMP_NUM_THREADS=1" in worker_script
    assert f"--subconfig '{special_name}'" in worker_script
    assert "sbatch --parsable -A soal -p soal" in submit_script
    assert "--ntasks=1" in submit_script
    assert "job_id=${raw_id%%;*}" in submit_script
    assert '--dependency="afterok:${dependency_0}"' in submit_script
    assert submit_script.count("assignment_ids_0+=(") == 2
    assert "srun" not in submit_script
    subprocess.run(["bash", "-n", str(submit_path)], check=True)


def _rng_market(tmp_path):
    market = MarketGenerator.__new__(MarketGenerator)
    market.config = {
        "policies": ["zones"],
        "iterations": {"start": 0, "end": 3},
        "random-seed": 2023,
        "utility-model": {"enable": False},
        "guard-rails-reserve-options": [
            {"guard-rails": -1, "reserve-settings": {}},
            {"guard-rails": 0, "reserve-settings": {"column": "frl"}},
        ],
        "restrict-zone": False,
        "citywide-or-lp": [],
        "ctip-options": [0],
        "rounds-merged-options": [0],
        "ties-options": ["MTB"],
        "reuse_assignments": False,
    }
    market._reset_zones = Mock()
    market._load_reusable_policy_run = Mock(return_value=None)
    market._assignment_save_path = Mock(
        side_effect=lambda _policy, iteration: tmp_path / f"iteration{iteration}.csv"
    )

    def simulate(_policy, iteration):
        yield pd.DataFrame({"iteration": [iteration], "draw": [np.random.random()]})

    market._simulate_policy = simulate
    return market


def test_independent_iteration_rng_matches_full_generation(tmp_path):
    market = _rng_market(tmp_path)
    sequential = list(market.create_iterations_generator())
    targeted = list(market.create_target_iteration_generator(2))

    assert sequential[2].loc[0, "draw"] == targeted[0].loc[0, "draw"]
    assert sequential[5].loc[0, "draw"] == targeted[1].loc[0, "draw"]
    assert sequential[2].loc[0, "draw"] == sequential[5].loc[0, "draw"]
    assert MarketGenerator.iteration_seed(2023, 2) != MarketGenerator.iteration_seed(
        2023, 1
    )


def test_targeted_retry_does_not_delete_sibling_iteration(tmp_path):
    market = _rng_market(tmp_path)
    market.config["guard-rails-reserve-options"] = [
        {"guard-rails": -1, "reserve-settings": {}}
    ]
    sibling = tmp_path / "iteration0.csv"
    sibling.write_text("complete sibling")

    list(market.create_target_iteration_generator(1))

    assert sibling.read_text() == "complete sibling"


def _reports(config_name, value):
    return {
        "program": pd.DataFrame(
            {
                "config_name": [config_name],
                "program_id": ["101-GE-KG"],
                "school_id": [101],
                "school_name": ["Alpha"],
                "school_category": ["Attendance"],
                "program_type": ["GE"],
                "new_metric": [value],
            }
        ),
        "zip_code": pd.DataFrame(
            {
                "config_name": [config_name],
                "zip_code": [94110],
                "new_metric": [value],
            }
        ),
        "attendance_area": pd.DataFrame(
            {
                "config_name": [config_name],
                "attendance_area": [101],
                "new_metric": [value],
            }
        ),
        "citywide": pd.DataFrame({"config_name": [config_name], "new_metric": [value]}),
    }


def test_locked_metric_merge_is_local_and_idempotent(tmp_path):
    aggregate_dir = tmp_path / "aggregate_metrics"
    aggregate_dir.mkdir()
    other_reports = _reports("other/variant", 7.0)
    owned_reports = _reports("policy+reserves_#3/old", 8.0)
    for report_name, filename in MarketGenerator.AGGREGATE_METRIC_FILES.items():
        existing = pd.concat(
            [other_reports[report_name], owned_reports[report_name]],
            ignore_index=True,
        ).rename(columns={"new_metric": "retired_metric"})
        existing.to_csv(aggregate_dir / filename, index=False)
    (aggregate_dir / "metrics_by_school.csv").write_text("obsolete")
    reports = _reports("policy+reserves_#3/variant", 2.0)

    MarketGenerator.merge_aggregate_metric_reports(
        tmp_path, "policy+reserves_#3", reports
    )
    MarketGenerator.merge_aggregate_metric_reports(
        tmp_path, "policy+reserves_#3", reports
    )

    citywide = pd.read_csv(aggregate_dir / "metrics_citywide.csv")
    assert citywide["config_name"].tolist() == [
        "other/variant",
        "policy+reserves_#3/variant",
    ]
    assert citywide.columns.tolist() == ["config_name", "new_metric"]
    assert pd.isna(citywide.loc[0, "new_metric"])
    assert citywide.loc[1, "new_metric"] == 2.0
    assert (tmp_path / ".aggregate_metrics.lock").is_file()
    assert {
        "metrics_by_program.csv",
        "metrics_by_zip_code.csv",
        "metrics_by_attendance_area.csv",
        "metrics_citywide.csv",
    } == {path.name for path in aggregate_dir.iterdir()}
    program = pd.read_csv(aggregate_dir / "metrics_by_program.csv")
    assert program.columns.tolist() == reports["program"].columns.tolist()
    assert program["config_name"].tolist() == [
        "other/variant",
        "policy+reserves_#3/variant",
    ]
    assert pd.isna(program.loc[0, "new_metric"])
    assert program.loc[1, "new_metric"] == 2.0
    for report_name, filename in MarketGenerator.AGGREGATE_METRIC_FILES.items():
        merged = pd.read_csv(aggregate_dir / filename)
        assert merged.columns.tolist() == reports[report_name].columns.tolist()
        assert "other/variant" in merged["config_name"].tolist()


def test_locked_metric_merge_removes_disabled_local_reports(tmp_path):
    aggregate_dir = tmp_path / "aggregate_metrics"
    aggregate_dir.mkdir()
    existing = _reports("other/variant", 7.0)
    for report_name, filename in MarketGenerator.AGGREGATE_METRIC_FILES.items():
        existing[report_name].to_csv(aggregate_dir / filename, index=False)

    MarketGenerator.merge_aggregate_metric_reports(
        tmp_path,
        "policy",
        {"citywide": _reports("policy/variant", 2.0)["citywide"]},
    )

    assert {path.name for path in aggregate_dir.iterdir()} == {
        "metrics_citywide.csv"
    }
    citywide = pd.read_csv(aggregate_dir / "metrics_citywide.csv")
    assert citywide["config_name"].tolist() == [
        "other/variant",
        "policy/variant",
    ]


def test_metrics_only_fails_before_evaluation_when_input_is_missing(tmp_path):
    market = MarketGenerator.__new__(MarketGenerator)
    market.config = {
        "subconfig-name": "missing",
        "export-aggregate-metrics": True,
    }
    market._expected_saved_assignment_specs = Mock(
        return_value=[(tmp_path / "missing.csv", "variant_iteration0.csv", 0)]
    )
    market._reset_aggregate_metric_reports = Mock()
    market._record_assignment_metric_reports = Mock()
    market._write_aggregate_metrics = False
    market._run_single_iteration_of_policy = Mock()

    with pytest.raises(FileNotFoundError, match="Missing 1 expected"):
        market.evaluate_saved_subconfig("missing")

    market._reset_aggregate_metric_reports.assert_not_called()
    market._record_assignment_metric_reports.assert_not_called()
    market._run_single_iteration_of_policy.assert_not_called()
