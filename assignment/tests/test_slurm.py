import json
import os
import pathlib
import subprocess
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import assignment.slurm as assignment_slurm
from assignment.slurm import (
    MAX_ASSIGNMENT_JOBS,
    MAX_CPUS_PER_NODE,
    MAX_METRICS_JOBS,
    MAX_SLURM_JOBS,
    PLAN_SCHEMA_VERSION,
    _build_allocations,
    build_slurm_plan,
    write_slurm_scripts,
)
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


def test_kumar_plan_batches_into_twelve_assignment_and_eight_metrics_jobs(tmp_path):
    config_path = pathlib.Path(__file__).parents[1] / "configs/kumar.config.yaml"

    plan, plan_path = build_slurm_plan(
        config_path,
        assignment_folder=tmp_path / "assignments",
        plan_dir=tmp_path / "plan",
    )

    assert len(plan["assignment_tasks"]) == 475
    assert len(plan["metrics_tasks"]) == 19
    assert len(plan["allocations"]) == MAX_SLURM_JOBS
    assignment_allocations = [
        allocation
        for allocation in plan["allocations"]
        if allocation["phase"] == "assignment"
    ]
    assert len(assignment_allocations) == MAX_ASSIGNMENT_JOBS
    assert {allocation["cpus"] for allocation in assignment_allocations} == {39, 40}
    assert sorted(
        index
        for allocation in assignment_allocations
        for index in allocation["task_indices"]
    ) == list(range(475))
    metrics_allocations = [
        allocation
        for allocation in plan["allocations"]
        if allocation["phase"] in {"metrics", "metrics-finalize"}
    ]
    assert len(metrics_allocations) == MAX_METRICS_JOBS
    assert sum(
        allocation["phase"] == "metrics-finalize"
        for allocation in metrics_allocations
    ) == 1
    assert {allocation["cpus"] for allocation in metrics_allocations} == {
        MAX_CPUS_PER_NODE
    }
    assert sorted(
        index
        for allocation in metrics_allocations
        for index in allocation["task_indices"]
    ) == list(range(19))
    dependencies = assignment_slurm._metrics_dependency_slots(plan)
    assert len(dependencies) == MAX_METRICS_JOBS
    assert len(set(map(tuple, dependencies.values()))) > 1
    assert all(0 < len(slots) < MAX_ASSIGNMENT_JOBS for slots in dependencies.values())
    assert set().union(*map(set, dependencies.values())) == set(
        range(MAX_ASSIGNMENT_JOBS)
    )
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
    submit_path = write_slurm_scripts(plan_path)
    submit_script = submit_path.read_text()
    assert submit_script.count("assignment_ids+=(") == MAX_ASSIGNMENT_JOBS
    assert submit_script.count("metrics_ids+=(") == MAX_METRICS_JOBS - 1
    assert submit_script.count('finalizer_id="$(submit_job') == 1
    assert (
        submit_script.count('--dependency="afterany:${metrics_dependency_')
        == MAX_METRICS_JOBS - 1
    )
    assert "assignment_dependency" not in submit_script
    assert '--dependency="afterany:${finalizer_dependency}"' in submit_script
    assert "This Slurm plan has already been submitted" in submit_script
    assert submit_script.count("$(submit_job") == MAX_SLURM_JOBS
    subprocess.run(["bash", "-n", str(submit_path)], check=True)


def test_generated_scripts_quote_names_and_wire_dependencies(tmp_path, monkeypatch):
    plan_path = (tmp_path / "plan.json").resolve()
    special_name = "policy+reserves_#3"
    fragment_dir = (tmp_path / "fragments").resolve()
    fragment_dir.mkdir()
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": "test-run",
        "workspace_root": str(pathlib.Path(__file__).parents[2]),
        "plan_path": str(plan_path),
        "assignment_folder": str((tmp_path / "assignments").resolve()),
        "metrics_fragment_dir": str(fragment_dir),
        "subconfigs": [{"name": special_name, "config": {}}],
        "assignment_tasks": [
            {
                "subconfig": special_name,
                "iteration": iteration,
                "write_utility_output": iteration == 0,
            }
            for iteration in range(2)
        ],
        "metrics_tasks": [{"subconfig": special_name, "report_names": ["citywide"]}],
        "allocations": _build_allocations(2, 1),
    }
    plan_path.write_text(json.dumps(plan))

    submit_path = write_slurm_scripts(plan_path)
    submit_script = submit_path.read_text()
    worker_scripts = sorted(submit_path.parent.glob("assignment-allocation-*.sh"))

    assert len(worker_scripts) == 2
    worker_script = worker_scripts[0].read_text()
    assert "#SBATCH -A soal" in worker_script
    assert "#SBATCH -p soal" in worker_script
    assert "#SBATCH --ntasks=1" in worker_script
    assert "#SBATCH --cpus-per-task=1" in worker_script
    assert "export OMP_NUM_THREADS=1" in worker_script
    assert "allocation-worker" in worker_script
    assert "--allocation-index 0" in worker_script
    assert "sbatch --parsable -A soal -p soal" in submit_script
    assert "--ntasks=1" in submit_script
    assert "job_id=${raw_id%%;*}" in submit_script
    assert "finalizer_dependency=" in submit_script
    assert "${assignment_ids[0]}" in submit_script
    assert "${assignment_ids[1]}" in submit_script
    assert submit_script.count("assignment_ids+=(") == 2
    assert submit_script.count("metrics_ids+=(") == 0
    assert submit_script.count('finalizer_id="$(submit_job') == 1
    assert submit_script.count("$(submit_job") == 3
    assert "srun" not in submit_script
    subprocess.run(["bash", "-n", str(submit_path)], check=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_count = tmp_path / "sbatch-count"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        f"if [[ ! -e {sbatch_count} ]]; then\n"
        f"  touch {sbatch_count}\n"
        "  printf '123\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    fake_sbatch.chmod(0o755)
    cancelled = tmp_path / "cancelled"
    fake_scancel = fake_bin / "scancel"
    fake_scancel.write_text(
        "#!/usr/bin/env bash\n" f"printf '%s\\n' \"$@\" > {cancelled}\n"
    )
    fake_scancel.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    result = subprocess.run(["bash", str(submit_path)], check=False)

    assert result.returncode != 0
    assert cancelled.read_text().splitlines() == ["123"]
    assert not (plan_path.parent / ".submitted").exists()


def test_worker_process_reuses_market_across_iterations_and_subconfigs(
    tmp_path, monkeypatch
):
    events = []

    class FakeMarketGenerator:
        def __init__(
            self, *, config, assignment_path, write_config, write_aggregate_metrics
        ):
            events.append(("initialize", config["subconfig-name"]))
            self.config = dict(config)

        def reconfigure(self, config, assignment_path, *, write_config):
            events.append(("reconfigure", config["subconfig-name"]))
            self.config = dict(config)

        def simulate_target(self, subconfig_name, iteration, *, write_utility_output):
            events.append(("simulate", subconfig_name, iteration))

    plan = {
        "plan_path": str(tmp_path / "plan.json"),
        "assignment_folder": str(tmp_path / "assignments"),
        "subconfigs": [
            {"name": name, "config": {"subconfig-name": name}}
            for name in ("first", "second")
        ],
        "assignment_tasks": [
            {
                "subconfig": subconfig,
                "iteration": iteration,
                "write_utility_output": False,
            }
            for subconfig, iteration in [
                ("first", 0),
                ("first", 1),
                ("second", 0),
                ("first", 2),
            ]
        ],
    }
    monkeypatch.setattr(assignment_slurm, "_WORKER_PLAN", plan)
    monkeypatch.setattr(assignment_slurm, "_WORKER_MARKET_GENERATOR", None)
    monkeypatch.setattr(assignment_slurm, "_WORKER_MARKET_KEY", None)
    monkeypatch.setattr(
        "assignment.student_assignment.market_generator."
        "school_choice_market_generator.MarketGenerator",
        FakeMarketGenerator,
    )

    for task_index in range(4):
        assignment_slurm._run_cached_assignment_task(task_index)

    assert events == [
        ("initialize", "first"),
        ("simulate", "first", 0),
        ("simulate", "first", 1),
        ("reconfigure", "second"),
        ("simulate", "second", 0),
        ("reconfigure", "first"),
        ("simulate", "first", 2),
    ]


def test_allocation_pool_initializes_each_process_with_the_plan(tmp_path, monkeypatch):
    plan_path = (tmp_path / "plan.json").resolve()
    plan = {
        "allocations": [{"phase": "assignment", "task_indices": [3, 7], "cpus": 2}],
    }
    seen = {}

    class FinishedFuture:
        def result(self):
            return None

    class FakeExecutor:
        def __init__(self, max_workers, initializer, initargs):
            seen["executor"] = (max_workers, initializer, initargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, task_index):
            seen.setdefault("submissions", []).append((function, task_index))
            return FinishedFuture()

    monkeypatch.setattr(assignment_slurm, "load_plan", lambda path: (plan, plan_path))
    monkeypatch.setattr(assignment_slurm, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(assignment_slurm, "as_completed", lambda futures: list(futures))

    assert assignment_slurm.run_allocation_worker(plan_path, 0) == 0
    assert seen["executor"] == (
        2,
        assignment_slurm._initialize_worker,
        (plan_path,),
    )
    assert seen["submissions"] == [
        (assignment_slurm._run_cached_assignment_task, 3),
        (assignment_slurm._run_cached_assignment_task, 7),
    ]


def test_final_metrics_allocation_evaluates_then_publishes(tmp_path, monkeypatch):
    plan_path = (tmp_path / "plan.json").resolve()
    plan = {
        "allocations": [
            {"phase": "metrics-finalize", "task_indices": [2], "cpus": 4}
        ]
    }
    run_metrics = Mock(return_value=False)
    finalize = Mock()
    monkeypatch.setattr(assignment_slurm, "load_plan", lambda _path: (plan, plan_path))
    monkeypatch.setattr(assignment_slurm, "_run_metrics_allocation", run_metrics)
    monkeypatch.setattr(assignment_slurm, "run_metrics_finalizer", finalize)

    assert assignment_slurm.run_allocation_worker(plan_path, 0) == 0
    run_metrics.assert_called_once_with(plan, plan_path, plan["allocations"][0])
    finalize.assert_called_once_with(plan_path)

    run_metrics.reset_mock()
    run_metrics.return_value = True
    finalize.reset_mock()
    assert assignment_slurm.run_allocation_worker(plan_path, 0) == 1
    finalize.assert_not_called()


def test_metrics_allocation_parallelizes_iterations_and_reduces_once(
    tmp_path, monkeypatch
):
    config = {
        "subconfig-name": "first",
        "policies": ["zones"],
        "iterations": {"start": 0, "end": 5},
        "export-local-metrics": False,
    }
    plan = {
        "run_id": "run-1",
        "metrics_fragment_dir": str(tmp_path / "fragments"),
        "metrics_tasks": [
            {"subconfig": "first", "report_names": ["citywide"]}
        ],
        "subconfigs": [{"name": "first", "config": config}],
        "assignment_folder": str(tmp_path / "assignments"),
    }
    allocation = {"phase": "metrics", "task_indices": [0], "cpus": 2}
    submitted = []

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class FakeExecutor:
        def __init__(self, max_workers, initializer, initargs):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, work_batch):
            submitted.append(work_batch)
            return ImmediateFuture(function(work_batch))

    payloads = {
        iteration: {
            "iteration": iteration,
            "expected_config_names": ["first/variant"],
        }
        for iteration in range(5)
    }
    combine = Mock(return_value={"citywide": pd.DataFrame()})
    write_fragment = Mock()
    monkeypatch.setattr(assignment_slurm, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(assignment_slurm, "as_completed", lambda futures: futures)
    monkeypatch.setattr(
        assignment_slurm,
        "_run_cached_metrics_iteration",
        lambda _task_index, iteration: payloads[iteration],
    )
    monkeypatch.setattr(MarketGenerator, "combine_metric_batch_payloads", combine)
    monkeypatch.setattr(MarketGenerator, "write_metric_fragment", write_fragment)

    failed = assignment_slurm._run_metrics_allocation(
        plan, tmp_path / "plan.json", allocation
    )

    assert not failed
    assert submitted == [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 3), (0, 4)],
    ]
    combine.assert_called_once_with(
        [payloads[index] for index in range(5)], include_local_metrics=False
    )
    write_fragment.assert_called_once_with(
        str(tmp_path / "fragments"),
        run_id="run-1",
        subconfig_name="first",
        reports=combine.return_value,
        expected_report_names=["citywide"],
        expected_config_names=["first/variant"],
    )


def test_saved_metric_batches_reuse_existing_evaluator(tmp_path):
    assignment_path = tmp_path / "assignment.csv"
    assignment_path.write_text("studentno\n1\n")
    market = MarketGenerator.__new__(MarketGenerator)
    market.config = {"subconfig-name": "test"}
    evaluator = object()
    market._aggregate_metric_evaluator = evaluator
    market._validate_reusable_assignment = Mock(return_value=pd.DataFrame())
    market._record_assignment_metric_reports = Mock(
        side_effect=lambda _assignment, save_name, iteration: market._aggregate_metric_batches[
            "citywide"
        ].append(
            pd.DataFrame(
                {
                    "config_name": [
                        market._metric_config_name(save_name, iteration)
                    ]
                }
            )
        )
    )

    for iteration in (0, 1):
        market._evaluate_saved_metric_specs(
            [(assignment_path, f"assignment_iteration{iteration}.csv", iteration)]
        )

    assert market._aggregate_metric_evaluator is evaluator
    assert market._record_assignment_metric_reports.call_count == 2


def test_parallel_metric_reduction_applies_frl_threshold_after_averaging():
    def payload(metric, first_frl):
        return {
            "reports": {
                "citywide": [
                    pd.DataFrame({"config_name": ["config/policy"], "metric": [metric]})
                ]
            },
            "frl_threshold_inputs": [
                pd.DataFrame(
                    {
                        "config_name": ["config/policy", "config/policy"],
                        "program_id": ["A", "B"],
                        "frl_assigned": [first_frl, 0.4],
                        "frl_non_designated": [first_frl, 0.4],
                        "district_frl": [0.5, 0.5],
                    }
                )
            ],
            "expected_config_names": ["config/policy"],
        }

    reports = MarketGenerator.combine_metric_batch_payloads(
        [payload(1.0, 0.8), payload(3.0, 0.4)],
        include_local_metrics=False,
    )

    citywide = reports["citywide"].iloc[0]
    assert citywide["metric"] == 2.0
    assert citywide["Alternative # of GE programs above +10% district FRL"] == 1


def test_parallel_metric_reduction_requires_every_expected_variant():
    payload = {
        "reports": {
            "citywide": [
                pd.DataFrame({"config_name": ["config/present"], "metric": [1.0]})
            ]
        },
        "frl_threshold_inputs": [],
        "expected_config_names": ["config/missing", "config/present"],
    }

    with pytest.raises(ValueError, match="every expected assignment variant"):
        MarketGenerator.combine_metric_batch_payloads(
            [payload], include_local_metrics=False
        )


def test_metric_report_frames_require_every_local_group():
    first = _reports("first/variant", 1.0)["program"]
    second = pd.concat(
        [
            _reports("second/variant", 2.0)["program"],
            _reports("second/variant", 3.0)["program"].assign(
                program_id="202-GE-KG", school_id=202, school_name="Beta"
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="inconsistent grouping keys"):
        MarketGenerator._validate_metric_report_frames(
            "program",
            [first, second],
            expected_config_names=["first/variant", "second/variant"],
        )


def test_large_assignment_plan_runs_multiple_waves_per_allocation():
    allocations = _build_allocations(12 * 40 * 3, 0)

    assert len(allocations) == MAX_ASSIGNMENT_JOBS
    assert {allocation["cpus"] for allocation in allocations} == {MAX_CPUS_PER_NODE}
    assert {len(allocation["task_indices"]) for allocation in allocations} == {120}


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


def test_metric_fragments_preserve_every_row_and_column(tmp_path):
    fragment_root = tmp_path / "fragments"
    first = _reports("first/variant", 1 / 13)
    second = _reports("second/variant", 2 / 13)
    first["citywide"]["first_only"] = 9_007_199_254_740_993
    first["citywide"]["second_only"] = pd.NA
    second["citywide"]["first_only"] = pd.NA
    second["citywide"]["second_only"] = 22

    for subconfig, reports in (("first", first), ("second", second)):
        MarketGenerator.write_metric_fragment(
            fragment_root,
            run_id="run-1",
            subconfig_name=subconfig,
            reports=reports,
            expected_report_names=reports,
            expected_config_names=[f"{subconfig}/variant"],
        )
        MarketGenerator.write_metric_fragment(
            fragment_root,
            run_id="run-1",
            subconfig_name=subconfig,
            reports=reports,
            expected_report_names=reports,
            expected_config_names=[f"{subconfig}/variant"],
        )

    conflicting = {name: report.copy() for name, report in first.items()}
    conflicting["citywide"].loc[0, "new_metric"] = 99
    with pytest.raises(RuntimeError, match="conflicts"):
        MarketGenerator.write_metric_fragment(
            fragment_root,
            run_id="run-1",
            subconfig_name="first",
            reports=conflicting,
            expected_report_names=conflicting,
            expected_config_names=["first/variant"],
        )

    combined, manifests = MarketGenerator.combine_metric_fragments(
        fragment_root,
        run_id="run-1",
        expected_fragments=[
            {"subconfig": "first", "report_names": list(first)},
            {"subconfig": "second", "report_names": list(second)},
        ],
    )

    citywide = combined["citywide"]
    assert citywide["config_name"].tolist() == ["first/variant", "second/variant"]
    assert citywide.columns.tolist() == [
        "config_name",
        "new_metric",
        "first_only",
        "second_only",
    ]
    assert citywide["new_metric"].tolist() == [1 / 13, 2 / 13]
    assert citywide["first_only"].iloc[0] == 9_007_199_254_740_993
    assert pd.isna(citywide["first_only"].iloc[1])
    assert pd.isna(citywide["second_only"].iloc[0])
    assert citywide["second_only"].iloc[1] == 22
    assert set(manifests) == {"first", "second"}
    for report_name in MarketGenerator.AGGREGATE_METRIC_FILES:
        assert len(combined[report_name]) == 2

    MarketGenerator.write_aggregate_metric_reports(
        tmp_path,
        combined,
        manifest={
            "schema_version": 1,
            "run_id": "run-1",
            "subconfigs": ["first", "second"],
            "fragments": manifests,
        },
    )
    aggregate_manifest = json.loads(
        (tmp_path / "aggregate_metrics/manifest.json").read_text()
    )
    assert aggregate_manifest["subconfigs"] == ["first", "second"]
    assert aggregate_manifest["aggregate_reports"]["citywide"]["row_count"] == 2
    published = pd.read_csv(
        tmp_path / "aggregate_metrics/metrics_citywide.csv",
        dtype={"first_only": "Int64"},
        float_precision="round_trip",
    )
    assert published.columns.tolist() == citywide.columns.tolist()
    assert len(published) == len(citywide)
    assert published.loc[0, "new_metric"] == 1 / 13
    assert published.loc[0, "first_only"] == 9_007_199_254_740_993


def test_metric_finalization_rejects_missing_or_corrupt_fragments(tmp_path):
    fragment_root = tmp_path / "fragments"
    fragment_root.mkdir()
    expected = [
        {"subconfig": "first", "report_names": ["citywide"]},
        {"subconfig": "second", "report_names": ["citywide"]},
    ]
    with pytest.raises(ValueError, match="does not cover every expected config"):
        MarketGenerator.write_metric_fragment(
            fragment_root,
            run_id="run-1",
            subconfig_name="first",
            reports={
                "citywide": pd.DataFrame(columns=["config_name", "new_metric"])
            },
            expected_report_names=["citywide"],
            expected_config_names=["first/variant"],
        )
    MarketGenerator.write_metric_fragment(
        fragment_root,
        run_id="run-1",
        subconfig_name="first",
        reports={"citywide": _reports("first/variant", 1.0)["citywide"]},
        expected_report_names=["citywide"],
        expected_config_names=["first/variant"],
    )

    with pytest.raises(ValueError, match="incomplete"):
        MarketGenerator.combine_metric_fragments(
            fragment_root, run_id="run-1", expected_fragments=expected
        )

    MarketGenerator.write_metric_fragment(
        fragment_root,
        run_id="run-1",
        subconfig_name="second",
        reports={"citywide": _reports("second/variant", 2.0)["citywide"]},
        expected_report_names=["citywide"],
        expected_config_names=["second/variant"],
    )
    first_dir = fragment_root / MarketGenerator.metric_fragment_id("first")
    with (first_dir / "metrics_citywide.csv").open("a") as stream:
        stream.write("corrupt,row\n")

    with pytest.raises(ValueError, match="checksum"):
        MarketGenerator.combine_metric_fragments(
            fragment_root, run_id="run-1", expected_fragments=expected
        )


def test_metric_finalization_rejects_inconsistent_fragment_columns(tmp_path):
    fragment_root = tmp_path / "fragments"
    first = {"citywide": _reports("first/variant", 1.0)["citywide"]}
    second = {"citywide": _reports("second/variant", 2.0)["citywide"]}
    first["citywide"]["missing_from_second"] = 3.0
    for subconfig, reports in (("first", first), ("second", second)):
        MarketGenerator.write_metric_fragment(
            fragment_root,
            run_id="run-1",
            subconfig_name=subconfig,
            reports=reports,
            expected_report_names=["citywide"],
            expected_config_names=[f"{subconfig}/variant"],
        )

    with pytest.raises(ValueError, match="inconsistent citywide columns"):
        MarketGenerator.combine_metric_fragments(
            fragment_root,
            run_id="run-1",
            expected_fragments=[
                {"subconfig": "first", "report_names": ["citywide"]},
                {"subconfig": "second", "report_names": ["citywide"]},
            ],
        )


def test_slurm_finalizer_does_not_publish_incomplete_run(tmp_path, monkeypatch):
    aggregate_dir = tmp_path / "assignments/aggregate_metrics"
    aggregate_dir.mkdir(parents=True)
    published_path = aggregate_dir / "metrics_citywide.csv"
    published_path.write_text("config_name,new_metric\nprevious/variant,7\n")
    fragment_root = tmp_path / "fragments"
    fragment_root.mkdir()
    plan_path = tmp_path / "plan.json"
    plan = {
        "run_id": "run-1",
        "metrics_fragment_dir": str(fragment_root),
        "assignment_folder": str(tmp_path / "assignments"),
        "metrics_tasks": [
            {"subconfig": "missing", "report_names": ["citywide"]}
        ],
    }
    monkeypatch.setattr(
        assignment_slurm, "load_plan", lambda _path: (plan, plan_path)
    )

    with pytest.raises(ValueError, match="incomplete"):
        assignment_slurm.run_metrics_finalizer(plan_path)

    assert published_path.read_text() == (
        "config_name,new_metric\nprevious/variant,7\n"
    )

    MarketGenerator.write_metric_fragment(
        fragment_root,
        run_id="run-1",
        subconfig_name="missing",
        reports={"citywide": _reports("missing/variant", 2.0)["citywide"]},
        expected_report_names=["citywide"],
        expected_config_names=["missing/variant"],
    )
    assignment_slurm.run_metrics_finalizer(plan_path)

    published = pd.read_csv(published_path)
    assert published["config_name"].tolist() == ["missing/variant"]
    manifest = json.loads((aggregate_dir / "manifest.json").read_text())
    assert manifest["run_id"] == "run-1"
    assert manifest["subconfigs"] == ["missing"]


def test_aggregate_publication_lock_preserves_existing_output(tmp_path):
    aggregate_dir = tmp_path / "aggregate_metrics"
    aggregate_dir.mkdir()
    published_path = aggregate_dir / "metrics_citywide.csv"
    published_path.write_text("config_name,new_metric\nprevious/variant,7\n")
    (tmp_path / ".aggregate_metrics.publish.lock").mkdir()

    with pytest.raises(RuntimeError, match="Another process"):
        MarketGenerator.write_aggregate_metric_reports(
            tmp_path,
            {"citywide": _reports("new/variant", 2.0)["citywide"]},
        )

    assert published_path.read_text() == (
        "config_name,new_metric\nprevious/variant,7\n"
    )


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
