import json
import pathlib
import subprocess
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import assignment.slurm as assignment_slurm
from assignment.slurm import (
    PLAN_SCHEMA_VERSION,
    build_slurm_plan,
    write_slurm_scripts,
)
from assignment.slurm_graph import (
    MAX_ASSIGNMENT_JOBS,
    MAX_CPUS_PER_NODE,
    MAX_METRICS_JOBS,
    MAX_SLURM_JOBS,
    build_job_graph,
    topological_jobs,
    validate_job_graph,
)
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


def test_kumar_plan_uses_one_task_per_run_with_targeted_dependencies(tmp_path):
    config_path = pathlib.Path(__file__).parents[1] / "configs/kumar.config.yaml"

    plan, plan_path = build_slurm_plan(
        config_path,
        assignment_folder=tmp_path / "assignments",
        plan_dir=tmp_path / "plan",
    )

    assert len(plan["assignment_tasks"]) == 19
    assert len(plan["metrics_tasks"]) == 19
    assert len(plan["jobs"]) == MAX_SLURM_JOBS
    assignment_jobs = [job for job in plan["jobs"] if job["kind"] == "assignment"]
    assert len(assignment_jobs) == MAX_ASSIGNMENT_JOBS
    assert {job["cpus"] for job in assignment_jobs} == {25, MAX_CPUS_PER_NODE}
    assert sorted(
        index for job in assignment_jobs for index in job["task_indices"]
    ) == list(range(19))
    metrics_jobs = [job for job in plan["jobs"] if job["kind"].startswith("metrics")]
    assert len(metrics_jobs) == MAX_METRICS_JOBS
    assert {job["cpus"] for job in metrics_jobs} == {MAX_CPUS_PER_NODE}
    assert sorted(
        index for job in metrics_jobs for index in job["task_indices"]
    ) == list(range(19))
    assignment_job_ids = {job["id"] for job in assignment_jobs}
    for job in metrics_jobs:
        required_subconfigs = {
            plan["metrics_tasks"][index]["subconfig"] for index in job["task_indices"]
        }
        expected_assignment_ids = {
            assignment_job["id"]
            for assignment_job in assignment_jobs
            if any(
                plan["assignment_tasks"][index]["subconfig"] in required_subconfigs
                for index in assignment_job["task_indices"]
            )
        }
        actual_assignment_ids = set(job["dependencies"]["afterok"]) & assignment_job_ids
        assert actual_assignment_ids == expected_assignment_ids
        assert actual_assignment_ids != assignment_job_ids
    finalizer = metrics_jobs[-1]
    assert finalizer["id"] == "metrics-finalize"
    assert set(finalizer["dependencies"]["afterok"]) >= {
        job["id"] for job in metrics_jobs[:-1]
    }
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
            "include_real_match": False,
            "write_utility_output": True,
        }
    ]
    submit_path = write_slurm_scripts(plan_path)
    submit_script = submit_path.read_text()
    assert "submit-plan" in submit_script
    assert "sbatch" not in submit_script
    assert len(list(submit_path.parent.glob("assignment-*.sh"))) == 12
    assert len(list(submit_path.parent.glob("metrics-*.sh"))) == 8
    assert (submit_path.parent / "metrics-finalize.sh").is_file()
    subprocess.run(["bash", "-n", str(submit_path)], check=True)


def test_submission_persists_ids_wires_dependencies_and_cancels_on_failure(
    tmp_path, monkeypatch
):
    plan_path = (tmp_path / "plan.json").resolve()
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": "test-run",
        "workspace_root": str(pathlib.Path(__file__).parents[2]),
        "plan_path": str(plan_path),
        "jobs": build_job_graph(2, 1, metric_assignment_dependencies=[[0, 1]]),
    }
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(assignment_slurm, "load_plan", lambda _path: (plan, plan_path))

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
    assert "job-worker" in worker_script
    assert "--job-id assignment-0" in worker_script
    assert "submit-plan" in submit_script
    assert "srun" not in submit_script
    subprocess.run(["bash", "-n", str(submit_path)], check=True)

    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[0] == "scancel":
            return subprocess.CompletedProcess(command, 0, "", "")
        if len([call for call in calls if call[0] == "sbatch"]) < 3:
            job_id = str(122 + len(calls))
            return subprocess.CompletedProcess(command, 0, f"{job_id}\n", "")
        return subprocess.CompletedProcess(command, 1, "", "scheduler unavailable")

    with pytest.raises(RuntimeError, match="metrics-finalize"):
        assignment_slurm.submit_slurm_plan(
            plan_path, script_dir=submit_path.parent, runner=fake_run
        )

    assert calls[2][0:2] == ["sbatch", "--parsable"]
    assert "--dependency=afterok:123:124" in calls[2]
    assert calls[3] == ["scancel", "123", "124"]
    state = json.loads((tmp_path / "submission.json").read_text())
    assert state["status"] == "submission-failed"
    assert state["jobs"] == [
        {"job_id": "assignment-0", "slurm_job_id": "123"},
        {"job_id": "assignment-1", "slurm_job_id": "124"},
    ]


def test_successful_submission_is_durable_and_cannot_repeat(tmp_path, monkeypatch):
    plan_path = (tmp_path / "plan.json").resolve()
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": "test-run",
        "workspace_root": str(pathlib.Path(__file__).parents[2]),
        "plan_path": str(plan_path),
        "jobs": build_job_graph(1, 0),
    }
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(assignment_slurm, "load_plan", lambda _path: (plan, plan_path))
    submit_path = write_slurm_scripts(plan_path)

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, "321;cluster\n", "")

    submission_path = assignment_slurm.submit_slurm_plan(
        plan_path, script_dir=submit_path.parent, runner=fake_run
    )

    state = json.loads(submission_path.read_text())
    assert state["status"] == "submitted"
    assert state["jobs"] == [{"job_id": "assignment-0", "slurm_job_id": "321"}]
    with pytest.raises(RuntimeError, match="already has submission state"):
        assignment_slurm.submit_slurm_plan(
            plan_path, script_dir=submit_path.parent, runner=fake_run
        )


def test_submission_adds_external_dependencies_to_assignment_jobs(
    tmp_path, monkeypatch
):
    plan_path = (tmp_path / "plan.json").resolve()
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": "test-run",
        "workspace_root": str(pathlib.Path(__file__).parents[2]),
        "plan_path": str(plan_path),
        "jobs": build_job_graph(1, 0),
    }
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(assignment_slurm, "load_plan", lambda _path: (plan, plan_path))
    submit_path = write_slurm_scripts(plan_path)
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "321\n", "")

    assignment_slurm.submit_slurm_plan(
        plan_path,
        script_dir=submit_path.parent,
        runner=fake_run,
        upstream_job_ids=["100", "101"],
    )

    assert calls[0][:2] == ["sbatch", "--parsable"]
    assert "--dependency=afterok:100:101" in calls[0]


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

        def simulate_target(
            self,
            subconfig_name,
            iteration,
            *,
            include_real_match,
            write_utility_output,
        ):
            events.append(("simulate", subconfig_name, iteration))

    plan = {
        "plan_path": str(tmp_path / "plan.json"),
        "assignment_folder": str(tmp_path / "assignments"),
        "subconfigs": [
            {
                "name": name,
                "config": {
                    "subconfig-name": name,
                    "policies": ["zones"],
                    "iterations": {"start": 0, "end": 3},
                },
            }
            for name in ("first", "second")
        ],
        "assignment_tasks": [
            {
                "subconfig": subconfig,
                "include_real_match": False,
                "write_utility_output": False,
            }
            for subconfig in ("first", "second")
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

    assert assignment_slurm._run_cached_assignment_batch(0, [0, 1]) == []
    assert assignment_slurm._run_cached_assignment_batch(1, [0]) == []
    assert assignment_slurm._run_cached_assignment_batch(0, [2]) == []

    assert events == [
        ("initialize", "first"),
        ("simulate", "first", 0),
        ("simulate", "first", 1),
        ("reconfigure", "second"),
        ("simulate", "second", 0),
        ("reconfigure", "first"),
        ("simulate", "first", 2),
    ]


def test_job_pool_initializes_each_process_with_the_plan(tmp_path, monkeypatch):
    plan_path = (tmp_path / "plan.json").resolve()
    plan = {
        "jobs": [
            {
                "id": "assignment-0",
                "kind": "assignment",
                "task_indices": [3, 7],
                "cpus": 2,
                "dependencies": {},
            }
        ],
    }
    seen = {}

    class FinishedFuture:
        def result(self):
            return []

    class FakeExecutor:
        def __init__(self, max_workers, initializer, initargs):
            seen["executor"] = (max_workers, initializer, initargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, task_index, iterations):
            seen.setdefault("submissions", []).append(
                (function, task_index, iterations)
            )
            return FinishedFuture()

    monkeypatch.setattr(assignment_slurm, "load_plan", lambda path: (plan, plan_path))
    monkeypatch.setattr(
        assignment_slurm,
        "_assignment_batches",
        lambda _plan, _job: [(3, [0, 1]), (7, [2])],
    )
    monkeypatch.setattr(assignment_slurm, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(assignment_slurm, "as_completed", lambda futures: list(futures))

    assert assignment_slurm.run_job_worker(plan_path, "assignment-0") == 0
    assert seen["executor"] == (
        2,
        assignment_slurm._initialize_worker,
        (plan_path,),
    )
    assert seen["submissions"] == [
        (assignment_slurm._run_cached_assignment_batch, 3, [0, 1]),
        (assignment_slurm._run_cached_assignment_batch, 7, [2]),
    ]


def test_final_metrics_job_evaluates_then_publishes(tmp_path, monkeypatch):
    plan_path = (tmp_path / "plan.json").resolve()
    plan = {
        "jobs": [
            {
                "id": "metrics-finalize",
                "kind": "metrics-finalize",
                "task_indices": [2],
                "cpus": 4,
                "dependencies": {"afterok": ["assignment-0"]},
            }
        ]
    }
    run_metrics = Mock(return_value=False)
    finalize = Mock()
    monkeypatch.setattr(assignment_slurm, "load_plan", lambda _path: (plan, plan_path))
    monkeypatch.setattr(assignment_slurm, "_run_metrics_job", run_metrics)
    monkeypatch.setattr(assignment_slurm, "run_metrics_finalizer", finalize)

    assert assignment_slurm.run_job_worker(plan_path, "metrics-finalize") == 0
    run_metrics.assert_called_once_with(plan, plan_path, plan["jobs"][0])
    finalize.assert_called_once_with(plan_path)

    run_metrics.reset_mock()
    run_metrics.return_value = True
    finalize.reset_mock()
    assert assignment_slurm.run_job_worker(plan_path, "metrics-finalize") == 1
    finalize.assert_not_called()


def test_metrics_job_parallelizes_iterations_and_reduces_once(tmp_path, monkeypatch):
    config = {
        "subconfig-name": "first",
        "policies": ["zones"],
        "iterations": {"start": 0, "end": 5},
        "export-local-metrics": False,
    }
    plan = {
        "run_id": "run-1",
        "metrics_fragment_dir": str(tmp_path / "fragments"),
        "metrics_tasks": [{"subconfig": "first", "report_names": ["citywide"]}],
        "subconfigs": [{"name": "first", "config": config}],
        "assignment_folder": str(tmp_path / "assignments"),
    }
    job = {"kind": "metrics", "task_indices": [0], "cpus": 2}
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

    failed = assignment_slurm._run_metrics_job(plan, tmp_path / "plan.json", job)

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
        side_effect=lambda _assignment, save_name, iteration: (
            market._aggregate_metric_batches["citywide"].append(
                pd.DataFrame(
                    {"config_name": [market._metric_config_name(save_name, iteration)]}
                )
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


def test_large_assignment_plan_runs_multiple_waves_per_job():
    jobs = build_job_graph(12 * 40 * 3, 0)

    assert len(jobs) == MAX_ASSIGNMENT_JOBS
    assert {job["cpus"] for job in jobs} == {MAX_CPUS_PER_NODE}
    assert {len(job["task_indices"]) for job in jobs} == {120}


def test_assignment_batches_keep_independent_runs_intact():
    plan = {
        "subconfigs": [
            {
                "name": f"run-{index}",
                "config": {
                    "policies": ["zones"],
                    "iterations": {"start": 0, "end": 5},
                },
            }
            for index in range(4)
        ],
        "assignment_tasks": [
            {
                "subconfig": f"run-{index}",
                "include_real_match": False,
                "write_utility_output": index == 0,
            }
            for index in range(4)
        ],
    }

    batches = assignment_slurm._assignment_batches(
        plan, {"task_indices": [0, 1, 2, 3], "cpus": 3}
    )

    assert batches == [
        (0, [0, 1, 2, 3, 4]),
        (1, [0, 1, 2, 3, 4]),
        (2, [0, 1, 2, 3, 4]),
        (3, [0, 1, 2, 3, 4]),
    ]


def test_assignment_batches_use_idle_cpus_within_one_run():
    plan = {
        "subconfigs": [
            {
                "name": "only-run",
                "config": {
                    "policies": ["zones"],
                    "iterations": {"start": 0, "end": 5},
                },
            }
        ],
        "assignment_tasks": [
            {
                "subconfig": "only-run",
                "include_real_match": False,
                "write_utility_output": True,
            }
        ],
    }

    batches = assignment_slurm._assignment_batches(
        plan, {"task_indices": [0], "cpus": 5}
    )

    assert batches == [(0, [index]) for index in range(5)]


def test_job_graph_rejects_missing_dependencies_and_cycles():
    metric_dependencies = [[0, 1, 2, 3]]
    jobs = build_job_graph(4, 1, metric_assignment_dependencies=metric_dependencies)
    jobs[-1]["dependencies"] = {"afterok": ["assignment-0"]}

    with pytest.raises(ValueError, match="exact required dependencies"):
        validate_job_graph(
            jobs,
            assignment_count=4,
            metrics_count=1,
            metric_assignment_dependencies=metric_dependencies,
        )

    cyclic = [
        {"id": "first", "dependencies": {"afterok": ["second"]}},
        {"id": "second", "dependencies": {"afterok": ["first"]}},
    ]
    with pytest.raises(ValueError, match="cycle"):
        topological_jobs(cyclic)


def test_assignment_planner_has_one_real_match_writer():
    def subconfig(name, policies):
        return {
            "name": name,
            "config": {
                "policies": policies,
                "iterations": {"start": 0, "end": 3},
            },
        }

    tasks = assignment_slurm._planned_assignment_tasks(
        [
            subconfig("historical", ["real_match"]),
            subconfig("mixed", ["real_match", "zones"]),
        ]
    )

    assert len(tasks) == 2
    assert sum(task["include_real_match"] for task in tasks) == 1
    assert [task["subconfig"] for task in tasks if task["write_utility_output"]] == [
        "mixed"
    ]
    assert assignment_slurm._assignment_work_counts(
        tasks,
        [
            subconfig("historical", ["real_match"]),
            subconfig("mixed", ["real_match", "zones"]),
        ],
    ) == [1, 3]


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
    market._reset_zones.reset_mock()
    targeted = list(market.create_target_iteration_generator(2))

    assert sequential[2].loc[0, "draw"] == targeted[0].loc[0, "draw"]
    assert sequential[5].loc[0, "draw"] == targeted[1].loc[0, "draw"]
    assert sequential[2].loc[0, "draw"] == sequential[5].loc[0, "draw"]
    market._reset_zones.assert_not_called()
    assert MarketGenerator.iteration_seed(2023, 2) != MarketGenerator.iteration_seed(
        2023, 1
    )


def test_targeted_iteration_resets_zones_when_restriction_changes(tmp_path):
    market = _rng_market(tmp_path)
    market.config["guard-rails-reserve-options"] = [
        {"guard-rails": -1, "reserve-settings": {}}
    ]
    market.config["restrict-zone-options"] = [
        {"restrict-zone": False, "citywide-or-lp": []},
        {"restrict-zone": True, "citywide-or-lp": []},
    ]

    list(market.create_target_iteration_generator(2))

    market._reset_zones.assert_called_once_with()


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
            reports={"citywide": pd.DataFrame(columns=["config_name", "new_metric"])},
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
        "subconfigs": [{"name": "missing"}],
        "metrics_tasks": [{"subconfig": "missing", "report_names": ["citywide"]}],
    }
    monkeypatch.setattr(assignment_slurm, "load_plan", lambda _path: (plan, plan_path))

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
