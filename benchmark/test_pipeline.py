"""Pipelines retain their semantics through sweep, storage and assignment."""

from dataclasses import replace

import pytest

from benchmark.assignment import planned_stage_names
from benchmark.config import (
    BenchmarkTask,
    ExecutionConfig,
    SimulationSweep,
    capacity_slots,
    optimization_config_hash,
    optimization_config_to_dict,
)
from benchmark.results import aggregate_results
from benchmark.runner import (
    MANIFEST_FILENAME,
    load_solutions,
    manifest_for,
    save_stage_artifacts,
    stage_names_for,
    write_json,
)
from benchmark.slurm import _optimization_cpus
from optimization.config import OptimizationConfig
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import FakeDataset, make_grid_problem


def _steps():
    return [
        {
            "name": "coarse",
            "level": "Block_1",
            "method": "cp_int",
            "objective": "cut_edges",
        },
        {
            "name": "refine",
            "level": "Block_0",
            "method": "cp_bool",
            "objective": "cut_edges",
        },
        {
            "name": "polish",
            "level": "Block_0",
            "method": "cp_int",
            "objective": "cut_edges",
        },
    ]


@pytest.mark.parametrize("location", ["sweep", "tasks"])
def test_sweep_treats_a_pipeline_as_one_sequence(location):
    kwargs = {
        location: {"steps": _steps()} if location == "sweep" else [{"steps": _steps()}]
    }
    sweep = SimulationSweep(**kwargs)

    tasks = sweep.generate_tasks()

    assert len(tasks) == 1
    assert len(tasks[0].config["steps"]) == 3
    assert planned_stage_names(tasks[0].config) == [
        "stage_00_Block_1",
        "stage_01_Block_0",
        "stage_02_Block_0",
    ]


def test_sweep_accepts_alternative_pipelines_and_tracks_nested_constraints():
    first = _steps()
    second = _steps()
    second[1]["constraints"] = {"frl_dev": 0.1}
    tasks = SimulationSweep(
        sweep={"steps": [first, second], "seed": [1, 2]}
    ).generate_tasks()

    assert len(tasks) == 4
    assert len({task.config_hash for task in tasks}) == 4


def test_scheduler_and_slurm_reserve_the_largest_step():
    config = {
        "workers": 2,
        "steps": [{"options": {"workers": 8}}, {"options": {"workers": 3}}],
    }
    task = BenchmarkTask("task", "hash", config, "/tmp/test-pipeline", 8)

    assert capacity_slots(config, ExecutionConfig()) == 8
    assert capacity_slots(config, ExecutionConfig(task_capacity=5)) == 5
    assert _optimization_cpus(task) == 8
    assert (
        capacity_slots(
            {"workers": 8, "steps": [{"options": {"workers": 3}}]}, ExecutionConfig()
        )
        == 3
    )


def test_nested_school_distance_objective_retains_its_weights():
    steps = [{"level": "Block_0", "method": "cp_int", "objective": "school_distance"}]
    first = OptimizationConfig(
        steps=steps, school_distance_area_weight=0.5, school_distance_student_weight=0.5
    )
    second = OptimizationConfig(
        steps=steps,
        school_distance_area_weight=0.25,
        school_distance_student_weight=0.75,
    )

    assert optimization_config_hash(first) != optimization_config_hash(second)


def test_legacy_hash_payload_omits_empty_pipeline_settings(monkeypatch):
    payloads = []

    def capture(payload):
        payloads.append(payload)
        return "legacy-task-hash"

    monkeypatch.setattr("benchmark.config.stable_hash", capture)
    result = optimization_config_hash(OptimizationConfig(), source_manifest={})

    assert result == "legacy-task-hash"
    assert (
        not {"method", "objective", "steps", "constraints"}
        & payloads[0]["optimization"].keys()
    )


def test_saved_pipeline_reconstructs_each_steps_problem(tmp_path):
    config = OptimizationConfig(
        steps=[
            {**step, "level": "Block_0", "constraints": {"frl_dev": 0.1 * (index + 1)}}
            for index, step in enumerate(_steps())
        ]
    )
    problem = make_grid_problem(2, 2)
    solutions = [
        ZoneSolution(
            problem=replace(problem, frl_dev=0.1 * (index + 1)),
            assignment={0: 0, 1: 0, 2: 1, 3: 1},
            status="FEASIBLE",
            metadata={
                "pipeline_step": name,
                "pipeline_step_index": index,
                "method": "cp_int",
                "objective_kind": "cut_edges",
            },
        )
        for index, name in enumerate(["coarse", "refine", "polish"])
    ]
    records = save_stage_artifacts(
        solutions, str(tmp_path), stage_names_for(solutions, config)
    )
    task = BenchmarkTask(
        "task", "hash", optimization_config_to_dict(config), str(tmp_path), 1
    )
    manifest = manifest_for(
        task=task,
        config=config,
        status="FEASIBLE",
        started_at="2026-01-01",
        completed_at="2026-01-01",
        stages=records,
        error_message=None,
    )
    calls = []

    class StageDataset(FakeDataset):
        def problem_for(self, level, *, config=None, **kwargs):
            calls.append(config.frl_dev)
            return replace(super().problem_for(level, **kwargs), frl_dev=config.frl_dev)

    write_json(str(tmp_path / MANIFEST_FILENAME), manifest)

    loaded, _, _ = load_solutions(str(tmp_path), dataset=StageDataset(problem))
    _, stages = aggregate_results(str(tmp_path))

    assert calls == pytest.approx([0.1, 0.2, 0.3])
    assert [solution.problem.frl_dev for solution in loaded] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert stages["pipeline_step"].tolist() == ["coarse", "refine", "polish"]
    assert stages["method"].tolist() == ["cp_int"] * 3
    assert stages["objective_kind"].tolist() == ["cut_edges"] * 3


def test_10_24_sweep_pairs_all_ideal_scenarios_and_freezes_shape(monkeypatch):
    from pathlib import Path
    import yaml

    monkeypatch.setattr(
        "benchmark.config._benchmark_source_manifest",
        lambda config, **kw: {"capacity_scenario": config.capacity_scenario},
    )
    path = Path(__file__).parent / "configs/attendance_areas/10-24.yaml"
    sweep = SimulationSweep.from_yaml(str(path))
    tasks = sweep.generate_tasks()
    assert len(tasks) == 120
    assert len({task.config_hash for task in tasks}) == 120
    assert {task.optimization_config().frl_dev for task in tasks} == {0.15, 0.2, 0.25, 0.3}
    catalog = yaml.safe_load(
        (path.parents[3] / "loaders/configs/base.yaml").read_text()
    )
    expected = {
        s
        for s in catalog["capacity_scenarios"]
        if s.startswith("ideal_") and s.endswith("2627")
    }
    assert {task.optimization_config().capacity_scenario for task in tasks} == expected
    for task in tasks:
        config = task.optimization_config()
        assert config.capacity_scenario == config.data_scenario.filter(
            "assignment", "capacity_scenario"
        )
        shape, distance, perimeter = config.steps
        assert shape.objective == "capacity_match"
        assert shape.options["recom_weights"] == "hedge"
        assert shape.options["recom_tree_sampler"] == "random_mst"
        assert shape.options["capacity_match_unassigned_weight"] == 1
        assert shape.options["capacity_match_designated_weight"] == 0
        assert shape.options["hints"] == "voronoi"
        assert shape.constraints == {"overage": -1, "shortage": -1}
        assert distance.capacity_reference == perimeter.capacity_reference == "shape"
        assert distance.method == perimeter.method == "cp_bool"
        from optimization.pipeline import step_config

        effective = [step_config(config, step, i) for i, step in enumerate(config.steps)]
        assert all(step.frl_dev == config.frl_dev for step in effective)
        assert effective[0].overage == effective[0].shortage == -1
        assert effective[0].recom_tree_sampler == "random_mst"
        assert all(step.overage == 0.10 and step.shortage == 0.05 for step in effective[1:])
