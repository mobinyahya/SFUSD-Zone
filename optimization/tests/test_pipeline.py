"""Pipeline contracts across real solver calls and persisted stage semantics."""

from dataclasses import asdict, replace

import networkx as nx
import pytest

from benchmark.config import optimization_config_from_dict, optimization_config_to_dict
from optimization.config import OptimizationConfig
from optimization.levels import LevelSpec
from optimization.pipeline import SolutionInput, SolveStep, step_config
from optimization.solution import ZoneSolution
from optimization.strategies.pipeline import selected_nodes
from optimization.tests.synthetic import make_grid_problem


class MemoryDataset:
    """Real problems and solver models, without production source files."""

    def __init__(self, config):
        self.config = config
        self.data = config.data_scenario
        self.base = make_grid_problem(4, 4, frl_dev=1.0, racial_dev=1.0)
        self.problems = []

    def graph_for(self, level):
        return self.base.G

    def problem_for(self, level, *, config=None, **kwargs):
        config = config or self.config
        problem = replace(
            self.base,
            level=LevelSpec.parse(level),
            frl_dev=config.frl_dev,
            racial_dev=config.racial_dev,
            overage=config.overage,
            shortage=config.shortage,
            max_distance=config.max_distance,
            boundary_prop=config.boundary_prop,
            optimization_config=config,
            **kwargs,
        )
        self.problems.append(problem)
        return problem


def _config(**kwargs):
    return OptimizationConfig(
        hints="none",
        workers=1,
        frl_dev=1,
        racial_dev=1,
        overage=5,
        shortage=0,
        **kwargs,
    )


def test_method_objective_shorthand_and_serialization():
    config = _config(method="cp_bool", objective="cut_edges")
    assert config.strategy == "pipeline"
    assert config.steps[0].method == "cp_bool"
    assert asdict(config)["steps"][0]["objective"] == "cut_edges"
    restored = optimization_config_from_dict(optimization_config_to_dict(config))
    assert restored.steps == config.steps


def test_typed_and_yaml_steps_share_default_input_semantics():
    typed = _config(
        steps=[
            SolveStep("a", "BlockGroup_0", "cp_int"),
            SolveStep("b", "BlockGroup_0", "cp_int"),
        ]
    )
    mapped = _config(steps=[{"name": "a"}, {"name": "b"}])
    assert typed.steps == mapped.steps
    assert typed.steps[0].inputs == []
    assert typed.steps[1].inputs == [SolutionInput()]


@pytest.mark.parametrize(
    "method",
    [
        "cp_bool",
        "cp_int",
        "recom",
        "relaxed_recom",
        "short_bursts",
        "adaptive_short_bursts",
    ],
)
def test_pipeline_solves_and_preserves_hard_inputs(method):
    config = _config(
        steps=[
            {
                "name": "seed",
                "level": "BlockGroup_0",
                "method": "cp_bool",
                "time_limit": 2,
            },
            {
                "name": "polish",
                "level": "BlockGroup_0",
                "method": method,
                "time_limit": 1,
                "options": {"recom_iterations": 4, "short_bursts_length": 2},
                "constraints": {"frl_dev": 0.2},
                "inputs": [{"mode": "hint"}, {"mode": "assign", "nodes": [1, 4, 10]}],
            },
        ]
    )
    dataset = MemoryDataset(config)
    results = config.make_strategy().run(dataset, config.make_solver())
    assert len(results) == 2
    assert all(result.feasible for result in results)
    assert results[1].problem.frl_dev == 0.2
    assert results[0].problem.frl_dev == 1
    assert all(results[1].assignment[n] == results[0].assignment[n] for n in (1, 4, 10))
    assert results[1].metadata["method"] == method
    assert results[1].metadata["pipeline_step"] == "polish"


def test_boundary_release_selects_hops_and_can_intersect_nodes():
    graph = nx.path_graph(8)
    assignment = {n: int(n >= 4) for n in graph}
    assert selected_nodes(
        graph, assignment, SolutionInput(select="boundary", radius=0)
    ) == {3, 4}
    assert selected_nodes(
        graph, assignment, SolutionInput(select="boundary", radius=1)
    ) == {2, 3, 4, 5}
    assert selected_nodes(
        graph, assignment, SolutionInput(select="interior", radius=1)
    ) == {0, 1, 6, 7}
    assert selected_nodes(
        graph, assignment, SolutionInput(select="boundary", radius=1, nodes=[0, 2, 4])
    ) == {2, 4}


def test_revisiting_levels_and_named_inputs(monkeypatch):
    config = _config(
        steps=[
            {"name": "first", "level": "BlockGroup_0"},
            {"name": "other", "level": "BlockGroup_1", "inputs": []},
            {
                "level": "BlockGroup_0",
                "inputs": [{"source": "first", "mode": "assign"}],
            },
        ]
    )
    dataset = MemoryDataset(config)
    calls = []

    def solve(problem, method, objective, **kwargs):
        assignment = {n: int(n >= 8) for n in problem.G}
        calls.append(problem)
        return ZoneSolution(problem, assignment, "FEASIBLE", 4, 0.01)

    monkeypatch.setattr("optimization.strategies.pipeline.solve_objective", solve)
    results = config.make_strategy().run(dataset, config.make_solver())
    assert config.levels == ["BlockGroup_0", "BlockGroup_1", "BlockGroup_0"]
    assert calls[1].hint is None
    assert calls[2].fixed == results[0].assignment


def test_failed_solve_stops_without_using_invalid_input(monkeypatch):
    config = _config(steps=[{"level": "BlockGroup_0"}, {"level": "BlockGroup_0"}])
    monkeypatch.setattr(
        "optimization.strategies.pipeline.solve_objective",
        lambda problem, *args, **kwargs: ZoneSolution(
            problem, {}, "INFEASIBLE", wall_time=0
        ),
    )
    results = config.make_strategy().run(MemoryDataset(config), config.make_solver())
    assert len(results) == 1
    assert results[-1].status == "INFEASIBLE"


@pytest.mark.parametrize(
    "steps,match",
    [
        ([{"name": "x"}, {"name": "x"}], "Duplicate"),
        ([{"inputs": [{"source": "later"}]}], "earlier"),
        ([{"inputs": [{"source": "previous"}]}], "earlier"),
        ([{"options": {"data": {}}}], "Unknown step options"),
        ([{"constraints": {"typo": 1}}], "Unknown step constraints"),
        ([{"inputs": [{"mode": "fixed"}]}], "input.mode"),
        ([{"time_limit": float("nan")}], "finite"),
    ],
)
def test_invalid_plans_fail_before_loading_graphs(steps, match):
    with pytest.raises(ValueError, match=match):
        _config(steps=steps)


def test_step_constraints_do_not_leak_and_sources_are_shared():
    config = _config(
        constraints={"frl_dev": 0.3},
        steps=[
            {"constraints": {"frl_dev": 0.1, "overage": -1}},
            {"objective": "cut_edges"},
        ],
    )
    first, second = [step_config(config, s, i) for i, s in enumerate(config.steps)]
    assert first.frl_dev == 0.1 and first.overage == -1
    assert second.frl_dev == 0.3 and second.overage == 5
    assert first.data_scenario is second.data_scenario is config.data_scenario


def test_saved_hard_assignments_survive_node_renumbering():
    config = _config(method="cp_bool")
    dataset = MemoryDataset(config)
    graph = dataset.base.G
    old_area = graph.nodes[1]["area_id"]
    dataset.base = replace(
        dataset.base,
        G=nx.relabel_nodes(graph, {n: 15 - n for n in graph}),
        centroids=[15, 0],
    )
    problem = config.make_strategy().problem_for_stage(
        dataset,
        {
            "pipeline_step_index": 0,
            "pipeline_fixed": {"1": 0},
            "pipeline_fixed_areas": {str(old_area): 0},
        },
    )
    assert problem.fixed == {14: 0}


def test_hard_inputs_cannot_override_a_steps_tighter_distance(monkeypatch):
    config = _config(
        steps=[
            {"name": "loose", "level": "BlockGroup_0"},
            {
                "level": "BlockGroup_0",
                "constraints": {"max_distance": 0.5},
                "inputs": [{"source": "loose", "mode": "assign"}],
            },
        ]
    )

    def solve(problem, *args, **kwargs):
        return ZoneSolution(problem, {n: int(n >= 8) for n in problem.G}, "FEASIBLE")

    monkeypatch.setattr("optimization.strategies.pipeline.solve_objective", solve)
    with pytest.raises(ValueError, match="violates the candidate constraints"):
        config.make_strategy().run(MemoryDataset(config), config.make_solver())


@pytest.mark.parametrize("objective", ["choice_welfare", "saa_welfare"])
@pytest.mark.parametrize(
    "options", [{"max_iterations": 0}, {"tolerance": float("nan")}]
)
def test_invalid_welfare_loop_settings_are_rejected_before_loading(objective, options):
    with pytest.raises(ValueError, match="max_iterations|tolerance"):
        _config(
            steps=[{"method": "cp_bool", "objective": objective, "options": options}]
        )


@pytest.mark.parametrize("values", [[], [True], [float("nan")], [-1]])
@pytest.mark.parametrize("name", ["solve_time_limits", "gap_limits"])
def test_pipeline_default_limits_are_validated(name, values):
    with pytest.raises(ValueError, match=name):
        _config(method="cp_bool", **{name: values})


def test_repeated_levels_keep_distinct_discoverable_solver_logs(tmp_path):
    from benchmark.solver_logs import collect

    config = _config(
        save_solver_logs=True,
        steps=[
            {"name": "first", "level": "BlockGroup_0", "method": "cp_bool"},
            {"name": "second", "level": "BlockGroup_0", "method": "cp_bool"},
        ],
    )
    results = config.make_strategy().run(
        MemoryDataset(config), config.make_solver(output_dir=str(tmp_path))
    )
    paths = [tmp_path / s.metadata["solver_log_path"] for s in results]
    assert paths[0] != paths[1]
    assert all(path.is_file() for path in paths)
    logs = collect(str(tmp_path))
    assert set(logs["pipeline_step_index"]) == {0, 1}
    assert logs["log_file"].nunique() == 2
