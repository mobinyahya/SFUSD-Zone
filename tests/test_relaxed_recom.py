import networkx as nx
import pytest

from Config.Constants import AREA_ETHNICITIES
from optimization.levels import LevelSpec
from optimization.problem import ZoneProblem
from optimization.solvers import get_solver


def test_relaxed_recom_registered():
    assert get_solver("relaxed_recom").name == "relaxed_recom"


def test_relaxed_recom_uses_hint_and_returns_feasible_solution():
    problem = _small_problem(hint={0: 0, 1: 0, 2: 1, 3: 1})
    solver = get_solver(
        "relaxed_recom",
        seed=3,
        solve_time_limit=10,
        recom_iterations=1,
        recom_cut_attempts=10,
        relaxed_recom_min_boundary_edges=0,
    )

    solution = solver.solve(problem)

    assert solution.status == "FEASIBLE"
    assert solution.is_contiguous()
    assert solution.metadata["hints"] == "provided"
    assert solution.metadata["accepted_moves"] == 1


def test_relaxed_recom_defaults_to_voronoi_initialization_without_hint():
    problem = _small_problem(hint=None)
    solver = get_solver(
        "relaxed_recom",
        seed=3,
        solve_time_limit=10,
        recom_iterations=0,
        recom_cut_attempts=10,
        relaxed_recom_min_boundary_edges=0,
    )

    solution = solver.solve(problem)

    assert solution.status == "FEASIBLE"
    assert solution.metadata["hints"] == "voronoi"


def test_relaxed_recom_rejects_choice_objective():
    problem = _small_problem(
        hint={0: 0, 1: 0, 2: 1, 3: 1},
        choice_objective=object(),
    )
    solver = get_solver("relaxed_recom")

    with pytest.raises(NotImplementedError, match="relaxed_recom"):
        solver.solve(problem)


def test_relaxed_recom_cut_weight_uses_configured_frl_constraint():
    loose = _small_problem(hint={0: 0, 1: 0, 2: 1, 3: 1})
    tight = _small_problem(hint={0: 0, 1: 0, 2: 1, 3: 1})
    for problem in (loose, tight):
        problem.G.nodes[0]["FRL"] = 1.0
        problem.G.nodes[1]["FRL"] = 1.0
        problem.G.nodes[2]["FRL"] = 0.0
        problem.G.nodes[3]["FRL"] = 0.0
    loose.frl_dev = 1.0
    tight.frl_dev = 0.1
    solver = get_solver("relaxed_recom")

    loose_weight = solver._relaxed_cut_log_weight(loose, {0, 1}, {2, 3})
    tight_weight = solver._relaxed_cut_log_weight(tight, {0, 1}, {2, 3})

    assert loose_weight == pytest.approx(0.0)
    assert tight_weight < loose_weight


def _small_problem(hint, choice_objective=None) -> ZoneProblem:
    graph = nx.path_graph(4)
    for node in graph.nodes:
        graph.nodes[node].update(
            {
                "area_id": node,
                "ge_students": 1.0,
                "ge_capacity": 1.0,
                "all_prog_students": 1.0,
                "all_prog_capacity": 1.0,
                "num_schools": 1,
                "school_ids": [node],
                "FRL": 0.5,
                "lat": 0.0,
                "lon": float(node),
            }
        )
        for ethnicity in AREA_ETHNICITIES:
            graph.nodes[node][ethnicity] = 0.2

    graph.graph["distance_dict"] = {
        source: {target: abs(source - target) for target in graph.nodes}
        for source in graph.nodes
    }
    graph.graph["F"] = 0.5
    graph.graph["R"] = {ethnicity: 0.2 for ethnicity in AREA_ETHNICITIES}

    return ZoneProblem(
        G=graph,
        level=LevelSpec.parse("BlockGroup_0"),
        centroids=[0, 3],
        frl_dev=1.0,
        racial_dev=1.0,
        overage=10.0,
        shortage=10.0,
        hint=hint,
        choice_objective=choice_objective,
    )
