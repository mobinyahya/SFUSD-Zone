"""Contracting fixed groups loses nothing: same zonings, costs and optimum."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from Config.Constants import AREA_ETHNICITIES
from optimization.data.closer_neighbors import CLOSER_NEIGHBORS_GRAPH_KEY
from optimization.data.contraction import contract_fixed
from optimization.data.edge_weights import BOUNDARY_WEIGHT_ATTR, boundary_cost
from optimization.data.feasibility import check_zoning
from optimization.solvers.base import get_solver
from optimization.solvers.cpsat import _assignment_objective
from optimization.strategies.pipeline import solve_step_problem
from optimization.tests.synthetic import make_grid_problem

ROWS = COLS = 6
SCHOOLS = {0: 100, 5: 300, 35: 200}  # zones 0, 1, 2 (sorted by node)
SUMMED = (
    "ge_students",
    "ge_capacity",
    "all_prog_students",
    "all_prog_capacity",
    "FRL",
    "num_schools",
    "land_area_m2",
    *AREA_ETHNICITIES,
)
REFERENCES = (
    "capacity_student_counts",
    "capacity_seat_counts",
    "frl_reference_student_counts",
    "frl_reference_seat_counts",
)


def _base_zoning(node):
    r, c = divmod(node, COLS)
    if c >= 3 and r <= 2:
        return 1
    return 0 if r + c < 6 else 2


# Zone 0 (centroid 0) holds its centroid in a fixed group; zone 2's group
# {21, 22, 27, 28} omits centroid 35, and 28 (nearest the school, supported
# only by free 29 and 34) is held out.
FIXED = {
    **{n: 0 for n in (0, 1, 2, 6, 7, 12, 13, 18)},
    **{n: 2 for n in (21, 22, 27, 28)},
    **{n: 1 for n in (5, 4, 11)},
}


def _problem(weight_edges=True, objective="cut_edges", seed=0, **overrides):
    rng = random.Random(seed)
    problem = make_grid_problem(
        ROWS,
        COLS,
        schools=SCHOOLS,
        frl_dev=0.25,
        racial_dev=-1,
        overage=0.6,
        shortage=0.4,
        min_zone_residents=4.0,
        weight_edges=weight_edges,
        objective=objective,
        **overrides,
    )
    G = problem.G
    for node in G:
        attrs = G.nodes[node]
        students = rng.randint(1, 5)
        attrs.update(
            ge_students=float(students),
            all_prog_students=float(students + rng.randint(0, 2)),
            ge_enrolling_students=0.8 * students,
            ge_capacity=float(rng.randint(0, 6)),
            all_prog_capacity=float(rng.randint(0, 8)),
            FRL=float(rng.randint(0, students)),
            land_area_m2=float(rng.randint(1, 9)),
        )
    for u, v in G.edges:
        G.edges[u, v][BOUNDARY_WEIGHT_ATTR] = rng.randint(1, 40)
    G.graph["weight_edges"] = weight_edges
    nodes = list(G)
    references = {
        name: {n: float(rng.randint(1, 6)) for n in nodes} for name in REFERENCES
    }
    problem = replace(problem, frl_reference_dev=0.5, **references)
    problem.fixed = dict(FIXED)
    problem.hint = {n: _base_zoning(n) for n in nodes}
    return problem


def _contract(problem):
    weighted = problem.weight_edges or problem.objective == "boundary_perimeter"
    contraction = contract_fixed(problem, weighted=weighted)
    assert contraction is not None
    return contraction


def _support_ok(problem, assignment):
    """The CP contiguity rows: each non-centroid node has a closer support."""
    relation = problem.G.graph[CLOSER_NEIGHBORS_GRAPH_KEY]
    for node, zone in assignment.items():
        if node == problem.centroids[zone]:
            continue
        school = problem.centroid_school_ids[zone]
        if not any(assignment.get(s) == zone for s in relation[node][school]):
            return False
    return True


def _random_assignments(reduced, count, seed):
    """Random zonings of the reduced problem, half near a feasible one."""
    rng = random.Random(seed)
    base = {
        node: next(iter(reduced.candidate_zones(node)))
        if len(reduced.candidate_zones(node)) == 1
        else reduced.hint[node]
        for node in reduced.nodes
    }
    free = [n for n in reduced.nodes if len(reduced.candidate_zones(n)) > 1]
    out = []
    for index in range(count):
        a = dict(base)
        flips = rng.randint(0, 3) if index % 2 else len(free)
        for node in rng.sample(free, min(flips, len(free))):
            a[node] = rng.choice(sorted(reduced.candidate_zones(node)))
        out.append(a)
    return out


def test_groups_hold_out_unsupported_nodes_and_move_the_centroid():
    problem = _problem()
    c = _contract(problem)
    reduced = c.reduced
    by_members = {tuple(nodes): node for node, nodes in c.members.items()}
    # Zone 0's group holds centroid 0 and becomes that zone's centroid node.
    zone0 = by_members[(0, 1, 2, 6, 7, 12, 13, 18)]
    assert reduced.centroids[0] == zone0
    assert reduced.centroid_school_ids == problem.centroid_school_ids
    # 28 is held out; {21, 22, 27} merge without zone 2's centroid 35.
    assert (28,) in by_members
    assert (21, 22, 27) in by_members
    assert reduced.centroids[2] == by_members[(35,)]
    assert reduced.centroids[1] == by_members[(4, 5, 11)]
    assert reduced.fixed[by_members[(28,)]] == 2
    assert reduced.fixed[by_members[(21, 22, 27)]] == 2
    assert c.held_out == 1
    assert c.nodes_after == problem.A - sum(len(m) - 1 for m in c.members.values())
    # Merged nodes carry every base area and school of their members.
    assert sorted(reduced.G.nodes[zone0]["block_ids"]) == sorted(
        1000 + n for n in (0, 1, 2, 6, 7, 12, 13, 18)
    )
    assert reduced.G.nodes[zone0]["school_ids"] == [100]


def test_sums_of_every_attribute_and_reference_count_are_preserved():
    problem = _problem()
    reduced = _contract(problem).reduced
    for name in SUMMED:
        original = sum(problem.G.nodes[n][name] for n in problem.G)
        contracted = sum(reduced.G.nodes[n][name] for n in reduced.G)
        assert contracted == pytest.approx(original), name
    for name in REFERENCES:
        assert sum(getattr(reduced, name).values()) == pytest.approx(
            sum(getattr(problem, name).values())
        )
    # Every edge weight leaving a group is kept; only internal edges vanish.
    internal = sum(
        problem.G.edges[u, v][BOUNDARY_WEIGHT_ATTR]
        for u, v in problem.G.edges
        if _contract(problem).node_map[u] == _contract(problem).node_map[v]
    )
    assert (
        sum(reduced.G.edges[u, v][BOUNDARY_WEIGHT_ATTR] for u, v in reduced.G.edges)
        == sum(problem.G.edges[u, v][BOUNDARY_WEIGHT_ATTR] for u, v in problem.G.edges)
        - internal
    )


def test_free_nodes_keep_candidates_and_distances():
    problem = _problem(max_distance=4.5)
    c = _contract(problem)
    for node, reduced_node in c.node_map.items():
        if len(c.members[reduced_node]) > 1:
            continue
        assert c.reduced.candidate_zones(reduced_node) == problem.candidate_zones(node)
        for zone in range(problem.Z):
            assert c.reduced.distance(
                c.reduced.centroids[zone], reduced_node
            ) == problem.distance(problem.centroids[zone], node)


@pytest.mark.parametrize("weight_edges", [True, False])
@pytest.mark.parametrize(
    "objective",
    ["cut_edges", "boundary_perimeter", "school_distance", "average_distance"],
)
def test_expansions_keep_feasibility_and_objective(weight_edges, objective):
    problem = _problem(weight_edges=weight_edges, objective=objective)
    if objective == "boundary_perimeter":
        problem = replace(problem, weight_edges=True)
    c = _contract(problem)
    reduced = c.reduced
    view = replace(
        problem, weight_edges=problem.weight_edges or objective == "boundary_perimeter"
    )
    offsets, feasible = set(), 0
    for a in _random_assignments(reduced, 400, seed=len(objective)):
        expanded = c.expand(a)
        assert set(expanded) == set(problem.nodes)
        ours = check_zoning(reduced, a)
        theirs = check_zoning(view, expanded)
        assert ours.feasible == theirs.feasible
        assert {v.kind for v in ours.violations} == {v.kind for v in theirs.violations}
        assert _support_ok(reduced, a) == _support_ok(view, expanded)
        feasible += ours.feasible
        # Boundary cost is identical; a distance objective moves by a constant.
        assert boundary_cost(reduced.G, a, weighted=True) == boundary_cost(
            view.G, expanded, weighted=view.weight_edges
        )
        offsets.add(
            round(
                _assignment_objective(view, expanded)
                - _assignment_objective(reduced, a),
                9,
            )
        )
    assert feasible > 10
    assert len(offsets) == 1
    if problem.model_cost == "cut_edges":
        assert offsets == {0.0}


@pytest.mark.parametrize(
    "objective,weight_edges",
    [("cut_edges", False), ("boundary_perimeter", False), ("school_distance", True)],
)
def test_cp_solve_reaches_the_same_optimum(objective, weight_edges):
    problem = _problem(weight_edges=weight_edges, objective=objective)
    options = get_solver("cp_bool", solve_time_limit=30, workers=2, seed=1).options

    def solve(contract):
        return solve_step_problem(
            replace(problem, fixed=dict(problem.fixed), hint=dict(problem.hint)),
            "cp_bool",
            objective,
            options,
            contract=contract,
        )

    on, off = solve(True), solve(False)
    assert on.status == off.status == "OPTIMAL"
    assert on.objective == pytest.approx(off.objective, abs=1e-6)
    assert set(on.assignment) == set(problem.nodes)
    assert check_zoning(on.problem, on.assignment).feasible
    assert (
        on.metadata["contracted_nodes_after"] < on.metadata["contracted_nodes_before"]
    )
    assert "contracted_nodes_before" not in off.metadata


def test_pipeline_refinement_contracts_by_default():
    from optimization.config import OptimizationConfig
    from optimization.tests.test_pipeline import MemoryDataset

    def run(**options):
        config = OptimizationConfig(
            hints="none",
            workers=1,
            frl_dev=1,
            racial_dev=1,
            overage=5,
            shortage=0,
            steps=[
                {"name": "loose", "level": "BlockGroup_0", "method": "cp_bool"},
                {
                    "level": "BlockGroup_0",
                    "method": "cp_bool",
                    "options": options,
                    "inputs": [
                        {
                            "source": "loose",
                            "mode": "assign",
                            "select": "interior",
                            "radius": 0,
                        },
                        {"source": "loose", "mode": "hint"},
                    ],
                },
            ],
        )
        dataset = MemoryDataset(config)
        return config.make_strategy().run(dataset, config.make_solver()), dataset

    results, dataset = run()
    refined = results[-1]
    assert refined.feasible
    assert set(refined.assignment) == set(dataset.base.G)
    assert refined.problem.G is dataset.base.G
    meta = refined.metadata
    assert meta["contracted_nodes_after"] < meta["contracted_nodes_before"] == 16
    assert "contraction_fallback" not in meta
    off, _ = run(contract_fixed=False)
    assert "contracted_nodes_before" not in off[-1].metadata
    assert off[-1].objective == pytest.approx(refined.objective)


def test_contract_fixed_default_keeps_task_ids():
    from benchmark.config import optimization_config_hash
    from optimization.config import OptimizationConfig

    def task_id(**kwargs):
        config = OptimizationConfig(method="cp_bool", objective="cut_edges", **kwargs)
        return optimization_config_hash(config, source_manifest={"sources": {}})

    assert task_id() == task_id(contract_fixed=True)
    assert task_id() != task_id(contract_fixed=False)
    with pytest.raises(ValueError):
        OptimizationConfig(contract_fixed="yes")
