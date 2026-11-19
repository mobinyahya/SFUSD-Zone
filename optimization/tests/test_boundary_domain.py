"""Boundary domains: nearby zones only, then exact closer-support pruning."""

from __future__ import annotations

import itertools
import random

import networkx as nx
import pytest

from optimization.data import contiguity
from optimization.data.boundary_domain import (
    prune_unsupported,
    restrict_boundary_domains,
    zones_within_hops,
)
from optimization.data.closer_neighbors import CLOSER_NEIGHBORS_GRAPH_KEY
from optimization.data.contraction import forced_zones
from optimization.solvers.base import get_solver
from optimization.strategies.pipeline import (
    release_unreachable_assignments,
    restrict_step_domains,
    solve_step_problem,
)
from optimization.tests.synthetic import make_grid_problem

ROWS = COLS = 6
# Four schools, one per corner: zones 0..3 in sorted node order.
CORNERS = {0: 100, 5: 200, 30: 300, 35: 400}


def _quadrant(node):
    r, c = divmod(node, COLS)
    return 2 * (r >= 3) + (c >= 3)


def _problem(**overrides):
    return make_grid_problem(ROWS, COLS, schools=CORNERS, **overrides)


def _band_split(problem, source):
    """Fix all but the cut-edge endpoints (band radius 0), as a refinement."""
    G = problem.G
    band = {n for u, v in G.edges if source[u] != source[v] for n in (u, v)}
    band -= set(problem.centroids)
    return {n: z for n, z in source.items() if n not in band}


def _supported(problem, assignment):
    """The CP contiguity rows: each non-centroid node has a closer support."""
    relation = problem.G.graph[CLOSER_NEIGHBORS_GRAPH_KEY]
    for node, zone in assignment.items():
        if node == problem.centroids[zone]:
            continue
        school = problem.centroid_school_ids[zone]
        if not any(assignment.get(u) == zone for u in relation[node][school]):
            return False
    return True


def _refinement(source, hops, prune=True):
    problem = _problem()
    problem.fixed = _band_split(problem, source)
    problem.hint = dict(source)
    before = {n: set(problem.candidate_zones(n)) for n in problem.G}
    meta = restrict_boundary_domains(problem, source, hops, prune=prune)
    return problem, before, meta


# --------------------------------------------------------------------------- #
# k-hop domains
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hops", [0, 1, 2, 3])
def test_zones_within_hops_matches_breadth_first_search(hops):
    G = _problem().G
    source = {n: _quadrant(n) for n in G if n != 14}  # one node without a zone
    got = zones_within_hops(G, source, list(G), hops)
    for node in G:
        ball = nx.single_source_shortest_path_length(G, node, cutoff=hops)
        assert got[node] == {source[u] for u in ball if u in source}


@pytest.mark.parametrize("hops", [0, 1, 2, 4])
def test_free_nodes_keep_their_zone_and_the_zones_within_k_hops(hops):
    source = {n: _quadrant(n) for n in range(ROWS * COLS)}
    problem, before, meta = _refinement(source, hops, prune=False)
    G = problem.G
    free = [n for n in G if n not in problem.fixed and n not in problem.centroids]
    assert meta["boundary_domain_free_nodes"] == len(free)
    for node in free:
        ball = nx.single_source_shortest_path_length(G, node, cutoff=hops)
        expected = before[node] & ({source[u] for u in ball} | {source[node]})
        assert problem.candidate_zones(node) == expected
    # Fixed nodes and centroids are untouched.
    for node in problem.fixed:
        assert problem.candidate_zones(node) == {source[node]}
    if hops == 1:
        # Node 8 (row 1, col 2) borders only zones 0 and 1.
        assert problem.candidate_zones(8) == {0, 1}
        assert meta["boundary_domain_pairs_after"] < meta["boundary_domain_pairs_before"]
    assert meta["boundary_domain_hint_outside"] == 0


def test_null_hops_leave_domains_unchanged():
    source = {n: _quadrant(n) for n in range(ROWS * COLS)}
    problem = _problem()
    problem.fixed = _band_split(problem, source)
    before = {n: set(problem.candidate_zones(n)) for n in problem.G}
    assert restrict_step_domains(problem, source, "cp_bool", None, {}, "shortest") == {}
    assert problem.candidates is None
    assert {n: problem.candidate_zones(n) for n in problem.G} == before
    # Methods without closer-support rows are left alone too.
    meta = restrict_step_domains(problem, source, "recom", 4, {}, "shortest")
    assert meta["boundary_domain_skipped"] == "method"
    assert problem.candidates is None


# --------------------------------------------------------------------------- #
# exact pruning
# --------------------------------------------------------------------------- #
def _all_supported(problem, domains):
    nodes = sorted(problem.G)
    for zones in itertools.product(*(sorted(domains[n]) for n in nodes)):
        assignment = dict(zip(nodes, zones))
        if _supported(problem, assignment):
            yield assignment


@pytest.mark.parametrize("seed", range(12))
def test_pruning_keeps_every_zoning_the_contiguity_rows_allow(seed):
    rng = random.Random(seed)
    problem = make_grid_problem(3, 3, schools={0: 100, 2: 200, 8: 300})
    centroid_zone = {c: z for z, c in enumerate(problem.centroids)}
    domains = {}
    for n in problem.G:
        if n in centroid_zone:
            domains[n] = {centroid_zone[n]}
        else:
            domains[n] = {z for z in range(3) if rng.random() < 0.7} or {0}
    protected = {(4, z) for z in domains[4]} if seed % 3 == 0 else set()
    original = {n: set(z) for n, z in domains.items()}
    feasible = list(_all_supported(problem, original))
    mutable = [n for n in problem.G if n not in centroid_zone]
    removed = prune_unsupported(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        domains,
        mutable,
        protected,
    )
    assert removed == sum(len(original[n]) - len(domains[n]) for n in problem.G)
    for assignment in feasible:
        assert all(assignment[n] in domains[n] for n in problem.G)
    for pair in protected:
        assert pair[1] in domains[pair[0]]
    # Fixed point: every surviving unprotected pair has a surviving support.
    relation = problem.G.graph[CLOSER_NEIGHBORS_GRAPH_KEY]
    for n in mutable:
        for z in domains[n]:
            if (n, z) not in protected:
                school = problem.centroid_school_ids[z]
                assert any(z in domains[u] for u in relation[n][school])


def _thin(problem):
    """Keep one closer neighbour per (node, school): the smallest id.

    Any subset of the strictly-closer relation is a valid relation, and on a
    plain grid the full one supports every pair, so nothing would be pruned.
    """
    relation = problem.G.graph[CLOSER_NEIGHBORS_GRAPH_KEY]
    for node, rows in relation.items():
        for school, closer in rows.items():
            rows[school] = frozenset(sorted(closer)[:1])
    return problem


def _hopped_and_pruned(hops, **overrides):
    """The quadrant refinement restricted to ``hops``, without and with pruning."""
    source = {n: _quadrant(n) for n in range(ROWS * COLS)}
    out = []
    for prune in (False, True):
        problem = _thin(
            _problem(overage=5.0, shortage=5.0, frl_dev=-1, racial_dev=-1, **overrides)
        )
        problem.fixed = _band_split(problem, source)
        problem.hint = dict(source)
        out.append((problem, restrict_boundary_domains(problem, source, hops, prune=prune)))
    return out


@pytest.mark.parametrize("hops", [2, 3, 12])
def test_pruning_keeps_every_cp_solution_of_a_refinement(hops):
    # Enumerate CP-SAT's zonings over the k-hop domains; every one survives
    # the pruning, which does remove pairs.
    (hopped, _), (pruned, meta) = _hopped_and_pruned(hops)
    assert meta["boundary_domain_pruned_pairs"] > 0
    solver = get_solver("cp_bool", workers=1, seed=0, solve_time_limit=60)
    solutions = solver.enumerate_solutions(hopped, 20000)
    assert 20 < len(solutions) < 20000
    for solution in solutions:
        for node, zone in solution.assignment.items():
            assert zone in pruned.candidate_zones(node)
    # Pruning only removes: each pruned domain is inside its k-hop domain.
    for node in hopped.G:
        assert pruned.candidate_zones(node) <= hopped.candidate_zones(node)


@pytest.mark.parametrize("objective", ["cut_edges", "average_distance"])
def test_exact_pruning_keeps_the_optimum(objective):
    # One worker without LNS proves these small optima at once.
    options = get_solver(
        "cp_bool",
        solve_time_limit=30,
        workers=1,
        seed=1,
        cp_sat_single_worker_lns=False,
    ).options
    (hopped, _), (pruned, meta) = _hopped_and_pruned(2, objective=objective)
    assert meta["boundary_domain_pruned_pairs"] > 0
    a = solve_step_problem(hopped, "cp_bool", objective, options)
    b = solve_step_problem(pruned, "cp_bool", objective, options)
    assert a.status == b.status == "OPTIMAL"
    assert b.objective == pytest.approx(a.objective, abs=1e-6)


# --------------------------------------------------------------------------- #
# relaxation, hints and contraction
# --------------------------------------------------------------------------- #
def _hooked_source():
    """Quadrants, but zone 1 takes 6, 7 and 8 (row 1, cols 0-2).

    Block 12 (row 2, col 0) stays in zone 0, but its only neighbour closer to
    school 100 (node 0) is 6, now zone 1: the pipeline's relaxation frees 12
    and opens zone 0 on 6, a free band block whose 0-hop domain is {1}.
    """
    source = {n: _quadrant(n) for n in range(ROWS * COLS)}
    source.update({6: 1, 7: 1, 8: 1})
    return source


def test_relaxation_zones_and_the_source_zone_survive_restriction():
    source = _hooked_source()
    problem = _problem()
    relation = problem.G.graph[CLOSER_NEIGHBORS_GRAPH_KEY]
    assert not any(source[u] == 0 for u in relation[12][100])  # unsupported
    hints = dict(source)
    fixed = _band_split(problem, source)
    fixed, widened, _ = release_unreachable_assignments(problem, fixed, hints)
    problem.fixed, problem.candidates = fixed, dict(widened) or None
    problem.hint = hints
    free = [n for n in problem.G if n not in fixed and n not in widened]
    free = [n for n in free if n not in problem.centroids]
    opened = contiguity.relax_unsupported_candidates(
        problem.G,
        source,
        problem.centroids,
        problem.centroid_school_ids,
        {n: set() for n in free},
        paths="shortest",
    )
    assert opened[6] == {0}  # the path crosses a free block
    meta = restrict_boundary_domains(problem, source, 0)
    for node, zones in widened.items():
        assert problem.candidate_zones(node) == zones
    for node in free:
        assert source[node] in problem.candidate_zones(node)
        assert set(opened.get(node, ())) <= problem.candidate_zones(node)
    assert problem.candidate_zones(6) == {0, 1}
    assert meta["boundary_domain_hint_outside"] == 0
    for node, zone in problem.hint.items():
        assert zone in problem.candidate_zones(node)


def test_single_zone_domains_contract_with_their_fixed_neighbours():
    source = {n: _quadrant(n) for n in range(ROWS * COLS)}
    problem = _problem(overage=5.0, shortage=5.0, frl_dev=-1, racial_dev=-1)
    problem.fixed = _band_split(problem, source)
    problem.hint = dict(source)
    meta = restrict_boundary_domains(problem, source, 0)
    single = {
        n: next(iter(z)) for n, z in problem.candidates.items() if len(z) == 1
    }
    assert meta["boundary_domain_single_zone_nodes"] == len(single) > 0
    forced = forced_zones(problem)
    assert single.items() <= forced.items()


# --------------------------------------------------------------------------- #
# pipeline and configuration
# --------------------------------------------------------------------------- #
def _pipeline_config(**options):
    from optimization.config import OptimizationConfig

    return OptimizationConfig(
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
                    {"source": "loose", "mode": "assign", "select": "interior"},
                    {"source": "loose", "mode": "hint"},
                ],
            },
        ],
    )


def _run(config):
    from optimization.tests.test_pipeline import MemoryDataset

    return config.make_strategy().run(MemoryDataset(config), config.make_solver())


def test_pipeline_restricts_refinement_domains_by_default():
    config = _pipeline_config()
    assert config.boundary_domain_hops == 4
    results = _run(config)
    first, refined = results
    assert "boundary_domain_hops" not in first.metadata  # no fixed nodes
    meta = refined.metadata
    assert refined.feasible
    assert meta["boundary_domain_hops"] == 4
    assert meta["boundary_domain_repair_radius"] == 2
    assert meta["boundary_domain_free_nodes"] > 0
    assert meta["boundary_domain_pairs_after"] <= meta["boundary_domain_pairs_before"]
    assert meta["boundary_domain_seconds"] >= 0
    # The saved stage rebuilds the restricted feasible set.
    assert meta["pipeline_candidates_areas"]
    off = _run(_pipeline_config(boundary_domain_hops=None))
    assert "boundary_domain_hops" not in off[-1].metadata
    assert not off[-1].metadata["pipeline_candidates_areas"]


def test_an_infeasible_restricted_step_is_solved_with_full_domains(monkeypatch):
    import optimization.strategies.pipeline as pipeline

    real = pipeline.solve_step_problem
    seen = []

    def solve(problem, *args, **kwargs):
        solution = real(problem, *args, **kwargs)
        if problem.fixed:
            seen.append(dict(problem.candidates or {}))
            if len(seen) == 1:
                solution.status = "INFEASIBLE"
        return solution

    monkeypatch.setattr(pipeline, "solve_step_problem", solve)
    results = _run(_pipeline_config(boundary_domain_hops=0))
    refined = results[-1]
    assert len(seen) == 2
    assert refined.feasible
    assert refined.metadata["boundary_domain_fallback"] is True
    assert "boundary_domain_restricted_wall_time" in refined.metadata
    # The retry gets the unrestricted candidates, and the stage saves them.
    assert seen[1] == {}
    assert not refined.metadata["pipeline_candidates_areas"]


def test_repair_radius_frees_every_zone_near_a_closer_rule_violation():
    from optimization.data.boundary_domain import closer_rule_violations

    source = _hooked_source()
    problem = _problem()
    assert closer_rule_violations(problem, source) == [12]
    problem.fixed = _band_split(problem, source)
    problem.hint = dict(source)
    full = {n: set(problem.candidate_zones(n)) for n in problem.G}
    meta = restrict_boundary_domains(problem, source, 0, prune=False, repair_radius=1)
    assert meta["boundary_domain_source_violations"] == 1
    near = {12, 6, 13, 18}
    free = [n for n in problem.G if n not in problem.fixed and n not in problem.centroids]
    assert meta["boundary_domain_repair_nodes"] == len(near & set(free))
    for node in free:
        if node in near:
            assert problem.candidate_zones(node) == full[node]
        else:
            assert len(problem.candidate_zones(node)) < len(full[node])


def test_saved_stage_rebuilds_the_restricted_domains():
    from optimization.tests.test_pipeline import MemoryDataset

    config = _pipeline_config(boundary_domain_hops=0)
    dataset = MemoryDataset(config)
    results = config.make_strategy().run(dataset, config.make_solver())
    refined = results[-1]
    rebuilt = config.make_strategy().problem_for_stage(dataset, refined.metadata)
    for node in rebuilt.G:
        assert rebuilt.candidate_zones(node) == refined.problem.candidate_zones(node)


def test_boundary_domain_hops_config_and_task_ids():
    from benchmark.config import optimization_config_hash
    from optimization.config import OptimizationConfig
    from optimization.pipeline import step_config

    def task_id(**kwargs):
        config = OptimizationConfig(method="cp_bool", objective="cut_edges", **kwargs)
        return optimization_config_hash(config, source_manifest={"sources": {}})

    assert task_id() == task_id(boundary_domain_hops=4)
    assert task_id() != task_id(boundary_domain_hops=None)
    assert task_id(boundary_domain_hops=2) != task_id()
    for bad in (-1, 1.5, True, "4"):
        with pytest.raises(ValueError):
            OptimizationConfig(boundary_domain_hops=bad)
    config = _pipeline_config(boundary_domain_hops=2)
    step = config.steps[1]
    assert step_config(config, step, 1).boundary_domain_hops == 2
    assert step_config(config, config.steps[0], 0).boundary_domain_hops == 4
