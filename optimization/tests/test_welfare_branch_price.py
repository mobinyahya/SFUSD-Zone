"""Exhaustive tiny tests for the branch-price Stage 1-3 building blocks."""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import networkx as nx
import pytest
from ortools.sat.python import cp_model

from optimization.branch_price import (
    PricingMultipliers,
    RestrictedPatternMaster,
    ZonePattern,
    assemble_lagrangian_certificate,
    quantize_multipliers,
    solve_access_pricing,
    solve_exact_pricing,
    solve_pattern_root,
    validate_zone_pattern,
    zone_perimeter,
    zone_raw_welfare,
)
from optimization.data import contiguity
from optimization.data.closer_neighbors import CLOSER_NEIGHBORS_GRAPH_KEY
from optimization.problem import CutoffMarket, CutoffStudent
from optimization.solvers.balance import balance_constraints
from optimization.tests.synthetic import make_grid_problem
import optimization.branch_price.exact_pricing as exact_pricing_module


def _complete_patterns(graph, centroids):
    optional_nodes = sorted(set(graph) - set(centroids))
    patterns = []
    for label, centroid in enumerate(centroids):
        for bits in itertools.product((0, 1), repeat=len(optional_nodes)):
            nodes = {centroid}
            nodes.update(
                node for node, selected in zip(optional_nodes, bits) if selected
            )
            if not nx.is_connected(graph.subgraph(nodes)):
                continue
            raw_welfare = (
                100 + 7 * label + sum((label + 1) * (node + 2) for node in nodes)
            )
            patterns.append(
                ZonePattern.from_graph(
                    label=label,
                    nodes=nodes,
                    raw_welfare=raw_welfare,
                    graph=graph,
                )
            )
    return patterns


def _partitions(graph, centroids, patterns, max_cut_edges):
    labels = range(len(centroids))
    optional_nodes = sorted(set(graph) - set(centroids))
    by_key = {pattern.key: pattern for pattern in patterns}
    for choices in itertools.product(labels, repeat=len(optional_nodes)):
        assignment = dict(enumerate(labels)) if list(centroids) == list(labels) else {}
        assignment.update({centroid: label for label, centroid in enumerate(centroids)})
        assignment.update(dict(zip(optional_nodes, choices)))
        keys = tuple(
            (
                label,
                frozenset(
                    node for node, assigned in assignment.items() if assigned == label
                ),
            )
            for label in labels
        )
        if any(key not in by_key for key in keys):
            continue
        selected = tuple(by_key[key] for key in keys)
        if sum(pattern.perimeter for pattern in selected) <= 2 * max_cut_edges:
            yield assignment, selected


def test_complete_master_matches_every_tiny_partition_and_perimeter_identity():
    graph = nx.path_graph(5)
    centroids = [0, 4]
    patterns = _complete_patterns(graph, centroids)
    max_cut_edges = 2
    partitions = list(_partitions(graph, centroids, patterns, max_cut_edges))
    brute_objective = max(
        sum(pattern.raw_welfare for pattern in selected) for _, selected in partitions
    )
    for assignment, selected in partitions:
        cut_edges = sum(
            assignment[left] != assignment[right] for left, right in graph.edges
        )
        assert sum(pattern.perimeter for pattern in selected) == 2 * cut_edges

    master = RestrictedPatternMaster(
        graph,
        centroids,
        patterns,
        max_cut_edges=max_cut_edges,
    )
    mip = master.solve_mip(time_limit=10, workers=1)
    lp = master.solve_lp()

    assert mip.status == "OPTIMAL"
    assert mip.objective == brute_objective
    assert mip.assignment is not None
    assert set(mip.assignment) == set(graph)
    assert mip.perimeter == 2 * sum(
        mip.assignment[left] != mip.assignment[right] for left, right in graph.edges
    )
    assert lp.status == "OPTIMAL"
    assert lp.objective + 1e-6 >= mip.objective
    assert lp.duals is not None
    assert set(lp.duals.coverage) == set(graph) - set(centroids)
    assert not (set(lp.duals.coverage) & set(centroids))
    dual_objective = (
        sum(lp.duals.convexity.values())
        + sum(lp.duals.coverage.values())
        + 2 * max_cut_edges * lp.duals.boundary
    )
    assert dual_objective == pytest.approx(lp.objective, abs=1e-6)
    assert lp.duals.boundary >= -1e-8
    assert max(lp.duals.reduced_cost(pattern) for pattern in patterns) <= 1e-6


def test_six_convexity_rows_and_noncentroid_assignment_reconstruction():
    graph = nx.star_graph(6)
    centroids = list(range(1, 7))
    noncentroid = 0
    patterns = []
    for label, centroid in enumerate(centroids):
        for nodes, welfare in [({centroid}, 10), ({centroid, noncentroid}, 20 + label)]:
            patterns.append(
                ZonePattern.from_graph(
                    label=label,
                    nodes=nodes,
                    raw_welfare=welfare,
                    graph=graph,
                )
            )
    master = RestrictedPatternMaster(
        graph,
        centroids,
        patterns,
        max_cut_edges=5,
    )

    result = master.solve_mip(time_limit=10, workers=1)
    lp = master.solve_lp()

    assert master.convexity_row_count == 6
    assert master.coverage_row_count == 1
    assert result.status == "OPTIMAL"
    assert lp.status == "OPTIMAL"
    assert lp.duals is not None
    assert len(lp.duals.convexity) == 6
    assert set(lp.duals.coverage) == {noncentroid}
    assert result.assignment == {
        **{centroid: label for label, centroid in enumerate(centroids)},
        0: 5,
    }
    assert result.perimeter == 10


def test_binding_perimeter_row_has_positive_dual_and_correct_scaling():
    graph = nx.grid_2d_graph(3, 3)
    graph = nx.convert_node_labels_to_integers(graph)
    centroids = [0, 8]
    optional_nodes = sorted(set(graph) - set(centroids))
    patterns = []
    for label, centroid in enumerate(centroids):
        for bits in itertools.product((0, 1), repeat=len(optional_nodes)):
            nodes = {centroid}
            nodes.update(
                node for node, selected in zip(optional_nodes, bits) if selected
            )
            if not nx.is_connected(graph.subgraph(nodes)):
                continue
            perimeter = zone_perimeter(graph, nodes)
            patterns.append(
                ZonePattern(
                    label=label,
                    nodes=frozenset(nodes),
                    raw_welfare=100 * perimeter,
                    perimeter=perimeter,
                )
            )

    master = RestrictedPatternMaster(
        graph,
        centroids,
        patterns,
        max_cut_edges=3,
    )
    lp = master.solve_lp()

    assert lp.status == "OPTIMAL"
    assert lp.duals is not None
    assert lp.perimeter == pytest.approx(6.0)
    assert lp.duals.boundary > 0
    assert lp.objective == pytest.approx(
        sum(lp.duals.convexity.values())
        + sum(lp.duals.coverage.values())
        + 2 * 3 * lp.duals.boundary
    )


def test_integer_lagrangian_certificate_bounds_all_enumerated_partitions():
    graph = nx.path_graph(5)
    centroids = [0, 4]
    patterns = _complete_patterns(graph, centroids)
    coverage_nodes = sorted(set(graph) - set(centroids))
    max_cut_edges = 2
    zone_perimeter_cap = 2 * max_cut_edges
    partitions = list(_partitions(graph, centroids, patterns, max_cut_edges))
    quantized = quantize_multipliers({1: -12.6, 2: 3.4}, 4.6)
    assert quantized == PricingMultipliers(node={1: -13, 2: 3}, boundary=5)

    for node_values in itertools.product((-13, 0, 17), repeat=len(coverage_nodes)):
        for boundary in (0, 5):
            multipliers = PricingMultipliers(
                node=dict(zip(coverage_nodes, node_values)),
                boundary=boundary,
            )
            pricing_bounds = {
                label: max(
                    pattern.raw_welfare
                    - sum(multipliers.node.get(node, 0) for node in pattern.nodes)
                    - boundary * pattern.perimeter
                    for pattern in patterns
                    if pattern.label == label
                )
                for label in range(len(centroids))
            }
            certificate = assemble_lagrangian_certificate(
                labels=range(len(centroids)),
                coverage_nodes=coverage_nodes,
                zone_perimeter_cap=zone_perimeter_cap,
                multipliers=multipliers,
                pricing_upper_bounds=pricing_bounds,
            )
            for _, selected in partitions:
                assert (
                    sum(pattern.raw_welfare for pattern in selected)
                    <= certificate.upper_bound
                )


def _welfare_student(studentno, node, preferences, utilities, priorities=None):
    return CutoffStudent(
        studentno=studentno,
        node=node,
        preferences=preferences,
        priorities=priorities or {school: 0 for school in preferences},
        utilities=utilities,
    )


def _legal_access_patterns(problem, label):
    centroid = problem.centroids[label]
    supports = contiguity.contiguity_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    total_schools = sum(problem.num_schools(node) for node in problem.nodes)
    average = total_schools / problem.Z
    lower_schools = round(100 * max(0.0, average - 1.0))
    upper_schools = round(100 * (average + 1.0))
    boundary_limit = math.floor(problem.boundary_prop * problem.G.number_of_edges())
    legal = []
    for bits in itertools.product((0, 1), repeat=problem.A):
        nodes = frozenset(node for node, selected in enumerate(bits) if selected)
        if centroid not in nodes:
            continue
        if any(
            other_centroid in nodes
            for other_label, other_centroid in enumerate(problem.centroids)
            if other_label != label
        ):
            continue
        if any(label not in problem.candidate_zones(node) for node in nodes):
            continue
        if any(
            node != centroid
            and not any(support in nodes for support in supports[(node, label)])
            for node in nodes
        ):
            continue
        demographic_feasible = True
        for constraint in balance_constraints(problem):
            if constraint.kind == "capacity":
                continue
            if constraint.lower_ratio is not None:
                lower = sum(
                    round(
                        100
                        * (
                            constraint.value(node)
                            - constraint.lower_ratio * problem.students(node)
                        )
                    )
                    for node in nodes
                )
                demographic_feasible &= lower >= 0
            if constraint.upper_ratio is not None:
                upper = sum(
                    round(
                        100
                        * (
                            constraint.value(node)
                            - constraint.upper_ratio * problem.students(node)
                        )
                    )
                    for node in nodes
                )
                demographic_feasible &= upper <= 0
        if not demographic_feasible:
            continue
        school_count = 100 * sum(problem.num_schools(node) for node in nodes)
        if not lower_schools <= school_count <= upper_schools:
            continue
        if zone_perimeter(problem.G, nodes) > boundary_limit:
            continue
        legal.append(nodes)
    return legal


def test_access_pricing_bounds_every_exact_tiny_pattern_and_reduced_cost():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=0,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        boundary_prop=1.0,
    )
    problem.G.nodes[1]["FRL"] = 1.0
    problem.G.nodes[2]["FRL"] = 0.0
    problem.cutoff_market = CutoffMarket(
        students=(
            _welfare_student(
                1,
                1,
                (100, 200),
                {100: 5.0, 200: 2.0},
                {100: 0, 200: 1},
            ),
            _welfare_student(
                2,
                2,
                (100, 200),
                {100: 4.0, 200: 3.0},
            ),
            _welfare_student(3, 3, (200,), {200: 6.0}),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=4,
    )
    utility_scale = 10
    multipliers = PricingMultipliers(node={1: -25, 2: 30}, boundary=7)

    for label in range(problem.Z):
        legal_nodes = _legal_access_patterns(problem, label)
        exact_reduced_costs = []
        for nodes in legal_nodes:
            raw_welfare = zone_raw_welfare(
                problem.cutoff_market,
                nodes,
                utility_scale=utility_scale,
            )
            selected_schools = {
                school
                for school, node in problem.cutoff_market.school_nodes.items()
                if node in nodes
            }
            access_welfare = sum(
                problem.cutoff_market.lottery_scale
                * max(
                    (
                        round(student.utilities[school] * utility_scale)
                        for school in student.preferences
                        if school in selected_schools
                    ),
                    default=0,
                )
                for student in problem.cutoff_market.students
                if student.node in nodes
            )
            node_price = sum(multipliers.node.get(node, 0) for node in nodes)
            perimeter_price = multipliers.boundary * zone_perimeter(problem.G, nodes)
            assert access_welfare >= raw_welfare
            assert access_welfare - node_price - perimeter_price >= (
                raw_welfare - node_price - perimeter_price
            )
            exact_reduced_costs.append(raw_welfare - node_price - perimeter_price)

        result = solve_access_pricing(
            problem,
            label,
            utility_scale=utility_scale,
            multipliers=multipliers,
            time_limit=10,
            workers=1,
            random_seed=42,
        )

        assert result.status == "OPTIMAL"
        assert result.candidate is not None
        assert result.candidate.nodes in legal_nodes
        assert result.candidate.raw_welfare == zone_raw_welfare(
            problem.cutoff_market,
            result.candidate.nodes,
            utility_scale=utility_scale,
        )
        assert (
            result.candidate_access_lagrangian_value
            >= result.candidate_lagrangian_value
        )
        assert result.pricing_lagrangian_upper_bound >= max(exact_reduced_costs)

        fallback = solve_access_pricing(
            problem,
            label,
            utility_scale=utility_scale,
            multipliers=multipliers,
            time_limit=0,
            workers=1,
            random_seed=42,
        )
        assert fallback.pricing_lagrangian_upper_bound >= max(exact_reduced_costs)


def test_exact_pricing_matches_every_tiny_legal_zone():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        boundary_prop=1.0,
    )
    problem.cutoff_market = CutoffMarket(
        students=(
            _welfare_student(
                1,
                1,
                (100, 200),
                {100: 5.0, 200: 2.0},
                {100: 0, 200: 1},
            ),
            _welfare_student(2, 2, (200, 100), {200: 4.0, 100: 1.0}),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=4,
    )
    utility_scale = 10
    multipliers = PricingMultipliers(node={1: -25, 2: 30}, boundary=7)

    for label in range(problem.Z):
        exact_values = {
            nodes: zone_raw_welfare(
                problem.cutoff_market,
                nodes,
                utility_scale=utility_scale,
            )
            - sum(multipliers.node.get(node, 0) for node in nodes)
            - multipliers.boundary * zone_perimeter(problem.G, nodes)
            for nodes in _legal_access_patterns(problem, label)
        }
        optimum = max(exact_values.values())

        result = solve_exact_pricing(
            problem,
            label,
            utility_scale=utility_scale,
            multipliers=multipliers,
            time_limit=10,
            workers=1,
            random_seed=42,
        )

        assert result.status == "OPTIMAL"
        assert result.candidate is not None
        assert result.candidate_lagrangian_value == optimum
        assert result.pricing_lagrangian_upper_bound == optimum


def test_exact_pricing_unknown_retains_analytic_bound(monkeypatch):
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        boundary_prop=1.0,
    )
    problem.cutoff_market = CutoffMarket(
        students=(_welfare_student(1, 1, (100,), {100: 5.0}),),
        school_nodes={100: 0},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )
    seed_nodes = _legal_access_patterns(problem, 0)[0]
    seed = ZonePattern.from_graph(
        label=0,
        nodes=seed_nodes,
        raw_welfare=zone_raw_welfare(
            problem.cutoff_market,
            seed_nodes,
            utility_scale=10,
        ),
        graph=problem.G,
    )

    class UnknownSolver:
        def __init__(self):
            self.parameters = SimpleNamespace()

        def Solve(self, _model):
            return cp_model.UNKNOWN

        def StatusName(self, _status):
            return "UNKNOWN"

        def BestObjectiveBound(self):
            raise AssertionError("UNKNOWN bounds must not be consumed")

    monkeypatch.setattr(exact_pricing_module.cp_model, "CpSolver", UnknownSolver)
    result = solve_exact_pricing(
        problem,
        0,
        utility_scale=10,
        time_limit=10,
        workers=1,
        seed_pattern=seed,
    )

    assert result.status == "UNKNOWN"
    assert result.candidate == seed
    assert (
        result.pricing_lagrangian_upper_bound == result.analytic_lagrangian_upper_bound
    )


def test_pattern_root_certificate_bounds_complete_tiny_master():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        boundary_prop=1.0,
    )
    problem.cutoff_market = CutoffMarket(
        students=(
            _welfare_student(1, 1, (100, 200), {100: 5.0, 200: 2.0}),
            _welfare_student(2, 2, (200, 100), {200: 4.0, 100: 1.0}),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=4,
    )
    utility_scale = 10
    patterns = []
    for label in range(problem.Z):
        for nodes in _legal_access_patterns(problem, label):
            patterns.append(
                ZonePattern.from_graph(
                    label=label,
                    nodes=nodes,
                    raw_welfare=zone_raw_welfare(
                        problem.cutoff_market,
                        nodes,
                        utility_scale=utility_scale,
                    ),
                    graph=problem.G,
                )
            )
    max_cut_edges = problem.G.number_of_edges()
    partitions = list(
        _partitions(problem.G, problem.centroids, patterns, max_cut_edges)
    )
    exact_optimum = max(
        sum(pattern.raw_welfare for pattern in selected) for _, selected in partitions
    )

    result = solve_pattern_root(
        problem,
        patterns,
        utility_scale=utility_scale,
        pricing_time_limit=10,
        mip_time_limit=10,
        workers=1,
        random_seed=42,
    )

    assert result.initial_lp.status == "OPTIMAL"
    assert result.enriched_lp.status == "OPTIMAL"
    assert result.restricted_mip.status == "OPTIMAL"
    assert result.restricted_mip.objective == exact_optimum
    assert result.certificate.upper_bound >= exact_optimum
    multipliers = result.certificate.multipliers
    assert all(
        pricing.pricing_lagrangian_upper_bound
        >= max(
            pattern.raw_welfare
            - sum(multipliers.node.get(node, 0) for node in pattern.nodes)
            - multipliers.boundary * pattern.perimeter
            for pattern in patterns
            if pattern.label == pricing.label
        )
        for pricing in result.pricing
    )


def test_master_rejects_disconnected_patterns():
    graph = nx.path_graph(4)
    patterns = [
        ZonePattern.from_graph(
            label=0,
            nodes={0, 2},
            raw_welfare=1,
            graph=graph,
        ),
        ZonePattern.from_graph(
            label=1,
            nodes={3},
            raw_welfare=1,
            graph=graph,
        ),
    ]

    with pytest.raises(ValueError, match="connected"):
        RestrictedPatternMaster(
            graph,
            [0, 3],
            patterns,
            max_cut_edges=graph.number_of_edges(),
        )


def test_problem_validator_rejects_support_invalid_pattern():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        boundary_prop=1.0,
    )
    centroid = problem.centroids[0]
    other_centroids = set(problem.centroids[1:])
    unsupported = next(
        node
        for node in problem.nodes
        if node != centroid
        and node not in other_centroids
        and problem.G.has_edge(centroid, node)
    )
    school_id = problem.centroid_school_ids[0]
    problem.G.graph[CLOSER_NEIGHBORS_GRAPH_KEY][unsupported][school_id] = []
    pattern = ZonePattern.from_graph(
        label=0,
        nodes={centroid, unsupported},
        raw_welfare=1,
        graph=problem.G,
    )

    with pytest.raises(ValueError, match="contiguity supports"):
        validate_zone_pattern(problem, pattern)
