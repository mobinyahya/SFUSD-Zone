"""CP-SAT one-zone submodular-access pricing relaxation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral

import networkx as nx
from ortools.sat.python import cp_model

from optimization.branch_price.certificate import PricingMultipliers
from optimization.branch_price.geography import add_zone_pattern_constraints
from optimization.branch_price.patterns import (
    ZonePattern,
    validate_zone_pattern,
    zone_perimeter,
)
from optimization.problem import CutoffMarket, CutoffStudent, ZoneProblem
from optimization.welfare_oracle import (
    MAX_EXACT_CP_SAT_OBJECTIVE,
    solve_zoned_welfare,
    validate_welfare_market,
)


@dataclass(frozen=True, slots=True)
class AccessPricingResult:
    """One exact-valued candidate and a safe bound on exact zone pricing."""

    label: int
    status: str
    candidate: ZonePattern | None
    candidate_lagrangian_value: int | None
    candidate_access_lagrangian_value: int | None
    pricing_lagrangian_upper_bound: int
    analytic_lagrangian_upper_bound: int


@dataclass(frozen=True, slots=True)
class AccessPricingTemplate:
    """Label-independent access variables shared by all pricing models."""

    problem: ZoneProblem
    utility_scale: int
    model: cp_model.CpModel
    selected_indices: dict[int, int]
    best_indices: tuple[int, ...]
    rounded_utilities: tuple[dict[int, int], ...]


def analytic_access_pricing_bound(
    problem: ZoneProblem,
    label: int,
    *,
    utility_scale: int,
    multipliers: PricingMultipliers,
    centroid_neighbor_radius: int = 0,
) -> int:
    """Return a model-free node-separable bound on one label's pricing value."""
    market = problem.cutoff_market
    if market is None:
        raise ValueError("Access pricing requires a cutoff market.")
    rounded_utilities = tuple(
        {
            school: round(student.utilities[school] * utility_scale)
            for school in student.preferences
        }
        for student in market.students
    )
    return _analytic_pricing_bound(
        problem,
        label,
        rounded_utilities,
        multipliers,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )


def build_access_pricing_template(
    problem: ZoneProblem,
    *,
    utility_scale: int,
) -> AccessPricingTemplate:
    """Build the expensive label-independent access model once."""
    market = problem.cutoff_market
    if market is None:
        raise ValueError("Access pricing requires a cutoff market.")
    if set(market.school_capacities) != set(market.zone_restricted_schools):
        raise ValueError("Access pricing requires fully isolated zone markets.")
    if (
        isinstance(utility_scale, bool)
        or not isinstance(utility_scale, Integral)
        or utility_scale <= 0
    ):
        raise ValueError("utility_scale must be a positive integer.")
    validate_welfare_market(market, utility_scale=int(utility_scale))

    model = cp_model.CpModel()
    selected = {node: model.NewBoolVar(f"selected_{node}") for node in problem.nodes}
    best_indices = []
    rounded_utilities = []
    for student_index, student in enumerate(market.students):
        coefficients = {
            school: round(student.utilities[school] * utility_scale)
            for school in student.preferences
        }
        rounded_utilities.append(coefficients)
        upper = max(coefficients.values(), default=0)
        best = model.NewIntVar(0, upper, f"access_best_{student_index}")
        alternatives = [0]
        for rank, school in enumerate(student.preferences):
            applicant_and_school = model.NewBoolVar(
                f"access_{student_index}_{rank}_{school}"
            )
            applicant = selected[student.node]
            school_selected = selected[market.school_nodes[school]]
            model.Add(applicant_and_school <= applicant)
            model.Add(applicant_and_school <= school_selected)
            model.Add(applicant_and_school >= applicant + school_selected - 1)
            alternatives.append(coefficients[school] * applicant_and_school)
        model.AddMaxEquality(best, alternatives)
        best_indices.append(best.Index())
    return AccessPricingTemplate(
        problem=problem,
        utility_scale=int(utility_scale),
        model=model,
        selected_indices={
            node: variable.Index() for node, variable in selected.items()
        },
        best_indices=tuple(best_indices),
        rounded_utilities=tuple(rounded_utilities),
    )


def solve_access_pricing(
    problem: ZoneProblem,
    label: int,
    *,
    utility_scale: int,
    multipliers: PricingMultipliers | None = None,
    centroid_neighbor_radius: int = 0,
    time_limit: float = 60.0,
    workers: int = 1,
    random_seed: int = 0,
    template: AccessPricingTemplate | None = None,
) -> AccessPricingResult:
    """Maximize uncongested selected-school access for one legal labeled zone.

    The CP objective upper-bounds exact stable welfare. Any returned membership
    is separately valued by the least-cutoff oracle before becoming a pattern.
    """
    market = problem.cutoff_market
    if market is None:
        raise ValueError("Access pricing requires a cutoff market.")
    if label not in range(problem.Z):
        raise ValueError(f"Unknown zone label {label}.")
    if (
        isinstance(utility_scale, bool)
        or not isinstance(utility_scale, Integral)
        or utility_scale <= 0
    ):
        raise ValueError("utility_scale must be a positive integer.")
    if (
        isinstance(centroid_neighbor_radius, bool)
        or not isinstance(centroid_neighbor_radius, int)
        or centroid_neighbor_radius < 0
    ):
        raise ValueError("centroid_neighbor_radius must be a non-negative integer.")
    if time_limit < 0:
        raise ValueError("Access-pricing time_limit must be nonnegative.")
    multipliers = multipliers or PricingMultipliers(node={}, boundary=0)
    coverage_nodes = set(problem.nodes) - set(problem.centroids)
    unknown_multipliers = set(multipliers.node) - coverage_nodes
    if unknown_multipliers:
        raise ValueError(
            "Node multipliers must correspond to noncentroid coverage rows: "
            f"{sorted(unknown_multipliers)}."
        )
    if time_limit == 0:
        analytic_bound = analytic_access_pricing_bound(
            problem,
            label,
            utility_scale=int(utility_scale),
            multipliers=multipliers,
            centroid_neighbor_radius=centroid_neighbor_radius,
        )
        return AccessPricingResult(
            label=label,
            status="NOT_SOLVED",
            candidate=None,
            candidate_lagrangian_value=None,
            candidate_access_lagrangian_value=None,
            pricing_lagrangian_upper_bound=analytic_bound,
            analytic_lagrangian_upper_bound=analytic_bound,
        )
    if template is None:
        template = build_access_pricing_template(
            problem,
            utility_scale=int(utility_scale),
        )
    elif template.problem is not problem or template.utility_scale != utility_scale:
        raise ValueError(
            "Access pricing template does not match the problem and scale."
        )
    analytic_bound = _analytic_pricing_bound(
        problem,
        label,
        template.rounded_utilities,
        multipliers,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    model = template.model.clone()
    selected = {
        node: model.get_bool_var_from_proto_index(index)
        for node, index in template.selected_indices.items()
    }
    perimeter = add_zone_pattern_constraints(
        model,
        problem,
        label,
        selected,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )

    rounded_utilities = template.rounded_utilities
    access_terms = [
        market.lottery_scale * model.get_int_var_from_proto_index(index)
        for index in template.best_indices
    ]

    node_cost = sum(
        multiplier * selected[node] for node, multiplier in multipliers.node.items()
    )
    objective = sum(access_terms) - node_cost - multipliers.boundary * perimeter
    objective_range = (
        sum(
            market.lottery_scale * max(coefficients.values(), default=0)
            for coefficients in rounded_utilities
        )
        + sum(abs(value) for value in multipliers.node.values())
        + multipliers.boundary * problem.G.number_of_edges()
    )
    if objective_range > MAX_EXACT_CP_SAT_OBJECTIVE:
        raise ValueError("Access-pricing objective exceeds exact reporting range.")
    model.Maximize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(random_seed)
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    candidate = None
    candidate_lagrangian_value = None
    candidate_access_lagrangian_value = None
    pricing_upper_bound = analytic_bound
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        candidate_nodes = frozenset(
            node for node, variable in selected.items() if solver.Value(variable) == 1
        )
        exact_perimeter = zone_perimeter(problem.G, candidate_nodes)
        candidate = ZonePattern(
            label=label,
            nodes=candidate_nodes,
            raw_welfare=zone_raw_welfare(
                market,
                candidate_nodes,
                utility_scale=int(utility_scale),
            ),
            perimeter=exact_perimeter,
        )
        validate_zone_pattern(
            problem,
            candidate,
            centroid_neighbor_radius=centroid_neighbor_radius,
        )
        node_price = sum(multipliers.node.get(node, 0) for node in candidate.nodes)
        candidate_lagrangian_value = (
            candidate.raw_welfare
            - node_price
            - multipliers.boundary * candidate.perimeter
        )
        selected_schools = {
            school
            for school, school_node in market.school_nodes.items()
            if school_node in candidate.nodes
        }
        access_welfare = sum(
            market.lottery_scale
            * max(
                (
                    rounded_utilities[index][school]
                    for school in student.preferences
                    if school in selected_schools
                ),
                default=0,
            )
            for index, student in enumerate(market.students)
            if student.node in candidate.nodes
        )
        candidate_access_lagrangian_value = (
            access_welfare - node_price - multipliers.boundary * candidate.perimeter
        )
        reported_bound = solver.BestObjectiveBound()
        if math.isfinite(reported_bound):
            outward_bound = math.ceil(math.nextafter(reported_bound, math.inf))
            pricing_upper_bound = min(analytic_bound, outward_bound)
        pricing_upper_bound = max(
            pricing_upper_bound,
            candidate_access_lagrangian_value,
            candidate_lagrangian_value,
        )
    return AccessPricingResult(
        label=label,
        status=status_name,
        candidate=candidate,
        candidate_lagrangian_value=candidate_lagrangian_value,
        candidate_access_lagrangian_value=candidate_access_lagrangian_value,
        pricing_lagrangian_upper_bound=pricing_upper_bound,
        analytic_lagrangian_upper_bound=analytic_bound,
    )


def zone_raw_welfare(
    market: CutoffMarket,
    nodes: frozenset[int] | set[int],
    *,
    utility_scale: int,
) -> int:
    """Evaluate exact least-cutoff welfare for one selected isolated market."""
    selected_nodes = frozenset(nodes)
    selected_schools = {
        school
        for school, school_node in market.school_nodes.items()
        if school_node in selected_nodes
    }
    students = []
    for student in market.students:
        if student.node not in selected_nodes:
            continue
        preferences = tuple(
            school for school in student.preferences if school in selected_schools
        )
        students.append(
            CutoffStudent(
                studentno=student.studentno,
                node=student.node,
                preferences=preferences,
                priorities={
                    school: student.priorities[school] for school in preferences
                },
                utilities={school: student.utilities[school] for school in preferences},
            )
        )
    reduced_market = CutoffMarket(
        students=tuple(students),
        school_nodes={
            school: market.school_nodes[school] for school in selected_schools
        },
        school_capacities={
            school: market.school_capacities[school] for school in selected_schools
        },
        zone_restricted_schools=frozenset(selected_schools),
        lottery_scale=market.lottery_scale,
        outside_option_utility=market.outside_option_utility,
        metadata=market.metadata,
    )
    assignment = {node: 0 for node in selected_nodes}
    return solve_zoned_welfare(
        reduced_market,
        assignment,
        num_zones=1,
        utility_scale=utility_scale,
    ).raw_scaled_welfare


def _analytic_pricing_bound(
    problem: ZoneProblem,
    label: int,
    rounded_utilities: tuple[dict[int, int], ...],
    multipliers: PricingMultipliers,
    *,
    centroid_neighbor_radius: int,
) -> int:
    market = problem.cutoff_market
    access_by_node = defaultdict(int)
    for student, coefficients in zip(market.students, rounded_utilities):
        access_by_node[student.node] += market.lottery_scale * max(
            coefficients.values(), default=0
        )
    required = set(
        nx.single_source_shortest_path_length(
            problem.G,
            problem.centroids[label],
            cutoff=centroid_neighbor_radius,
        )
    )
    other_centroids = set(problem.centroids) - {problem.centroids[label]}
    bound = 0
    for node in problem.nodes:
        if node in other_centroids or label not in problem.candidate_zones(node):
            continue
        value = access_by_node[node] - multipliers.node.get(node, 0)
        bound += value if node in required else max(0, value)
    return bound
