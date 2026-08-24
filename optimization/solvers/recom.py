"""ReCom-family stochastic zoning solvers.

The three solvers in this module share one proposal kernel.  A proposal merges
two adjacent zones, samples a uniform spanning tree with Wilson's algorithm,
and cuts one tree edge.  Zone statistics and constraint residuals are additive,
so every possible cut can be evaluated without constructing each candidate
subgraph.

Centroids determine the number of zones and may seed a Voronoi hint, but they
and distance-derived candidates are intentionally not hard constraints here.
Explicit candidate and fixed assignments remain hard constraints.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Mapping

from optimization.data import contiguity
from optimization.data.initial_solutions import initial_solution, normalize_hints
from optimization.problem import ZoneProblem
from optimization.solution import ZoneSolution
from optimization.solvers.balance import (
    BalanceConstraint,
    balance_constraints,
)
from optimization.solvers.base import Solver, register

_EPS = 1e-6
_WEIGHT_EPS = 1e-12
_RELAXED_WEIGHTS = {
    "trees": 1,
    "nodes": 1,
    "frl": 3,
    "students": 1,
    "seats": 1,
    "shortage%": 10,
    "sch_count": 45,
}


class _HintError(ValueError):
    """A supplied/generated hint cannot initialize a ReCom chain."""


class _NoProposal(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class _DeadlineReached(RuntimeError):
    pass


@dataclass(frozen=True)
class _ZoneStats:
    node_count: int
    students: float
    values: tuple[float, ...]
    schools: float
    internal_edges: int
    seat_value: float | None = None
    frl_value: float | None = None

    @property
    def seats(self) -> float:
        if self.seat_value is not None:
            return self.seat_value
        return self.values[0] if self.values else 0.0

    @property
    def frl(self) -> float:
        if self.frl_value is not None:
            return self.frl_value
        return self.values[1] if len(self.values) > 1 else 0.0

    @property
    def cycle_rank(self) -> int:
        return max(0, self.internal_edges - self.node_count + 1)


@dataclass(frozen=True)
class _BalanceRow:
    constraint: BalanceConstraint
    sense: str
    ratio: float


@dataclass
class _State:
    assignment: list[int]
    zone_nodes: list[set[int]]
    zone_stats: list[_ZoneStats]
    zone_violations: list[tuple[float, ...]]
    violations: tuple[float, ...]
    boundary_pairs: dict[tuple[int, int], int]
    boundary_costs: dict[tuple[int, int], int]
    boundary_cost: int

    @property
    def feasible(self) -> bool:
        return all(value <= _EPS for value in self.violations)

    def clone(self) -> "_State":
        return _State(
            assignment=list(self.assignment),
            zone_nodes=[set(nodes) for nodes in self.zone_nodes],
            zone_stats=list(self.zone_stats),
            zone_violations=list(self.zone_violations),
            violations=tuple(self.violations),
            boundary_pairs=dict(self.boundary_pairs),
            boundary_costs=dict(self.boundary_costs),
            boundary_cost=self.boundary_cost,
        )


@dataclass(frozen=True)
class _CutCandidate:
    tin: int
    size: int
    subtree_to_a: bool
    stats_a: _ZoneStats
    stats_b: _ZoneStats
    violations_a: tuple[float, ...]
    violations_b: tuple[float, ...]
    global_violations: tuple[float, ...]
    boundary_cost: int

    @property
    def pair_feasible(self) -> bool:
        return all(value <= _EPS for value in self.violations_a) and all(
            value <= _EPS for value in self.violations_b
        )

    @property
    def globally_feasible(self) -> bool:
        return all(value <= _EPS for value in self.global_violations)


@dataclass(frozen=True)
class _Move:
    zone_a: int
    zone_b: int
    subtree: tuple[int, ...]
    subtree_to_a: bool
    stats_a: _ZoneStats
    stats_b: _ZoneStats
    violations_a: tuple[float, ...]
    violations_b: tuple[float, ...]
    global_violations: tuple[float, ...]
    boundary_cost: int

    @property
    def globally_feasible(self) -> bool:
        return all(value <= _EPS for value in self.global_violations)


@dataclass(frozen=True)
class _Snapshot:
    assignment: tuple[int, ...]
    violations: tuple[float, ...]
    boundary_cost: int

    @property
    def feasible(self) -> bool:
        return all(value <= _EPS for value in self.violations)


@dataclass(frozen=True)
class _Setup:
    context: "_ReComContext"
    state: _State
    rng: random.Random
    max_iterations: int | None
    deadline: float | None
    hint_metadata: dict[str, object]


class _DynamicMaxNormalizer:
    """Running max linearization for short-burst constraint penalties."""

    def __init__(self, size: int) -> None:
        self.maxima = [0.0] * size

    def observe(self, violations: tuple[float, ...]) -> None:
        for idx, value in enumerate(violations):
            if value > self.maxima[idx]:
                self.maxima[idx] = value

    def penalty(self, violations: tuple[float, ...]) -> float:
        return sum(
            value / maximum
            for value, maximum in zip(violations, self.maxima)
            if maximum > _EPS
        )


class _ReComContext:
    """Problem data converted to array-backed structures for hot loops."""

    def __init__(self, problem: ZoneProblem) -> None:
        self.problem = problem
        self.nodes = tuple(problem.nodes)
        self.node_to_pos = {node: idx for idx, node in enumerate(self.nodes)}
        self.zone_count = problem.Z
        if self.zone_count <= 0:
            raise ValueError("ReCom requires at least one zone.")

        self.adjacency: list[tuple[int, ...]] = []
        for node in self.nodes:
            self.adjacency.append(
                tuple(
                    self.node_to_pos[neighbor] for neighbor in problem.G.neighbors(node)
                )
            )
        graph_edges = tuple(problem.G.edges())
        self.edges = tuple(
            (self.node_to_pos[u], self.node_to_pos[v]) for u, v in graph_edges
        )
        self.edge_weights = tuple(
            problem.boundary_weight(u, v) for u, v in graph_edges
        )
        incident: list[list[int]] = [[] for _ in self.nodes]
        for edge_id, (u, v) in enumerate(self.edges):
            incident[u].append(edge_id)
            incident[v].append(edge_id)
        self.incident_edges = tuple(tuple(edge_ids) for edge_ids in incident)

        self.constraints = tuple(balance_constraints(problem))
        self.balance_rows = tuple(
            _BalanceRow(
                constraint,
                sense,
                ratio,
            )
            for constraint in self.constraints
            for sense, ratio in (
                ("lower", constraint.lower_ratio),
                ("upper", constraint.upper_ratio),
            )
            if ratio is not None
        )
        self.students = tuple(problem.students(node) for node in self.nodes)
        self.seats = tuple(problem.capacity(node) for node in self.nodes)
        self.frl = tuple(problem.frl(node) for node in self.nodes)
        self.values = tuple(
            tuple(row.constraint.value(node) for row in self.balance_rows)
            for node in self.nodes
        )
        self.schools = tuple(float(problem.num_schools(node)) for node in self.nodes)

        total_schools = sum(self.schools)
        if total_schools > 0:
            average = total_schools / self.zone_count
            self.school_bounds = (max(0.0, average - 1.0), average + 1.0)
        else:
            self.school_bounds = None

        all_zones = frozenset(range(self.zone_count))
        allowed: list[frozenset[int]] = []
        for node in self.nodes:
            if problem.candidates is not None and node in problem.candidates:
                zones = frozenset(
                    int(zone)
                    for zone in problem.candidates[node]
                    if 0 <= int(zone) < self.zone_count
                )
            elif problem.fixed is not None and node in problem.fixed:
                zone = int(problem.fixed[node])
                zones = (
                    frozenset({zone}) if 0 <= zone < self.zone_count else frozenset()
                )
            else:
                zones = all_zones
            if not zones:
                raise ValueError(f"Node {node} has no explicit ReCom candidate zones.")
            allowed.append(zones)
        self.allowed = tuple(allowed)

    @property
    def violation_count(self) -> int:
        return len(self.balance_rows) + (2 if self.school_bounds else 0)

    def zone_violations(self, stats: _ZoneStats) -> tuple[float, ...]:
        violations: list[float] = []
        for value, row in zip(stats.values, self.balance_rows, strict=True):
            if row.sense == "lower":
                violations.append(max(0.0, row.ratio * stats.students - value))
            else:
                violations.append(max(0.0, value - row.ratio * stats.students))
        if self.school_bounds is not None:
            lower, upper = self.school_bounds
            violations.append(max(0.0, lower - stats.schools))
            violations.append(max(0.0, stats.schools - upper))
        return tuple(violations)

    def build_state(self, assignment: list[int]) -> _State:
        zone_nodes = [set() for _ in range(self.zone_count)]
        node_counts = [0] * self.zone_count
        students = [0.0] * self.zone_count
        schools = [0.0] * self.zone_count
        seats = [0.0] * self.zone_count
        frl = [0.0] * self.zone_count
        values = [[0.0] * len(self.balance_rows) for _ in range(self.zone_count)]
        internal_edges = [0] * self.zone_count

        for pos, zone in enumerate(assignment):
            zone_nodes[zone].add(pos)
            node_counts[zone] += 1
            students[zone] += self.students[pos]
            schools[zone] += self.schools[pos]
            seats[zone] += self.seats[pos]
            frl[zone] += self.frl[pos]
            for idx, value in enumerate(self.values[pos]):
                values[zone][idx] += value

        boundary_pairs: dict[tuple[int, int], int] = {}
        boundary_costs: dict[tuple[int, int], int] = {}
        boundary_cost = 0
        for edge_id, (u, v) in enumerate(self.edges):
            zone_u = assignment[u]
            zone_v = assignment[v]
            if zone_u == zone_v:
                internal_edges[zone_u] += 1
                continue
            weight = self.edge_weights[edge_id]
            boundary_cost += weight
            pair = _zone_pair(zone_u, zone_v)
            boundary_pairs[pair] = boundary_pairs.get(pair, 0) + 1
            boundary_costs[pair] = boundary_costs.get(pair, 0) + weight

        zone_stats = [
            _ZoneStats(
                node_count=node_counts[zone],
                students=students[zone],
                values=tuple(values[zone]),
                schools=schools[zone],
                internal_edges=internal_edges[zone],
                seat_value=seats[zone],
                frl_value=frl[zone],
            )
            for zone in range(self.zone_count)
        ]
        zone_violations = [self.zone_violations(stats) for stats in zone_stats]
        violations = tuple(
            sum(zone_values[idx] for zone_values in zone_violations)
            for idx in range(self.violation_count)
        )
        return _State(
            assignment=assignment,
            zone_nodes=zone_nodes,
            zone_stats=zone_stats,
            zone_violations=zone_violations,
            violations=violations,
            boundary_pairs=boundary_pairs,
            boundary_costs=boundary_costs,
            boundary_cost=boundary_cost,
        )

    def assignment_dict(
        self, assignment: tuple[int, ...] | list[int]
    ) -> dict[int, int]:
        return {node: int(assignment[pos]) for pos, node in enumerate(self.nodes)}

    def validate_hint(self, hint: Mapping[int, int]) -> list[int]:
        if set(hint) != set(self.nodes):
            missing = len(set(self.nodes) - set(hint))
            extra = len(set(hint) - set(self.nodes))
            raise _HintError(
                f"ReCom hint must assign every problem node (missing={missing}, extra={extra})."
            )

        assignment: list[int] = []
        for pos, node in enumerate(self.nodes):
            raw_zone = hint[node]
            if not isinstance(raw_zone, int) or isinstance(raw_zone, bool):
                raise _HintError(
                    f"ReCom hint assigns non-integer zone {raw_zone!r} to node {node}."
                )
            zone = int(raw_zone)
            if not 0 <= zone < self.zone_count:
                raise _HintError(
                    f"ReCom hint assigns invalid zone {zone} to node {node}."
                )
            if zone not in self.allowed[pos]:
                raise _HintError(
                    f"ReCom hint assigns node {node} to explicitly forbidden zone {zone}."
                )
            assignment.append(zone)

        represented = set(assignment)
        expected = set(range(self.zone_count))
        if represented != expected:
            raise _HintError(
                f"ReCom hint must represent every zone; got {sorted(represented)}, expected {sorted(expected)}."
            )
        for zone in range(self.zone_count):
            zone_nodes = {
                idx for idx, assigned in enumerate(assignment) if assigned == zone
            }
            if not self._connected(zone_nodes):
                raise _HintError(f"ReCom hint zone {zone} is not contiguous.")
        return assignment

    def _connected(self, nodes: set[int]) -> bool:
        if not nodes:
            return False
        start = next(iter(nodes))
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in self.adjacency[node]:
                if neighbor in nodes and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return len(seen) == len(nodes)


class _ReComKernel:
    def __init__(
        self,
        context: _ReComContext,
        rng: random.Random,
        deadline: float | None,
    ) -> None:
        self.context = context
        self.rng = rng
        self.deadline = deadline
        self._deadline_checks = 0

    def propose(self, state: _State, selector: str) -> _Move:
        adjacent_pairs = sorted(
            pair for pair, count in state.boundary_pairs.items() if count > 0
        )
        if not adjacent_pairs:
            raise _NoProposal("no_adjacent_zone_pairs")
        zone_a, zone_b = self.rng.choice(adjacent_pairs)
        union = state.zone_nodes[zone_a] | state.zone_nodes[zone_b]
        pair_adjacency, pair_edges = self._pair_graph(union)
        tree = self._random_spanning_tree(sorted(union), pair_adjacency)
        preorder, parent, depth = self._root_tree(tree)
        candidates = self._cut_candidates(
            state,
            zone_a,
            zone_b,
            preorder,
            parent,
            depth,
            pair_adjacency,
            pair_edges,
        )
        if not candidates:
            raise _NoProposal("no_explicitly_legal_cut")

        feasible = [candidate for candidate in candidates if candidate.pair_feasible]
        pool = feasible or candidates
        if selector == "uniform":
            selected = self.rng.choice(pool)
        elif selector == "relaxed":
            probabilities = self._relaxed_probabilities(pool)
            selected = self.rng.choices(pool, weights=probabilities, k=1)[0]
        else:  # pragma: no cover - guarded by config/class callers
            raise ValueError(f"Unknown ReCom cut selector {selector!r}.")

        subtree = tuple(preorder[selected.tin : selected.tin + selected.size])
        return _Move(
            zone_a=zone_a,
            zone_b=zone_b,
            subtree=subtree,
            subtree_to_a=selected.subtree_to_a,
            stats_a=selected.stats_a,
            stats_b=selected.stats_b,
            violations_a=selected.violations_a,
            violations_b=selected.violations_b,
            global_violations=selected.global_violations,
            boundary_cost=selected.boundary_cost,
        )

    def apply(self, state: _State, move: _Move) -> None:
        zone_a = move.zone_a
        zone_b = move.zone_b
        union = state.zone_nodes[zone_a] | state.zone_nodes[zone_b]
        subtree = set(move.subtree)
        if move.subtree_to_a:
            nodes_a = subtree
            nodes_b = union - subtree
        else:
            nodes_b = subtree
            nodes_a = union - subtree

        affected_edges: set[int] = set()
        for node in union:
            affected_edges.update(self.context.incident_edges[node])
        for edge_id in affected_edges:
            u, v = self.context.edges[edge_id]
            old_u = state.assignment[u]
            old_v = state.assignment[v]
            if old_u != old_v:
                _change_boundary_value(state.boundary_pairs, old_u, old_v, -1)
                _change_boundary_value(
                    state.boundary_costs,
                    old_u,
                    old_v,
                    -self.context.edge_weights[edge_id],
                )

        for node in nodes_a:
            state.assignment[node] = zone_a
        for node in nodes_b:
            state.assignment[node] = zone_b

        for edge_id in affected_edges:
            u, v = self.context.edges[edge_id]
            new_u = state.assignment[u]
            new_v = state.assignment[v]
            if new_u != new_v:
                _change_boundary_value(state.boundary_pairs, new_u, new_v, 1)
                _change_boundary_value(
                    state.boundary_costs,
                    new_u,
                    new_v,
                    self.context.edge_weights[edge_id],
                )

        state.zone_nodes[zone_a] = nodes_a
        state.zone_nodes[zone_b] = nodes_b
        state.zone_stats[zone_a] = move.stats_a
        state.zone_stats[zone_b] = move.stats_b
        state.zone_violations[zone_a] = move.violations_a
        state.zone_violations[zone_b] = move.violations_b
        state.violations = move.global_violations
        state.boundary_cost = move.boundary_cost

    def _pair_graph(
        self, union: set[int]
    ) -> tuple[dict[int, tuple[int, ...]], list[int]]:
        pair_adjacency = {
            node: tuple(
                neighbor
                for neighbor in self.context.adjacency[node]
                if neighbor in union
            )
            for node in union
        }
        edge_ids: set[int] = set()
        for node in union:
            edge_ids.update(self.context.incident_edges[node])
        pair_edges = [
            edge_id
            for edge_id in edge_ids
            if self.context.edges[edge_id][0] in union
            and self.context.edges[edge_id][1] in union
        ]
        return pair_adjacency, pair_edges

    def _random_spanning_tree(
        self,
        nodes: list[int],
        adjacency: dict[int, tuple[int, ...]],
    ) -> dict[int, list[int]]:
        if not nodes:
            raise _NoProposal("empty_zone_pair")
        tree = {node: [] for node in nodes}
        root = self.rng.choice(nodes)
        in_tree = {root}
        unvisited = set(nodes)
        unvisited.remove(root)

        while unvisited:
            start = min(unvisited)
            path = [start]
            path_index = {start: 0}
            current = start
            while current not in in_tree:
                neighbors = adjacency[current]
                if not neighbors:
                    raise _NoProposal("disconnected_zone_pair")
                nxt = self.rng.choice(neighbors)
                if nxt in path_index:
                    loop_start = path_index[nxt]
                    for removed in path[loop_start + 1 :]:
                        path_index.pop(removed, None)
                    path = path[: loop_start + 1]
                else:
                    path_index[nxt] = len(path)
                    path.append(nxt)
                current = nxt
                self._check_deadline()

            for u, v in zip(path, path[1:]):
                tree[u].append(v)
                tree[v].append(u)
            for node in path[:-1]:
                in_tree.add(node)
                unvisited.discard(node)
        return tree

    def _root_tree(
        self, tree: dict[int, list[int]]
    ) -> tuple[list[int], list[int], list[int]]:
        root = min(tree)
        preorder: list[int] = []
        parent_by_node = {root: root}
        depth_by_node = {root: 0}
        stack = [root]
        while stack:
            node = stack.pop()
            preorder.append(node)
            children = sorted(
                neighbor for neighbor in tree[node] if neighbor != parent_by_node[node]
            )
            for child in reversed(children):
                parent_by_node[child] = node
                depth_by_node[child] = depth_by_node[node] + 1
                stack.append(child)
        if len(preorder) != len(tree):
            raise _NoProposal("disconnected_spanning_tree")
        local = {node: idx for idx, node in enumerate(preorder)}
        parent = [local[parent_by_node[node]] for node in preorder]
        depth = [depth_by_node[node] for node in preorder]
        return preorder, parent, depth

    def _cut_candidates(
        self,
        state: _State,
        zone_a: int,
        zone_b: int,
        preorder: list[int],
        parent: list[int],
        depth: list[int],
        pair_adjacency: dict[int, tuple[int, ...]],
        pair_edges: list[int],
    ) -> list[_CutCandidate]:
        context = self.context
        count = len(preorder)
        local = {node: idx for idx, node in enumerate(preorder)}
        subtree_nodes = [1] * count
        subtree_students = [context.students[node] for node in preorder]
        subtree_schools = [context.schools[node] for node in preorder]
        subtree_seats = [context.seats[node] for node in preorder]
        subtree_frl = [context.frl[node] for node in preorder]
        subtree_values = [list(context.values[node]) for node in preorder]
        subtree_volume = [len(pair_adjacency[node]) for node in preorder]
        illegal_a = [int(zone_a not in context.allowed[node]) for node in preorder]
        illegal_b = [int(zone_b not in context.allowed[node]) for node in preorder]

        for idx in range(count - 1, 0, -1):
            parent_idx = parent[idx]
            subtree_nodes[parent_idx] += subtree_nodes[idx]
            subtree_students[parent_idx] += subtree_students[idx]
            subtree_schools[parent_idx] += subtree_schools[idx]
            subtree_seats[parent_idx] += subtree_seats[idx]
            subtree_frl[parent_idx] += subtree_frl[idx]
            subtree_volume[parent_idx] += subtree_volume[idx]
            illegal_a[parent_idx] += illegal_a[idx]
            illegal_b[parent_idx] += illegal_b[idx]
            for value_idx, value in enumerate(subtree_values[idx]):
                subtree_values[parent_idx][value_idx] += value

        up = [parent]
        for _ in range(1, max(1, count.bit_length())):
            previous = up[-1]
            up.append([previous[previous[idx]] for idx in range(count)])

        delta = [0] * count
        cost_delta = [0] * count
        for edge_id in pair_edges:
            u, v = context.edges[edge_id]
            weight = context.edge_weights[edge_id]
            idx_u = local[u]
            idx_v = local[v]
            ancestor = _lca(idx_u, idx_v, depth, up)
            delta[idx_u] += 1
            delta[idx_v] += 1
            delta[ancestor] -= 2
            cost_delta[idx_u] += weight
            cost_delta[idx_v] += weight
            cost_delta[ancestor] -= 2 * weight
        for idx in range(count - 1, 0, -1):
            delta[parent[idx]] += delta[idx]
            cost_delta[parent[idx]] += cost_delta[idx]

        total_students = subtree_students[0]
        total_schools = subtree_schools[0]
        total_seats = subtree_seats[0]
        total_frl = subtree_frl[0]
        total_values = subtree_values[0]
        total_illegal_a = illegal_a[0]
        total_illegal_b = illegal_b[0]
        pair_edge_count = len(pair_edges)
        old_pair_cost = state.boundary_costs.get(_zone_pair(zone_a, zone_b), 0)
        candidates: list[_CutCandidate] = []

        for idx in range(1, count):
            self._check_deadline()
            nodes_sub = subtree_nodes[idx]
            nodes_other = count - nodes_sub
            crossing = delta[idx]
            crossing_cost = cost_delta[idx]
            internal_sub = (subtree_volume[idx] - crossing) // 2
            internal_other = pair_edge_count - crossing - internal_sub
            stats_sub = _ZoneStats(
                node_count=nodes_sub,
                students=subtree_students[idx],
                values=tuple(subtree_values[idx]),
                schools=subtree_schools[idx],
                internal_edges=internal_sub,
                seat_value=subtree_seats[idx],
                frl_value=subtree_frl[idx],
            )
            stats_other = _ZoneStats(
                node_count=nodes_other,
                students=total_students - subtree_students[idx],
                values=tuple(
                    total_values[value_idx] - value
                    for value_idx, value in enumerate(subtree_values[idx])
                ),
                schools=total_schools - subtree_schools[idx],
                internal_edges=internal_other,
                seat_value=total_seats - subtree_seats[idx],
                frl_value=total_frl - subtree_frl[idx],
            )
            boundary_cost = state.boundary_cost - old_pair_cost + crossing_cost

            if illegal_a[idx] == 0 and total_illegal_b - illegal_b[idx] == 0:
                candidates.append(
                    self._candidate(
                        state,
                        zone_a,
                        zone_b,
                        idx,
                        nodes_sub,
                        True,
                        stats_sub,
                        stats_other,
                        boundary_cost,
                    )
                )
            if illegal_b[idx] == 0 and total_illegal_a - illegal_a[idx] == 0:
                candidates.append(
                    self._candidate(
                        state,
                        zone_a,
                        zone_b,
                        idx,
                        nodes_sub,
                        False,
                        stats_other,
                        stats_sub,
                        boundary_cost,
                    )
                )
        return candidates

    def _candidate(
        self,
        state: _State,
        zone_a: int,
        zone_b: int,
        tin: int,
        size: int,
        subtree_to_a: bool,
        stats_a: _ZoneStats,
        stats_b: _ZoneStats,
        boundary_cost: int,
    ) -> _CutCandidate:
        violations_a = self.context.zone_violations(stats_a)
        violations_b = self.context.zone_violations(stats_b)
        global_violations = tuple(
            max(
                0.0,
                state.violations[idx]
                - state.zone_violations[zone_a][idx]
                - state.zone_violations[zone_b][idx]
                + violations_a[idx]
                + violations_b[idx],
            )
            for idx in range(self.context.violation_count)
        )
        return _CutCandidate(
            tin=tin,
            size=size,
            subtree_to_a=subtree_to_a,
            stats_a=stats_a,
            stats_b=stats_b,
            violations_a=violations_a,
            violations_b=violations_b,
            global_violations=global_violations,
            boundary_cost=boundary_cost,
        )

    def _relaxed_probabilities(self, candidates: list[_CutCandidate]) -> list[float]:
        log_weights = [self._relaxed_log_weight(candidate) for candidate in candidates]
        positive_infinity = [
            idx for idx, value in enumerate(log_weights) if value == float("inf")
        ]
        if positive_infinity:
            probability = 1.0 / len(positive_infinity)
            return [
                probability if idx in positive_infinity else 0.0
                for idx in range(len(candidates))
            ]

        finite = [value for value in log_weights if math.isfinite(value)]
        if not finite:
            return [1.0 / len(candidates)] * len(candidates)
        maximum = max(finite)
        weights = [
            math.exp(value - maximum) if math.isfinite(value) else 0.0
            for value in log_weights
        ]
        total = sum(weights)
        if not math.isfinite(total) or total <= 0:
            return [1.0 / len(candidates)] * len(candidates)
        return [weight / total for weight in weights]

    def _relaxed_log_weight(self, candidate: _CutCandidate) -> float:
        stats_a = candidate.stats_a
        stats_b = candidate.stats_b
        # exp(cycle rank) is the fast approximation of each zone's tree count.
        log_weight = _RELAXED_WEIGHTS["trees"] * (
            stats_a.cycle_rank + stats_b.cycle_rank
        )
        metrics_a = self._relaxed_metrics(stats_a)
        metrics_b = self._relaxed_metrics(stats_b)
        for metric, weight in _RELAXED_WEIGHTS.items():
            if metric == "trees" or metric not in metrics_a:
                continue
            value_a = metrics_a[metric]
            value_b = metrics_b[metric]
            if value_a <= 0 or value_b <= 0:
                return float("-inf")
            log_weight += weight * (math.log(value_a) + math.log(value_b))
        return log_weight

    @staticmethod
    def _relaxed_metrics(stats: _ZoneStats) -> dict[str, float]:
        shortage_percent = max(
            abs(stats.students - stats.seats) / max(stats.students, _WEIGHT_EPS),
            _WEIGHT_EPS,
        )
        metrics = {
            "nodes": float(stats.node_count),
            "frl": stats.frl,
            "students": stats.students,
            "seats": stats.seats,
            "shortage%": shortage_percent,
            "sch_count": stats.schools,
        }
        return metrics

    def _check_deadline(self) -> None:
        if self.deadline is None:
            return
        self._deadline_checks += 1
        if self._deadline_checks % 128 == 0 and time.monotonic() >= self.deadline:
            raise _DeadlineReached


class _ReComSolverBase(Solver):
    def _initialize(self, problem: ZoneProblem, start: float) -> _Setup:
        context = _ReComContext(problem)
        if problem.hint is not None:
            hint = problem.hint
            hint_metadata = {"hints": "provided", "hint_source": "problem_hint"}
        else:
            method = normalize_hints(self.options.get("hints", "voronoi"))
            if method == "none":
                raise _HintError("ReCom solvers require a provided or generated hint.")
            if method == "feasible":
                generated = initial_solution(
                    problem,
                    method,
                    solver_options=self.options,
                )
                assert generated is not None
                hint = generated.assignment
                hint_metadata = {**generated.metadata, "hint_source": "generated"}
            else:
                hint = self._voronoi_hint(context)
                hint_metadata = {"hints": "voronoi", "hint_source": "generated"}

        assignment = context.validate_hint(hint)
        max_iterations, deadline = self._limits(start)
        return _Setup(
            context=context,
            state=context.build_state(assignment),
            rng=random.Random(int(self.options.get("seed", 42))),
            max_iterations=max_iterations,
            deadline=deadline,
            hint_metadata=hint_metadata,
        )

    def _voronoi_hint(self, context: _ReComContext) -> dict[int, int]:
        problem = context.problem
        assignment: dict[int, int] = {}
        for pos, node in enumerate(context.nodes):
            assignment[node] = min(
                context.allowed[pos],
                key=lambda zone: problem.distance(problem.centroids[zone], node),
            )
        return contiguity.repair(problem.G, assignment, problem.centroids)

    def _limits(self, start: float) -> tuple[int | None, float | None]:
        iterations = int(self.options.get("recom_iterations", 1000))
        raw_time_limit = self.options.get("solve_time_limit", 60.0)
        time_limit = None if raw_time_limit is None else max(0.0, float(raw_time_limit))
        if iterations < 0 and time_limit is None:
            raise ValueError(
                "solve_time_limit must be supplied when recom_iterations is negative."
            )
        deadline = start + time_limit if time_limit is not None else None
        return (None if iterations < 0 else max(0, iterations)), deadline

    @staticmethod
    def _budget_available(
        attempted: int,
        max_iterations: int | None,
        deadline: float | None,
    ) -> bool:
        if max_iterations is not None and attempted >= max_iterations:
            return False
        return deadline is None or time.monotonic() < deadline

    @staticmethod
    def _snapshot(state: _State) -> _Snapshot:
        return _Snapshot(
            assignment=tuple(state.assignment),
            violations=tuple(state.violations),
            boundary_cost=state.boundary_cost,
        )

    def _error_solution(
        self,
        problem: ZoneProblem,
        start: float,
        message: str,
    ) -> ZoneSolution:
        return ZoneSolution(
            problem=problem,
            assignment={},
            status="ERROR",
            objective=None,
            wall_time=time.monotonic() - start,
            metadata={"solver": self.name, "error_message": message},
        )

    def _result(
        self,
        problem: ZoneProblem,
        context: _ReComContext,
        start: float,
        best: _Snapshot | None,
        metadata: dict[str, object],
    ) -> ZoneSolution:
        if best is None:
            status = "UNKNOWN"
            assignment = {}
            objective = None
        else:
            status = "FEASIBLE"
            assignment = context.assignment_dict(best.assignment)
            objective = float(best.boundary_cost)
        if problem.weight_edges:
            metadata = {
                **metadata,
                "objective_kind": "weighted_boundary_length",
                "objective_unit": "meter",
            }
        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status=status,
            objective=objective,
            wall_time=time.monotonic() - start,
            metadata={"solver": self.name, **metadata},
        )

    @staticmethod
    def _better_feasible(candidate: _Snapshot, best: _Snapshot | None) -> bool:
        return candidate.feasible and (
            best is None or candidate.boundary_cost < best.boundary_cost
        )

    @staticmethod
    def _check_choice_objective(problem: ZoneProblem, solver_name: str) -> None:
        if problem.choice_objective is not None:
            raise NotImplementedError(
                f"{solver_name} does not support iterative choice objectives; "
                "use cp_int, cp_bool, or mip."
            )


@register("recom")
class ReComSolver(_ReComSolverBase):
    """Feasibility-rejecting ReCom random walk."""

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        return self._solve_chain(problem, selector="uniform", reject_infeasible=True)

    def _solve_chain(
        self,
        problem: ZoneProblem,
        *,
        selector: str,
        reject_infeasible: bool,
    ) -> ZoneSolution:
        self._check_choice_objective(problem, self.name)
        start = time.monotonic()
        try:
            setup = self._initialize(problem, start)
        except _HintError as exc:
            return self._error_solution(problem, start, str(exc))

        state = setup.state
        kernel = _ReComKernel(setup.context, setup.rng, setup.deadline)
        initial = self._snapshot(state)
        best = initial if initial.feasible else None
        attempted = 0
        accepted = 0
        rejected = 0
        proposal_failures = 0
        stop_reason = "iteration_limit"

        while self._budget_available(attempted, setup.max_iterations, setup.deadline):
            attempted += 1
            try:
                move = kernel.propose(state, selector)
            except _DeadlineReached:
                stop_reason = "time_limit"
                break
            except _NoProposal as exc:
                proposal_failures += 1
                if exc.reason == "no_adjacent_zone_pairs":
                    stop_reason = exc.reason
                    break
                continue

            if reject_infeasible and not move.globally_feasible:
                rejected += 1
                continue
            kernel.apply(state, move)
            accepted += 1
            snapshot = self._snapshot(state)
            if self._better_feasible(snapshot, best):
                best = snapshot

        if setup.deadline is not None and time.monotonic() >= setup.deadline:
            stop_reason = "time_limit"
        metadata = {
            **setup.hint_metadata,
            "recom_iterations": self.options.get("recom_iterations", 1000),
            "attempted_moves": attempted,
            "accepted_moves": accepted,
            "rejected_moves": rejected,
            "proposal_failures": proposal_failures,
            "stop_reason": stop_reason,
            "initial_feasible": initial.feasible,
            "cut_selector": selector,
            "tree_sampler": "wilson_uniform",
            "tree_count_approximation": "exp_cycle_rank",
        }
        return self._result(problem, setup.context, start, best, metadata)


@register("relaxed_recom")
class RelaxedReComSolver(ReComSolver):
    """Biased ReCom random walk with no feasibility rejection."""

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        return self._solve_chain(problem, selector="relaxed", reject_infeasible=False)


@register("short_bursts")
class ShortBurstsSolver(_ReComSolverBase):
    """Unrejected ReCom walks with deterministic short-burst restarts."""

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        self._check_choice_objective(problem, self.name)
        start = time.monotonic()
        try:
            setup = self._initialize(problem, start)
        except _HintError as exc:
            return self._error_solution(problem, start, str(exc))

        method = str(self.options.get("short_bursts_method", "recom"))
        if method not in {"recom", "relaxed_recom"}:
            raise ValueError(
                "short_bursts_method must be one of: recom, relaxed_recom."
            )
        selector = "uniform" if method == "recom" else "relaxed"
        burst_length = int(self.options.get("short_bursts_length", 25))
        if burst_length <= 0:
            raise ValueError("short_bursts_length must be positive.")

        context = setup.context
        current = setup.state
        kernel = _ReComKernel(context, setup.rng, setup.deadline)
        normalizer = _DynamicMaxNormalizer(context.violation_count)
        initial = self._snapshot(current)
        normalizer.observe(initial.violations)
        best_feasible = initial if initial.feasible else None
        attempted = 0
        accepted = 0
        proposal_failures = 0
        completed_bursts = 0
        selected_improvements = 0
        stop_reason = "iteration_limit"

        while self._budget_available(attempted, setup.max_iterations, setup.deadline):
            base = self._snapshot(current)
            walk = current.clone()
            samples: list[_Snapshot] = []
            remaining = burst_length
            if setup.max_iterations is not None:
                remaining = min(remaining, setup.max_iterations - attempted)

            deadline_reached = False
            no_adjacent_pairs = False
            for _ in range(remaining):
                if not self._budget_available(
                    attempted, setup.max_iterations, setup.deadline
                ):
                    deadline_reached = True
                    break
                attempted += 1
                try:
                    move = kernel.propose(walk, selector)
                except _DeadlineReached:
                    deadline_reached = True
                    break
                except _NoProposal as exc:
                    proposal_failures += 1
                    if exc.reason == "no_adjacent_zone_pairs":
                        no_adjacent_pairs = True
                        break
                    continue

                kernel.apply(walk, move)
                accepted += 1
                snapshot = self._snapshot(walk)
                normalizer.observe(snapshot.violations)
                samples.append(snapshot)
                if self._better_feasible(snapshot, best_feasible):
                    best_feasible = snapshot

            selected = base
            for sample in samples:
                if _burst_better(sample, selected, normalizer):
                    selected = sample
            if _burst_better(selected, base, normalizer):
                current = context.build_state(list(selected.assignment))
                selected_improvements += 1

            if deadline_reached:
                stop_reason = "time_limit"
                break
            if no_adjacent_pairs:
                stop_reason = "no_adjacent_zone_pairs"
                break
            completed_bursts += 1

        if setup.deadline is not None and time.monotonic() >= setup.deadline:
            stop_reason = "time_limit"
        metadata = {
            **setup.hint_metadata,
            "recom_iterations": self.options.get("recom_iterations", 1000),
            "attempted_moves": attempted,
            "accepted_moves": accepted,
            "rejected_moves": 0,
            "proposal_failures": proposal_failures,
            "completed_bursts": completed_bursts,
            "selected_burst_improvements": selected_improvements,
            "short_bursts_length": burst_length,
            "short_bursts_method": method,
            "short_bursts_score": (
                "running_max_linear_violations_else_boundary_cost"
                if problem.weight_edges
                else "running_max_linear_violations_else_cut_edges"
            ),
            "stop_reason": stop_reason,
            "initial_feasible": initial.feasible,
            "cut_selector": selector,
            "tree_sampler": "wilson_uniform",
            "tree_count_approximation": "exp_cycle_rank",
        }
        return self._result(problem, context, start, best_feasible, metadata)


def _zone_pair(zone_a: int, zone_b: int) -> tuple[int, int]:
    return (zone_a, zone_b) if zone_a < zone_b else (zone_b, zone_a)


def _change_boundary_value(
    values: dict[tuple[int, int], int],
    zone_a: int,
    zone_b: int,
    change: int,
) -> None:
    pair = _zone_pair(zone_a, zone_b)
    updated = values.get(pair, 0) + change
    if updated > 0:
        values[pair] = updated
    else:
        values.pop(pair, None)


def _lca(node_a: int, node_b: int, depth: list[int], up: list[list[int]]) -> int:
    if depth[node_a] < depth[node_b]:
        node_a, node_b = node_b, node_a
    difference = depth[node_a] - depth[node_b]
    bit = 0
    while difference:
        if difference & 1:
            node_a = up[bit][node_a]
        difference >>= 1
        bit += 1
    if node_a == node_b:
        return node_a
    for level in range(len(up) - 1, -1, -1):
        if up[level][node_a] != up[level][node_b]:
            node_a = up[level][node_a]
            node_b = up[level][node_b]
    return up[0][node_a]


def _burst_better(
    candidate: _Snapshot,
    incumbent: _Snapshot,
    normalizer: _DynamicMaxNormalizer,
) -> bool:
    if candidate.feasible != incumbent.feasible:
        return candidate.feasible
    if candidate.feasible:
        return candidate.boundary_cost < incumbent.boundary_cost
    return normalizer.penalty(candidate.violations) < (
        normalizer.penalty(incumbent.violations) - _EPS
    )
