"""ReCom: one stochastic zoning solver, parameterized by ``ReComSettings``.

A move merges two adjacent zones, draws a spanning tree of the merged region,
and cuts one tree edge. By default the tree minimizes fresh independent random
edge weights; Wilson's uniform spanning tree is optional. Zone statistics and
constraint residuals are additive, so every cut of the tree is scored at once
without building any candidate subgraph.

A walk is a sequence of moves in bursts of ``recom_burst_length``. After each
burst the walk restarts from the best state it visited (feasible beats
infeasible, then the objective, else the policy's merit), and the policy
updates whatever it learns. ``recom_burst_length: 0`` never restarts: one
continuous chain, which with ``recom_reject_infeasible`` is classic ReCom.
How pairs and cuts are chosen, and how constraint weights adapt, is the
``Policy`` of ``optimization.solvers.recom_policy``.

Centroids set the number of zones and seed the Voronoi hint, but they and
distance-derived candidates are not hard constraints here; explicit candidate
and fixed assignments are.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace

from optimization.data import contiguity
from optimization.data.initial_solutions import initial_solution, normalize_hints
from optimization.problem import ZoneProblem
from optimization.solution import ZoneSolution
from optimization.solvers.balance import BalanceConstraint, balance_constraints
from optimization.solvers.base import Solver, register
from optimization.solvers.heuristic_log import HeuristicProgressLog
from optimization.solvers.recom_policy import PRESETS, Policy, ReComSettings

_EPS = 1e-6
# Moves between objective batches when the walk never restarts.
_CHAIN_BLOCK = 25

# What ``solve`` ranks feasible partitions by (``recom_objective``):
# ``cut_edges`` is the boundary cost (metres under ``weight_edges``),
# ``school_distance`` the weighted squared distance to each zone's school,
# ``welfare`` the discrete MID program welfare, ``capacity_match`` the
# weighted count of students left unassigned or designated by the MID match,
# ``aa_capacity_match`` the same count with the match confined to
# attendance-area seats.
RECOM_OBJECTIVES = (
    "cut_edges",
    "school_distance",
    "welfare",
    "capacity_match",
    "aa_capacity_match",
)

ShortBurstBatchScorer = Callable[
    [tuple[Mapping[int, int], ...], Mapping[int, int] | None],
    tuple[float, ...],
]


class _HintError(ValueError):
    """A hint that cannot start a walk, with whatever is known of its origin."""

    def __init__(self, message: str, metadata: Mapping[str, object] | None = None):
        super().__init__(message)
        self.metadata = dict(metadata or {})


class _NoProposal(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class _DeadlineReached(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ZoneStats:
    node_count: int
    students: float
    # One value per balance row, then the student count its ratio is against.
    values: tuple[float, ...]
    schools: float


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
    # Signed per-zone residuals (``g <= 0`` feasible) and their clipped parts.
    zone_residuals: list[tuple[float, ...]]
    zone_violations: list[tuple[float, ...]]
    # Per-row totals of the clipped residuals, then the cut-edge cap's.
    violations: tuple[float, ...]
    boundary_pairs: dict[tuple[int, int], int]
    boundary_costs: dict[tuple[int, int], int]
    boundary_cost: int
    # Unweighted cut edges, what ``boundary_prop`` caps even in metres.
    cut_edges: int = 0

    @property
    def feasible(self) -> bool:
        return all(value <= _EPS for value in self.violations)

    def clone(self) -> "_State":
        return _State(
            assignment=list(self.assignment),
            zone_nodes=[set(nodes) for nodes in self.zone_nodes],
            zone_stats=list(self.zone_stats),
            zone_residuals=list(self.zone_residuals),
            zone_violations=list(self.zone_violations),
            violations=self.violations,
            boundary_pairs=dict(self.boundary_pairs),
            boundary_costs=dict(self.boundary_costs),
            boundary_cost=self.boundary_cost,
            cut_edges=self.cut_edges,
        )


@dataclass(frozen=True)
class _Cut:
    """One cut of a merged pair's tree: the two zones it would leave."""

    tin: int
    size: int
    subtree_to_a: bool
    stats_a: _ZoneStats
    stats_b: _ZoneStats
    residuals_a: tuple[float, ...]
    residuals_b: tuple[float, ...]
    violations_a: tuple[float, ...]
    violations_b: tuple[float, ...]
    violations: tuple[float, ...]
    boundary_cost: int
    cut_edges: int

    @property
    def pair_feasible(self) -> bool:
        return all(v <= _EPS for v in self.violations_a) and all(
            v <= _EPS for v in self.violations_b
        )

    @property
    def feasible(self) -> bool:
        return all(v <= _EPS for v in self.violations)


@dataclass(frozen=True)
class _Move:
    zone_a: int
    zone_b: int
    subtree: tuple[int, ...]
    cut: _Cut


@dataclass(frozen=True)
class _Snapshot:
    assignment: tuple[int, ...]
    violations: tuple[float, ...]
    boundary_cost: int
    cut_edges: int = 0
    # The policy's merit when the snapshot was taken (weights change only
    # between bursts, so snapshots of one burst compare).
    merit: float = 0.0

    @property
    def feasible(self) -> bool:
        return all(value <= _EPS for value in self.violations)


@dataclass
class _Tally:
    attempted_moves: int = 0
    accepted_moves: int = 0
    rejected_moves: int = 0
    proposal_failures: int = 0
    completed_bursts: int = 0
    restarts: int = 0


def _snapshot(state: _State, merit: float = 0.0) -> _Snapshot:
    return _Snapshot(
        tuple(state.assignment),
        state.violations,
        state.boundary_cost,
        state.cut_edges,
        merit,
    )


# ---------------------------------------------------------------------------
# Context: the problem as arrays
# ---------------------------------------------------------------------------


class _ReComContext:
    """Problem data converted to array-backed structures for hot loops.

    Residuals are in percent of the zone's students (school counts in
    percent of the average count), so one unit means the same in a small
    zone and a large one.
    """

    def __init__(self, problem: ZoneProblem) -> None:
        self.problem = problem
        self.nodes = tuple(problem.nodes)
        self.node_to_pos = {node: idx for idx, node in enumerate(self.nodes)}
        self.zone_count = problem.Z
        if self.zone_count <= 0:
            raise ValueError("ReCom requires at least one zone.")

        self.adjacency = [
            tuple(self.node_to_pos[n] for n in problem.G.neighbors(node))
            for node in self.nodes
        ]
        graph_edges = tuple(problem.G.edges())
        self.edges = tuple(
            (self.node_to_pos[u], self.node_to_pos[v]) for u, v in graph_edges
        )
        self.edge_weights = tuple(problem.boundary_weight(u, v) for u, v in graph_edges)
        incident: list[list[int]] = [[] for _ in self.nodes]
        for edge_id, (u, v) in enumerate(self.edges):
            incident[u].append(edge_id)
            incident[v].append(edge_id)
        self.incident_edges = tuple(tuple(ids) for ids in incident)

        self.constraints = tuple(balance_constraints(problem))
        self.balance_rows = tuple(
            _BalanceRow(constraint, sense, ratio)
            for constraint in self.constraints
            for sense, ratio in (
                ("lower", constraint.lower_ratio),
                ("upper", constraint.upper_ratio),
            )
            if ratio is not None
        )
        self.students = tuple(problem.students(node) for node in self.nodes)
        # The capacity rows count enrolling students, so each row carries the
        # student count its ratio is taken against.
        self.values = tuple(
            tuple(row.constraint.value(node) for row in self.balance_rows)
            + tuple(row.constraint.students(problem, node) for row in self.balance_rows)
            for node in self.nodes
        )
        self.schools = tuple(float(problem.num_schools(node)) for node in self.nodes)
        # Mirrors the CP/MIP models and ``check_zoning``: an unweighted count
        # against a floored proportion of the edge count.
        self.cut_limit = (
            math.floor(problem.boundary_prop * problem.G.number_of_edges())
            if problem.boundary_prop >= 0
            else None
        )
        total_schools = sum(self.schools)
        self.average_schools = total_schools / self.zone_count
        self.school_bounds = (
            problem.school_count_bounds() if total_schools > 0 else None
        )

        all_zones = frozenset(range(self.zone_count))
        allowed: list[frozenset[int]] = []
        for node in self.nodes:
            if problem.candidates is not None and node in problem.candidates:
                zones = frozenset(
                    int(z)
                    for z in problem.candidates[node]
                    if 0 <= int(z) < self.zone_count
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

    # The violation vector is the per-zone rows (balance, then school count)
    # summed over zones, followed by the cut-edge cap, which only exists for
    # the whole partition. Every consumer treats each row alike.
    @property
    def zone_row_count(self) -> int:
        return len(self.balance_rows) + (2 if self.school_bounds else 0)

    @property
    def cut_row(self) -> int | None:
        return self.zone_row_count if self.cut_limit is not None else None

    @property
    def violation_count(self) -> int:
        return self.zone_row_count + (self.cut_limit is not None)

    def residuals(self, stats: _ZoneStats) -> tuple[float, ...]:
        """Signed residuals of one zone, in percent; ``<= 0`` is feasible."""
        out: list[float] = []
        rows = len(self.balance_rows)
        for value, students, row in zip(
            stats.values[:rows], stats.values[rows:], self.balance_rows, strict=True
        ):
            diff = (
                row.ratio * students - value
                if row.sense == "lower"
                else value - row.ratio * students
            )
            if students > 0:
                out.append(diff / students * 100.0)
            else:
                out.append(0.0 if value == 0 else 100.0)
        if self.school_bounds is not None:
            lower, upper = self.school_bounds
            scale = 100.0 / self.average_schools
            out.append((lower - stats.schools) * scale)
            out.append((stats.schools - upper) * scale)
        return tuple(out)

    def row_students(self, stats: _ZoneStats) -> tuple[float, ...]:
        """Per row, what its residual is a percentage of: ``count = g s / 100``."""
        rows = len(self.balance_rows)
        schools = (self.average_schools,) * (2 if self.school_bounds else 0)
        return (*stats.values[rows:], *schools)

    def cut_residual(self, cut_edges: int) -> float:
        return (cut_edges - self.cut_limit) / max(self.cut_limit, 1) * 100.0

    def total_violations(
        self, zone_totals: list[float], cut_edges: int
    ) -> tuple[float, ...]:
        """Append the cut-edge cap's violation to already-summed zone rows."""
        if self.cut_limit is None:
            return tuple(zone_totals)
        return (*zone_totals, max(0.0, self.cut_residual(cut_edges)))

    def violation_labels(self) -> tuple[str, ...]:
        """Name every entry of the violation vector, index-aligned with it."""
        ethnicities = list(getattr(self.problem, "ethnicities", ()) or ())
        names: list[str] = []
        racial = 0
        for constraint in self.constraints:
            if constraint.kind == "racial":
                names.append(
                    f"racial_{ethnicities[racial] if racial < len(ethnicities) else racial}"
                )
                racial += 1
            else:
                names.append(constraint.kind)
        labels = [
            f"{name}_{sense}"
            for name, constraint in zip(names, self.constraints, strict=True)
            for sense, ratio in (
                ("lower", constraint.lower_ratio),
                ("upper", constraint.upper_ratio),
            )
            if ratio is not None
        ]
        if self.school_bounds is not None:
            labels += ["schools_lower", "schools_upper"]
        if self.cut_limit is not None:
            labels.append("boundary")
        return tuple(labels)

    def build_state(self, assignment: list[int]) -> _State:
        zone_nodes: list[set[int]] = [set() for _ in range(self.zone_count)]
        for pos, zone in enumerate(assignment):
            zone_nodes[zone].add(pos)
        zone_stats = [self.stats(nodes) for nodes in zone_nodes]
        boundary_pairs: dict[tuple[int, int], int] = {}
        boundary_costs: dict[tuple[int, int], int] = {}
        for edge_id, (u, v) in enumerate(self.edges):
            if assignment[u] != assignment[v]:
                pair = _zone_pair(assignment[u], assignment[v])
                boundary_pairs[pair] = boundary_pairs.get(pair, 0) + 1
                boundary_costs[pair] = (
                    boundary_costs.get(pair, 0) + self.edge_weights[edge_id]
                )
        residuals = [self.residuals(stats) for stats in zone_stats]
        violations = [_clip(g) for g in residuals]
        cut_edges = sum(boundary_pairs.values())
        return _State(
            assignment=assignment,
            zone_nodes=zone_nodes,
            zone_stats=zone_stats,
            zone_residuals=residuals,
            zone_violations=violations,
            violations=self.total_violations(
                [sum(v[k] for v in violations) for k in range(self.zone_row_count)],
                cut_edges,
            ),
            boundary_pairs=boundary_pairs,
            boundary_costs=boundary_costs,
            boundary_cost=sum(boundary_costs.values()),
            cut_edges=cut_edges,
        )

    def stats(self, nodes) -> _ZoneStats:
        values = [0.0] * (2 * len(self.balance_rows))
        for node in nodes:
            for idx, value in enumerate(self.values[node]):
                values[idx] += value
        return _ZoneStats(
            node_count=len(nodes),
            students=sum(self.students[node] for node in nodes),
            values=tuple(values),
            schools=sum(self.schools[node] for node in nodes),
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
            zone = hint[node]
            if not isinstance(zone, int) or isinstance(zone, bool):
                raise _HintError(
                    f"ReCom hint assigns non-integer zone {zone!r} to node {node}."
                )
            if not 0 <= zone < self.zone_count:
                raise _HintError(
                    f"ReCom hint assigns invalid zone {zone} to node {node}."
                )
            if zone not in self.allowed[pos]:
                raise _HintError(
                    f"ReCom hint assigns node {node} to explicitly forbidden zone {zone}."
                )
            assignment.append(int(zone))
        if set(assignment) != set(range(self.zone_count)):
            raise _HintError(
                f"ReCom hint must represent every zone; got {sorted(set(assignment))}, "
                f"expected {list(range(self.zone_count))}."
            )
        for zone in range(self.zone_count):
            if not self._connected({i for i, z in enumerate(assignment) if z == zone}):
                raise _HintError(f"ReCom hint zone {zone} is not contiguous.")
        return assignment

    def _connected(self, nodes: set[int]) -> bool:
        if not nodes:
            return False
        start = next(iter(nodes))
        seen, stack = {start}, [start]
        while stack:
            for neighbor in self.adjacency[stack.pop()]:
                if neighbor in nodes and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return len(seen) == len(nodes)


def _clip(residuals: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(g if g > 0.0 else 0.0 for g in residuals)


# ---------------------------------------------------------------------------
# Kernel: proposals and their application
# ---------------------------------------------------------------------------


class _ReComKernel:
    def __init__(
        self,
        context: _ReComContext,
        rng: random.Random,
        deadline: float | None,
        *,
        tree_sampler: str | None = None,
    ):
        self.context = context
        self.rng = rng
        self.deadline = deadline
        self.tree_sampler = (
            ReComSettings().tree_sampler if tree_sampler is None else tree_sampler
        )
        self._deadline_checks = 0

    def propose(self, state: _State, policy: Policy) -> _Move:
        """Merge the policy's pair, draw a tree, and cut where the policy says."""
        pairs = sorted(
            pair for pair, count in state.boundary_pairs.items() if count > 0
        )
        if not pairs:
            raise _NoProposal("no_adjacent_zone_pairs")
        zone_a, zone_b = _zone_pair(*policy.choose_pair(state, pairs))
        union = state.zone_nodes[zone_a] | state.zone_nodes[zone_b]
        adjacency = {
            node: tuple(n for n in self.context.adjacency[node] if n in union)
            for node in union
        }
        preorder, parent, depth = self._root_tree(
            self._random_spanning_tree(sorted(union), adjacency)
        )
        cuts = self._cuts(state, zone_a, zone_b, preorder, parent, depth, union)
        if not cuts:
            raise _NoProposal("no_explicitly_legal_cut")
        probabilities = policy.cut_probabilities(state, zone_a, zone_b, cuts)
        cut = self.rng.choices(cuts, weights=probabilities, k=1)[0]
        return _Move(zone_a, zone_b, tuple(preorder[cut.tin : cut.tin + cut.size]), cut)

    def apply(self, state: _State, move: _Move) -> None:
        context, cut = self.context, move.cut
        zone_a, zone_b = move.zone_a, move.zone_b
        union = state.zone_nodes[zone_a] | state.zone_nodes[zone_b]
        subtree = set(move.subtree)
        nodes_a, nodes_b = (
            (subtree, union - subtree)
            if cut.subtree_to_a
            else (union - subtree, subtree)
        )
        affected = {edge for node in union for edge in context.incident_edges[node]}

        def boundary(sign: int) -> None:
            for edge_id in affected:
                u, v = context.edges[edge_id]
                if state.assignment[u] != state.assignment[v]:
                    pair = _zone_pair(state.assignment[u], state.assignment[v])
                    _add(state.boundary_pairs, pair, sign)
                    _add(
                        state.boundary_costs, pair, sign * context.edge_weights[edge_id]
                    )

        boundary(-1)
        for node in nodes_a:
            state.assignment[node] = zone_a
        for node in nodes_b:
            state.assignment[node] = zone_b
        boundary(+1)
        state.zone_nodes[zone_a], state.zone_nodes[zone_b] = nodes_a, nodes_b
        state.zone_stats[zone_a], state.zone_stats[zone_b] = cut.stats_a, cut.stats_b
        state.zone_residuals[zone_a], state.zone_residuals[zone_b] = (
            cut.residuals_a,
            cut.residuals_b,
        )
        state.zone_violations[zone_a], state.zone_violations[zone_b] = (
            cut.violations_a,
            cut.violations_b,
        )
        state.violations = cut.violations
        state.boundary_cost = cut.boundary_cost
        state.cut_edges = cut.cut_edges

    def _random_spanning_tree(
        self, nodes: list[int], adjacency: dict[int, tuple[int, ...]]
    ) -> dict[int, list[int]]:
        """Draw a fresh tree using the selected sampler and the run's RNG."""
        if not nodes:
            raise _NoProposal("empty_zone_pair")
        if self.tree_sampler == "random_mst":
            return self._random_weight_mst(nodes, adjacency)
        if self.tree_sampler == "wilson_uniform":
            return self._wilson_spanning_tree(nodes, adjacency)
        raise ValueError(f"Unknown ReCom tree sampler {self.tree_sampler!r}.")

    def _random_weight_mst(
        self, nodes: list[int], adjacency: dict[int, tuple[int, ...]]
    ) -> dict[int, list[int]]:
        """Kruskal's MST with a fresh independent U(0, 1) weight per edge."""
        tree: dict[int, list[int]] = {node: [] for node in nodes}
        edges = []
        for u in nodes:
            for v in sorted(adjacency[u]):
                if u < v:
                    edges.append((self.rng.random(), u, v))
                self._check_deadline()
        edges.sort()
        parents = {node: node for node in nodes}
        sizes = dict.fromkeys(nodes, 1)

        def root(node: int) -> int:
            while parents[node] != node:
                parents[node] = parents[parents[node]]
                node = parents[node]
            return node

        components = len(nodes)
        for _, u, v in edges:
            self._check_deadline()
            a, b = root(u), root(v)
            if a == b:
                continue
            if sizes[a] < sizes[b]:
                a, b = b, a
            parents[b] = a
            sizes[a] += sizes[b]
            tree[u].append(v)
            tree[v].append(u)
            components -= 1
            if components == 1:
                break
        if components != 1:
            raise _NoProposal("disconnected_zone_pair")
        return tree

    def _wilson_spanning_tree(
        self, nodes: list[int], adjacency: dict[int, tuple[int, ...]]
    ) -> dict[int, list[int]]:
        """Wilson's algorithm: a uniformly random spanning tree."""
        tree: dict[int, list[int]] = {node: [] for node in nodes}
        in_tree = {self.rng.choice(nodes)}
        unvisited = set(nodes) - in_tree
        while unvisited:
            path = [min(unvisited)]
            index = {path[0]: 0}
            current = path[0]
            while current not in in_tree:  # loop-erased random walk
                if not adjacency[current]:
                    raise _NoProposal("disconnected_zone_pair")
                current = self.rng.choice(adjacency[current])
                if current in index:
                    for removed in path[index[current] + 1 :]:
                        del index[removed]
                    del path[index[current] + 1 :]
                else:
                    index[current] = len(path)
                    path.append(current)
                self._check_deadline()
            for u, v in zip(path, path[1:]):
                tree[u].append(v)
                tree[v].append(u)
            in_tree.update(path[:-1])
            unvisited.difference_update(path[:-1])
        return tree

    @staticmethod
    def _root_tree(
        tree: dict[int, list[int]],
    ) -> tuple[list[int], list[int], list[int]]:
        """Preorder of the tree from its smallest node, with parent and depth."""
        root = min(tree)
        preorder: list[int] = []
        parent_of, depth_of = {root: root}, {root: 0}
        stack = [root]
        while stack:
            node = stack.pop()
            preorder.append(node)
            for child in sorted(
                (n for n in tree[node] if n != parent_of[node]), reverse=True
            ):
                parent_of[child] = node
                depth_of[child] = depth_of[node] + 1
                stack.append(child)
        if len(preorder) != len(tree):
            raise _NoProposal("disconnected_spanning_tree")
        local = {node: idx for idx, node in enumerate(preorder)}
        return (
            preorder,
            [local[parent_of[n]] for n in preorder],
            [depth_of[n] for n in preorder],
        )

    def _cuts(
        self,
        state: _State,
        zone_a: int,
        zone_b: int,
        preorder: list[int],
        parent: list[int],
        depth: list[int],
        union: set[int],
    ) -> list[_Cut]:
        """Every legal cut of the tree, scored from subtree sums."""
        context = self.context
        count = len(preorder)
        local = {node: idx for idx, node in enumerate(preorder)}
        sub_nodes = [1] * count
        sub_students = [context.students[n] for n in preorder]
        sub_schools = [context.schools[n] for n in preorder]
        sub_values = [list(context.values[n]) for n in preorder]
        illegal_a = [int(zone_a not in context.allowed[n]) for n in preorder]
        illegal_b = [int(zone_b not in context.allowed[n]) for n in preorder]
        for idx in range(count - 1, 0, -1):  # children follow parents in preorder
            p = parent[idx]
            sub_nodes[p] += sub_nodes[idx]
            sub_students[p] += sub_students[idx]
            sub_schools[p] += sub_schools[idx]
            illegal_a[p] += illegal_a[idx]
            illegal_b[p] += illegal_b[idx]
            for k, value in enumerate(sub_values[idx]):
                sub_values[p][k] += value

        # Edges and weight crossing each subtree's boundary, by LCA differences.
        up = [parent]
        for _ in range(1, max(1, count.bit_length())):
            up.append([up[-1][up[-1][idx]] for idx in range(count)])
        crossing = [0] * count
        crossing_cost = [0] * count
        for edge_id in {e for node in union for e in context.incident_edges[node]}:
            u, v = context.edges[edge_id]
            if u not in union or v not in union:
                continue
            iu, iv = local[u], local[v]
            ancestor = _lca(iu, iv, depth, up)
            weight = context.edge_weights[edge_id]
            crossing[iu] += 1
            crossing[iv] += 1
            crossing[ancestor] -= 2
            crossing_cost[iu] += weight
            crossing_cost[iv] += weight
            crossing_cost[ancestor] -= 2 * weight
        for idx in range(count - 1, 0, -1):
            crossing[parent[idx]] += crossing[idx]
            crossing_cost[parent[idx]] += crossing_cost[idx]

        pair = _zone_pair(zone_a, zone_b)
        base_cost = state.boundary_cost - state.boundary_costs.get(pair, 0)
        base_edges = state.cut_edges - state.boundary_pairs.get(pair, 0)
        others = [
            state.violations[k]
            - state.zone_violations[zone_a][k]
            - state.zone_violations[zone_b][k]
            for k in range(context.zone_row_count)
        ]
        total = _ZoneStats(
            sub_nodes[0], sub_students[0], tuple(sub_values[0]), sub_schools[0]
        )
        cuts: list[_Cut] = []

        def add(idx: int, to_a: bool, stats_a: _ZoneStats, stats_b: _ZoneStats) -> None:
            g_a, g_b = context.residuals(stats_a), context.residuals(stats_b)
            v_a, v_b = _clip(g_a), _clip(g_b)
            cut_edges = base_edges + crossing[idx]
            violations = context.total_violations(
                [max(0.0, o + a + b) for o, a, b in zip(others, v_a, v_b)], cut_edges
            )
            cuts.append(
                _Cut(
                    idx,
                    sub_nodes[idx],
                    to_a,
                    stats_a,
                    stats_b,
                    g_a,
                    g_b,
                    v_a,
                    v_b,
                    violations,
                    base_cost + crossing_cost[idx],
                    cut_edges,
                )
            )

        for idx in range(1, count):
            self._check_deadline()
            sub = _ZoneStats(
                sub_nodes[idx],
                sub_students[idx],
                tuple(sub_values[idx]),
                sub_schools[idx],
            )
            rest = _ZoneStats(
                total.node_count - sub.node_count,
                total.students - sub.students,
                tuple(t - s for t, s in zip(total.values, sub.values)),
                total.schools - sub.schools,
            )
            if illegal_a[idx] == 0 and illegal_b[0] == illegal_b[idx]:
                add(idx, True, sub, rest)
            if illegal_b[idx] == 0 and illegal_a[0] == illegal_a[idx]:
                add(idx, False, rest, sub)
        return cuts

    def _check_deadline(self) -> None:
        if self.deadline is None:
            return
        self._deadline_checks += 1
        if self._deadline_checks % 128 == 0 and time.monotonic() >= self.deadline:
            raise _DeadlineReached


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


class _BurstObjective:
    """What feasible partitions are ranked by, as a cost (lower is better).

    The boundary cost is known for every cut, so the cut-edge objective is
    ``local``. Every other objective is scored in one batch per burst on its
    feasible samples only, never inside a proposal.
    """

    def __init__(
        self,
        kind: str,
        scorer: ShortBurstBatchScorer | None = None,
        *,
        maximize: bool = False,
    ) -> None:
        if (scorer is None) != (kind == "cut_edges"):
            raise ValueError("Only the cut_edges objective is scored without a scorer.")
        self.kind = kind
        self.scorer = scorer
        self.sign = -1.0 if maximize else 1.0
        self._values: dict[tuple[int, ...], float] = {}

    @property
    def local(self) -> bool:
        return self.scorer is None

    def prime(
        self, context: _ReComContext, snapshots, base: _Snapshot | None = None
    ) -> None:
        """Score every feasible, not yet scored snapshot in one batch."""
        if self.scorer is None:
            return
        pending = tuple(
            dict.fromkeys(
                s.assignment
                for s in snapshots
                if s.feasible and s.assignment not in self._values
            )
        )
        if not pending:
            return
        values = self.scorer(
            tuple(context.assignment_dict(a) for a in pending),
            None if base is None else context.assignment_dict(base.assignment),
        )
        if len(values) != len(pending):
            raise ValueError(
                "Burst objective scorer returned the wrong number of scores."
            )
        for assignment, value in zip(pending, values, strict=True):
            if not math.isfinite(value):
                raise ValueError("Burst objective scorer returned a non-finite score.")
            self._values[assignment] = float(value)

    def value(self, snapshot: _Snapshot) -> float:
        """The objective in its own units."""
        if self.scorer is None:
            return float(snapshot.boundary_cost)
        return self._values[snapshot.assignment]

    def cost(self, snapshot: _Snapshot) -> float:
        return self.sign * self.value(snapshot)

    def details(self, context: _ReComContext, snapshot: _Snapshot) -> dict[str, object]:
        describe = getattr(self.scorer, "describe", None)
        return (
            {}
            if describe is None
            else dict(describe(context.assignment_dict(snapshot.assignment)))
        )

    def close(self) -> None:
        close = getattr(self.scorer, "close", None)
        if close is not None:
            close()


def _better(sample: _Snapshot, incumbent: _Snapshot | None, cost) -> bool:
    """Deb's rules: feasible beats infeasible, then cost, else lower merit."""
    if incumbent is None:
        return True
    if sample.feasible != incumbent.feasible:
        return sample.feasible
    if sample.feasible:
        return cost(sample) < cost(incumbent)
    return sample.merit < incumbent.merit - _EPS


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


@register("recom")
class ReComSolver(Solver):
    """ReCom walks; ``ReComSettings`` (``recom_*`` options) pick the method."""

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        """Rank feasible partitions by ``recom_objective`` (default cut edges)."""
        kind = str(self.options.get("recom_objective", "cut_edges"))
        if kind not in RECOM_OBJECTIVES:
            raise ValueError(
                f"recom_objective must be one of: {', '.join(RECOM_OBJECTIVES)}."
            )
        return self._run(problem, lambda: _make_objective(problem, kind, self.options))

    def solve_with_scorer(
        self,
        problem: ZoneProblem,
        scorer: ShortBurstBatchScorer,
        *,
        objective_kind: str,
        maximize: bool = True,
    ) -> ZoneSolution:
        """Rank feasible partitions by an externally owned batch scorer."""
        return self._run(
            problem,
            lambda: _BurstObjective(objective_kind, scorer, maximize=maximize),
            close_objective=False,
        )

    def sample_feasible(self, problem: ZoneProblem, visitor) -> ZoneSolution:
        """Walk a rejection chain with the configured tree sampler and visit the
        initial and every accepted feasible partition, improving or not, as a
        fresh assignment dict. Other policy settings use the chain preset.
        A visitor returning False stops it.
        """
        return self._run(
            problem,
            lambda: _BurstObjective("cut_edges"),
            visitor=visitor,
            settings=replace(
                PRESETS["chain"],
                tree_sampler=ReComSettings.from_options(self.options).tree_sampler,
            ),
        )

    def _run(
        self,
        problem: ZoneProblem,
        objective_factory,
        *,
        close_objective=True,
        visitor=None,
        settings: ReComSettings | None = None,
    ) -> ZoneSolution:
        if problem.choice_objective is not None:
            raise NotImplementedError(
                "recom does not support iterative choice objectives; use cp_int, cp_bool, or mip."
            )
        settings = settings or ReComSettings.from_options(self.options)
        start = time.monotonic()
        try:
            context, state, hint_metadata = self._initial_state(problem)
        except _HintError as exc:
            return self._error_solution(problem, start, str(exc), exc.metadata)
        # Built inside the budget: a choice objective's market is part of
        # what the solve costs under wall-clock accounting.
        objective = objective_factory()
        try:
            return self._walk(
                problem,
                context,
                state,
                start,
                settings,
                objective,
                visitor,
                hint_metadata,
            )
        finally:
            if close_objective:
                objective.close()

    def _walk(
        self,
        problem: ZoneProblem,
        context: _ReComContext,
        current: _State,
        start: float,
        settings: ReComSettings,
        objective: _BurstObjective,
        visitor,
        hint_metadata: dict[str, object],
    ) -> ZoneSolution:
        max_moves, deadline = self._limits(start)
        rng = random.Random(int(self.options.get("seed", 42)))
        kernel = _ReComKernel(context, rng, deadline, tree_sampler=settings.tree_sampler)
        policy = Policy(context, rng, settings, objective_local=objective.local)
        log = self._open_log(problem, context, start, settings)
        tally = _Tally()
        initial = _snapshot(current, policy.merit(current))
        objective.prime(context, (initial,))
        best = initial if initial.feasible else None
        first_feasible = 0.0 if initial.feasible else None
        min_violation = sum(initial.violations)
        stop_reason = "iteration_limit"

        def budget() -> bool:
            if max_moves is not None and tally.attempted_moves >= max_moves:
                return False
            return deadline is None or time.monotonic() < deadline

        def visit(snapshot: _Snapshot) -> bool:
            return (
                visitor is None
                or not snapshot.feasible
                or visitor(context.assignment_dict(snapshot.assignment)) is not False
            )

        if not visit(initial):
            stop_reason = "visitor_stop"
        restarting = settings.burst_length > 0
        while stop_reason == "iteration_limit" and budget():
            base = _snapshot(current, policy.merit(current))
            walk = current.clone() if restarting else current
            walk_merit = policy.walk_merit(walk, base.merit)
            samples: list[_Snapshot] = []
            for _ in range(settings.burst_length or _CHAIN_BLOCK):
                if not budget():
                    break
                tally.attempted_moves += 1
                try:
                    move = kernel.propose(walk, policy)
                except _DeadlineReached:
                    break
                except _NoProposal as exc:
                    tally.proposal_failures += 1
                    if exc.reason == "no_adjacent_zone_pairs":
                        stop_reason = exc.reason
                        break
                    continue
                if settings.reject_infeasible and not move.cut.feasible:
                    tally.rejected_moves += 1
                    policy.moved(walk, walk_merit, walk_merit)
                    continue
                kernel.apply(walk, move)
                tally.accepted_moves += 1
                sample = _snapshot(walk, policy.merit(walk))
                after = policy.walk_merit(walk, sample.merit)
                policy.moved(walk, walk_merit, after)
                walk_merit = after
                samples.append(sample)
                if log is not None:
                    log.record(
                        feasible=sample.feasible,
                        boundary_cost=sample.boundary_cost,
                        violations=sample.violations,
                        iteration=tally.attempted_moves,
                    )
                min_violation = min(min_violation, sum(sample.violations))
                if sample.feasible and first_feasible is None:
                    first_feasible = time.monotonic() - start
                if not visit(sample):
                    stop_reason = "visitor_stop"
                    break

            objective.prime(context, samples, base)
            for sample in samples:
                if sample.feasible and _better(sample, best, objective.cost):
                    best = sample
            if restarting:
                selected = base
                for sample in samples:
                    if _better(sample, selected, objective.cost):
                        selected = sample
                if selected is not base:
                    current = context.build_state(list(selected.assignment))
                    tally.restarts += 1
            policy.end_burst(current)
            tally.completed_bursts += 1

        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = "time_limit"
        summary = {
            **asdict(tally),
            "stop_reason": stop_reason,
            "feasible_found": best is not None,
            "time_to_first_feasible": first_feasible,
            "min_total_violation": min_violation,
        }
        metadata = {
            **hint_metadata,
            **settings.as_options(),
            **summary,
            "initial_feasible": initial.feasible,
            "violation_labels": context.violation_labels(),
            "tree_sampler": settings.tree_sampler,
            **policy.metadata(),
        }
        if best is not None:
            metadata.update(
                best_boundary_cost=best.boundary_cost, best_cut_edges=best.cut_edges
            )
            metadata.update(objective.details(context, best))
        if initial.feasible:
            metadata["initial_objective"] = objective.value(initial)
        if log is not None:
            log.finish(**summary)
            metadata.update(
                self._solver_log_metadata(log.path, log_format="jsonl"),
                solver_log_improvements=log.improvements,
            )
        return self._result(problem, context, start, best, metadata, objective)

    # -- setup ---------------------------------------------------------

    def _initial_state(
        self, problem: ZoneProblem
    ) -> tuple[_ReComContext, _State, dict[str, object]]:
        context = _ReComContext(problem)
        if problem.hint is not None:
            hint, metadata = (
                problem.hint,
                {"hints": "provided", "hint_source": "problem_hint"},
            )
        else:
            method = normalize_hints(self.options.get("hints", "voronoi"))
            if method == "none":
                raise _HintError("ReCom requires a provided or generated hint.")
            if method == "feasible":
                generated = initial_solution(
                    problem, method, solver_options=self.options
                )
                assert generated is not None
                hint, metadata = (
                    generated.assignment,
                    {**generated.metadata, "hint_source": "generated"},
                )
            else:
                hint, metadata = (
                    voronoi_hint(context),
                    {"hints": "voronoi", "hint_source": "generated"},
                )
        try:
            assignment = context.validate_hint(hint)
        except _HintError as exc:  # add the provenance validate_hint cannot know
            raise _HintError(str(exc), metadata) from exc
        return context, context.build_state(assignment), metadata

    def _limits(self, start: float) -> tuple[int | None, float | None]:
        moves = int(self.options.get("recom_iterations", -1))
        raw = self.options.get("solve_time_limit", 60.0)
        time_limit = None if raw is None else max(0.0, float(raw))
        if moves < 0 and time_limit is None:
            raise ValueError(
                "solve_time_limit must be supplied when recom_iterations is negative."
            )
        return (None if moves < 0 else moves), (
            None if time_limit is None else start + time_limit
        )

    def _open_log(
        self, problem, context, start, settings
    ) -> HeuristicProgressLog | None:
        path = self._next_solver_log_path(problem, suffix=".jsonl")
        if path is None:
            return None
        return HeuristicProgressLog(
            path,
            solver=self.name,
            level=getattr(getattr(problem, "level", None), "name", "unknown_level"),
            labels=context.violation_labels(),
            start=start,
            header={
                "num_zones": problem.Z,
                "weighted_edges": bool(problem.weight_edges),
                "objective_unit": "meter" if problem.weight_edges else "cut_edges",
                "penalty_scale": "percent",
                **settings.as_options(),
            },
        )

    # -- results -------------------------------------------------------

    def _error_solution(self, problem, start, message, metadata=None) -> ZoneSolution:
        return ZoneSolution(
            problem=problem,
            assignment={},
            status="ERROR",
            objective=None,
            wall_time=time.monotonic() - start,
            # Hint provenance first, so the solver and error always win.
            metadata={
                **dict(metadata or {}),
                "solver": self.name,
                "error_message": message,
            },
        )

    def _result(
        self, problem, context, start, best, metadata, objective
    ) -> ZoneSolution:
        if not objective.local:
            metadata["objective_kind"] = objective.kind
        elif problem.weight_edges:
            metadata.update(
                objective_kind="weighted_boundary_length", objective_unit="meter"
            )
        return ZoneSolution(
            problem=problem,
            assignment={} if best is None else context.assignment_dict(best.assignment),
            status="UNKNOWN" if best is None else "FEASIBLE",
            objective=None if best is None else objective.value(best),
            wall_time=time.monotonic() - start,
            metadata={"solver": self.name, **metadata},
        )


def voronoi_hint(context: _ReComContext) -> dict[int, int]:
    """Each node to its nearest allowed centroid, repaired for contiguity."""
    problem = context.problem
    assignment = {
        node: min(
            context.allowed[pos],
            key=lambda z: problem.distance(problem.centroids[z], node),
        )
        for pos, node in enumerate(context.nodes)
    }
    return contiguity.repair(problem.G, assignment, problem.centroids)


def _make_objective(problem: ZoneProblem, kind: str, options) -> _BurstObjective:
    if kind == "cut_edges":
        return _BurstObjective(kind)
    from optimization.objectives import get_objective, heuristic_scorer

    if kind == "school_distance":
        # Unanchored school identities; pipeline dispatch passes anchored costs.
        from optimization.solvers.burst_objectives import school_distance_burst_scorer

        return _BurstObjective(kind, school_distance_burst_scorer(problem))
    return _BurstObjective(
        kind,
        heuristic_scorer(problem, kind, options),
        maximize=get_objective(kind).maximize,
    )


def _zone_pair(zone_a: int, zone_b: int) -> tuple[int, int]:
    return (zone_a, zone_b) if zone_a < zone_b else (zone_b, zone_a)


def _add(values: dict[tuple[int, int], int], pair: tuple[int, int], change) -> None:
    updated = values.get(pair, 0) + change
    if updated > 0:
        values[pair] = updated
    else:
        values.pop(pair, None)


def _lca(a: int, b: int, depth: list[int], up: list[list[int]]) -> int:
    if depth[a] < depth[b]:
        a, b = b, a
    difference, bit = depth[a] - depth[b], 0
    while difference:
        if difference & 1:
            a = up[bit][a]
        difference >>= 1
        bit += 1
    if a == b:
        return a
    for level in range(len(up) - 1, -1, -1):
        if up[level][a] != up[level][b]:
            a, b = up[level][a], up[level][b]
    return up[0][a]
