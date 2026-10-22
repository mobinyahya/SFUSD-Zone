"""The admissible single-zone families, written once and shared.

Dantzig--Wolfe needs one description of "which sets are legal zones for label
``z``" and it needs the *same* one in three places: the seeding filter, the
column pool's validator, and the pricing model. When those disagree the method
breaks silently in both directions -- a pool that admits a zone the pricer
cannot see makes the pricing-corrected bound invalid, and a pricer that returns
a zone the pool rejects wastes the round.

This module is that description. :class:`ZoneFamily` holds, per label, the
candidate node set, the forced memberships, the closer-neighbour support lists,
and the balance rows -- and the balance rows are the *integer* rows
``cp_bool`` writes, not float rows with a feasibility slack. Two consequences
worth stating:

* ``ZoneFamily.feasible`` and the pricing model test literally the same
  inequalities over the same integers, so no tolerance is needed anywhere and
  none is used.
* the family is exactly the projection of the base zoning model onto one label.
  Every row of that model involves a single label except the assignment row
  (which becomes the master's cover row) and the boundary cap (which becomes
  the master's boundary row), so a tuple of sets, one per label, is admissible
  for the base model precisely when each set lies in its own
  :class:`ZoneFamily` and the sets partition ``V``. That is the statement
  Appendix D of the paper calls exact decomposition, and it is why the
  decomposition now describes the same feasible set as ``cp_bool``.

Geometry follows ``cp_bool``/``cp_single_zone``: anchored centroids and the
closer-neighbour contiguity relation, not an unanchored rooted flow. A
non-centroid member needs a same-zone neighbour whose geometry is strictly
closer to the label's school point, so every member has a distance-decreasing
path to the anchor. That implies connectedness and is strictly stronger than
it: a zone that wraps around and re-approaches its anchor from the far side is
connected and is not admissible. The gain is that connectedness costs
``|A|`` clauses rather than a rooted single-commodity flow, and that the
family is anchored, so pricing searches near one centroid rather than over
every connected subset of the graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from optimization.data import contiguity
from optimization.problem import ZoneProblem
from optimization.solvers.balance import balance_constraints
from optimization.solvers.base import centroid_neighborhoods
from optimization.solvers.cpsat import scaled_row


@dataclass(frozen=True)
class IntegerRow:
    """One label's balance or school-count row, in integer arithmetic.

    ``cp_bool`` scales every float coefficient by ``CP_SAT_SCALE`` and rounds
    it away from feasibility, so that is the row this repo actually solves.
    Reading it from the same :func:`~optimization.solvers.cpsat.scaled_row`
    -- rather than re-deriving the float row and allowing a slack -- is what
    makes the family's membership test, the pricing model and ``cp_bool``
    agree exactly.
    """

    kind: str
    sense: str
    rhs: int
    coefficients: dict[int, int]

    def value(self, nodes) -> int:
        return sum(
            coefficient
            for node, coefficient in self.coefficients.items()
            if node in nodes
        )

    def satisfied(self, nodes) -> bool:
        total = self.value(nodes)
        return total >= self.rhs if self.sense == ">=" else total <= self.rhs


def boundary_limit(problem: ZoneProblem) -> int:
    """The district cut-edge budget, ``floor(boundary_prop * |E|)``.

    Unweighted whatever ``weight_edges`` says, as every other backend and
    :func:`~optimization.data.feasibility.check_zoning` count it: edge weights
    enter the boundary *objective* (:func:`perimeter`), never the cap. ``b(A)``
    in the master's boundary row is half of :func:`cut_edges`, so summing it
    over a partition counts each cut edge once.
    """

    return math.floor(problem.boundary_prop * problem.G.number_of_edges())


def cut_edges(problem: ZoneProblem, nodes) -> int:
    """How many edges leave ``nodes``, unweighted -- what the cap counts."""

    return sum(1 for u, v in problem.G.edges if (u in nodes) != (v in nodes))


def perimeter(problem: ZoneProblem, nodes) -> int:
    """The cut weight leaving ``nodes``, which the boundary objective prices."""

    return sum(
        problem.boundary_weight(u, v)
        for u, v in problem.G.edges
        if (u in nodes) != (v in nodes)
    )


@dataclass(frozen=True)
class ZoneFamily:
    """``F_z`` for every label, as data the pricer and the pool both read."""

    problem: ZoneProblem
    centroid_neighbor_radius: int
    candidates: tuple[frozenset[int], ...]
    forced: tuple[frozenset[int], ...]
    supports: tuple[dict[int, tuple[int, ...]], ...]
    rows: tuple[tuple[IntegerRow, ...], ...]

    @property
    def Z(self) -> int:
        return len(self.candidates)

    def centroid(self, zone: int) -> int:
        return self.problem.centroids[zone]

    def feasible(self, zone: int, nodes) -> bool:
        """Is ``nodes`` an admissible zone for ``zone``?

        Exactly the rows the pricing model carries, in the same arithmetic.
        """

        if not 0 <= zone < self.Z or not nodes:
            return False
        nodes = frozenset(nodes)
        if not nodes <= self.candidates[zone]:
            return False
        if not self.forced[zone] <= nodes:
            return False
        anchor = self.centroid(zone)
        supports = self.supports[zone]
        for node in nodes:
            if node == anchor:
                continue
            support = supports.get(node)
            if not support or nodes.isdisjoint(support):
                return False
        return all(row.satisfied(nodes) for row in self.rows[zone])

    def metadata(self) -> dict:
        return {
            "dw_zone_family": "anchored_closer_neighbor",
            "dw_centroid_neighbor_radius": self.centroid_neighbor_radius,
            "dw_candidate_nodes": [len(nodes) for nodes in self.candidates],
            "dw_forced_nodes": [len(nodes) for nodes in self.forced],
            "dw_family_rows": [len(rows) for rows in self.rows],
        }


def build_zone_family(
    problem: ZoneProblem, *, centroid_neighbor_radius: int = 0
) -> ZoneFamily:
    """Derive every label's family from ``problem``.

    The candidate sets are pruned to a fixed point. A non-anchor node whose
    support list is empty can never be a member, which can empty another
    node's support list, and so on. ``cp_bool`` reaches the same fixings by
    propagation -- its variables exist and are pinned to zero -- so the pruning
    changes no feasible set; it just makes the family's own description closed
    under the implication, which is what lets the membership test be a single
    pass.
    """

    neighborhoods = centroid_neighborhoods(problem, centroid_neighbor_radius)
    nodes = list(problem.nodes)
    supports = contiguity.contiguity_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )

    candidates: list[frozenset[int]] = []
    forced: list[frozenset[int]] = []
    pruned_supports: list[dict[int, tuple[int, ...]]] = []
    for zone in range(problem.Z):
        anchor = problem.centroids[zone]
        excluded = {
            node
            for other, ball in neighborhoods.items()
            if other != zone
            for node in ball
        }
        live = {
            node
            for node in nodes
            if zone in problem.candidate_zones(node) and node not in excluded
        }
        # Fixed point of "a member needs a live closer support".
        changed = True
        while changed:
            changed = False
            for node in sorted(live):
                if node == anchor:
                    continue
                if not any(
                    support in live for support in supports.get((node, zone), ())
                ):
                    live.discard(node)
                    changed = True
        required = {node for node in neighborhoods[zone] if node in live}
        required.add(anchor)
        if problem.fixed:
            required.update(
                node for node, value in problem.fixed.items() if value == zone
            )
        candidates.append(frozenset(live))
        forced.append(frozenset(required))
        pruned_supports.append(
            {
                node: tuple(
                    support
                    for support in supports.get((node, zone), ())
                    if support in live
                )
                for node in sorted(live)
                if node != anchor
            }
        )

    return ZoneFamily(
        problem=problem,
        centroid_neighbor_radius=centroid_neighbor_radius,
        candidates=tuple(candidates),
        forced=tuple(forced),
        supports=tuple(pruned_supports),
        rows=tuple(
            _integer_rows(problem, candidates[zone]) for zone in range(problem.Z)
        ),
    )


def _integer_rows(problem: ZoneProblem, nodes) -> tuple[IntegerRow, ...]:
    """The label's balance and school-count rows, scaled exactly as CP-SAT does."""

    nodes = sorted(nodes)
    rows: list[IntegerRow] = []

    def add(kind, sense, values, rhs):
        coefficients, bound = scaled_row(values, sense, rhs)
        rows.append(IntegerRow(kind, sense, bound, dict(zip(nodes, coefficients))))

    for constraint in balance_constraints(problem):
        for sense, bound, ratio in (
            (">=", "lower", constraint.lower_ratio),
            ("<=", "upper", constraint.upper_ratio),
        ):
            if ratio is None:
                continue
            values = [
                constraint.value(node) - ratio * problem.students(node)
                for node in nodes
            ]
            add(f"{constraint.kind}_{bound}", sense, values, 0.0)
    bounds = problem.school_count_bounds()
    if bounds is not None:
        floor_, ceiling = bounds
        values = [float(problem.num_schools(node)) for node in nodes]
        add("schools_lower", ">=", values, floor_)
        add("schools_upper", "<=", values, ceiling)
    return tuple(rows)
