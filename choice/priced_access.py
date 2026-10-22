"""Congestion-priced access welfare: a capacity-aware zoning surrogate.

The MNL surrogate in :mod:`choice.mnl` scores a zoning by the schools it makes
available and ignores capacity entirely, so it happily rewards a zone built
around one oversubscribed school. This module scores the same zoning after
charging each program a congestion price.

For any non-negative price vector ``beta``, dropping stability and
Lagrangian-relaxing the capacity constraints of the assignment LP gives

    W(x) <= sum_p c_p beta_p + sum_i max(0, max_{p in zone(i)} (u_ip - beta_p))

for *every* zoning ``x`` and every tie-breaking scenario -- a valid upper bound
on stable-matching welfare, unlike the MNL logsum, which is neither an upper nor
a lower bound. The leading term does not depend on ``x``, so it shifts the
objective without changing the argmax; :attr:`PricedAccessUtility.price_constant`
carries it for reporting.

What makes the bound usable as a *master objective* is that the second term is
separable across students, and each student's term depends only on that
student's own access row. So it admits an exact, finite, closed-form linear
description -- no LP per candidate:

    theta_i <= v_it + sum_{q: v_iq > v_it} (v_iq - v_it) a_iq     for every option t

where ``v_ip = u_ip - beta_p``. Taking ``t`` to be the student's best currently
accessible option makes the cut tight at the incumbent; taking successively
worse ``t`` prices what the student loses when that option is taken away, which
is the direction a cut built only at the incumbent is blind to.

Students sharing a graph node share an access row, so their cuts sum into one
cut per node -- the granularity the master actually reasons about.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from choice.objective import ChoiceCut, ChoiceEvaluation, ChoiceTerm
from optimization.data.mid import MidProgram, MidStudent
from optimization.data.saa import SaaMarket
from optimization.problem import ZoneProblem
from optimization.saa_oracle import access_state


def transport_prices(
    programs: Sequence[MidProgram], students: Sequence[MidStudent]
) -> tuple[dict[str, float], float]:
    """Capacity prices from the zone-blind transportation LP.

    Returns ``(prices, objective)``. The duals of

        max  sum_ip u_ip y_ip
        s.t. sum_p y_ip <= 1,  sum_i y_ip <= c_p,  0 <= y <= 1

    are the natural prices to charge: they are exactly the marginal value of a
    seat when zones are ignored, and at ``a == 1`` the resulting bound collapses
    back to :func:`optimization.welfare_bounds.transport_upper_bound`. They are
    computed once, offline -- re-optimising them per candidate would reproduce
    the zone-restricted LP value exactly, but costs an LP per iteration.
    """
    import gurobipy as gp
    from gurobipy import GRB

    capacity = {program.program_id: program.capacity for program in programs}
    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as model:
        model.ModelSense = GRB.MAXIMIZE
        seats: dict[str, list] = {program_id: [] for program_id in capacity}
        for student in students:
            share = []
            for program_id, utility in zip(student.programs, student.utilities):
                if program_id not in capacity:
                    raise ValueError(f"Unknown program {program_id!r} in preferences.")
                variable = model.addVar(lb=0.0, ub=1.0, obj=float(utility))
                share.append(variable)
                seats[program_id].append(variable)
            if share:
                model.addConstr(gp.quicksum(share) <= 1.0)
        constraints = {}
        for program_id, variables in seats.items():
            if variables:
                constraints[program_id] = model.addConstr(
                    gp.quicksum(variables) <= float(capacity[program_id])
                )
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Transport price LP did not solve (status {model.Status})."
            )
        prices = {
            program_id: max(0.0, constraint.Pi)
            for program_id, constraint in constraints.items()
        }
        return prices, float(model.ObjVal)


@dataclass(frozen=True)
class _Option:
    """One priced option for one student."""

    value: float
    # ``None`` for options no zoning can take away (citywide programs, and
    # schools sited at the student's own node).
    school_node: int | None


class PricedAccessUtility:
    """Evaluate the priced-access surrogate and build per-node cuts.

    ``cut_levels`` controls how many thresholds each student contributes per
    evaluation. Level 0 is tight at the incumbent; level ``k`` prices the loss of
    the student's ``k`` best accessible options, which is what stops the master
    from revoking access it believes to be free.
    """

    def __init__(
        self,
        market: SaaMarket,
        problem: ZoneProblem,
        prices: dict[str, float],
        *,
        cut_levels: int = 3,
    ) -> None:
        if cut_levels <= 0:
            raise ValueError("priced-access cut_levels must be positive.")
        self.market = market
        self.prices = dict(prices)
        self.cut_levels = int(cut_levels)
        self.price_constant = sum(
            program.capacity * max(0.0, self.prices.get(program.program_id, 0.0))
            for program in market.programs
        )

        # Per student: their node, and their options sorted by priced value.
        # Options priced at or below the outside option are dropped -- they can
        # never be the best accessible one, so they carry no information.
        self._student_nodes: list[int] = []
        self._student_options: list[tuple[_Option, ...]] = []
        # Value no zoning can take away, so no threshold may drop below it.
        self._student_floors: list[float] = []
        modelled = set(problem.nodes)
        for student in market.students:
            self._require_modelled(student.node, modelled)
            options: list[_Option] = []
            for program_id, utility in zip(student.programs, student.utilities):
                program = market.program_by_id[program_id]
                value = float(utility) - max(0.0, self.prices.get(program_id, 0.0))
                if value <= 0.0:
                    continue
                pair, fixed = access_state(problem, student.node, program)
                if pair is None:
                    if fixed:
                        options.append(_Option(value=value, school_node=None))
                    continue
                options.append(_Option(value=value, school_node=pair[1]))
            options.sort(key=lambda option: -option.value)
            self._student_nodes.append(student.node)
            self._student_options.append(tuple(options))
            self._student_floors.append(
                max(
                    (option.value for option in options if option.school_node is None),
                    default=0.0,
                )
            )

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, problem: ZoneProblem, assignment: dict[int, int]) -> float:
        """Total priced-access utility, excluding :attr:`price_constant`."""
        total = 0.0
        for index in range(len(self._student_options)):
            accessible = self._accessible(index, assignment)
            total += accessible[0].value if accessible else 0.0
        return total

    def initial_cuts(self, problem: ZoneProblem) -> tuple[ChoiceCut, ...]:
        """Zoning-independent cuts, one per node, for the first master solve.

        The solver gives every node its own utility variable bounded only by
        :meth:`node_utility_bounds`, so a node with no cut floats at the loosest
        node's bound and the first master objective is meaningless. These cuts
        take each student's threshold to be their floor -- the value no zoning
        can revoke -- which is valid without knowing any zoning at all.

        A node with no students contributes exactly zero welfare under every
        zoning, and gets ``constant == 0`` with no terms. Together with the zero
        lower bound from :meth:`node_utility_bounds` that pins its variable to
        precisely 0 rather than merely bounding it.
        """
        by_node: dict[int, tuple[float, dict[int, float]]] = {
            node: (0.0, {}) for node in problem.nodes
        }
        for index, options in enumerate(self._student_options):
            node = self._student_nodes[index]
            self._require_modelled(node, by_node)
            threshold = self._student_floors[index]
            constant, coefficients = by_node[node]
            self._accumulate(options, threshold, coefficients)
            by_node[node] = (constant + threshold, coefficients)
        return tuple(
            self._make_cut(node, constant, coefficients)
            for node, (constant, coefficients) in sorted(by_node.items())
        )

    def evaluate_with_cuts(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> ChoiceEvaluation:
        """Utility at ``assignment`` plus one cut per node per threshold level.

        Every cut is valid for every zoning, and the level-0 cut is tight at
        ``assignment``, so the master's model utility meets the evaluated utility
        exactly once no better zoning is left to find. Only nodes with students
        are cut here; the rest are pinned once by :meth:`initial_cuts`.
        """
        total = 0.0
        # level -> node -> (constant, {school_node: coefficient})
        levels: list[dict[int, tuple[float, dict[int, float]]]] = [
            {} for _ in range(self.cut_levels)
        ]
        modelled = set(problem.nodes)
        for index, options in enumerate(self._student_options):
            node = self._student_nodes[index]
            self._require_modelled(node, modelled)
            accessible = self._accessible(index, assignment)
            total += accessible[0].value if accessible else 0.0
            for level in range(self.cut_levels):
                # Threshold: the student's (level+1)-th best accessible option,
                # or the outside option once they run out. Level 0 reproduces
                # the incumbent value. Clamped at the floor, because a threshold
                # below an option no zoning can revoke would make the cut cut
                # off welfare the student is guaranteed to keep.
                threshold = max(
                    self._student_floors[index],
                    accessible[level].value if level < len(accessible) else 0.0,
                )
                constant, coefficients = levels[level].setdefault(node, (0.0, {}))
                self._accumulate(options, threshold, coefficients)
                levels[level][node] = (constant + threshold, coefficients)

        cuts: list[ChoiceCut] = []
        for level in levels:
            for node, (constant, coefficients) in sorted(level.items()):
                cuts.append(self._make_cut(node, constant, coefficients))
        return ChoiceEvaluation(utility=total, cuts=tuple(cuts))

    def node_utility_bounds(self, problem: ZoneProblem) -> tuple[float, float]:
        """Per-node bounds on the priced utility variable.

        The upper bound is the node's students all taking their best option with
        no capacity contention, which is the loosest any zoning can be.
        """
        best: dict[int, float] = {}
        for index, options in enumerate(self._student_options):
            node = self._student_nodes[index]
            best[node] = best.get(node, 0.0) + (options[0].value if options else 0.0)
        return 0.0, max(best.values(), default=0.0)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_modelled(node: int, modelled) -> None:
        """Fail loudly if a student sits on a node the master does not model.

        Silently dropping the student would leave the evaluated utility and the
        cuts describing different populations, which quietly breaks validity.
        """
        if node not in modelled:
            raise ValueError(
                f"Student node {node} is absent from the zoning problem, so its "
                "utility could not be cut. The market and the problem must come "
                "from the same level."
            )

    @staticmethod
    def _accumulate(
        options: tuple[_Option, ...],
        threshold: float,
        coefficients: dict[int, float],
    ) -> None:
        """Add one student's cut coefficients at ``threshold`` into ``coefficients``."""
        for option in options:
            if option.school_node is None or option.value <= threshold:
                continue
            coefficients[option.school_node] = coefficients.get(
                option.school_node, 0.0
            ) + (option.value - threshold)

    @staticmethod
    def _make_cut(
        node: int, constant: float, coefficients: dict[int, float]
    ) -> ChoiceCut:
        return ChoiceCut(
            node=node,
            constant=constant,
            terms=tuple(
                ChoiceTerm(coefficient=coefficient, node=school_node)
                for school_node, coefficient in sorted(coefficients.items())
                if abs(coefficient) > 1e-12
            ),
        )

    def _accessible(
        self, index: int, assignment: dict[int, int]
    ) -> tuple[_Option, ...]:
        """Student ``index``'s options under ``assignment``, best first."""
        node = self._student_nodes[index]
        zone = assignment.get(node)
        return tuple(
            option
            for option in self._student_options[index]
            if option.school_node is None
            or (zone is not None and assignment.get(option.school_node) == zone)
        )
