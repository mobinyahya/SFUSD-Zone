"""Redraw two zones at once: an extra column source, never a bound.

What this is for
----------------
Pricing asks a question about one label at a time, and the answer is a bound.
That is what Proposition 9 needs and it is why the pricer must range over all
of ``F_z``. But it is also why the pool stopped growing in any useful
direction. Measured on ``Block_2`` (501 units, ``Z = 6``, 4,154 applicants)
seeded with one ``cp_bool`` feasibility hint: 21 column-generation rounds
admitted 542 to 555 zones of strictly positive reduced cost in 600s, and the
restricted LP sat at the hint value to four decimal places on every single
round, with the resulting pool still admitting **exactly one tiling**. The
same thing on a synthetic instance, where it was first found, is confirmed
more sharply still -- deleting the ``Z`` seed columns and re-solving the
integer master returns INFEASIBLE. Z labels priced independently are under no
pressure to be mutually complementary, so nothing in that loop ever
manufactures a second way to cover ``V``.

This module manufactures them. Fix an admissible partition
``(A_1, ..., A_Z)``, choose two labels ``a, b``, and re-partition their joint
territory ``U = A_a u A_b`` optimally, leaving the other ``Z - 2`` zones alone.
Three facts make that the right move:

* **Every feasible solution is a tiling.** ``A'_a`` and ``A'_b`` partition
  ``U`` by construction and the untouched zones partition ``V \\ U``, so any
  solution of the pair model -- not just the optimum, and not just an
  improving one -- is a complete admissible partition of ``V``. Each one
  contributes two columns that *provably* complete the ``Z - 2`` columns
  already in the pool. After ``k`` successful redraws the pool holds at least
  ``k + 1`` tilings, which is precisely the dimension the master was missing.
* **No duals are needed.** Because the two new zones cover ``U`` exactly once,
  ``sum_{v in U} pi_v`` is the same number for every feasible solution, so it
  cannot influence which solution is best. The pair's summed reduced cost and
  its summed welfare differ by a constant, and maximizing welfare maximizes
  both. The redraw therefore runs with no master, no LP and no dual point: it
  is simultaneously an exact primal local search on the map and the
  reduced-cost-maximizing pair split. The boundary cap is handled exactly, as
  a hard row, rather than priced (see below).
* **A pair's territory is invariant under its own redraw.** ``A'_a u A'_b = U``,
  so once a pair is solved to optimality it stays optimal until some *other*
  pair moves a block into or out of one of its two zones. Memoizing on
  ``(pair, U)`` therefore gives exact "don't look" bookkeeping for free: a
  sweep that reaches a state where every adjacent pair is solved has produced
  a tiling that no two-zone move can improve.

This is ReCom's move with the guesswork removed -- the same neighbourhood, but
the split is chosen by CP-SAT under the real welfare objective instead of by a
spanning-tree cut, and admissibility is a constraint rather than a rejection
test.

What it deliberately is not
---------------------------
It is **not** a pricing problem and it produces no ``R_z``. Its search space
is ``{A in F_a : A subset U}``, a subset of ``F_a``, so its optimum is a lower
bound on the label's true pricing optimum, not an upper bound; and it couples
two labels, so it does not answer the per-label question at all. Proposition 9
and the finite-convergence argument of Theorem 3 consume the single-zone
pricer's bounds and nothing else. The redraw is therefore strictly
*additive*: it supplies columns and incumbents alongside pricing, which can
only enlarge the master's feasible set and tighten the LP, and every column it
offers is re-validated by ``ZoneFamily`` through :meth:`ZonePool.admit` before
entering. If it ever replaced pricing, both results would be lost.

Structure of the model
----------------------
One Boolean per block per admissible label over ``U``, tied by an exactly-one
row -- which presolve immediately substitutes into a *single* Boolean per
block wherever both labels are candidates, so the solver sees the same
one-variable-per-block zoning model the single-zone pricer does. Contiguity,
balance and anchoring are then :mod:`optimization.zone_family`'s rows applied
once per label, and welfare is one block per label, built by the same
functions :mod:`optimization.zone_pricing` uses:

* ``stable_matching`` keeps its all-Boolean core exactly. The block is per
  label either way, so the seat, clearing and prefix variables stay Boolean
  and the drawn-order chain still encodes the cutoff with no cutoff variable.
* ``mid`` keeps its small-domain integers (cutoffs, thresholds, the mass
  recurrence), again per label.
* The access conjunction is where two labels cost more than one: it becomes
  ``x_{i,z} AND x_{l(s),z}`` for each of the two labels rather than a single
  conjunction, which is the halfway point between the single-zone form and the
  ``Z``-joint form a whole-district master would need. Against that, ``U`` is
  two zones' worth of blocks rather than a label's whole candidate set, which
  on the measured instance is 1.5 to 2.5 times smaller.

The incumbent split is supplied as a CP-SAT hint, so an interrupted solve
still returns at least the incumbent and the harvest is never empty.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from optimization.zone_family import boundary_limit
from optimization.zone_pricing import (
    add_cut_indicators,
    add_welfare_block,
    compatible,
    solver_status,
    welfare_allowance_units,
)


@dataclass(frozen=True)
class RedrawResult:
    """What one pair's redraw proved, and the tilings it passed through.

    ``incumbent_score`` and ``score`` are the pair's *exact* welfare -- the
    welfare oracle's, via :meth:`ZonePool.admit` -- before and after, so
    ``score - incumbent_score`` is the improvement the whole partition gains.
    ``bound`` is CP-SAT's proven optimum for the pair over ``U``, in the same
    exact units up to ``allowance``; it bounds this *neighbourhood* and says
    nothing about the label's pricing problem.

    ``splits`` holds every admissible pair split the search passed through,
    best first. Each is a pair of columns that completes the incumbent's other
    ``Z - 2`` columns into a tiling.
    """

    status: str
    pair: tuple[int, int]
    incumbent_score: float
    score: float | None = None
    bound: float | None = None
    columns: tuple = ()
    splits: tuple[tuple, ...] = ()
    allowance: float = 0.0
    territory: int = 0
    wall_time: float = 0.0
    tolerance: float = 1e-9

    @property
    def improved(self) -> bool:
        return (
            self.score is not None
            and self.score > self.incumbent_score + self.tolerance
        )

    @property
    def gain(self) -> float:
        return 0.0 if self.score is None else self.score - self.incumbent_score


@dataclass
class _PairModel:
    """One pair's model plus the handles needed to read a solution back."""

    model: cp_model.CpModel
    territory: frozenset[int]
    members: tuple = ()
    allowance_units: int = 0
    stats: dict = field(default_factory=dict)
    infeasible: bool = False


class _Splits(cp_model.CpSolverSolutionCallback):
    """Record both labels' membership at every solution the search passes."""

    def __init__(self, members, limit):
        super().__init__()
        self._members = tuple(members)
        self._labels = tuple(sorted({label for label, _, _ in self._members}))
        self._limit = int(limit)
        self.found: list[tuple[float, dict[int, frozenset[int]]]] = []

    def on_solution_callback(self) -> None:
        selected: dict[int, set[int]] = {label: set() for label in self._labels}
        for label, node, var in self._members:
            if self.Value(var) == 1:
                selected[label].add(node)
        self.found.append(
            (
                self.ObjectiveValue(),
                {label: frozenset(nodes) for label, nodes in selected.items()},
            )
        )
        if len(self.found) > self._limit:
            # Keep the best ``limit`` seen so far. Every split is re-validated
            # by the pool, so this caps work rather than correctness.
            self.found.sort(key=lambda item: -item[0])
            del self.found[self._limit :]

    def best_first(self) -> tuple[tuple[float, dict[int, frozenset[int]]], ...]:
        seen: set = set()
        ordered = []
        for value, split in sorted(self.found, key=lambda item: -item[0]):
            key = tuple(sorted((label, split[label]) for label in split))
            if key in seen:
                continue
            seen.add(key)
            ordered.append((value, split))
        return tuple(ordered)


class ZoneRedraw:
    """Optimal two-zone re-partitions of an admissible partition."""

    def __init__(
        self,
        pool,
        *,
        workers=0,
        seed=42,
        scale=1000,
        splits_per_call=8,
        non_wastefulness=True,
        aggregate_stability=True,
        tolerance=1e-9,
    ):
        if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
            raise ValueError("dw_pricing_scale must be a positive integer.")
        if (
            isinstance(splits_per_call, bool)
            or not isinstance(splits_per_call, int)
            or splits_per_call <= 0
        ):
            raise ValueError("dw_redraw_splits_per_call must be a positive integer.")
        self.pool = pool
        self.workers = max(0, int(workers))
        self.seed = int(seed)
        self.scale = int(scale)
        self.splits_per_call = int(splits_per_call)
        self.non_wastefulness = bool(non_wastefulness)
        self.aggregate_stability = bool(aggregate_stability)
        self.tolerance = float(tolerance)
        # ``(pair, territory, branch fixings)`` proved optimal. The territory
        # is in the key because a pair's own redraw does not change it, so an
        # entry expires exactly when another pair moves a block across one of
        # this pair's two zones.
        self._exhausted: set = set()
        # How often each key has been attempted without being proved. A pair
        # too big to finish inside one sweep's budget must not monopolize
        # every later sweep, so attempts order the queue and unattempted pairs
        # go first.
        self._attempts: dict = {}
        self.pairs_solved = 0
        self.improvements = 0
        self.columns_added = 0
        self.welfare_gain = 0.0

    # ------------------------------------------------------------------ #
    # Neighbourhood
    # ------------------------------------------------------------------ #
    @property
    def objective_scale(self) -> int:
        """``K``: the integer units the pair objective is measured in."""

        objective = self.pool.objective
        if objective.kind == "mid":
            return self.scale * objective.lottery_scale
        if objective.kind == "boundary":
            return 2 * self.scale
        return self.scale

    def pairs(self, incumbent) -> tuple[tuple[int, int], ...]:
        """Graph-adjacent label pairs, most shared boundary first.

        Non-adjacent pairs are provably no-ops, so they are not enumerated. If
        no edge joins ``A_a`` and ``A_b`` then every ``U``-neighbour of a block
        of ``A_a`` lies in ``A_a``; a block joining ``A'_b`` needs a strictly
        closer support in ``A'_b``, and iterating that requirement produces a
        support chain that must terminate at ``c_b``, which is not in ``A_a``.
        So no block can cross, in either direction, and the incumbent split is
        the pair model's only solution.
        """

        problem = self.pool.problem
        label_of = {node: column.zone for column in incumbent for node in column.nodes}
        shared: dict[tuple[int, int], int] = {}
        for u, v in problem.G.edges:
            first, second = label_of.get(u), label_of.get(v)
            if first is None or second is None or first == second:
                continue
            key = (min(first, second), max(first, second))
            shared[key] = shared.get(key, 0) + 1
        return tuple(sorted(shared, key=lambda key: (-shared[key], key)))

    # ------------------------------------------------------------------ #
    # One pair
    # ------------------------------------------------------------------ #
    def __call__(self, incumbent, pair, *, deadline, decisions=None):
        """Optimally re-partition ``pair``'s joint territory."""

        start = time.monotonic()
        incumbent = tuple(incumbent)
        by_label = {column.zone: column for column in incumbent}
        if len(by_label) != len(incumbent) or set(by_label) != set(
            range(self.pool.problem.Z)
        ):
            raise ValueError("A redraw needs exactly one incumbent column per label.")
        pair = (min(pair), max(pair))
        if pair[0] == pair[1]:
            raise ValueError("A redraw needs two distinct labels.")
        base = sum(by_label[label].score for label in pair)
        territory = by_label[pair[0]].nodes | by_label[pair[1]].nodes
        if deadline - start <= 0:
            return RedrawResult(
                "TIME_LIMIT",
                pair,
                base,
                territory=len(territory),
                tolerance=self.tolerance,
            )

        built = self._build(by_label, pair, dict(decisions or {}))
        if built.infeasible:
            return RedrawResult(
                "INFEASIBLE",
                pair,
                base,
                bound=-math.inf,
                territory=len(built.territory),
                wall_time=time.monotonic() - start,
                tolerance=self.tolerance,
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return RedrawResult(
                "TIME_LIMIT",
                pair,
                base,
                territory=len(built.territory),
                tolerance=self.tolerance,
            )
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = self.seed
        # One pair at a time, so the redraw is welcome to the whole machine.
        solver.parameters.num_search_workers = max(1, self.workers)
        if math.isfinite(remaining):
            solver.parameters.max_time_in_seconds = max(1e-3, remaining)
        harvest = _Splits(built.members, self.splits_per_call)
        status = solver_status(solver.Solve(built.model, harvest))
        return self._read(
            solver, status, harvest, pair, base, built, time.monotonic() - start
        )

    def _read(self, solver, status, harvest, pair, base, built, wall):
        allowance = built.allowance_units / self.objective_scale
        if status == "INFEASIBLE":
            return RedrawResult(
                "INFEASIBLE",
                pair,
                base,
                bound=-math.inf,
                territory=len(built.territory),
                wall_time=wall,
                tolerance=self.tolerance,
            )
        if status == "MODEL_INVALID":
            return RedrawResult(
                "ERROR",
                pair,
                base,
                territory=len(built.territory),
                wall_time=wall,
                tolerance=self.tolerance,
            )
        bound = solver.BestObjectiveBound() / self.objective_scale
        if not math.isfinite(bound):
            bound = math.inf
        splits = []
        for _, membership in harvest.best_first():
            columns = tuple(
                self.pool.admit(label, membership.get(label, frozenset()))
                for label in pair
            )
            if any(column is None for column in columns):
                # A rounding artefact or a family row the split misses: the
                # pool is the authority, so drop it and keep going.
                continue
            covered = frozenset().union(*(column.nodes for column in columns))
            if covered != built.territory or sum(
                len(column.nodes) for column in columns
            ) != len(built.territory):
                continue
            splits.append(columns)
        # The harvest is ordered by the scaled objective; re-order by the
        # oracle's exact score, which is what the master will see.
        splits.sort(key=lambda columns: -sum(column.score for column in columns))
        score = sum(column.score for column in splits[0]) if splits else None
        return RedrawResult(
            status if status in {"OPTIMAL", "FEASIBLE"} else "TIME_LIMIT",
            pair,
            base,
            score,
            bound,
            splits[0] if splits else (),
            tuple(splits),
            allowance,
            len(built.territory),
            wall,
            self.tolerance,
        )

    # ------------------------------------------------------------------ #
    # Model construction
    # ------------------------------------------------------------------ #
    def _build(self, by_label, pair, decisions) -> _PairModel:
        pool = self.pool
        family = pool.family
        problem = pool.problem
        objective = pool.objective
        territory = by_label[pair[0]].nodes | by_label[pair[1]].nodes
        model = cp_model.CpModel()
        built = _PairModel(model, territory)

        # One Boolean per (block, admissible label), tied by exactly-one. That
        # single row is the whole coupling between the two labels, and it is
        # what makes this a re-partition rather than two independent zones;
        # presolve substitutes it away into one Boolean per block.
        x: dict[int, dict[int, cp_model.IntVar]] = {label: {} for label in pair}
        for node in sorted(territory):
            literals = []
            for label in pair:
                if node not in family.candidates[label]:
                    continue
                variable = model.NewBoolVar(f"x{label}_{node}")
                x[label][node] = variable
                literals.append(variable)
            if not literals:
                built.infeasible = True
                return built
            model.AddExactlyOne(literals)

        for (node, label), value in sorted(decisions.items()):
            if label not in pair:
                # A pin into a third label is already honoured: the redraw
                # never moves a block out of the pair's territory, and a
                # compatible incumbent keeps such a block outside it.
                if value and node in territory:
                    built.infeasible = True
                    return built
                continue
            if value and node not in x[label]:
                built.infeasible = True
                return built
            if node in x[label]:
                model.Add(x[label][node] == int(bool(value)))

        for label in pair:
            if not family.forced[label] <= family.candidates[label]:
                built.infeasible = True
                return built
            for node in sorted(family.forced[label]):
                if node not in x[label]:
                    built.infeasible = True
                    return built
                model.Add(x[label][node] == 1)
            anchor = family.centroid(label)
            supports = family.supports[label]
            for node in sorted(x[label]):
                if node == anchor:
                    continue
                support = [
                    x[label][other]
                    for other in supports.get(node, ())
                    if other in x[label]
                ]
                if support:
                    model.AddBoolOr([x[label][node].Not(), *support])
                else:
                    # Nothing inside the territory can support it, and the
                    # zone cannot reach outside the territory.
                    model.Add(x[label][node] == 0)
            for row in family.rows[label]:
                terms = [
                    (node, coefficient)
                    for node, coefficient in sorted(row.coefficients.items())
                    if coefficient and node in x[label]
                ]
                expression = cp_model.LinearExpr.WeightedSum(
                    [x[label][node] for node, _ in terms],
                    [value for _, value in terms],
                )
                if row.sense == ">=":
                    model.Add(expression >= row.rhs)
                else:
                    model.Add(expression <= row.rhs)

        need_cuts = objective.kind == "boundary" or problem.boundary_prop >= 0
        cut_terms: dict[int, tuple] = {}
        welfare = []
        for label in pair:
            tag = f"z{label}_"
            cut_terms[label] = (
                add_cut_indicators(model, problem, x[label], tag=tag)
                if need_cuts
                else ()
            )
            block, stats = add_welfare_block(
                model,
                x[label],
                objective,
                self.scale,
                cut_terms=cut_terms[label],
                non_wastefulness=self.non_wastefulness,
                aggregate_stability=self.aggregate_stability,
                tag=tag,
            )
            welfare.append(block)
            # No duals are charged here, so the only rounding is the welfare
            # block's own ceiling inflation.
            built.allowance_units += welfare_allowance_units(objective, stats)
            built.stats[f"zone_{label}"] = {
                "candidate_nodes": len(x[label]),
                "cut_indicators": len(cut_terms[label]),
                **stats,
            }

        if problem.boundary_prop >= 0:
            # The other Z - 2 zones' perimeters are fixed, and so is the part
            # of this pair's perimeter that faces them, so the district cap
            # becomes an exact row on the pair rather than a priced term.
            others = sum(
                column.perimeter
                for label, column in by_label.items()
                if label not in pair
            )
            budget = 2 * int(boundary_limit(problem)) - others
            if budget < 0:
                built.infeasible = True
                return built
            variables = []
            coefficients = []
            for label in pair:
                for _, weight, variable in cut_terms[label]:
                    variables.append(variable)
                    coefficients.append(int(weight))
            model.Add(
                cp_model.LinearExpr.WeightedSum(variables, coefficients) <= budget
            )

        model.Maximize(sum(welfare))
        for label in pair:
            held = by_label[label].nodes
            for node, variable in sorted(x[label].items()):
                model.AddHint(variable, 1 if node in held else 0)
        built.members = tuple(
            (label, node, variable)
            for label in pair
            for node, variable in sorted(x[label].items())
        )
        built.stats["territory"] = len(territory)
        built.stats["allowance_units"] = built.allowance_units
        return built

    # ------------------------------------------------------------------ #
    # Sweeps
    # ------------------------------------------------------------------ #
    def sweep(
        self,
        incumbent,
        *,
        deadline,
        decisions=None,
        offer=None,
        pair_time_limit=math.inf,
    ):
        """Redraw every adjacent pair until no two-zone move improves.

        Returns the improved partition and a metadata dictionary. Every split
        the solver passed through is added to the pool, and every one of them
        -- improving or not -- is handed to ``offer`` as a complete partition,
        because that is what the pair model's feasible solutions are.
        """

        current = tuple(incumbent)
        records: list[dict] = []
        added = improvements = solved = passes = 0
        gain = 0.0
        start = time.monotonic()
        blocked = None
        if len(current) != self.pool.problem.Z or self.pool.problem.Z < 2:
            blocked = "no_incumbent"
        elif decisions and not all(compatible(column, decisions) for column in current):
            blocked = "incumbent_incompatible"
        if blocked is not None:
            return current, self._metadata(
                records, solved, improvements, added, gain, passes, start, blocked
            )

        while time.monotonic() < deadline:
            passes += 1
            progress = False
            pending = [
                pair
                for pair in self.pairs(current)
                if self._key(current, pair, decisions) not in self._exhausted
            ]
            # Stable, so the shared-boundary order survives inside a tier.
            pending.sort(
                key=lambda pair: self._attempts.get(
                    self._key(current, pair, decisions), 0
                )
            )
            for index, pair in enumerate(pending):
                now = time.monotonic()
                if now >= deadline:
                    break
                frozen = current
                key = self._key(frozen, pair, decisions)
                if key in self._exhausted:
                    # An earlier improvement in this pass already settled it.
                    continue
                # Each still-pending pair gets a share of what is left, so one
                # expensive pair cannot consume the sweep and starve the rest.
                share = (deadline - now) / (len(pending) - index)
                self._attempts[key] = self._attempts.get(key, 0) + 1
                result = self(
                    frozen,
                    pair,
                    deadline=min(deadline, now + min(pair_time_limit, share)),
                    decisions=decisions,
                )
                solved += 1
                self.pairs_solved += 1
                for columns in result.splits:
                    replacement = {column.zone: column for column in columns}
                    for column in columns:
                        if self.pool.add(column):
                            added += 1
                            self.columns_added += 1
                    if offer is not None:
                        offer(
                            tuple(
                                replacement.get(column.zone, column)
                                for column in frozen
                            )
                        )
                if result.status == "OPTIMAL":
                    # The pair's territory is unchanged by its own redraw, so
                    # this entry stays valid until another pair disturbs it.
                    self._exhausted.add(key)
                if result.improved:
                    replacement = {column.zone: column for column in result.columns}
                    current = tuple(
                        replacement.get(column.zone, column) for column in frozen
                    )
                    gain += result.gain
                    self.welfare_gain += result.gain
                    improvements += 1
                    self.improvements += 1
                    progress = True
                records.append(
                    {
                        "pair": list(pair),
                        "territory": result.territory,
                        "status": result.status,
                        "splits": len(result.splits),
                        "incumbent_score": result.incumbent_score,
                        "score": result.score,
                        "gain": result.gain,
                        "bound": result.bound
                        if result.bound is not None and math.isfinite(result.bound)
                        else None,
                        "allowance": result.allowance,
                        "wall_time": result.wall_time,
                    }
                )
            if not progress:
                break
        return current, self._metadata(
            records, solved, improvements, added, gain, passes, start, "swept"
        )

    def _key(self, incumbent, pair, decisions):
        """What identifies a solved pair: its labels, territory and fixings."""

        by_label = {column.zone: column for column in incumbent}
        territory = by_label[pair[0]].nodes | by_label[pair[1]].nodes
        return (pair, territory, self._fingerprint(decisions, pair))

    @staticmethod
    def _fingerprint(decisions, pair):
        if not decisions:
            return ()
        return tuple(
            sorted(
                (node, label, int(bool(value)))
                for (node, label), value in decisions.items()
                if label in pair
            )
        )

    def _metadata(
        self, records, solved, improvements, added, gain, passes, start, status
    ) -> dict:
        return {
            "dw_redraw_method": "two_zone_cpsat_redraw",
            "dw_redraw_status": status,
            "dw_redraw_passes": passes,
            "dw_redraw_pairs_solved": solved,
            "dw_redraw_improvements": improvements,
            "dw_redraw_columns_added": added,
            "dw_redraw_welfare_gain": gain,
            "dw_redraw_exhausted_pairs": len(self._exhausted),
            "dw_redraw_wall_time": time.monotonic() - start,
            "dw_redraw_records": records,
        }

    def metadata(self) -> dict:
        return {
            "dw_redraw_method": "two_zone_cpsat_redraw",
            "dw_redraw_scale": self.scale,
            "dw_redraw_objective_scale": self.objective_scale,
            "dw_redraw_splits_per_call": self.splits_per_call,
            "dw_redraw_total_pairs_solved": self.pairs_solved,
            "dw_redraw_total_improvements": self.improvements,
            "dw_redraw_total_columns_added": self.columns_added,
            "dw_redraw_total_welfare_gain": self.welfare_gain,
        }


def redraw_pair(pool, incumbent, pair, *, deadline, decisions=None, **options):
    """One-shot redraw. Inside a loop, hold a :class:`ZoneRedraw` instead."""

    return ZoneRedraw(pool, **options)(
        incumbent, pair, deadline=deadline, decisions=decisions
    )
