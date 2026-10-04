"""MID-market scorers for short-burst objectives.

A short-burst objective that depends on student choice needs a full matching
per partition, so these scorers evaluate a whole burst's feasible samples in
one batch, optionally across a process pool, and cache every outcome by
zoning.  One least-cutoff oracle run per zoning yields everything the choice
objectives need:

* ``welfare`` -- the discrete MID program welfare.
* ``capacity_match`` -- ``unassigned_weight * unassigned +
  designated_weight * designated``.  Students the match leaves without any of
  their ranked programs are *designated* to the seats their own zone's
  general-education programs still have free, and are *unassigned* once those
  run out.  Designation is pooled per zone rather than walked school by
  school, which is exact for the count because any free in-zone GE seat is a
  designation the district can make.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from optimization.data.mid import MidMarket, build_mid_market, preprocess_mid_market
from optimization.mid_oracle import MidOracleResult, finite_grid_oracle

ZoningKey = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DesignationSeat:
    """A zoned general-education program that can take designated students."""

    program_id: str
    school_node: int
    capacity: int


@dataclass(frozen=True)
class MidOutcome:
    welfare: float
    unassigned: float
    designated: float
    cutoffs: dict[str, int]


def is_general_education(program_id: str) -> bool:
    """Whether an id of the form ``<school>-<type>-<grade>`` names a GE program."""

    parts = str(program_id).split("-")
    return len(parts) >= 2 and parts[1] == "GE"


def designation_seats(market: MidMarket) -> tuple[DesignationSeat, ...]:
    """Zoned GE programs of ``market``, which should be the unpreprocessed one.

    Preprocessing drops programs nobody ranks, but their seats still take
    designated students.  Citywide programs belong to no zone, so they are not
    designation targets.
    """

    return tuple(
        DesignationSeat(program.program_id, int(program.school_node), program.capacity)
        for program in market.programs
        if not program.citywide
        and program.school_node is not None
        and program.capacity > 0
        and is_general_education(program.program_id)
    )


def designation_split(
    market: MidMarket,
    seats: tuple[DesignationSeat, ...],
    zoning: Mapping[int, int],
    result: MidOracleResult,
    lottery_scale: int,
) -> tuple[float, float]:
    """Return ``(unassigned, designated)`` student counts for one matching."""

    unmatched: dict[int, float] = defaultdict(float)
    for student_type, remaining in zip(
        market.types, result.remaining_masses, strict=True
    ):
        mass = remaining[-1] if remaining else lottery_scale
        if mass > 0:
            unmatched[zoning[student_type.node]] += student_type.count * mass
    free: dict[int, float] = defaultdict(float)
    for seat in seats:
        taken = result.demands.get(seat.program_id, 0.0)
        free[zoning[seat.school_node]] += max(0.0, seat.capacity - taken)

    total = 0.0
    designated = 0.0
    for zone, mass in unmatched.items():
        students = mass / lottery_scale
        total += students
        designated += min(students, free.get(zone, 0.0))
    return max(0.0, total - designated), designated


def mid_outcome(
    market: MidMarket,
    seats: tuple[DesignationSeat, ...],
    zoning: Mapping[int, int],
    lottery_scale: int,
    warm_cutoffs: Mapping[str, int] | None = None,
) -> MidOutcome:
    result = finite_grid_oracle(
        market,
        dict(zoning),
        lottery_scale,
        check_minimality=False,
        warm_cutoffs=warm_cutoffs,
    )
    unassigned, designated = designation_split(
        market, seats, zoning, result, lottery_scale
    )
    return MidOutcome(
        welfare=float(result.welfare),
        unassigned=unassigned,
        designated=designated,
        cutoffs={key: int(value) for key, value in result.cutoffs.items()},
    )


_worker_state: tuple[MidMarket, tuple[DesignationSeat, ...]] | None = None


def _initialize_worker(market: MidMarket, seats: tuple[DesignationSeat, ...]) -> None:
    global _worker_state
    _worker_state = (market, seats)


def _outcome_in_worker(
    args: tuple[ZoningKey, int, dict[str, int] | None],
) -> tuple[ZoningKey, MidOutcome]:
    zoning_items, lottery_scale, warm_cutoffs = args
    if _worker_state is None:
        raise RuntimeError("MID worker was not initialized.")
    market, seats = _worker_state
    return zoning_items, mid_outcome(
        market, seats, dict(zoning_items), lottery_scale, warm_cutoffs
    )


class MidBatchEvaluator:
    """Cached, optionally parallel MID outcomes for batches of zonings.

    Each batch is warm-started from the cutoffs of its base zoning, which is
    the partition the burst walked away from and so is usually cached.
    """

    def __init__(
        self,
        market: MidMarket,
        lottery_scale: int,
        workers: int,
        seats: tuple[DesignationSeat, ...] = (),
    ) -> None:
        self.market = market
        self.lottery_scale = lottery_scale
        self.workers = workers
        self.seats = seats
        self.cache: dict[ZoningKey, MidOutcome] = {}
        self.executor = self._make_executor() if workers > 1 else None

    def _make_executor(self) -> ProcessPoolExecutor:
        try:
            context = mp.get_context("fork")
        except ValueError:
            context = mp.get_context()
        return ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(self.market, self.seats),
        )

    @staticmethod
    def key(assignment: Mapping[int, int]) -> ZoningKey:
        return tuple(
            sorted((int(node), int(zone)) for node, zone in assignment.items())
        )

    def outcomes(
        self,
        assignments: tuple[Mapping[int, int] | Any, ...],
        base_assignment: Mapping[int, int] | Any | None,
    ) -> tuple[MidOutcome, ...]:
        keys = tuple(self.key(assignment) for assignment in assignments)
        base = (
            self.cache.get(self.key(base_assignment))
            if base_assignment is not None
            else None
        )
        warm_cutoffs = None if base is None else base.cutoffs
        uncached = tuple(dict.fromkeys(key for key in keys if key not in self.cache))

        if self.executor is not None and len(uncached) > 1:
            args = tuple((key, self.lottery_scale, warm_cutoffs) for key in uncached)
            for key, outcome in self.executor.map(_outcome_in_worker, args):
                self.cache[key] = outcome
        else:
            for key in uncached:
                self.cache[key] = mid_outcome(
                    self.market, self.seats, dict(key), self.lottery_scale, warm_cutoffs
                )
        return tuple(self.cache[key] for key in keys)

    def __call__(
        self,
        assignments: tuple[Mapping[int, int] | Any, ...],
        base_assignment: Mapping[int, int] | Any | None,
    ) -> tuple[float, ...]:
        """Score by welfare, the short-bursts batch-scorer contract."""

        return tuple(
            outcome.welfare for outcome in self.outcomes(assignments, base_assignment)
        )

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


CHOICE_METRICS = ("welfare", "capacity_match")


class MidBurstScorer:
    """Batch scorer reporting one choice metric in its own units."""

    def __init__(
        self,
        evaluator: MidBatchEvaluator,
        metric: str,
        *,
        unassigned_weight: float = 1.0,
        designated_weight: float = 1.0,
    ) -> None:
        if metric not in CHOICE_METRICS:
            raise ValueError(
                f"MID burst metric must be one of: {', '.join(CHOICE_METRICS)}."
            )
        self.evaluator = evaluator
        self.metric = metric
        self.unassigned_weight = float(unassigned_weight)
        self.designated_weight = float(designated_weight)

    def _value(self, outcome: MidOutcome) -> float:
        if self.metric == "welfare":
            return outcome.welfare
        return (
            self.unassigned_weight * outcome.unassigned
            + self.designated_weight * outcome.designated
        )

    def __call__(
        self,
        assignments: tuple[Mapping[int, int] | Any, ...],
        base_assignment: Mapping[int, int] | Any | None,
    ) -> tuple[float, ...]:
        return tuple(
            self._value(outcome)
            for outcome in self.evaluator.outcomes(assignments, base_assignment)
        )

    def describe(self, assignment: Mapping[int, int]) -> dict[str, float]:
        """Breakdown of a partition this scorer has already scored."""

        outcome = self.evaluator.cache[self.evaluator.key(assignment)]
        return {
            "best_welfare": outcome.welfare,
            "best_unassigned_students": outcome.unassigned,
            "best_designated_students": outcome.designated,
        }

    def close(self) -> None:
        self.evaluator.close()


def _weight(options: Mapping[str, Any], name: str) -> float:
    raw = options.get(name, 1.0)
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be finite and non-negative.")
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def mid_burst_scorer(
    problem, metric: str, options: Mapping[str, Any]
) -> MidBurstScorer:
    """Build the MID market for ``problem`` and wrap it as a burst scorer."""

    if metric not in CHOICE_METRICS:
        raise ValueError(
            f"MID burst metric must be one of: {', '.join(CHOICE_METRICS)}."
        )
    config = getattr(problem, "optimization_config", None)
    if config is None:
        raise ValueError(
            f"The {metric} objective builds a MID market from the problem's "
            "optimization_config, which this problem does not carry."
        )
    if problem.program_population != "All":
        raise ValueError(f"The {metric} objective requires program_population='All'.")
    lottery_scale = options.get("mid_lottery_scale", 20)
    if (
        isinstance(lottery_scale, bool)
        or not isinstance(lottery_scale, int)
        or lottery_scale <= 0
    ):
        raise ValueError("mid_lottery_scale must be a positive integer.")
    unassigned_weight = _weight(options, "adaptive_short_bursts_unassigned_weight")
    designated_weight = _weight(options, "adaptive_short_bursts_designated_weight")
    if metric == "capacity_match" and unassigned_weight == designated_weight == 0:
        raise ValueError(
            "capacity_match needs a positive unassigned or designated weight."
        )

    raw_market = build_mid_market(problem, config)
    if raw_market.utility_student_count == 0:
        # Every student would be outside-only, so welfare is 0 and capacity
        # match the same constant for every zoning: the search is blind.
        raise ValueError(
            f"The {metric} objective's MID market has no student with a "
            "choice-utility row: the choice estimate does not cover the "
            "assignment year's students."
        )
    evaluator = MidBatchEvaluator(
        preprocess_mid_market(raw_market, problem),
        lottery_scale,
        max(1, int(options.get("workers", 1))),
        designation_seats(raw_market),
    )
    return MidBurstScorer(
        evaluator,
        metric,
        unassigned_weight=unassigned_weight,
        designated_weight=designated_weight,
    )
