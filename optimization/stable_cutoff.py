"""Per-student cutoff model of zoned deferred acceptance: market preparation.

Every other welfare-aware master in this repo reasons about *lottery mass*. The
MID solver spends one variable per (type, rank) on a finite grid and the SAA
master recourses through the continuous stable-admissions LP; both hand their
linear relaxation a continuum, and a fractional co-zoning weight then buys a
pro-rata slice of an oversubscribed school that no integral zoning could hand
out. ``analysis/probe_access_convexification`` measured where that leads: a root
relaxation of 17,377 against a best-known zoning of about 14,950, leaking at a
perfectly consistent co-zoning matrix, so nothing writable in access space cuts
it off.

This module prepares the other end of the design space. Fix the lottery -- draw
a finite number of strict priority orders and average over them -- and the
matching stops being a mass flow and becomes a *set of seats*, which can be
written exactly with per-student binaries. That is what
:mod:`optimization.solvers.stable_cutoff` builds; this module is the data half:
one market, ``S`` sampled priority orders, and the access structure they share.

Why a cutoff *chain* and no cutoff variable
-------------------------------------------
A program's cutoff, in a fixed lottery draw, is not a number that needs
representing: it is a *position* in that program's realised priority order.
Once the order ``Gamma(s)`` is drawn, "student ``i`` clears ``s``'s cutoff" is a
per-pair indicator ``z[i, s]``, and the only thing that makes those indicators a
cutoff is that they are monotone down the order,

    z[i, s] <= z[i', s]   for i' the immediate predecessor of i in Gamma(s),

which is the (F5) chain. A set of ``z`` satisfying it is exactly the indicator
of a prefix of ``Gamma(s)``, i.e. a cutoff, so an explicit cutoff variable would
be a redundant encoding of the same prefix -- and a strictly worse one, because
a cutoff variable has to live on a grid (MID) or carry big-M rows linking it to
per-student admission (a threshold formulation), whereas the chain is
``|Gamma(s)| - 1`` two-variable rows with a tight relaxation and nothing to
scale. The price is that ``z`` is only meaningful for one drawn order, which is
why the sample orders and the market are prepared together here and must stay
aligned: :func:`optimization.data.saa.sample_school_preferences` indexes
``school_orders`` by position in ``market.programs``, so any preprocessing that
drops a program has to happen *before* the draw.

Access structure
----------------
The zoning enters the matching model in exactly one place, as an upper bound
``y[i, s] <= A_is`` on an assignment variable, where ``A_is`` is 1 for citywide
programs and the co-zoning indicator of ``i``'s node with ``s``'s school node
otherwise. Which pairs are genuinely undecided is read from
:func:`optimization.saa_oracle.access_state`, the same helper the SAA recourse
oracle uses, so the two models see the same access structure by construction and
their bounds are directly comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from optimization.data.saa import (
    SAA_TIE_BREAKING_METHODS,
    SaaMarket,
    SaaSample,
    build_saa_market,
    preprocess_saa_market,
    sample_school_preferences,
)
from optimization.problem import ZoneProblem
from optimization.saa_oracle import AccessPair, access_state, required_access_pairs


STABLE_CUTOFF_TIE_BREAKING_METHODS = SAA_TIE_BREAKING_METHODS


@dataclass(frozen=True)
class StableCutoffInstance:
    """A preprocessed market plus the sampled priority orders drawn on it.

    The two fields are not independently valid: ``samples`` index programs by
    their position in ``market.programs``, so a market and a sample set built
    from different preprocessing passes describe different programs under the
    same index. Always construct through
    :func:`prepare_stable_cutoff_instance`, which preprocesses first, draws
    second, and then checks the alignment.
    """

    market: SaaMarket
    samples: tuple[SaaSample, ...]
    tie_breaking_method: str
    preprocessing_seconds: float = 0.0

    @property
    def num_seeds(self) -> int:
        return len(self.samples)

    @property
    def sample_seeds(self) -> tuple[int, ...]:
        return tuple(sample.seed for sample in self.samples)

    @property
    def gamma_size(self) -> int:
        """``|Gamma|``: the number of (student, program) pairs in one sample.

        This is the model's unit of size. Each pair costs one ``y`` and one
        ``z`` binary per sample, so the whole matching block is ``2 S |Gamma|``
        binaries and the constraint families are all ``O(S |Gamma|)`` rows.
        """

        return self.market.preference_count


def prepare_stable_cutoff_instance(
    market: SaaMarket,
    problem: ZoneProblem,
    *,
    num_seeds: int = 3,
    tie_breaking_method: str = "STB",
    base_seed: int = 42,
    preprocessing_seconds: float = 0.0,
) -> StableCutoffInstance:
    """Preprocess ``market`` against ``problem``, then draw its priority orders.

    Order matters. :func:`preprocess_saa_market` drops zero-capacity programs
    and alternatives no zoning could ever make accessible, which renumbers
    ``market.programs``; drawing first and preprocessing after would silently
    attach each program the priority order of a different one.
    """

    prepared = preprocess_saa_market(market, problem)
    samples = sample_school_preferences(
        prepared, num_seeds, tie_breaking_method, base_seed
    )
    validate_sample_orders(prepared, samples)
    return StableCutoffInstance(
        market=prepared,
        samples=samples,
        tie_breaking_method=str(tie_breaking_method).upper(),
        preprocessing_seconds=preprocessing_seconds,
    )


def build_stable_cutoff_instance(
    problem: ZoneProblem,
    optimization_config,
    *,
    num_seeds: int = 3,
    tie_breaking_method: str = "STB",
    base_seed: int = 42,
) -> StableCutoffInstance:
    """Load the individual market for ``problem`` and sample its priorities.

    Uses MID's utility and access semantics through
    :func:`optimization.data.saa.build_saa_market`, so a ``stable_cutoff`` run
    and an ``saa`` run on the same config score the same students over the same
    programs with the same utilities.
    """

    start = time.perf_counter()
    market = build_saa_market(problem, optimization_config)
    instance = prepare_stable_cutoff_instance(
        market,
        problem,
        num_seeds=num_seeds,
        tie_breaking_method=tie_breaking_method,
        base_seed=base_seed,
    )
    return StableCutoffInstance(
        market=instance.market,
        samples=instance.samples,
        tie_breaking_method=instance.tie_breaking_method,
        preprocessing_seconds=time.perf_counter() - start,
    )


def validate_sample_orders(
    market: SaaMarket, samples: tuple[SaaSample, ...] | list[SaaSample]
) -> None:
    """Check that every sampled order is a permutation of its own ``Gamma(s)``.

    The (F5) chain and the (S1)/(S2) families are built by walking
    ``school_orders[program_index]``, so a program whose order is missing an
    interested student would simply never constrain that student's ``z`` -- an
    unstable matching the model would happily report as optimal. Cheap to check
    once, impossible to notice later.
    """

    if not samples:
        raise ValueError("stable_cutoff needs at least one priority sample.")
    interested: dict[str, set[int]] = {
        program.program_id: set() for program in market.programs
    }
    for student_index, student in enumerate(market.students):
        for program_id in student.programs:
            interested[program_id].add(student_index)
    for sample in samples:
        if len(sample.school_orders) != len(market.programs):
            raise ValueError(
                "stable_cutoff priority orders must align with market programs."
            )
        for program, order in zip(market.programs, sample.school_orders):
            expected = interested[program.program_id]
            if len(order) != len(expected) or set(order) != expected:
                raise ValueError(
                    "stable_cutoff priority order for "
                    f"{program.program_id!r} is not Gamma(s)."
                )


def undecided_access_pairs(
    problem: ZoneProblem, market: SaaMarket
) -> tuple[AccessPair, ...]:
    """The (node, node) pairs whose co-zoning the model has to carry a variable for.

    Everything else is settled by construction and enters the matching block as
    a constant: citywide programs, a school sitting on the student's own node,
    and a pair with no candidate zone in common. Sorted and deduplicated in the
    ``(min, max)`` orientation :meth:`MipSolver._get_or_create_access_var` keys
    its cache by.
    """

    programs = market.program_by_id
    pairs: set[AccessPair] = set()
    for student in market.students:
        for program_id in student.programs:
            pair, _ = access_state(problem, student.node, programs[program_id])
            if pair is not None:
                pairs.add((min(pair), max(pair)))
    return tuple(sorted(pairs))


def restricted_access_pair_count(market: SaaMarket) -> int:
    """How many (student node, school node) pairs a zoning could decide at all."""

    return len(required_access_pairs(market))
