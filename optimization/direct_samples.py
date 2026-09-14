"""Market preparation and matching replay for the ``direct_samples`` model.

The ``saa`` strategy solves the stable-matching access polytope (the appendix
"The Stable-Matching Access Polytope" in paper.tex) *outside* the zoning model:
one continuous recourse LP per sampled tie-breaking seed, whose dual becomes a
cut in the master's access variables. This module is the data half of the other
arrangement -- put a copy of that polytope, one per seed, *inside* the zoning
model and maximize the sample average of the copies directly. There is then no
outer loop, no dual, and no cut pool; branch and bound is the whole search.

Why the surrogate disappears
----------------------------
The implementation the ``saa`` oracle loads is not the comb family but the
aggregated one-row-per-pair surrogate of that appendix -- the many-to-one
aggregation of Rothblum's (1992) inequality,

    sum_{i' >_s^psi i} D_{i',s} + q_s sum_{s' >=_i^psi s} D_{i,s'} >= q_s a_{i,s},

which is valid at every zoning but incomplete: it has fractional extreme points
that no stable matching reaches, and on Block_2 it over-reports realized
deferred-acceptance welfare by 195 to 252 units. Cutting those points off is
what the comb inequalities of Proposition~2 are for, and separating them is why
the oracle would need column generation.

None of that applies once ``D`` is Boolean. Read the row at an integral ``D``
and it is exactly Lemma~1: if ``i`` has access to ``s`` and holds nothing
weakly preferred to ``s``, the second sum is zero and the first must reach
``q_s``, i.e. ``s`` is full of applicants it strictly prefers to ``i``; and
conversely a matching with a blocking pair violates that pair's row. So the
integral points of the aggregated system are already *exactly* the stable
matchings of the market the zoning leaves, the fractional points the comb
family was invented to remove are not in the feasible set to begin with, and no
separation is needed. By Corollary~1 the maximum over that set is the welfare
of applicant-proposing deferred acceptance, because deferred acceptance returns
the applicant-optimal stable matching and the utilities are consistent with the
preference order. :func:`deferred_acceptance` below is that replay, and the
solver checks its own optimum against it.

What it costs
-------------
``|Gamma|`` Booleans and ``O(|Gamma|)`` rows *per seed*, all sharing one copy of
the co-zoning indicators, against ``saa``'s one master row per seed per
iteration. The trade is the whole point of the strategy and ``saa_num_seeds``
is the knob that decides whether the model fits.

The zone-restricted ``deferred_acceptance`` in :mod:`optimization.zone_welfare`
is a different routine for a different purpose: it prices one *zone* in
isolation for the Dantzig-Wolfe pricing problem, forbids citywide programs
because their seats are contested across zones, and never sees a zoning. This
one replays a whole zoning, citywide programs included.
"""

from __future__ import annotations

import heapq
import math

from optimization.data.saa import SaaMarket, SaaSample
from optimization.problem import ZoneProblem
from optimization.saa_oracle import access_state


#: Per student, the rank they hold in a matching, or ``-1`` when unassigned.
Matching = tuple[int, ...]


def access_mask(
    market: SaaMarket, problem: ZoneProblem, zoning: dict[int, int]
) -> tuple[tuple[bool, ...], ...]:
    """Which of each student's ranked alternatives ``zoning`` makes reachable.

    Read through :func:`optimization.saa_oracle.access_state`, the same helper
    the SAA recourse oracle and the matching model use, so a replay and the
    model it checks cannot disagree about what a zoning grants.
    """

    programs = market.program_by_id
    mask = []
    for student in market.students:
        row = []
        for program_id in student.programs:
            pair, fixed = access_state(problem, student.node, programs[program_id])
            row.append(
                bool(fixed)
                if pair is None
                else zoning[pair[0]] == zoning[pair[1]]
            )
        mask.append(tuple(row))
    return tuple(mask)


def deferred_acceptance(
    market: SaaMarket,
    sample: SaaSample,
    mask: tuple[tuple[bool, ...], ...],
) -> Matching:
    """Applicant-proposing deferred acceptance on the market ``mask`` leaves.

    Returns each student's matched rank in their own preference list, ``-1``
    when unassigned. Ranks rather than program ids because every consumer here
    -- welfare, hints, the exactness check -- indexes the market's utility and
    variable tables by rank anyway.

    Each program holds the best ``q_s`` proposals it has seen under its drawn
    order, so a max-heap keyed by position in that order is the whole data
    structure. The outcome is the applicant-optimal stable matching, which
    every applicant weakly prefers to every other stable matching, so with
    utilities consistent with the preference order it also carries the largest
    welfare of any stable matching.
    """

    if len(sample.school_orders) != len(market.programs):
        raise ValueError("Sampled priority orders must align with market programs.")
    if len(mask) != len(market.students):
        raise ValueError("Access mask must cover every student.")

    capacity = {
        program.program_id: int(program.capacity) for program in market.programs
    }
    position = {
        program.program_id: {
            student_index: place
            for place, student_index in enumerate(sample.school_orders[index])
        }
        for index, program in enumerate(market.programs)
    }
    held: dict[str, list[tuple[int, int]]] = {
        program_id: [] for program_id in capacity
    }

    matched = [-1] * len(market.students)
    cursor = [0] * len(market.students)
    pending = list(range(len(market.students)))
    while pending:
        student_index = pending.pop()
        student = market.students[student_index]
        rank = cursor[student_index]
        while rank < len(student.programs):
            program_id = student.programs[rank]
            reachable = mask[student_index][rank]
            rank += 1
            if not reachable or capacity[program_id] <= 0:
                continue
            place = position[program_id][student_index]
            heap = held[program_id]
            if len(heap) < capacity[program_id]:
                heapq.heappush(heap, (-place, student_index))
            else:
                worst_place, worst = heap[0]
                if -worst_place <= place:
                    # Even the worst held applicant outranks this proposal.
                    continue
                heapq.heapreplace(heap, (-place, student_index))
                matched[worst] = -1
                pending.append(worst)
            matched[student_index] = rank - 1
            break
        cursor[student_index] = rank
    return tuple(matched)


def replay(
    market: SaaMarket,
    samples: tuple[SaaSample, ...],
    problem: ZoneProblem,
    zoning: dict[int, int],
) -> tuple[Matching, ...]:
    """Realized matchings of ``zoning``, one per sampled priority order."""

    mask = access_mask(market, problem, zoning)
    return tuple(deferred_acceptance(market, sample, mask) for sample in samples)


def matching_welfare(market: SaaMarket, matching: Matching) -> float:
    """Utilitarian welfare of one matching, in the market's own utilities."""

    return math.fsum(
        market.students[student_index].utilities[rank]
        for student_index, rank in enumerate(matching)
        if rank >= 0
    )


def scaled_matching_welfare(market: SaaMarket, matching: Matching) -> int:
    """The same welfare in the integer units the CP-SAT objective accumulates.

    Scaling is monotone in utility, so the matching that maximizes one
    maximizes the other, which is what makes this comparable to the solver's
    raw objective without any rounding tolerance.
    """

    return sum(
        market.students[student_index].scaled_utilities[rank]
        for student_index, rank in enumerate(matching)
        if rank >= 0
    )
