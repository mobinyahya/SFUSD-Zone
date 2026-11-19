"""Deferred acceptance with reserves on one fixed lottery, for MID markets.

A matching policy with reserves (``guard-rails`` 0 or 1) splits each reserved
program's seats into one pool per student class plus an open pool, which least
cutoffs on a single seat pool cannot express. This module matches such a
market the way ``assignment.student_assignment.da.da_with_guardrails`` does,
on one lottery: every student-program pair draws a uniform tie-breaker
(multiple tie-breaking, ``ties-options: MTB``) from the reserve seed, and a
student's score at a program is their priority tier plus that draw.

Seats follow ``DeferredAcceptance_with_GuardRails`` exactly. Class ``c`` holds
``int(capacity * fraction_c)`` reserved seats; the open pool starts with the
whole capacity and loses one seat each time a class pool admits into a free
seat, so reserved seats a class leaves empty stay open to everyone. A proposal
tries the student's class pool first, then the open pool. A student displaced
from a class pool proposes to the same program again (now to its open pool);
one displaced from an open pool moves on. With ``strict`` (guard-rails 1) a
program with reserves has no open pool. The outcome is the student-proposing
stable match of these choice functions, whatever the proposal order.

The result is a ``MidOracleResult`` in lottery-scale units -- a matched student
puts the whole ``lottery_scale`` of mass on their program -- so designation,
FRL dissimilarity and capacity references read it like any MID match. Each
market type is one student (``ReserveSpec``).
"""

from __future__ import annotations

import hashlib

import numba
import numpy as np

from optimization.data.mid import MidMarket
from optimization.mid_oracle import MidOracleResult

_PREPARED: dict[int, tuple[MidMarket, "_Prepared"]] = {}


class _Prepared:
    """The market flattened into arrays, built once per market."""

    def __init__(self, market: MidMarket) -> None:
        reserves = market.reserves
        if reserves is None:
            raise ValueError("The reserve DA needs a reserve market.")
        programs = market.programs
        index = {program.program_id: i for i, program in enumerate(programs)}
        self.program_ids = tuple(program.program_id for program in programs)
        classes = reserves.num_classes
        self.capacity = np.array([p.capacity for p in programs], dtype=np.int64)
        self.citywide = np.array([p.citywide for p in programs], dtype=np.bool_)
        self.school_node = np.array(
            [-1 if p.school_node is None else int(p.school_node) for p in programs],
            dtype=np.int64,
        )
        pool_capacity = np.zeros((len(programs), classes), dtype=np.int64)
        has_reserves = np.zeros(len(programs), dtype=np.bool_)
        for program_id, fractions in reserves.fractions.items():
            if program_id not in index:
                continue
            row = index[program_id]
            if round(sum(fractions), 10) > 1:
                raise ValueError(f"Reserves of {program_id} exceed its seats.")
            pool_capacity[row] = [
                int(self.capacity[row] * fraction) for fraction in fractions
            ]
            has_reserves[row] = any(fraction > 0 for fraction in fractions)
        self.pool_capacity = pool_capacity
        self.has_reserves = has_reserves
        self.strict = bool(reserves.strict)

        types = market.types
        self.node = np.array([t.node for t in types], dtype=np.int64)
        self.student_class = np.array([t.reserve_class for t in types], dtype=np.int64)
        lengths = [len(t.programs) for t in types]
        self.start = np.zeros(len(types) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self.start[1:])
        total = int(self.start[-1])
        self.alternative = np.empty(total, dtype=np.int64)
        self.tier = np.empty(total, dtype=np.float64)
        self.open_tier = np.empty(total, dtype=np.float64)
        self.draw = np.empty(total, dtype=np.float64)
        position = 0
        for student_type in types:
            open_tiers = student_type.open_priorities
            for rank, program_id in enumerate(student_type.programs):
                self.alternative[position] = index[program_id]
                self.tier[position] = student_type.priorities[rank]
                self.open_tier[position] = (
                    -1.0 if open_tiers is None else open_tiers[rank]
                )
                self.draw[position] = lottery_draw(
                    reserves.seed, student_type.student_id, program_id
                )
                position += 1
        self.max_node = int(
            max(
                int(self.node.max(initial=-1)),
                int(self.school_node.max(initial=-1)),
            )
        )


def lottery_draw(seed: int, student_id: str, program_id: str) -> float:
    """The student's tie-breaker at the program, in [0, 1).

    Keyed by the identities, not by position, so it does not move when
    preprocessing drops alternatives or reorders types.
    """
    digest = hashlib.blake2b(
        f"{seed}:{student_id}:{program_id}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2.0**64


def _prepared(market: MidMarket) -> _Prepared:
    entry = _PREPARED.get(id(market))
    if entry is None or entry[0] is not market:
        entry = (market, _Prepared(market))
        _PREPARED[id(market)] = entry
    return entry[1]


@numba.njit(cache=True)
def _push(scores, students, offset, length, score, student):
    """Push onto the min-heap ``[offset, offset + length)``; return new length."""
    i = length
    scores[offset + i] = score
    students[offset + i] = student
    while i > 0:
        parent = (i - 1) // 2
        if scores[offset + parent] <= scores[offset + i]:
            break
        scores[offset + parent], scores[offset + i] = (
            scores[offset + i],
            scores[offset + parent],
        )
        students[offset + parent], students[offset + i] = (
            students[offset + i],
            students[offset + parent],
        )
        i = parent
    return length + 1


@numba.njit(cache=True)
def _pop(scores, students, offset, length):
    """Pop the minimum; return ``(student, new length)``."""
    top = students[offset]
    length -= 1
    if length > 0:
        scores[offset] = scores[offset + length]
        students[offset] = students[offset + length]
        i = 0
        while True:
            left = 2 * i + 1
            if left >= length:
                break
            child = left
            right = left + 1
            if right < length and scores[offset + right] < scores[offset + left]:
                child = right
            if scores[offset + i] <= scores[offset + child]:
                break
            scores[offset + child], scores[offset + i] = (
                scores[offset + i],
                scores[offset + child],
            )
            students[offset + child], students[offset + i] = (
                students[offset + i],
                students[offset + child],
            )
            i = child
    return top, length


@numba.njit(cache=True)
def _deferred_acceptance(
    start,
    alternative,
    tier,
    open_tier,
    draw,
    node,
    student_class,
    capacity,
    citywide,
    school_node,
    pool_capacity,
    has_reserves,
    strict,
    zone_of_node,
):
    """Return each student's matched rank (-1: unmatched)."""
    students = node.shape[0]
    programs = capacity.shape[0]
    classes = pool_capacity.shape[1]
    pools = classes + 1
    # Heap segments: class pools, then the open pool, of each program; each
    # segment holds at most the program's capacity.
    offsets = np.zeros(programs * pools + 1, dtype=np.int64)
    for p in range(programs):
        for c in range(pools):
            size = capacity[p] if c == classes else pool_capacity[p, c]
            offsets[p * pools + c + 1] = offsets[p * pools + c] + size
    heap_score = np.empty(offsets[-1], dtype=np.float64)
    heap_student = np.empty(offsets[-1], dtype=np.int64)
    heap_length = np.zeros(programs * pools, dtype=np.int64)
    open_capacity = np.empty(programs, dtype=np.int64)
    for p in range(programs):
        open_capacity[p] = 0 if (strict and has_reserves[p]) else capacity[p]

    rank = np.zeros(students, dtype=np.int64)
    matched = np.full(students, -1, dtype=np.int64)
    queue = np.empty(students, dtype=np.int64)
    head = 0
    tail = 0
    for s in range(students):
        queue[tail] = s
        tail += 1
    pending = students

    while pending > 0:
        s = queue[head % students]
        head += 1
        pending -= 1
        zone = zone_of_node[node[s]]
        c = student_class[s]
        while True:
            r = rank[s]
            position = start[s] + r
            if position >= start[s + 1]:
                break
            p = alternative[position]
            if citywide[p] or zone == zone_of_node[school_node[p]]:
                priority = tier[position]
            else:
                priority = open_tier[position]
            if priority < 0 or capacity[p] <= 0:
                rank[s] = r + 1
                continue
            score = priority + draw[position]
            if has_reserves[p] and pool_capacity[p, c] > 0:
                segment = p * pools + c
                length = heap_length[segment]
                if length < pool_capacity[p, c]:
                    heap_length[segment] = _push(
                        heap_score, heap_student, offsets[segment], length, score, s
                    )
                    matched[s] = r
                    open_capacity[p] -= 1
                    open_segment = p * pools + classes
                    if heap_length[open_segment] > max(open_capacity[p], 0):
                        evicted, open_length = _pop(
                            heap_score,
                            heap_student,
                            offsets[open_segment],
                            heap_length[open_segment],
                        )
                        heap_length[open_segment] = open_length
                        matched[evicted] = -1
                        rank[evicted] += 1
                        queue[tail % students] = evicted
                        tail += 1
                        pending += 1
                    break
                if heap_score[offsets[segment]] < score:
                    displaced, _ = _pop(
                        heap_score, heap_student, offsets[segment], length
                    )
                    heap_length[segment] = _push(
                        heap_score,
                        heap_student,
                        offsets[segment],
                        length - 1,
                        score,
                        s,
                    )
                    matched[s] = r
                    # The displaced student tries the same program's open pool.
                    matched[displaced] = -1
                    queue[tail % students] = displaced
                    tail += 1
                    pending += 1
                    break
            segment = p * pools + classes
            length = heap_length[segment]
            if length < open_capacity[p]:
                heap_length[segment] = _push(
                    heap_score, heap_student, offsets[segment], length, score, s
                )
                matched[s] = r
                break
            if (
                open_capacity[p] > 0
                and length > 0
                and heap_score[offsets[segment]] < score
            ):
                evicted, length = _pop(
                    heap_score, heap_student, offsets[segment], length
                )
                heap_length[segment] = _push(
                    heap_score, heap_student, offsets[segment], length, score, s
                )
                matched[s] = r
                matched[evicted] = -1
                rank[evicted] += 1
                queue[tail % students] = evicted
                tail += 1
                pending += 1
                break
            rank[s] = r + 1
    return matched


def reserve_da_ranks(market: MidMarket, zoning) -> np.ndarray:
    """Each type's matched rank under ``zoning`` (-1: unmatched)."""
    prepared = _prepared(market)
    zone_of_node = np.full(prepared.max_node + 1, -1, dtype=np.int64)
    for node, zone in zoning.items():
        if 0 <= int(node) <= prepared.max_node:
            zone_of_node[int(node)] = int(zone)
    return _deferred_acceptance(
        prepared.start,
        prepared.alternative,
        prepared.tier,
        prepared.open_tier,
        prepared.draw,
        prepared.node,
        prepared.student_class,
        prepared.capacity,
        prepared.citywide,
        prepared.school_node,
        prepared.pool_capacity,
        prepared.has_reserves,
        prepared.strict,
        zone_of_node,
    )


def reserve_da_oracle(market: MidMarket, zoning, lottery_scale: int) -> MidOracleResult:
    """The reserve DA's match as a ``MidOracleResult`` (see the module)."""
    ranks = reserve_da_ranks(market, zoning)
    prepared = _prepared(market)
    demand_masses = {program_id: 0 for program_id in prepared.program_ids}
    assignment_rows = []
    remaining_rows = []
    type_values = []
    welfare = 0.0
    outside = 0
    for student_type, rank in zip(market.types, ranks.tolist(), strict=True):
        length = len(student_type.programs)
        if rank < 0:
            assignment_rows.append((0,) * length)
            remaining_rows.append((lottery_scale,) * length)
            type_values.append(0)
            outside += lottery_scale
            continue
        assignment_rows.append(
            tuple(lottery_scale if k == rank else 0 for k in range(length))
        )
        remaining_rows.append(
            tuple(lottery_scale if k < rank else 0 for k in range(length))
        )
        demand_masses[student_type.programs[rank]] += lottery_scale
        welfare += student_type.utility_sums[rank]
        type_values.append(student_type.scaled_utility_sums[rank] * lottery_scale)
    fixed_point = sum(type_values)
    return MidOracleResult(
        cutoffs={program_id: 0 for program_id in prepared.program_ids},
        demands={key: mass / lottery_scale for key, mass in demand_masses.items()},
        demand_masses=demand_masses,
        assignment_masses=tuple(assignment_rows),
        remaining_masses=tuple(remaining_rows),
        outside_mass=outside / lottery_scale,
        welfare=welfare,
        fixed_point_value=fixed_point,
        fixed_point_welfare=fixed_point / (lottery_scale * market.utility_scale),
        type_fixed_point_values=tuple(type_values),
        stable=True,
        minimal=True,
    )
