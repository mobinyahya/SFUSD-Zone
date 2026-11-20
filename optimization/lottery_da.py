"""Deferred acceptance on fixed lotteries for matching-policy MID markets.

A matching_policy market (optimization.data.mid.policy_student_market)
holds, for every student, the list the assignment simulator builds -- their
submitted programs, their zone's GE program (add_aa_schools), and their
designation order -- with every zone-dependent candidate tagged by kind. This
module matches it the way assignment.student_assignment does, on
LotterySpec.lotteries multiple-tie-breaking lotteries, and averages the
outcomes:

* A student's score at a program is their priority tier plus a uniform draw
  per student and program (ties-options: MTB), the same form as the
  simulator's priorities + lottery.
* The zoning decides which candidates exist. A citywide program is always
  reached at its tier; a zoned one in the student's zone at its tier, except a
  designation there when add_aa_schools already appended it (the
  simulator drops that duplicate); a zoned one outside it at the policy's
  open tier (-1: closed), never for an appended program.
* Seats follow DeferredAcceptance_with_GuardRails exactly. Class c
  holds int(capacity * fraction_c) reserved seats; the open pool starts
  with the whole capacity and loses one seat each time a class pool admits
  into a free seat, so reserved seats a class leaves empty stay open to
  everyone. A proposal tries the student's class pool, then the open pool; a
  student displaced from a class pool proposes to the same program again (now
  to its open pool); one displaced from the open pool moves on. With
  strict (guard-rails 1) a reserved program has no open pool. Without
  reserves every program is its open pool: plain deferred acceptance.

Designation happens inside the match, as in the simulator: a student is
*designated* when placed through a designation candidate and *unassigned*
when placed nowhere.

A policy with the utility model (``utility-model.enable``) gives every
student a list per lottery (``LotterySpec.lists``), drawn as the simulator
draws its iteration ``mid_lottery_seed + l``, lottery numbers included. Two
more things then depend on the zoning, and the match settles them before it
starts: the list keeps only its first ``budget`` *eligible* listed programs
(those the zoning lets the student enter), and an appended or designated
program the zoning leaves on that list is the duplicate the simulator drops.
"""

from __future__ import annotations

from collections import defaultdict

import numba
import numpy as np

from optimization.data.mid import DESIGNATED, UNLIMITED_BUDGET, MidMarket

_PREPARED: dict[int, tuple[MidMarket, "_Prepared"]] = {}


class _Prepared:
    """The market flattened into arrays, built once per market."""

    def __init__(self, market: MidMarket) -> None:
        spec = market.lottery
        if spec is None:
            raise ValueError("The lottery DA needs a matching-policy market.")
        programs = market.programs
        index = {program.program_id: i for i, program in enumerate(programs)}
        self.programs = programs
        classes = spec.num_classes
        self.capacity = np.array([p.capacity for p in programs], dtype=np.int64)
        self.citywide = np.array([p.citywide for p in programs], dtype=np.bool_)
        self.school_node = np.array(
            [-1 if p.school_node is None else int(p.school_node) for p in programs],
            dtype=np.int64,
        )
        pool_capacity = np.zeros((len(programs), classes), dtype=np.int64)
        has_reserves = np.zeros(len(programs), dtype=np.bool_)
        for program_id, fractions in spec.fractions.items():
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
        self.strict = bool(spec.strict)
        self.skip_designation_in_zone = bool(spec.add_aa_schools)

        types = market.types
        self.node = np.array([t.node for t in types], dtype=np.int64)
        self.student_class = np.array([t.reserve_class for t in types], dtype=np.int64)
        self.frl = np.array([t.frl_sum for t in types], dtype=np.float64)
        self.lists = spec.lists is not None
        if self.lists:
            self._flatten_lists(market, index)
        else:
            self._flatten_types(market, index)
        self.max_node = int(
            max(int(self.node.max(initial=-1)), int(self.school_node.max(initial=-1)))
        )
        # Outcome bookkeeping: each program's school, and zoned GE seats.
        school_ids = sorted({p.school_id for p in programs})
        school_index = {school: i for i, school in enumerate(school_ids)}
        self.school_ids = school_ids
        self.program_school = np.array(
            [school_index[p.school_id] for p in programs], dtype=np.int64
        )
        self.zoned_ge = np.array(
            [
                (not p.citywide)
                and p.school_node is not None
                and str(p.program_id).split("-")[1:2] == ["GE"]
                for p in programs
            ],
            dtype=np.bool_,
        )

    def _flatten_types(self, market: MidMarket, index) -> None:
        """One fixed list per student, one lottery draw per alternative."""
        spec = market.lottery
        types = market.types
        lengths = [len(t.programs) for t in types]
        self.start = np.zeros(len(types) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self.start[1:])
        flat_programs = [index[p] for t in types for p in t.programs]
        self.alternative = np.array(flat_programs, dtype=np.int64)
        self.tier = np.array(
            [tier for t in types for tier in t.priorities], dtype=np.float64
        )
        self.open_tier = np.array(
            [tier for t in types for tier in t.open_priorities], dtype=np.float64
        )
        self.kind = np.array([k for t in types for k in t.kinds], dtype=np.int64)
        self.utility = np.array(
            [u for t in types for u in t.utility_sums], dtype=np.float64
        )
        # One uniform per student and program and lottery, indexed by the
        # market's full student and program sets so preprocessing cannot
        # move them.
        student_row = {sid: i for i, sid in enumerate(spec.student_ids)}
        program_col = {pid: j for j, pid in enumerate(spec.program_ids)}
        rows = np.repeat(
            np.array([student_row[t.student_id] for t in types], dtype=np.int64),
            lengths,
        )
        cols = np.array(
            [program_col[p] for t in types for p in t.programs], dtype=np.int64
        )
        self.draws = np.empty((spec.lotteries, len(flat_programs)), dtype=np.float64)
        for lottery in range(spec.lotteries):
            rng = np.random.default_rng([spec.seed, lottery])
            full = rng.random((len(spec.student_ids), len(spec.program_ids)))
            self.draws[lottery] = full[rows, cols]
        # Every lottery reads the same lists, uncapped and without duplicates.
        self.starts = self.start[None, :]
        self.copy = np.full(len(flat_programs), -1, dtype=np.int64)
        self.budget = np.full(len(types), UNLIMITED_BUDGET, dtype=np.int64)

    def _flatten_lists(self, market: MidMarket, index) -> None:
        """Per-lottery lists (``LotteryLists``), rows gathered in type order."""
        spec = market.lottery
        lists = spec.lists
        n = len(lists.student_ids)
        row_of = {sid: i for i, sid in enumerate(lists.student_ids)}
        rows = np.array([row_of[t.student_id] for t in market.types], dtype=np.int64)
        segments = lists.segments
        lotteries = lists.lotteries
        types = len(rows)
        picked = (np.arange(lotteries)[:, None] * n + rows[None, :]).ravel()
        lengths = segments[picked + 1] - segments[picked]
        offsets = np.zeros(len(picked) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        gather = np.repeat(segments[picked] - offsets[:-1], lengths) + np.arange(
            offsets[-1]
        )
        # Absolute offsets of each lottery's rows, in type order.
        self.starts = np.zeros((lotteries, types + 1), dtype=np.int64)
        for lottery in range(lotteries):
            self.starts[lottery] = offsets[lottery * types : (lottery + 1) * types + 1]
        self.start = (
            self.starts[0] if lotteries else np.zeros(types + 1, dtype=np.int64)
        )
        to_market = np.array(
            [index.get(program_id, -1) for program_id in spec.program_ids],
            dtype=np.int64,
        )
        alternative = to_market[lists.program[gather]]
        if np.any(alternative < 0):
            raise ValueError("A lottery list names a program the market dropped.")
        self.alternative = alternative
        self.tier = lists.tier[gather].astype(np.float64)
        self.open_tier = lists.open_tier[gather].astype(np.float64)
        self.kind = lists.kind[gather].astype(np.int64)
        self.copy = lists.copy[gather].astype(np.int64)
        self.utility = lists.utility[gather].astype(np.float64)
        self.draws = lists.draw[gather].astype(np.float64)
        self.budget = lists.budget[rows].astype(np.int64)

    def lottery_inputs(self, lottery: int, draws=None):
        """``(start, draw)`` of one lottery."""
        if self.lists:
            if draws is not None:
                raise ValueError(
                    "A utility-model market draws its own lotteries with its lists."
                )
            return self.starts[lottery], self.draws
        draw = self.draws[lottery] if draws is None else draws[lottery]
        return self.start, draw

    def lottery_count(self, draws=None) -> int:
        if draws is not None:
            return int(np.asarray(draws).shape[0])
        return int(self.starts.shape[0]) if self.lists else int(self.draws.shape[0])


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
    kind,
    skip_designation_in_zone,
    copy,
    budget,
):
    """Return each student's matched rank (-1: unmatched).

    kind and skip_designation_in_zone select which alternatives the
    zoning allows (see lottery_market_access). A listed alternative counts
    only among the student's first budget eligible ones; an appended or
    designated one whose listed copy (copy, relative to the student's
    start; -1 none) counts is the dropped duplicate.
    """
    students = node.shape[0]
    programs = capacity.shape[0]
    classes = pool_capacity.shape[1]
    # Which listed alternatives this zoning puts on each list.
    included = np.zeros(alternative.shape[0], dtype=np.bool_)
    for s in range(students):
        zone = zone_of_node[node[s]]
        counted = 0
        for position in range(start[s], start[s + 1]):
            if kind[position] != 0:
                continue
            if counted >= budget[s]:
                break
            p = alternative[position]
            if citywide[p] or zone == zone_of_node[school_node[p]]:
                eligible = True
            else:
                eligible = open_tier[position] >= 0
            if eligible:
                included[position] = True
                counted += 1
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
            if kind[position] == 0:
                if not included[position]:
                    rank[s] = r + 1
                    continue
            elif copy[position] >= 0 and included[start[s] + copy[position]]:
                rank[s] = r + 1
                continue
            if citywide[p]:
                priority = tier[position]
            elif zone == zone_of_node[school_node[p]]:
                if kind[position] == 2 and skip_designation_in_zone:
                    priority = -1.0
                else:
                    priority = tier[position]
            elif kind[position] == 1:
                priority = -1.0
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


def lottery_ranks(market: MidMarket, zoning, draws=None) -> np.ndarray:
    """Each student's matched position in each lottery: (lotteries, students),
    an index into the student's alternatives, -1 when unmatched.

    draws (lotteries x alternatives) replaces the market's own lotteries,
    e.g. to replay the simulator's.
    """
    prepared = _prepared(market)
    zone_of_node = np.full(prepared.max_node + 1, -1, dtype=np.int64)
    for node, zone in zoning.items():
        if 0 <= int(node) <= prepared.max_node:
            zone_of_node[int(node)] = int(zone)
    if draws is not None:
        draws = np.asarray(draws, dtype=np.float64)
    lotteries = prepared.lottery_count(draws)
    ranks = np.empty((lotteries, len(prepared.node)), dtype=np.int64)
    for lottery in range(lotteries):
        start, draw = prepared.lottery_inputs(lottery, draws)
        ranks[lottery] = _deferred_acceptance(
            start,
            prepared.alternative,
            prepared.tier,
            prepared.open_tier,
            draw,
            prepared.node,
            prepared.student_class,
            prepared.capacity,
            prepared.citywide,
            prepared.school_node,
            prepared.pool_capacity,
            prepared.has_reserves,
            prepared.strict,
            zone_of_node,
            prepared.kind,
            prepared.skip_designation_in_zone,
            prepared.copy,
            prepared.budget,
        )
    return ranks


def lottery_placements(market: MidMarket, zoning, draws=None):
    """Per lottery and student: the matched program index (into
    market.programs) and alternative kind, both -1 when unmatched."""
    prepared = _prepared(market)
    ranks = lottery_ranks(market, zoning, draws)
    positions = prepared.starts[:, :-1] + ranks
    matched = ranks >= 0
    safe = np.where(matched, positions, 0)
    program = np.where(matched, prepared.alternative[safe], -1)
    kind = np.where(matched, prepared.kind[safe], -1)
    utility = np.where(matched, prepared.utility[safe], 0.0)
    return program, kind, utility


def lottery_outcome(market: MidMarket, zoning, draws=None):
    """The MidOutcome of a matching-policy market, averaged over lotteries.

    unassigned counts students placed nowhere and designated those
    placed through a designation; frl_dissimilarity is the two-group FRL
    index across the schools of everyone placed (see
    optimization.solvers.burst_objectives); fill_polarization is
    sum_z u_z (C_z - u_z) / C_z over the zoned GE seats C_z of each
    zone and the seats u_z the match fills.
    """
    from optimization.solvers.burst_objectives import MidOutcome

    prepared = _prepared(market)
    program, kind, utility = lottery_placements(market, zoning, draws)
    lotteries = program.shape[0]
    schools = len(prepared.school_ids)
    zone_of_program = np.array(
        [
            zoning[p.school_node] if p.school_node is not None else -1
            for p in prepared.programs
        ],
        dtype=np.int64,
    )
    zones = sorted(set(int(z) for z in zone_of_program if z >= 0))
    zone_index = {zone: i for i, zone in enumerate(zones)}
    program_zone = np.array(
        [zone_index.get(int(z), -1) for z in zone_of_program], dtype=np.int64
    )
    ge_capacity = np.bincount(
        program_zone[prepared.zoned_ge],
        weights=prepared.capacity[prepared.zoned_ge],
        minlength=len(zones),
    )
    unassigned = designated = welfare = dissimilarity = polarization = 0.0
    for lottery in range(lotteries):
        placed = program[lottery] >= 0
        unassigned += float((~placed).sum())
        designated += float((kind[lottery] == DESIGNATED).sum())
        welfare += float(utility[lottery].sum())
        at = program[lottery][placed]
        school = prepared.program_school[at]
        n = np.bincount(school, minlength=schools)
        f = np.bincount(school, weights=prepared.frl[placed], minlength=schools)
        total_frl, total = f.sum(), n.sum()
        if total_frl > 0 and total - total_frl > 0:
            dissimilarity += 50.0 * float(
                np.abs(f / total_frl - (n - f) / (total - total_frl)).sum()
            )
        ge = prepared.zoned_ge[at]
        used = np.bincount(program_zone[at[ge]], minlength=len(zones))
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                ge_capacity > 0, used * (ge_capacity - used) / ge_capacity, 0.0
            )
        polarization += float(terms.sum())
    return MidOutcome(
        welfare=welfare / lotteries,
        unassigned=unassigned / lotteries,
        designated=designated / lotteries,
        cutoffs={},
        fill_polarization=polarization / lotteries,
        frl_dissimilarity=dissimilarity / lotteries,
    )


def lottery_capacity_reference(problem, zoning, market, lottery_scale):
    """build_capacity_reference for a matching-policy market.

    The same quantities, averaged over the lotteries, read off the placements
    directly: designation already happened inside the match, so a student
    placed in the attendance-area seats of a school in their own zone --
    listed, appended or designated -- counts for their home area and occupies
    those seats, and one placed anywhere else departs. A sibling-access seat in
    another zone is a departure too, so each zone's counted residents equal
    the seats they occupy in it, as they do in the status-quo reference, and
    the source zoning meets its own capacity rows.
    """
    from optimization.data.capacity_reference import CAPACITY_REFERENCE_SCHEMA_VERSION
    from optimization.solvers.burst_objectives import attendance_area_seats

    prepared = _prepared(market)
    program, kind, _ = lottery_placements(market, zoning)
    lotteries = program.shape[0]
    share = np.array(
        [
            attendance_area_seats(p) / p.capacity
            if p.capacity and str(p.program_id).split("-")[1:2] == ["GE"]
            else 0.0
            for p in prepared.programs
        ]
    )
    assigned = defaultdict(float)
    assigned_frl = defaultdict(float)
    departures = 0.0
    used = defaultdict(float)
    used_frl = defaultdict(float)
    school_zone = np.array(
        [
            zoning[p.school_node] if p.school_node is not None else -1
            for p in prepared.programs
        ],
        dtype=np.int64,
    )
    for lottery in range(lotteries):
        for s in np.flatnonzero(program[lottery] >= 0):
            p = program[lottery, s]
            node = int(prepared.node[s])
            in_zone = school_zone[p] >= 0 and school_zone[p] == zoning[node]
            to_aa = share[p] / lotteries if in_zone else 0.0
            frl = prepared.frl[s]
            assigned[node] += to_aa
            assigned_frl[node] += frl * to_aa
            departures += 1.0 / lotteries - to_aa
            if to_aa:
                school = prepared.programs[p].school_id
                used[school] += to_aa
                used_frl[school] += frl * to_aa
    for p in prepared.programs:
        if attendance_area_seats(p) and str(p.program_id).split("-")[1:2] == ["GE"]:
            used.setdefault(p.school_id, 0.0)
    return {
        "schema_version": CAPACITY_REFERENCE_SCHEMA_VERSION,
        "unit": problem.level.unit,
        "lottery_scale": lottery_scale,
        "students_by_area": {
            int(problem.G.nodes[n]["area_id"]): assigned[n] for n in problem.nodes
        },
        "seats_by_school": dict(used),
        "frl_students_by_area": {
            int(problem.G.nodes[n]["area_id"]): assigned_frl[n] for n in problem.nodes
        },
        "frl_seats_by_school": {school: used_frl.get(school, 0.0) for school in used},
        "non_aa_students": departures,
        "designated_students": float((kind == DESIGNATED).sum()) / lotteries,
        "unassigned_students": float((program < 0).sum()) / lotteries,
    }

