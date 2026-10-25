"""Frozen MID demand and used AA seats for subsequent capacity balance rows.

Counts travel by base geography and school identity, independently of graph
aggregation and node numbering. Shared graphs and reporting counts stay intact.
"""

from __future__ import annotations

import math
from collections import defaultdict

from optimization.data.conversion import _node_base_ids
from optimization.mid_oracle import finite_grid_oracle
from optimization.solvers.burst_objectives import (
    attendance_area_seats,
    is_general_education,
)

CAPACITY_REFERENCE_SCHEMA_VERSION = 2


def build_capacity_reference(problem, zoning, market, lottery_scale):
    """Expected AA GE assignments by residence and occupied seats by school.

    Match the full market, including citywide alternatives. For a program with
    mixed AA/citywide seats, attribute its demand proportionally to the seat
    split; MID does not distinguish seats within a program. Remaining students
    are designated to free AA GE seats in-zone, proportionally to each school's
    free seats (exact with one school per zone). Attribute those designations
    to residents proportionally to their unmatched mass. Students left
    unassigned contribute to neither side of the capacity constraints, so each
    source zone's counted students equal its occupied seats by construction.
    """
    if not problem.level.is_base or problem.program_population != "All":
        raise ValueError(
            "A capacity reference requires a base-level All population problem."
        )
    if set(zoning) != set(problem.G):
        raise ValueError("A capacity reference requires a complete source zoning.")
    residents = {n: 0.0 for n in problem.nodes}
    for student_type in market.types:
        residents[student_type.node] += student_type.count
    if any(
        not math.isclose(residents[n], problem.students(n), abs_tol=1e-6)
        for n in problem.nodes
    ):
        raise ValueError(
            "Capacity reference assignment and optimization populations must agree "
            "at every base geographic unit. Check years, grades and population selectors."
        )

    result = finite_grid_oracle(market, zoning, lottery_scale, check_minimality=False)
    aa = {
        p.program_id: attendance_area_seats(p)
        if is_general_education(p.program_id)
        else 0
        for p in market.programs
    }
    shares = {
        p.program_id: aa[p.program_id] / p.capacity if p.capacity else 0.0
        for p in market.programs
    }
    if any(not 0 <= share <= 1 for share in shares.values()):
        raise ValueError(
            "Attendance-area seats must be between zero and program capacity."
        )
    departures = defaultdict(float)
    assigned = defaultdict(float)
    unmatched_by_node = defaultdict(float)
    unmatched = defaultdict(float)
    for student_type, masses, remaining in zip(
        market.types, result.assignment_masses, result.remaining_masses, strict=True
    ):
        for program_id, mass in zip(student_type.programs, masses, strict=True):
            assigned[student_type.node] += (
                student_type.count * mass / lottery_scale * shares[program_id]
            )
            departures[student_type.node] += (
                student_type.count * mass / lottery_scale * (1 - shares[program_id])
            )
        remaining_count = (
            student_type.count
            * (remaining[-1] if remaining else lottery_scale)
            / lottery_scale
        )
        unmatched_by_node[student_type.node] += remaining_count
        unmatched[zoning[student_type.node]] += remaining_count

    used = defaultdict(float)
    free = defaultdict(float)
    school_zones = {}
    for program in market.programs:
        seats = aa[program.program_id]
        if not seats:
            continue
        taken = result.demands[program.program_id] * shares[program.program_id]
        used[program.school_id] += taken
        free[program.school_id] += max(0.0, seats - taken)
        school_zones[program.school_id] = zoning[program.school_node]
    free_by_zone = defaultdict(float)
    for school, seats in free.items():
        free_by_zone[school_zones[school]] += seats
    designated_by_zone = {
        zone: min(count, free_by_zone[zone]) for zone, count in unmatched.items()
    }
    for node, count in unmatched_by_node.items():
        zone = zoning[node]
        if unmatched[zone]:
            assigned[node] += count * designated_by_zone[zone] / unmatched[zone]
    designated = 0.0
    for school, seats in free.items():
        zone = school_zones[school]
        total_free = free_by_zone[zone]
        taken = (
            designated_by_zone.get(zone, 0.0) * seats / total_free
            if total_free
            else 0.0
        )
        used[school] += taken
        designated += taken

    return {
        "schema_version": CAPACITY_REFERENCE_SCHEMA_VERSION,
        "unit": problem.level.unit,
        "lottery_scale": lottery_scale,
        "students_by_area": {
            int(problem.G.nodes[n]["area_id"]): assigned[n] for n in problem.nodes
        },
        "seats_by_school": dict(used),
        "non_aa_students": sum(departures.values()),
        "designated_students": designated,
        "unassigned_students": max(0.0, sum(unmatched.values()) - designated),
    }


def apply_capacity_reference(problem, reference):
    """Attach portable reference counts to a solve, including a saved stage."""
    if (
        reference.get("schema_version") != CAPACITY_REFERENCE_SCHEMA_VERSION
        or reference.get("unit") != problem.level.unit
    ):
        raise ValueError(
            "Capacity reference schema/geography does not match the problem."
        )
    students = {int(a): float(v) for a, v in reference["students_by_area"].items()}
    seats = {int(s): float(v) for s, v in reference["seats_by_school"].items()}
    if any(
        not math.isfinite(v) or v < 0 for v in (*students.values(), *seats.values())
    ):
        raise ValueError("Capacity reference counts must be finite and non-negative.")
    areas = {a for n in problem.nodes for a in _node_base_ids(problem.G, n)}
    schools = {
        int(s) for n in problem.nodes for s in problem.G.nodes[n].get("school_ids", [])
    }
    if set(students) != areas or not set(seats) <= schools:
        raise ValueError(
            "Capacity reference geography/schools do not cover the target graph."
        )
    problem.capacity_student_counts = {
        n: sum(students[a] for a in _node_base_ids(problem.G, n)) for n in problem.nodes
    }
    problem.capacity_seat_counts = {
        n: sum(seats.get(int(s), 0.0) for s in problem.G.nodes[n].get("school_ids", []))
        for n in problem.nodes
    }
