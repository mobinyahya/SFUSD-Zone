"""Distance and GE proximity metrics.

Distances are ``distance_dict`` great-circle miles between node points; a
school sits at its node's point. ``avg_any``/``avg_farthest`` are unweighted
node means. The ``nearest`` fields pair each node with its nearest in-zone GE
school and weight it by GE students (``ge_*``) or by the problem's own
``student_attribute`` (``students``); a populated node is one with a positive
weight.
"""

from __future__ import annotations

from metrics.base import MetricOutput, MetricsContext
from metrics.columns import MetricColumns
from metrics.programs import ge_schools

GE_PROXIMITY_RADIUS = 0.5


def compute(context: MetricsContext) -> MetricOutput:
    distance_dict = context.G.graph.get("distance_dict", {})
    school_data = context.G.graph.get("school_data", {})
    ge_school_ids = ge_schools(context)
    ge_school_to_zone = _ge_school_zones(context, ge_school_ids)

    zone_data: dict[int, dict] = {}
    avg_any_values: list[float] = []
    avg_farthest_values: list[float] = []
    out_of_zone_values: list[float] = []
    attendance_area_values: list[int] = []
    nearby_ge_values: list[float] = []
    nearest_totals = {
        population: {"weighted": 0.0, "weight": 0.0, "max": None, "unserved": 0.0}
        for population in _NEAREST_POPULATIONS
    }

    for zone_id, nodes in context.zone_nodes.items():
        schools = context.zone_schools.get(zone_id, [])
        ge_in_zone = [sid for sid in schools if sid in ge_school_ids]

        avg_any, avg_farthest = _in_zone_distances(
            nodes, ge_in_zone, context.school_to_node, distance_dict
        )
        out_of_zone = _nearby_out_of_zone_ge(
            nodes, zone_id, ge_school_to_zone, context.school_to_node, distance_dict
        )
        nearby_ge = _student_weighted_nearby_ge(
            context, nodes, ge_in_zone, context.school_to_node, distance_dict
        )
        attendance_area = _schools_in_attendance_area(schools, school_data)
        nearest = _nearest_ge_school_distances(
            nodes, ge_in_zone, context.school_to_node, distance_dict
        )

        zone_data[zone_id] = {
            "school_ids": sorted(int(sid) for sid in schools),
            "attendance_school_ids": sorted(
                int(sid)
                for sid in schools
                if school_data.get(sid, {}).get("category") == "Attendance"
            ),
            "ge_school_ids": sorted(int(sid) for sid in ge_in_zone),
            "avg_any_ge_school_distance": avg_any,
            "avg_farthest_ge_school_distance": avg_farthest,
            "avg_out_of_zone_ge_schools": out_of_zone,
            "schools_in_attendance_area": attendance_area,
            "ge_schools_within_half_mile": nearby_ge,
        }
        for population, attribute in _NEAREST_POPULATIONS.items():
            weights = {
                node: float(
                    context.G.nodes[node].get(
                        attribute or context.problem.student_attribute, 0.0
                    )
                )
                for node in nodes
            }
            summary = _nearest_summary(nearest, weights)
            p = "ge_" if population == "ge" else ""
            zone_data[zone_id].update(
                {
                    f"{p}weighted_avg_nearest_ge_school_distance": summary["avg"],
                    f"max_{p}populated_node_nearest_ge_school_distance": summary["max"],
                    f"{p}populated_nodes": summary["populated"],
                    f"{p}students_without_in_zone_ge_school": summary["unserved"],
                }
            )
            totals = nearest_totals[population]
            totals["weighted"] += summary["weighted"]
            totals["weight"] += summary["weight"]
            totals["unserved"] += summary["unserved"]
            if summary["max"] is not None:
                totals["max"] = max(totals["max"] or 0.0, summary["max"])
        avg_any_values.append(avg_any)
        avg_farthest_values.append(avg_farthest)
        out_of_zone_values.append(out_of_zone)
        attendance_area_values.append(attendance_area)
        nearby_ge_values.append(nearby_ge)

    metrics = {
        MetricColumns.AVG_ANY_ZONE_GE_SCHOOL_DISTANCE: _mean(avg_any_values),
        MetricColumns.AVG_FARTHEST_ZONE_GE_SCHOOL_DISTANCE: _mean(avg_farthest_values),
        MetricColumns.AVG_OUT_OF_ZONE_GE_SCHOOLS: _mean(out_of_zone_values),
        MetricColumns.AVG_SCHOOLS_IN_ATTENDANCE_AREA: _mean(attendance_area_values),
        MetricColumns.AVG_GE_SCHOOLS_WITHIN_HALF_MILE: _mean(nearby_ge_values),
    }
    for population, columns in _NEAREST_COLUMNS.items():
        totals = nearest_totals[population]
        avg_column, max_column, unserved_column = columns
        metrics[avg_column] = (
            totals["weighted"] / totals["weight"] if totals["weight"] > 0 else None
        )
        metrics[max_column] = totals["max"]
        metrics[unserved_column] = totals["unserved"]
    return MetricOutput(metrics=metrics, zone_data=zone_data)


# Weight attribute per nearest-school population; ``None`` is the problem's own
# ``student_attribute``.
_NEAREST_POPULATIONS = {"ge": "ge_students", "students": None}

_NEAREST_COLUMNS = {
    "ge": (
        MetricColumns.GE_WEIGHTED_AVG_NEAREST_GE_SCHOOL_DISTANCE,
        MetricColumns.MAX_GE_POPULATED_NODE_NEAREST_GE_SCHOOL_DISTANCE,
        MetricColumns.GE_STUDENTS_WITHOUT_IN_ZONE_GE_SCHOOL,
    ),
    "students": (
        MetricColumns.WEIGHTED_AVG_NEAREST_GE_SCHOOL_DISTANCE,
        MetricColumns.MAX_POPULATED_NODE_NEAREST_GE_SCHOOL_DISTANCE,
        MetricColumns.STUDENTS_WITHOUT_IN_ZONE_GE_SCHOOL,
    ),
}


def _nearest_ge_school_distances(
    nodes: list[int],
    ge_in_zone: list[int],
    school_to_node: dict[int, int],
    distance_dict: dict,
) -> dict[int, float]:
    """Each node's distance to its nearest in-zone GE school, where one exists."""
    nearest: dict[int, float] = {}
    for node in nodes:
        distances = [
            distance_dict[node][school_to_node[sid]]
            for sid in ge_in_zone
            if node in distance_dict
            and sid in school_to_node
            and school_to_node[sid] in distance_dict[node]
        ]
        if distances:
            nearest[node] = min(distances)
    return nearest


def _nearest_summary(
    nearest: dict[int, float], weights: dict[int, float]
) -> dict[str, float | int | None]:
    """Weighted mean and populated-node max of ``nearest`` over one zone.

    Populated nodes with no in-zone GE school count toward ``unserved`` only.
    """
    weighted = weight = unserved = 0.0
    populated = 0
    largest: float | None = None
    for node, students in weights.items():
        if students <= 0:
            continue
        populated += 1
        if node not in nearest:
            unserved += students
            continue
        weighted += students * nearest[node]
        weight += students
        largest = nearest[node] if largest is None else max(largest, nearest[node])
    return {
        "avg": weighted / weight if weight > 0 else None,
        "max": largest,
        "populated": populated,
        "unserved": unserved,
        "weighted": weighted,
        "weight": weight,
    }


def _ge_school_zones(
    context: MetricsContext, ge_school_ids: set[int]
) -> dict[int, int]:
    out: dict[int, int] = (
        {school_id: -1 for school_id in ge_school_ids}
        if context.solution.metadata.get("partial_assignment")
        else {}
    )
    for zone_id, schools in context.zone_schools.items():
        for sid in schools:
            if sid in ge_school_ids:
                out[sid] = zone_id
    return out


def _in_zone_distances(
    nodes: list[int],
    ge_in_zone: list[int],
    school_to_node: dict[int, int],
    distance_dict: dict,
) -> tuple[float, float]:
    avg_any_values = []
    farthest_values = []
    for node in nodes:
        distances = [
            distance_dict[node][school_to_node[sid]]
            for sid in ge_in_zone
            if node in distance_dict
            and sid in school_to_node
            and school_to_node[sid] in distance_dict[node]
        ]
        if distances:
            avg_any_values.append(sum(distances) / len(distances))
            farthest_values.append(max(distances))
    return _mean(avg_any_values), _mean(farthest_values)


def _nearby_out_of_zone_ge(
    nodes: list[int],
    zone_id: int,
    ge_school_to_zone: dict[int, int],
    school_to_node: dict[int, int],
    distance_dict: dict,
) -> float:
    counts = []
    for node in nodes:
        if node not in distance_dict:
            continue
        nearby = 0
        for school_id, school_zone in ge_school_to_zone.items():
            if school_zone == zone_id or school_id not in school_to_node:
                continue
            school_node = school_to_node[school_id]
            if (
                distance_dict[node].get(school_node, float("inf"))
                <= GE_PROXIMITY_RADIUS
            ):
                nearby += 1
        counts.append(nearby)
    return _mean(counts)


def _student_weighted_nearby_ge(
    context: MetricsContext,
    nodes: list[int],
    ge_in_zone: list[int],
    school_to_node: dict[int, int],
    distance_dict: dict,
) -> float:
    weighted_count = 0.0
    total_students = 0.0
    for node in nodes:
        if node not in distance_dict:
            continue
        students = float(
            context.G.nodes[node].get(context.problem.student_attribute, 0.0)
        )
        nearby = 0
        for school_id in ge_in_zone:
            if school_id not in school_to_node:
                continue
            school_node = school_to_node[school_id]
            if (
                distance_dict[node].get(school_node, float("inf"))
                <= GE_PROXIMITY_RADIUS
            ):
                nearby += 1
        weighted_count += nearby * students
        total_students += students
    return weighted_count / total_students if total_students > 0 else 0.0


def _schools_in_attendance_area(schools: list[int], school_data: dict) -> int:
    school_set = set(schools)
    count = 0
    for school_id in schools:
        attendance_area = school_data.get(school_id, {}).get("attendance_area")
        if attendance_area in school_set:
            count += 1
    return count


def _mean(values) -> float:
    return sum(values) / len(values) if values else 0.0
