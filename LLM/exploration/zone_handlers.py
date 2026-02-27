"""Zone query and comparison tool handlers."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional

from .state import ToolResult

if TYPE_CHECKING:
    from .zoning_agent import ZoningAgent


def _load_zone_data(solution_path: str) -> Optional[dict]:
    """Load zone-level data from a solution's result.json.

    Returns a dict with:
      - "zone_data": {internal_zone_id (int): zone data dict}
      - "zone_index_map": {internal_zone_id (int): display_number (int, 1-indexed)}
      - "zone_colors": {internal_zone_id (int): color_name (str)}
      - "reverse_map": {display_number (int): internal_zone_id (int)}
    Or None on failure.
    """
    try:
        from Zone_Generation.Config.Constants import zone_colors

        result_path = os.path.join(solution_path, "result.json")
        with open(result_path, "r") as f:
            result = json.load(f)
        raw_zone_data = result.get("zone_data", {})
        normalized = {}
        for zone_id, data in raw_zone_data.items():
            d = data.copy()
            if "frl_pct" in d:
                d["FRL_pct"] = d["frl_pct"] * 100
            normalized[int(zone_id)] = d

        # Build display-number mapping: sorted internal IDs -> 1-indexed
        sorted_ids = sorted(normalized.keys())
        zone_index_map = {zid: idx + 1 for idx, zid in enumerate(sorted_ids)}
        reverse_map = {idx + 1: zid for idx, zid in enumerate(sorted_ids)}
        colors_map = {zid: zone_colors.get(idx, "#808080") for idx, zid in enumerate(sorted_ids)}

        return {
            "zone_data": normalized,
            "zone_index_map": zone_index_map,
            "zone_colors": colors_map,
            "reverse_map": reverse_map,
        }
    except Exception:
        return None


def handle_query_zone_data(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Query detailed data for specific zones in the current mapping."""
    display_ids = arguments.get("zone_ids", [])
    metrics_requested = arguments.get("metrics", [])

    centroid, path, count = agent._get_current_centroid()
    if centroid is None:
        return ToolResult("No current solution. Apply filters first.")

    loaded = _load_zone_data(path)
    if loaded is None:
        return ToolResult("No zone data available for this solution.")

    zone_data = loaded["zone_data"]
    zone_index_map = loaded["zone_index_map"]
    zone_colors_map = loaded["zone_colors"]
    reverse_map = loaded["reverse_map"]

    # Resolve display numbers to internal IDs
    if not display_ids:
        internal_ids = sorted(zone_data.keys())
    else:
        internal_ids = []
        for did in display_ids:
            if did in reverse_map:
                internal_ids.append(reverse_map[did])
            elif did in zone_data:
                internal_ids.append(did)
        if not internal_ids:
            valid = sorted(reverse_map.keys())
            return ToolResult(f"No matching zones found. Valid zone numbers: {valid}")

    result_lines = []
    for zone_id in internal_ids:
        if zone_id not in zone_data:
            continue

        data = zone_data[zone_id]
        display_num = zone_index_map.get(zone_id, zone_id)
        color = zone_colors_map.get(zone_id, "gray")
        result_lines.append(f"**Zone {display_num} ({color}):**")

        if not metrics_requested or 'demographics' in metrics_requested:
            frl = data.get('FRL_pct', 0)
            students = data.get('ge_students', 0)
            result_lines.append(f"  - Students: {students:.0f}, FRL: {frl:.1f}%")
            eth = data.get('ethnicity_pcts', {})
            if eth:
                top_eth = sorted(eth.items(), key=lambda x: x[1], reverse=True)[:2]
                eth_str = ", ".join([f"{k}: {v:.1f}%" for k, v in top_eth])
                result_lines.append(f"  - Ethnicity: {eth_str}")

        if not metrics_requested or 'programs' in metrics_requested:
            progs = data.get('total_programs', 0)
            lang = data.get('language_immersion_count', 0)
            result_lines.append(f"  - Programs: {progs}, Language immersion: {lang}")

        if not metrics_requested or 'quality' in metrics_requested:
            math = data.get('avg_math_score', 0)
            eng = data.get('avg_eng_score', 0)
            result_lines.append(f"  - Math: {math:.1f}, English: {eng:.1f}")

        if not metrics_requested or 'distance' in metrics_requested:
            dist = data.get('avg_closest_school_distance', 0)
            schools = data.get('schools_in_attendance_area', 0)
            result_lines.append(f"  - Avg distance: {dist:.2f}mi, Schools: {schools}")

    return ToolResult("\n".join(result_lines), solution_path=path)


def handle_compare_zones(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Compare two or more zones side-by-side on key metrics."""
    display_ids = arguments.get("zone_ids", [])
    if len(display_ids) < 2:
        return ToolResult("Need at least 2 zone numbers to compare.")

    centroid, path, count = agent._get_current_centroid()
    if centroid is None:
        return ToolResult("No current solution.")

    loaded = _load_zone_data(path)
    if loaded is None:
        return ToolResult("No zone data available for this solution.")

    zone_data = loaded["zone_data"]
    zone_index_map = loaded["zone_index_map"]
    zone_colors_map = loaded["zone_colors"]
    reverse_map = loaded["reverse_map"]

    # Resolve display numbers to internal IDs
    internal_ids = []
    for did in display_ids:
        if did in reverse_map:
            internal_ids.append(reverse_map[did])
        elif did in zone_data:
            internal_ids.append(did)
    if len(internal_ids) < 2:
        valid = sorted(reverse_map.keys())
        return ToolResult(f"Could not find enough matching zones. Valid zone numbers: {valid}")

    # Build header with display numbers and colors
    zone_labels = []
    for zid in internal_ids:
        dn = zone_index_map.get(zid, zid)
        color = zone_colors_map.get(zid, "gray")
        zone_labels.append(f"Zone {dn} ({color})")

    result_lines = [f"**Comparing {', '.join(zone_labels)}:**\n"]
    metrics_to_compare = [
        ('FRL_pct', 'FRL %', '{:.1f}%'),
        ('ge_students', 'Students', '{:.0f}'),
        ('total_programs', 'Programs', '{:.0f}'),
        ('avg_math_score', 'Math', '{:.1f}'),
        ('avg_closest_school_distance', 'Avg Dist', '{:.2f}mi'),
    ]
    for field, label, fmt in metrics_to_compare:
        values = []
        for zid in internal_ids:
            if zid in zone_data:
                val = zone_data[zid].get(field, 0)
                values.append(fmt.format(val))
            else:
                values.append("N/A")
        result_lines.append(f"{label}: {' vs '.join(values)}")

    return ToolResult("\n".join(result_lines), solution_path=path)
