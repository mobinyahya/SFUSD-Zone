"""Filter manipulation tool handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .state import ToolResult, _direction_text, save_or_defer_version
from .filters import (
    FilterState,
    adjust_filter_bound,
    find_relaxation_needed,
    get_filter_summary,
    percentile_to_value,
    set_filter_bound,
)
from .metrics_config import CORE_METRICS, METRIC_BY_NAME

if TYPE_CHECKING:
    from .zoning_agent import ZoningAgent


def handle_tighten_filter(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Tighten the constraint for a specific metric."""
    metric_name = arguments["metric_name"]
    strength = arguments.get("strength", "moderate")

    pct_map = {"mild": 0.05, "moderate": 0.10, "aggressive": 0.25}
    pct = pct_map.get(strength, 0.10)

    before_centroid, _, before_count = agent._get_current_centroid()
    if before_count <= 1:
        return ToolResult(f"Cannot tighten: only {before_count} solution(s) remaining. Consider loosening other filters first.")

    filtered = agent._get_filtered_solutions()
    metric = METRIC_BY_NAME[metric_name]
    new_bound = adjust_filter_bound(
        agent.pareto_original, filtered, agent.filter_state,
        metric_name, "tighten", pct,
        current_value=float(before_centroid[metric.column]),
    )

    if metric.direction == "minimize":
        agent.filter_state.bounds[metric_name].max_bound = new_bound
    else:
        agent.filter_state.bounds[metric_name].min_bound = new_bound

    agent._invalidate_centroid()
    after_centroid, after_path, after_count = agent._get_current_centroid()

    before_val = before_centroid[metric.column]
    after_val = after_centroid[metric.column] if after_centroid is not None else before_val

    strength_text = {"mild": "mildly", "moderate": "moderately", "aggressive": "aggressively"}[strength]
    pending_version_id = len(agent.state.versions)
    result_lines = [
        f"v{pending_version_id}: Tightened {metric_name} ({strength_text})",
        f"- Solutions: {before_count} -> {after_count}",
        f"- {metric_name} improved: {before_val:.3f} -> {after_val:.3f}"
    ]

    save_or_defer_version(
        agent,
        description=f"Tightened {metric_name} ({strength})",
        solution_path=after_path,
        solution_count=after_count,
    )

    return ToolResult("\n".join(result_lines), solution_path=after_path)


def handle_loosen_filter(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Loosen the constraint for a specific metric."""
    metric_name = arguments["metric_name"]
    strength = arguments.get("strength", "moderate")

    pct_map = {"mild": 0.10, "moderate": 0.25, "aggressive": 0.50}
    pct = pct_map.get(strength, 0.25)

    filtered = agent._get_filtered_solutions()
    new_bound = adjust_filter_bound(
        agent.pareto_original, filtered, agent.filter_state,
        metric_name, "loosen", pct,
    )

    if new_bound is None:
        return ToolResult(f"'{metric_name}' is already unconstrained.")

    metric = METRIC_BY_NAME[metric_name]
    _, _, before_count = agent._get_current_centroid()

    if metric.direction == "minimize":
        agent.filter_state.bounds[metric_name].max_bound = new_bound
    else:
        agent.filter_state.bounds[metric_name].min_bound = new_bound

    agent._invalidate_centroid()
    _, after_path, after_count = agent._get_current_centroid()

    pending_version_id = len(agent.state.versions)
    desc = f"Loosened {metric_name} ({strength})"

    save_or_defer_version(agent, description=desc, solution_path=after_path,
                          solution_count=after_count)

    return ToolResult(
        f"v{pending_version_id}: Loosened {metric_name} ({strength})\n- {before_count} -> {after_count} solutions",
        solution_path=after_path,
    )


def handle_set_filter(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Set an explicit filter bound for a metric."""
    metric_name = arguments["metric_name"]
    raw_value = arguments.get("value")
    pctl = arguments.get("percentile")

    if raw_value is not None and pctl is not None:
        return ToolResult("Provide either 'value' or 'percentile', not both.")

    metric = METRIC_BY_NAME[metric_name]
    _, _, before_count = agent._get_current_centroid()

    if pctl is not None:
        bound_value = percentile_to_value(agent.pareto_original, metric_name, pctl)
    else:
        bound_value = raw_value

    set_filter_bound(agent.filter_state, metric_name, bound_value)
    agent._invalidate_centroid()
    _, after_path, after_count = agent._get_current_centroid()

    pending_version_id = len(agent.state.versions)
    if bound_value is None:
        desc = f"Cleared filter on {metric_name}"
        detail = "Filter removed (unconstrained)"
    elif pctl is not None:
        desc = f"Set {metric_name} to {pctl:.0f}th percentile ({bound_value:.4f})"
        detail = f"Bound set to {bound_value:.4f} ({pctl:.0f}th percentile)"
    else:
        desc = f"Set {metric_name} to {bound_value:.4f}"
        detail = f"Bound set to {bound_value:.4f}"

    save_or_defer_version(agent, description=desc, solution_path=after_path,
                          solution_count=after_count)

    return ToolResult(
        f"v{pending_version_id}: {desc}\n- {detail}\n- Solutions: {before_count} -> {after_count}",
        solution_path=after_path,
    )


def handle_get_filter_bounds(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Get current filter bounds and statistics for metrics."""
    filtered = agent._get_filtered_solutions()
    category = arguments.get("category")
    return ToolResult(get_filter_summary(agent.filter_state, agent.pareto_original, filtered, show_category=category))


def handle_find_feasible_relaxation(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Find which filters need to be relaxed to restore feasibility."""
    suggestions = find_relaxation_needed(
        agent.pareto_original, agent.filter_state
    )

    if not suggestions:
        return ToolResult("Unable to find relaxations that would restore feasibility. Try loosening multiple filters manually.")

    lines = ["**Suggested Relaxations** (relaxing any ONE of these could restore feasibility):\n"]
    for metric_name, new_bound in suggestions.items():
        metric = METRIC_BY_NAME.get(metric_name)
        if metric and metric.direction == "minimize":
            current = agent.filter_state.bounds[metric_name].max_bound
            lines.append(f"- **{metric_name}**: relax max from {current:.4f} to {new_bound:.4f}")
        elif metric:
            current = agent.filter_state.bounds[metric_name].min_bound
            lines.append(f"- **{metric_name}**: relax min from {current:.4f} to {new_bound:.4f}")
    lines.append("\nAsk the user which metric they're willing to compromise on.")
    return ToolResult("\n".join(lines))


def handle_apply_feedback_filters(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Reset all filters and apply multiple metric constraints in one batch."""
    adjustments = arguments.get("adjustments", [])
    if not adjustments:
        return ToolResult("No adjustments provided. Analyze the user's feedback and specify metric adjustments.")

    agent.filter_state = FilterState()
    agent._invalidate_centroid()
    _, _, before_count = agent._get_current_centroid()

    tighten_pct_map = {"mild": 0.05, "moderate": 0.10, "aggressive": 0.25}
    loosen_pct_map = {"mild": 0.10, "moderate": 0.25, "aggressive": 0.50}
    applied = []

    for adj in adjustments:
        metric_name = adj.get("metric_name")
        direction = adj.get("direction", "tighten")
        strength = adj.get("strength", "moderate")

        if metric_name not in METRIC_BY_NAME:
            continue

        metric = METRIC_BY_NAME[metric_name]
        if metric.direction is None:
            applied.append(f"- {metric_name}: skipped (informational metric)")
            continue
        filtered = agent._get_filtered_solutions()
        if len(filtered) <= 1:
            applied.append(f"- {metric_name}: skipped (only {len(filtered)} solution left)")
            break

        centroid_val = None
        if direction == "tighten":
            centroid, _, _ = agent._get_current_centroid()
            if centroid is not None:
                centroid_val = float(centroid[metric.column])

        pct = (tighten_pct_map if direction == "tighten" else loosen_pct_map).get(strength, 0.10)
        new_bound = adjust_filter_bound(
            agent.pareto_original, filtered, agent.filter_state,
            metric_name, direction, pct,
            current_value=centroid_val,
        )

        if new_bound is not None:
            if metric.direction == "minimize":
                if direction == "tighten":
                    agent.filter_state.bounds[metric_name].max_bound = new_bound
                else:
                    agent.filter_state.bounds[metric_name].max_bound = new_bound
            else:
                if direction == "tighten":
                    agent.filter_state.bounds[metric_name].min_bound = new_bound
                else:
                    agent.filter_state.bounds[metric_name].min_bound = new_bound
            agent._invalidate_centroid()
            applied.append(f"- {'Tightened' if direction == 'tighten' else 'Loosened'} {metric_name} ({strength})")

    agent._invalidate_centroid()
    after_centroid, after_path, after_count = agent._get_current_centroid()

    metric_lines = []
    if after_centroid is not None:
        for m in CORE_METRICS[:6]:
            if m.column in after_centroid.index:
                val = after_centroid[m.column]
                metric_lines.append(f"- {m.display_name}: {val:.3f} {_direction_text(m)}")

    pending_version_id = len(agent.state.versions)
    desc = f"Applied {len(applied)} feedback-based filters"

    save_or_defer_version(agent, description=desc, solution_path=after_path,
                          solution_count=after_count)

    result_lines = [
        f"v{pending_version_id}: Applied filters from feedback analysis",
        f"- Solutions: {before_count} -> {after_count}",
        "",
        "Adjustments applied:",
        *applied,
    ]
    if metric_lines:
        result_lines.extend(["", "New centroid metrics:", *metric_lines])
    elif after_count == 0:
        result_lines.append("\nNo solutions remain. Consider loosening some constraints.")

    return ToolResult("\n".join(result_lines), solution_path=after_path)
