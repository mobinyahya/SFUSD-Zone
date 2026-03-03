"""Solution display and versioning tool handlers."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from .state import ToolResult, _direction_text
from .metrics_config import (
    ALL_METRICS,
    CATEGORIES,
    CORE_METRICS,
    get_metrics_by_category,
    get_metric_summary,
    search_metrics as search_metrics_func,
)
from .pareto import normalize_metrics, get_centroid_solution

if TYPE_CHECKING:
    from .zoning_agent import ZoningAgent


# ============================================================================
# Solution display
# ============================================================================

def handle_get_solution(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Get the balanced mapping for a given version (defaults to current)."""
    version = arguments.get("version")

    if version is not None:
        if version < 0 or version >= len(agent.state.versions):
            return ToolResult(f"Invalid version {version}. Valid versions: 0-{len(agent.state.versions) - 1}.")
        ver = agent.state.versions[version]
        path = ver.solution_path
        count = ver.solution_count
        version_id = ver.version_id
        if path is None:
            return ToolResult(f"Version {version} has no solution path.")
        match = agent.pareto_original[agent.pareto_original["path"] == path]
        if match.empty:
            return ToolResult(f"Could not load solution for version {version}.")
        centroid = match.iloc[0]
    else:
        centroid, path, count = agent._get_current_centroid()
        version_id = agent.state.current_version
        if centroid is None:
            return ToolResult("No solutions match the current filters. Use find_feasible_relaxation to see which constraints to relax.")

    show_all = arguments.get("show_all_metrics", False)
    metrics_to_show = ALL_METRICS if show_all else CORE_METRICS

    percentile_ranks = agent._compute_percentile_for_solution(centroid)

    if show_all:
        lines = [f"v{version_id}: Complete metrics ({count} solutions available)\n"]
        for category_key, category_name in CATEGORIES.items():
            category_metrics = get_metrics_by_category(category_key)
            if not category_metrics:
                continue
            lines.append(f"\n**{category_name}:**")
            for metric in category_metrics:
                if metric.column not in centroid.index:
                    continue
                value = centroid[metric.column]
                direction_text = _direction_text(metric)
                pct_info = percentile_ranks.get(metric.column)
                pct_text = f" ({pct_info['percentile']}th percentile)" if pct_info else ""
                lines.append(f"- {metric.display_name}: {value:.3f}{pct_text} {direction_text}")
        lines.append("\nWould you like to adjust any of these metrics?")
    else:
        lines = [f"v{version_id}: {count} solutions\n"]
        for metric in metrics_to_show[:8]:
            if metric.column not in centroid.index:
                continue
            value = centroid[metric.column]
            direction_text = _direction_text(metric)
            pct_info = percentile_ranks.get(metric.column)
            pct_text = f" ({pct_info['percentile']}th percentile)" if pct_info else ""
            lines.append(f"- {metric.display_name}: {value:.3f}{pct_text} {direction_text}")
        lines.append("\nAdjust metrics?")

    return ToolResult("\n".join(lines), solution_path=path)


def handle_list_all_metrics(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """List all available metrics organized by category."""
    return ToolResult(get_metric_summary())


def handle_search_metrics(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Search for metrics by keyword."""
    query = arguments.get("query", "")
    matches = search_metrics_func(query)

    if not matches:
        return ToolResult(f"No metrics found matching '{query}'. Use list_all_metrics to see all available metrics.")

    lines = [f"**Metrics matching '{query}':**\n"]
    for m in matches:
        lines.append(f"- **{m.display_name}** ({m.category}): {m.description} ({_direction_text(m)})")

    return ToolResult("\n".join(lines))


# ============================================================================
# Versioning
# ============================================================================

def handle_undo_action(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Undo the last filter change and restore previous mapping."""
    steps = arguments.get("steps", 1)
    version = agent.state.undo(steps)

    if version is None:
        return ToolResult(f"Cannot undo {steps} steps. Only {agent.state.current_version} versions available.")

    agent.filter_state = copy.deepcopy(version.filter_state)
    agent._invalidate_centroid()
    _, path, count = agent._get_current_centroid()

    return ToolResult(
        f"Undid {steps} step(s) to v{version.version_id}\n- {version.description}\n- {count} solutions available",
        solution_path=path,
    )


def handle_show_version_history(agent: ZoningAgent, arguments: dict) -> ToolResult:
    """Show the history of filter changes and mapping states."""
    if not agent.state.versions:
        return ToolResult("No version history.")

    lines = ["**Version History:**\n"]
    for v in agent.state.versions:
        marker = ">" if v.version_id == agent.state.current_version else " "
        lines.append(f"{marker} v{v.version_id}: {v.description} ({v.solution_count} solutions)")

    return ToolResult("\n".join(lines))


def build_clusters_response(agent: ZoningAgent) -> list:
    """Build cluster data for frontend response."""
    if agent.state.cluster_labels is None:
        return []

    clusters = []
    n_clusters = len(agent.state.cluster_centers)

    for cluster_id in range(n_clusters):
        mask = agent.state.cluster_labels == cluster_id
        cluster_solutions = agent.state.clustered_solutions[mask]

        if len(cluster_solutions) == 0:
            continue

        normalized = normalize_metrics(cluster_solutions)
        centroid_solution, _ = get_centroid_solution(cluster_solutions, normalized)

        direction_info = agent.state.cluster_directions.get(cluster_id, {})

        clusters.append({
            "id": cluster_id + 1,
            "label": direction_info.get("direction_label", f"Cluster {cluster_id + 1}"),
            "count": int(mask.sum()),
            "path": centroid_solution["path"],
        })

    return clusters
