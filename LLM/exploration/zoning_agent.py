"""
School Zoning Exploration Agent.

An LLM-powered agent that helps users iteratively explore school zoning proposals
using adjustable filters on a Pareto frontier of solutions.
"""

import os
import json
import copy
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from .metrics_config import (
    ALL_METRICS,
    CORE_METRICS,
    METRIC_BY_NAME,
    CATEGORIES,
    get_metrics_by_category,
    get_metric_summary,
    search_metrics,
)
from .pareto import (
    load_solutions,
    normalize_metrics,
    compute_pareto_frontier,
    get_centroid_solution,
)
from .filters import (
    FilterState,
    apply_filters,
    get_filter_summary,
    adjust_filter_bound,
    find_relaxation_needed,
)
from .clusters import (
    vectorize_solutions,
    cluster_solutions,
    compute_cluster_directions,
    get_cluster_bounds,
    format_cluster_summary,
)


def _direction_text(metric) -> str:
    if metric.direction == "minimize":
        return "(lower better)"
    elif metric.direction == "maximize":
        return "(higher better)"
    return "(informational)"


# ============================================================================
# TOOL RESULT: STRUCTURED RETURN FROM TOOL EXECUTION
# ============================================================================

@dataclass
class ToolResult:
    """Structured result from a tool execution.
    
    text: The string sent to the LLM as tool output
    solution_path: If the tool produced/changed a solution, the path to it
    clusters: If the tool produced cluster data, the list of cluster dicts
    """
    text: str
    solution_path: Optional[str] = None
    clusters: Optional[list] = None


# ============================================================================
# STATE MANAGEMENT FOR VERSIONED ZONING PROPOSALS
# ============================================================================

@dataclass
class ProposalVersion:
    """A versioned snapshot of a zoning proposal state."""
    version_id: int
    timestamp: str
    filter_state: FilterState
    solution_path: Optional[str] = None
    solution_count: int = 0
    description: str = ""


@dataclass
class AgentState:
    """Complete state for the zoning agent session."""
    # Version history
    versions: list[ProposalVersion] = field(default_factory=list)
    current_version: int = 0
    
    # Clustering state
    cluster_labels: Optional[list] = None
    cluster_centers: Optional[list] = None
    cluster_directions: Optional[dict] = None
    clustered_solutions: Optional[object] = None
    clustered_vectors: Optional[object] = None
    
    # Interaction state
    awaiting_confirmation: bool = False
    pending_action: Optional[dict] = None
    last_action: str = ""
    
    def save_version(self, filter_state: FilterState, solution_path: str = None, 
                     solution_count: int = 0, description: str = "") -> int:
        """Save a new version and return version ID."""
        version_id = len(self.versions)
        version = ProposalVersion(
            version_id=version_id,
            timestamp=datetime.now().isoformat(),
            filter_state=copy.deepcopy(filter_state),
            solution_path=solution_path,
            solution_count=solution_count,
            description=description
        )
        self.versions.append(version)
        self.current_version = version_id
        return version_id
    
    def undo(self, steps: int = 1) -> Optional[ProposalVersion]:
        """Undo to a previous version. Returns the version or None if not possible."""
        target_version = self.current_version - steps
        if target_version < 0 or target_version >= len(self.versions):
            return None
        self.current_version = target_version
        return self.versions[target_version]
    
    def get_current_version(self) -> Optional[ProposalVersion]:
        """Get the current version."""
        if 0 <= self.current_version < len(self.versions):
            return self.versions[self.current_version]
        return None


# ============================================================================
# DYNAMIC TOOL DEFINITIONS
# ============================================================================

def build_tools():
    """Build tool definitions with current metric names."""
    # Get all metric display names for enum
    all_metric_names = [m.display_name for m in ALL_METRICS]
    category_names = list(CATEGORIES.keys())
    
    return [
        {
            "type": "function",
            "function": {
                "name": "query_zone_data",
                "description": "Query detailed data for specific zones in the current solution. Returns demographics, programs, quality metrics, and distances for requested zones.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "zone_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of zone display numbers as shown on the map (e.g., [1, 2, 3]). If empty, returns summary for all zones.",
                        },
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Which metric categories to include: 'demographics', 'programs', 'quality', 'distance'. If empty, returns all.",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_zones",
                "description": "Compare two or more zones side-by-side on key metrics. Useful for understanding differences between zones.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "zone_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of 2+ zone display numbers as shown on the map (e.g., [1, 3]).",
                            "minItems": 2,
                        }
                    },
                    "required": ["zone_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "undo_action",
                "description": "Undo the last filter change and restore the previous solution state. Can undo multiple steps.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "integer",
                            "description": "Number of steps to undo (default 1)",
                            "minimum": 1,
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "show_version_history",
                "description": "Show the history of filter changes and solution states in this session.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_solution",
                "description": "Get the current 'balanced' centroid solution based on the current filters. Use show_all_metrics based on user intent: if they're asking for overview/details/depth about metrics, set to true. If they just want current status or quick check, set to false.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "show_all_metrics": {
                            "type": "boolean",
                            "description": "Set to true when user is asking for detailed information, comprehensive view, or wants to understand all available metrics. Set to false for quick status checks. Use your judgment based on user intent, not specific trigger phrases.",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_all_metrics",
                "description": "List all available metrics organized by category with their descriptions and directions (higher/lower is better). Use this to understand what metrics are available before filtering.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_metrics",
                "description": "Search for metrics by keyword. Returns matching metrics with their details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term to find in metric names or descriptions (e.g., 'spanish', 'diversity', 'math').",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tighten_filter",
                "description": "Tighten the constraint for a specific metric to improve it. IMPORTANT: Before calling this tool, ALWAYS explain the trade-offs to the user and ask them to choose a strength level (mild/moderate/aggressive). Only call this tool AFTER the user confirms their choice.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "type": "string",
                            "description": "The display name of the metric to tighten.",
                            "enum": all_metric_names,
                        },
                        "strength": {
                            "type": "string",
                            "description": "How aggressively to tighten: 'mild' (~5% of range), 'moderate' (~10%), or 'aggressive' (~25%)",
                            "enum": ["mild", "moderate", "aggressive"],
                        },
                    },
                    "required": ["metric_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "loosen_filter",
                "description": "Loosen the constraint for a specific metric to allow more diverse solutions. IMPORTANT: Before calling this tool, ALWAYS explain what loosening this constraint will enable (more solutions, ability to improve other metrics) and what the trade-off is (accepting worse values for this metric).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "type": "string",
                            "description": "The display name of the metric to loosen.",
                            "enum": all_metric_names,
                        },
                        "strength": {
                            "type": "string",
                            "description": "How much to loosen: 'mild' (~10% toward unconstrained), 'moderate' (~25%), or 'aggressive' (~50%)",
                            "enum": ["mild", "moderate", "aggressive"],
                        },
                    },
                    "required": ["metric_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_filter_bounds",
                "description": "Get current filter bounds and statistics for metrics. Shows the current constraints, ranges in all solutions, and ranges in filtered solutions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Optional: only show metrics in this category.",
                            "enum": category_names,
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_feasible_relaxation",
                "description": "When there are no feasible solutions with current filters, find which filters need to be relaxed and by how much to restore feasibility.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "show_solution_clusters",
                "description": "Group the current feasible solutions into clusters and show a representative solution from each cluster with an interpretable direction label. Useful when there are many solutions and the user wants to see different 'types' of solutions available.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n_clusters": {
                            "type": "integer",
                            "description": "Number of clusters to create. Default is automatically chosen based on solution count.",
                            "minimum": 2,
                            "maximum": 8,
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "select_cluster",
                "description": "Select a cluster from the previous show_solution_clusters results. This will tighten all metric filters to only include solutions within that cluster.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {
                            "type": "integer",
                            "description": "The cluster number to select (1 to N, as shown in show_solution_clusters)",
                        },
                    },
                    "required": ["cluster_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_feedback_filters",
                "description": "Reset all filters and apply multiple metric constraints in one batch based on analysis of user feedback. Use this when the user asks to generate a new solution from their feedback, or when you identify clear metric preferences from their pros/cons notes. Be aggressive - apply a constraint for every identifiable preference.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "adjustments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "metric_name": {
                                        "type": "string",
                                        "description": "The display name of the metric to adjust.",
                                        "enum": all_metric_names,
                                    },
                                    "direction": {
                                        "type": "string",
                                        "description": "'tighten' to improve this metric (stricter filter), 'loosen' to relax it.",
                                        "enum": ["tighten", "loosen"],
                                    },
                                    "strength": {
                                        "type": "string",
                                        "description": "How aggressively to adjust.",
                                        "enum": ["mild", "moderate", "aggressive"],
                                    },
                                },
                                "required": ["metric_name", "direction", "strength"],
                            },
                            "description": "List of metric adjustments to apply. Include one entry for EVERY preference you can identify from the user's feedback.",
                        },
                    },
                    "required": ["adjustments"],
                },
            },
        },
    ]


def build_system_prompt():
    """Build system prompt with current metric information."""
    
    return """You are an AI administrator assistant for SFUSD school zoning optimization.

## Response Style
- 80-100 words by default. Expand when user asks for detail ("tell me more", "explain", "why").
- Bullets for lists. Lead with action/summary.
- Speak to administrators as policy experts. No code references, function names, file paths, or technical jargon.
- Use clear text for metric directions: "lower FRL deviation", "more programs" -- never arrows.

## State System
Each filter change creates a versioned snapshot. Always show version number and solution count.
Users can undo to previous versions.

## Adjustment Flow
When user requests a change:
1. Explain trade-offs first (what improves, what might worsen)
2. Ask strength: mild / moderate / aggressive
3. Apply and confirm with before/after counts and key metric changes

Example:
User: "Prioritize math scores"
Agent: "Higher math scores typically means fewer solutions and slightly longer distances.
How much: mild, moderate, or aggressive?"
User: "moderate"
Agent: "v3: Tightened Math Scores (moderately). Solutions: 289 -> 183. Math improved: 2409 -> 2543."

## Metrics
Minimize (lower better): FRL deviation, racial deviation, distances, boundary cost
Maximize (higher better): programs, quality scores, school count

## Feedback Context
When saved solutions with pros/cons are provided in context, use that feedback as your primary signal.
Map complaints to metric tightening, praise to maintaining. Reference solutions by number.

## Zone Numbering
Zones are numbered 1 through N as shown on the map. Each zone has a color (e.g., Zone 1 (red), Zone 2 (midnightblue)).
When referencing zones, always use the display number and color, never internal IDs.

## Never
- Show file paths or internal details
- Use arrows or code syntax
- Be verbose when concise suffices
- List 20+ metrics unprompted
- Reference internal zone IDs -- always use display numbers (1, 2, 3...)"""


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
        colors_map = {zid: zone_colors.get(zid, "#808080") for zid in sorted_ids}

        return {
            "zone_data": normalized,
            "zone_index_map": zone_index_map,
            "zone_colors": colors_map,
            "reverse_map": reverse_map,
        }
    except Exception:
        return None


class ZoningAgent:
    """Interactive agent for exploring school zoning solutions with state management."""

    def __init__(self, csv_path: str | Path):
        """
        Initialize the agent with zoning solution data.

        Args:
            csv_path: Path to the CSV file with zoning solutions
        """
        load_dotenv()

        # Initialize OpenAI client with Google's compatible endpoint
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model = "gemini-2.5-flash"

        # Deferred version saving: when True, filter tool calls accumulate descriptions
        # instead of saving a new version per call. One LLM turn = one version save.
        self._defer_version_save: bool = False
        self._pending_descriptions: list[str] = []
        self._pending_solution_path: Optional[str] = None
        
        # Load and process solutions
        self.all_solutions = load_solutions(csv_path)
        
        metric_cols = [m.column for m in ALL_METRICS if m.column in self.all_solutions.columns]
        before_count = len(self.all_solutions)
        self.all_solutions = self.all_solutions.dropna(subset=metric_cols)
        if before_count > len(self.all_solutions):
            print(f"Dropped {before_count - len(self.all_solutions)} solutions with missing metrics")
            
        before_count = len(self.all_solutions)
        self.all_solutions = self.all_solutions.drop_duplicates(subset=metric_cols)
        if before_count > len(self.all_solutions):
            print(f"Dropped {before_count - len(self.all_solutions)} duplicate solutions")
            
        self.normalized_solutions = normalize_metrics(self.all_solutions)
        self.pareto_frontier = compute_pareto_frontier(self.normalized_solutions)
        pareto_indices = self.pareto_frontier.index
        self.pareto_original = self.all_solutions.loc[pareto_indices].copy()
        
        # Centroid cache: invalidated when filters change
        self._centroid_dirty: bool = True
        self._cached_centroid = None       # pandas Series or None
        self._cached_centroid_path: Optional[str] = None
        self._cached_filtered_count: int = 0
        
        # Initialize state management
        self.state = AgentState()
        self.filter_state = FilterState()
        
        # Compute initial centroid and save version
        centroid, path, count = self._get_current_centroid()
        self.state.save_version(
            self.filter_state,
            solution_path=path,
            solution_count=count,
            description="Initial state"
        )
        
        self.tools = build_tools()
        self.history = [{"role": "system", "content": build_system_prompt()}]
        
        print(f"Loaded {len(self.all_solutions)} total solutions")
        print(f"Computed Pareto frontier with {len(self.pareto_frontier)} solutions")
        print(f"Available metrics: {len(ALL_METRICS)}")
    
    def _get_filtered_solutions(self):
        """Get currently filtered solutions from Pareto frontier."""
        return apply_filters(self.pareto_original, self.filter_state)
    
    def _invalidate_centroid(self):
        """Mark the centroid cache as stale (call after any filter change)."""
        self._centroid_dirty = True
    
    def _get_current_centroid(self) -> tuple:
        """Return (centroid_series, solution_path, filtered_count).
        
        Uses a cache that is invalidated when filters change.
        Returns (None, None, 0) when no solutions match.
        """
        if not self._centroid_dirty and self._cached_centroid is not None:
            return self._cached_centroid, self._cached_centroid_path, self._cached_filtered_count
        
        filtered = self._get_filtered_solutions()
        if len(filtered) == 0:
            self._cached_centroid = None
            self._cached_centroid_path = None
            self._cached_filtered_count = 0
        else:
            normalized = normalize_metrics(filtered)
            solution, _ = get_centroid_solution(filtered, normalized)
            self._cached_centroid = solution
            self._cached_centroid_path = solution.get('path')
            self._cached_filtered_count = len(filtered)
        
        self._centroid_dirty = False
        return self._cached_centroid, self._cached_centroid_path, self._cached_filtered_count

    def _compute_percentile_for_solution(self, solution_series) -> dict:
        """Compute true empirical percentiles for a solution against the Pareto set.

        Uses pandas rank to determine where each metric value falls among all
        Pareto solutions. For 'minimize' metrics the percentile is inverted so
        higher always means better.

        Returns dict mapping metric column -> {percentile, raw_value}.
        """
        import pandas as pd

        ranks = {}
        for metric in CORE_METRICS:
            col = metric.column
            if col not in self.pareto_original.columns or col not in solution_series.index:
                continue
            value = solution_series[col]
            if pd.isna(value):
                continue

            all_values = self.pareto_original[col].dropna()
            if len(all_values) == 0:
                continue

            if metric.direction is None:
                continue

            raw_pct = (all_values <= value).sum() / len(all_values) * 100
            if metric.direction == 'minimize':
                normalized = 100 - raw_pct
            else:
                normalized = raw_pct

            ranks[col] = {
                'percentile': round(normalized),
                'raw_value': float(value),
            }
        return ranks

    def _execute_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute a tool and return a ToolResult."""
        
        if tool_name == "query_zone_data":
            display_ids = arguments.get("zone_ids", [])
            metrics_requested = arguments.get("metrics", [])

            centroid, path, count = self._get_current_centroid()
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
        
        elif tool_name == "compare_zones":
            display_ids = arguments.get("zone_ids", [])
            if len(display_ids) < 2:
                return ToolResult("Need at least 2 zone numbers to compare.")

            centroid, path, count = self._get_current_centroid()
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
        
        elif tool_name == "undo_action":
            steps = arguments.get("steps", 1)
            version = self.state.undo(steps)
            
            if version is None:
                return ToolResult(f"Cannot undo {steps} steps. Only {self.state.current_version} versions available.")
            
            self.filter_state = copy.deepcopy(version.filter_state)
            self._invalidate_centroid()
            _, path, count = self._get_current_centroid()
            
            return ToolResult(
                f"Undid {steps} step(s) to v{version.version_id}\n- {version.description}\n- {count} solutions available",
                solution_path=path,
            )
        
        elif tool_name == "show_version_history":
            if not self.state.versions:
                return ToolResult("No version history.")
            
            lines = ["**Version History:**\n"]
            for v in self.state.versions:
                marker = ">" if v.version_id == self.state.current_version else " "
                lines.append(f"{marker} v{v.version_id}: {v.description} ({v.solution_count} solutions)")
            
            return ToolResult("\n".join(lines))
        
        elif tool_name == "get_current_solution":
            centroid, path, count = self._get_current_centroid()

            if centroid is None:
                return ToolResult("No solutions match the current filters. Use find_feasible_relaxation to see which constraints to relax.")

            show_all = arguments.get("show_all_metrics", False)
            metrics_to_show = ALL_METRICS if show_all else CORE_METRICS

            # Compute empirical percentiles for the centroid solution
            percentile_ranks = self._compute_percentile_for_solution(centroid)

            if show_all:
                lines = [f"v{self.state.current_version}: Complete metrics for current solution ({count} solutions available)\n"]
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
                lines = [f"v{self.state.current_version}: {count} solutions\n"]
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
        
        elif tool_name == "list_all_metrics":
            return ToolResult(get_metric_summary())
        
        elif tool_name == "search_metrics":
            query = arguments.get("query", "")
            matches = search_metrics(query)
            
            if not matches:
                return ToolResult(f"No metrics found matching '{query}'. Use list_all_metrics to see all available metrics.")
            
            lines = [f"**Metrics matching '{query}':**\n"]
            for m in matches:
                lines.append(f"- **{m.display_name}** ({m.category}): {m.description} ({_direction_text(m)})")
            
            return ToolResult("\n".join(lines))
        
        elif tool_name == "tighten_filter":
            metric_name = arguments["metric_name"]
            strength = arguments.get("strength", "moderate")

            pct_map = {"mild": 0.05, "moderate": 0.10, "aggressive": 0.25}
            pct = pct_map.get(strength, 0.10)

            before_centroid, _, before_count = self._get_current_centroid()
            if before_count <= 1:
                return ToolResult(f"Cannot tighten: only {before_count} solution(s) remaining. Consider loosening other filters first.")

            filtered = self._get_filtered_solutions()
            metric = METRIC_BY_NAME[metric_name]
            new_bound = adjust_filter_bound(
                self.pareto_original, filtered, self.filter_state,
                metric_name, "tighten", pct,
                current_value=float(before_centroid[metric.column]),
            )

            if metric.direction == "minimize":
                self.filter_state.bounds[metric_name].max_bound = new_bound
            else:
                self.filter_state.bounds[metric_name].min_bound = new_bound

            self._invalidate_centroid()
            after_centroid, after_path, after_count = self._get_current_centroid()

            before_val = before_centroid[metric.column]
            after_val = after_centroid[metric.column] if after_centroid is not None else before_val

            strength_text = {"mild": "mildly", "moderate": "moderately", "aggressive": "aggressively"}[strength]
            pending_version_id = len(self.state.versions)
            result_lines = [
                f"v{pending_version_id}: Tightened {metric_name} ({strength_text})",
                f"- Solutions: {before_count} -> {after_count}",
                f"- {metric_name} improved: {before_val:.3f} -> {after_val:.3f}"
            ]

            desc = f"Tightened {metric_name} ({strength})"
            if self._defer_version_save:
                self._pending_descriptions.append(desc)
                if after_path:
                    self._pending_solution_path = after_path
            else:
                self.state.save_version(
                    self.filter_state,
                    solution_path=after_path,
                    solution_count=after_count,
                    description=desc,
                )

            return ToolResult("\n".join(result_lines), solution_path=after_path)
        
        elif tool_name == "loosen_filter":
            metric_name = arguments["metric_name"]
            strength = arguments.get("strength", "moderate")

            pct_map = {"mild": 0.10, "moderate": 0.25, "aggressive": 0.50}
            pct = pct_map.get(strength, 0.25)

            filtered = self._get_filtered_solutions()
            new_bound = adjust_filter_bound(
                self.pareto_original, filtered, self.filter_state,
                metric_name, "loosen", pct,
            )

            if new_bound is None:
                return ToolResult(f"'{metric_name}' is already unconstrained.")

            metric = METRIC_BY_NAME[metric_name]
            _, _, before_count = self._get_current_centroid()

            if metric.direction == "minimize":
                self.filter_state.bounds[metric_name].max_bound = new_bound
            else:
                self.filter_state.bounds[metric_name].min_bound = new_bound

            self._invalidate_centroid()
            _, after_path, after_count = self._get_current_centroid()

            pending_version_id = len(self.state.versions)
            desc = f"Loosened {metric_name} ({strength})"

            if self._defer_version_save:
                self._pending_descriptions.append(desc)
                if after_path:
                    self._pending_solution_path = after_path
            else:
                self.state.save_version(
                    self.filter_state,
                    solution_path=after_path,
                    solution_count=after_count,
                    description=desc,
                )

            return ToolResult(
                f"v{pending_version_id}: Loosened {metric_name} ({strength})\n- {before_count} -> {after_count} solutions",
                solution_path=after_path,
            )
        
        elif tool_name == "get_filter_bounds":
            filtered = self._get_filtered_solutions()
            category = arguments.get("category")
            return ToolResult(get_filter_summary(self.filter_state, self.pareto_original, filtered, show_category=category))
        
        elif tool_name == "find_feasible_relaxation":
            suggestions = find_relaxation_needed(
                self.pareto_original, self.filter_state
            )
            
            if not suggestions:
                return ToolResult("Unable to find relaxations that would restore feasibility. Try loosening multiple filters manually.")
            
            lines = ["**Suggested Relaxations** (relaxing any ONE of these could restore feasibility):\n"]
            for metric_name, new_bound in suggestions.items():
                metric = METRIC_BY_NAME.get(metric_name)
                if metric and metric.direction == "minimize":
                    current = self.filter_state.bounds[metric_name].max_bound
                    lines.append(f"- **{metric_name}**: relax max from {current:.4f} to {new_bound:.4f}")
                elif metric:
                    current = self.filter_state.bounds[metric_name].min_bound
                    lines.append(f"- **{metric_name}**: relax min from {current:.4f} to {new_bound:.4f}")
            lines.append("\nAsk the user which metric they're willing to compromise on.")
            return ToolResult("\n".join(lines))
        
        elif tool_name == "show_solution_clusters":
            filtered = self._get_filtered_solutions()
            
            if len(filtered) < 3:
                return ToolResult(f"Not enough solutions to cluster (only {len(filtered)}). Need at least 3 solutions.")
            
            n_clusters = arguments.get("n_clusters", 3)
            n_clusters = min(n_clusters, len(filtered) // 2)
            n_clusters = max(2, n_clusters)
            
            vectors = vectorize_solutions(filtered)
            labels, centers = cluster_solutions(vectors, n_clusters)
            directions = compute_cluster_directions(vectors, centers)
            
            self.state.clustered_solutions = filtered
            self.state.clustered_vectors = vectors
            self.state.cluster_labels = labels
            self.state.cluster_centers = centers
            self.state.cluster_directions = directions
            
            clusters_data = self._build_clusters_response()
            return ToolResult(
                format_cluster_summary(filtered, vectors, labels, centers, directions),
                clusters=clusters_data,
            )
        
        elif tool_name == "select_cluster":
            if self.state.cluster_labels is None:
                return ToolResult("No clustering results available. Call show_solution_clusters first.")
            
            cluster_id = arguments["cluster_id"] - 1
            
            n_clusters = len(self.state.cluster_centers)
            if cluster_id < 0 or cluster_id >= n_clusters:
                return ToolResult(f"Invalid cluster ID. Please choose between 1 and {n_clusters}.")
            
            cluster_bounds = get_cluster_bounds(
                self.state.clustered_solutions,
                self.state.cluster_labels,
                cluster_id
            )
            
            for metric_name, bounds in cluster_bounds.items():
                if metric_name in METRIC_BY_NAME:
                    metric = METRIC_BY_NAME[metric_name]
                    if metric.direction is None:
                        continue
                    if metric.direction == "minimize":
                        self.filter_state.bounds[metric_name].max_bound = bounds.max_bound
                    else:
                        self.filter_state.bounds[metric_name].min_bound = bounds.min_bound
            
            direction_label = self.state.cluster_directions[cluster_id]["direction_label"]
            self.state.cluster_labels = None
            self.state.cluster_centers = None
            self.state.cluster_directions = None
            self.state.clustered_solutions = None
            self.state.clustered_vectors = None
            
            self._invalidate_centroid()
            _, after_path, after_count = self._get_current_centroid()
            
            pending_version_id = len(self.state.versions)
            desc = f"Selected cluster: {direction_label}"

            if self._defer_version_save:
                self._pending_descriptions.append(desc)
                if after_path:
                    self._pending_solution_path = after_path
            else:
                self.state.save_version(
                    self.filter_state,
                    solution_path=after_path,
                    solution_count=after_count,
                    description=desc,
                )

            return ToolResult(
                f"v{pending_version_id}: Cluster {cluster_id + 1} selected\n- {direction_label}\n- {after_count} solutions",
                solution_path=after_path,
            )

        elif tool_name == "apply_feedback_filters":
            adjustments = arguments.get("adjustments", [])
            if not adjustments:
                return ToolResult("No adjustments provided. Analyze the user's feedback and specify metric adjustments.")

            self.filter_state = FilterState()
            self._invalidate_centroid()
            _, _, before_count = self._get_current_centroid()

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
                filtered = self._get_filtered_solutions()
                if len(filtered) <= 1:
                    applied.append(f"- {metric_name}: skipped (only {len(filtered)} solution left)")
                    break

                centroid_val = None
                if direction == "tighten":
                    centroid, _, _ = self._get_current_centroid()
                    if centroid is not None:
                        centroid_val = float(centroid[metric.column])

                pct = (tighten_pct_map if direction == "tighten" else loosen_pct_map).get(strength, 0.10)
                new_bound = adjust_filter_bound(
                    self.pareto_original, filtered, self.filter_state,
                    metric_name, direction, pct,
                    current_value=centroid_val,
                )

                if new_bound is not None:
                    if metric.direction == "minimize":
                        if direction == "tighten":
                            self.filter_state.bounds[metric_name].max_bound = new_bound
                        else:
                            self.filter_state.bounds[metric_name].max_bound = new_bound
                    else:
                        if direction == "tighten":
                            self.filter_state.bounds[metric_name].min_bound = new_bound
                        else:
                            self.filter_state.bounds[metric_name].min_bound = new_bound
                    self._invalidate_centroid()
                    applied.append(f"- {'Tightened' if direction == 'tighten' else 'Loosened'} {metric_name} ({strength})")

            self._invalidate_centroid()
            after_centroid, after_path, after_count = self._get_current_centroid()

            metric_lines = []
            if after_centroid is not None:
                for m in CORE_METRICS[:6]:
                    if m.column in after_centroid.index:
                        val = after_centroid[m.column]
                        metric_lines.append(f"- {m.display_name}: {val:.3f} {_direction_text(m)}")

            pending_version_id = len(self.state.versions)
            desc = f"Applied {len(applied)} feedback-based filters"

            if self._defer_version_save:
                self._pending_descriptions.append(desc)
                if after_path:
                    self._pending_solution_path = after_path
            else:
                self.state.save_version(
                    self.filter_state,
                    solution_path=after_path,
                    solution_count=after_count,
                    description=desc,
                )

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

        else:
            return ToolResult(f"Unknown tool: {tool_name}")
    
    def reset_filters(self):
        """Reset all filters to allow all solutions."""
        self.filter_state = FilterState()
        self._invalidate_centroid()

    def _build_solution_context(self, solution_context: dict) -> str:
        """Build a context string from the user's saved solutions and notes."""
        lines = []

        current_idx = solution_context.get("current_solution_index")
        saved = solution_context.get("saved_solutions", [])

        if not saved:
            return ""

        if current_idx is not None:
            lines.append(f"The user is currently viewing Solution #{current_idx}.")

        lines.append("\nSaved solutions:")
        for sol in saved:
            idx = sol.get("index", "?")
            label = sol.get("label", "Untitled")
            pros = sol.get("pros", "")
            cons = sol.get("cons", "")
            viewing = " [CURRENTLY VIEWING]" if idx == current_idx else ""

            annotations = []
            if pros:
                annotations.append(f'LIKES: "{pros}"')
            if cons:
                annotations.append(f'DISLIKES: "{cons}"')
            annotation_text = " -- " + "; ".join(annotations) if annotations else ""
            lines.append(f'- #{idx}: "{label}"{annotation_text}{viewing}')

        all_pros = []
        all_cons = []
        for sol in saved:
            if sol.get("pros"):
                all_pros.append(f'Solution #{sol.get("index", "?")}: {sol["pros"]}')
            if sol.get("cons"):
                all_cons.append(f'Solution #{sol.get("index", "?")}: {sol["cons"]}')

        if all_pros or all_cons:
            lines.append("\n== USER FEEDBACK (use this to drive recommendations) ==")
            lines.append("Map feedback to metric adjustments: complaints = tighten, praise = maintain/improve.")
            lines.append("For generate-from-feedback requests, call apply_feedback_filters with ALL identified preferences.")
            if all_pros:
                lines.append("LIKES (maintain or improve):")
                lines.extend(f"  + {p}" for p in all_pros)
            if all_cons:
                lines.append("DISLIKES (aggressively fix):")
                lines.extend(f"  - {c}" for c in all_cons)

        return "\n".join(lines)

    def chat(self, user_message: str, solution_context: dict = None) -> dict:
        """Process a user message and return structured response.

        Returns a dict with:
        - text: The LLM response text
        - response_type: "text" | "clusters" | "solution_update"
        - clusters: List of cluster info (if applicable)
        - solution_path: Path to solution (if applicable)
        - description: Current version description
        """
        clusters_data = None
        solution_path = None
        any_tool_called = False

        self._defer_version_save = True
        self._pending_descriptions = []
        self._pending_solution_path = None

        if solution_context:
            context_text = self._build_solution_context(solution_context)
            if context_text:
                enhanced_message = f"[Context: {context_text}]\n\n{user_message}"
            else:
                enhanced_message = user_message
        else:
            enhanced_message = user_message

        self.history.append({"role": "user", "content": enhanced_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            tools=self.tools,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message

        while assistant_message.tool_calls:
            self.history.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ],
            })

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                any_tool_called = True
                tool_result = self._execute_tool(tool_name, arguments)

                if tool_result.solution_path:
                    solution_path = tool_result.solution_path
                if tool_result.clusters:
                    clusters_data = tool_result.clusters

                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result.text,
                })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                tools=self.tools,
                tool_choice="auto",
            )
            assistant_message = response.choices[0].message

        final_content = assistant_message.content or ""
        self.history.append({"role": "assistant", "content": final_content})

        # Flush deferred version: one version per chat turn
        self._defer_version_save = False
        if self._pending_descriptions:
            combined_desc = "; ".join(self._pending_descriptions)
            _, flush_path, flush_count = self._get_current_centroid()
            flush_path = self._pending_solution_path or flush_path
            self.state.save_version(
                self.filter_state,
                solution_path=flush_path,
                solution_count=flush_count,
                description=combined_desc,
            )
            if solution_path is None:
                solution_path = flush_path
        self._pending_descriptions = []
        self._pending_solution_path = None

        # Fallback: if tools were called but no path captured, get from cache
        if solution_path is None and clusters_data is None and any_tool_called:
            _, solution_path, _ = self._get_current_centroid()

        if clusters_data is not None:
            response_type = "clusters"
        elif solution_path is not None:
            response_type = "solution_update"
        else:
            response_type = "text"

        current_version = self.state.get_current_version()
        description = current_version.description if current_version else ""

        return {
            "text": final_content,
            "response_type": response_type,
            "clusters": clusters_data,
            "solution_path": solution_path,
            "description": description,
        }

    def _build_clusters_response(self) -> list:
        """Build cluster data for frontend response."""
        if self.state.cluster_labels is None:
            return []

        clusters = []
        n_clusters = len(self.state.cluster_centers)

        for cluster_id in range(n_clusters):
            mask = self.state.cluster_labels == cluster_id
            cluster_solutions = self.state.clustered_solutions[mask]

            if len(cluster_solutions) == 0:
                continue

            normalized = normalize_metrics(cluster_solutions)
            centroid_solution, _ = get_centroid_solution(cluster_solutions, normalized)

            direction_info = self.state.cluster_directions.get(cluster_id, {})

            clusters.append({
                "id": cluster_id + 1,
                "label": direction_info.get("direction_label", f"Cluster {cluster_id + 1}"),
                "count": int(mask.sum()),
                "path": centroid_solution["path"],
            })

        return clusters
