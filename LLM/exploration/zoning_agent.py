"""
School Zoning Exploration Agent.

An LLM-powered agent that helps users iteratively explore school zoning proposals
using adjustable filters on a Pareto frontier of solutions.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

from .metrics_config import ALL_METRICS, CORE_METRICS
from .pareto import (
    load_solutions,
    normalize_metrics,
    compute_pareto_frontier,
    get_centroid_solution,
)
from .filters import FilterState, apply_filters
from .state import AgentState, ToolResult
from .tool_defs import build_tools
from .prompts import build_system_prompt

# Handler imports
from .zone_handlers import handle_query_zone_data, handle_compare_zones
from .filter_handlers import (
    handle_tighten_filter,
    handle_loosen_filter,
    handle_set_filter,
    handle_get_filter_bounds,
    handle_find_feasible_relaxation,
    handle_apply_feedback_filters,
)
from .solution_handlers import (
    handle_get_solution,
    handle_list_all_metrics,
    handle_search_metrics,
    handle_undo_action,
    handle_show_version_history,
    handle_save_feedback,
    build_clusters_response,
)
from .clusters import (
    themed_cluster_solutions,
    vectorize_solutions,
    format_cluster_summary,
    get_cluster_bounds,
)
from .metrics_config import METRIC_BY_NAME

DEFAULT_MODEL = "gemini-3-flash-preview"


# ============================================================================
# TOOL DISPATCH TABLE
# ============================================================================

TOOL_HANDLERS = {
    "query_zone_data": handle_query_zone_data,
    "compare_zones": handle_compare_zones,
    "tighten_filter": handle_tighten_filter,
    "loosen_filter": handle_loosen_filter,
    "set_filter": handle_set_filter,
    "get_filter_bounds": handle_get_filter_bounds,
    "find_feasible_relaxation": handle_find_feasible_relaxation,
    "apply_feedback_filters": handle_apply_feedback_filters,
    "get_solution": handle_get_solution,
    "list_all_metrics": handle_list_all_metrics,
    "search_metrics": handle_search_metrics,
    "undo_action": handle_undo_action,
    "show_version_history": handle_show_version_history,
    "save_feedback": handle_save_feedback,
}


class ZoningAgent:
    """Interactive agent for exploring school zoning solutions with state management."""

    def __init__(self, csv_path: str | Path):
        """
        Initialize the agent with zoning solution data.

        Args:
            csv_path: Path to the CSV file with zoning solutions
        """
        load_dotenv()

        # Initialize native Google GenAI client
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model = DEFAULT_MODEL

        # Session-level token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_api_calls = 0

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

        self.tools = build_tools(mode="all")
        self.feedback_tools = build_tools(mode="feedback")
        self.generate_tools = build_tools(mode="generate")
        self.system_instruction = build_system_prompt(mode="feedback")
        self.history: list[types.Content] = []

        # Pre-compute themed clusters for instant initial display
        self._initial_clusters = self._compute_initial_clusters()

        print(f"Loaded {len(self.all_solutions)} total solutions")
        print(f"Computed Pareto frontier with {len(self.pareto_frontier)} solutions")
        print(f"Available metrics: {len(ALL_METRICS)}")

    def _compute_initial_clusters(self) -> dict | None:
        """Pre-compute themed clusters for instant initial display.

        Returns a response dict with clusters data, or None if not enough solutions.
        """
        filtered = self._get_filtered_solutions()
        if len(filtered) < 3:
            return None

        try:
            labels, centers, directions, columns = themed_cluster_solutions(filtered)
            vectors = vectorize_solutions(filtered, columns=columns)

            # Store clustering state so select_cluster works
            self.state.clustered_solutions = filtered
            self.state.clustered_vectors = vectors
            self.state.cluster_labels = labels
            self.state.cluster_centers = centers
            self.state.cluster_directions = directions
            self.state.cluster_columns = columns

            clusters_data = build_clusters_response(self)
            text = format_cluster_summary(filtered, vectors, labels, centers, directions)

            return {
                "text": text,
                "response_type": "clusters",
                "clusters": clusters_data,
                "solution_path": None,
                "description": "Initial state",
            }
        except Exception as e:
            logger.error("Failed to compute initial clusters: %s", e)
            return None

    def get_initial_clusters(self) -> dict | None:
        """Return pre-computed clusters and consume them (one-shot)."""
        result = self._initial_clusters
        self._initial_clusters = None
        return result

    def select_cluster(self, cluster_id: int) -> dict:
        """Select a cluster by 1-based ID and tighten filters to match it.

        This is a pure-code path -- no LLM involved.
        """
        if self.state.cluster_labels is None:
            return {"text": "No clustering results available.", "response_type": "text",
                    "solution_path": None, "description": ""}

        cluster_idx = cluster_id - 1
        n_clusters = len(self.state.cluster_centers)
        if cluster_idx < 0 or cluster_idx >= n_clusters:
            return {"text": f"Invalid cluster ID. Choose between 1 and {n_clusters}.",
                    "response_type": "text", "solution_path": None, "description": ""}

        cluster_bounds = get_cluster_bounds(
            self.state.clustered_solutions,
            self.state.cluster_labels,
            cluster_idx,
            columns=self.state.cluster_columns,
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

        direction_label = self.state.cluster_directions[cluster_idx]["direction_label"]

        self.state.cluster_labels = None
        self.state.cluster_centers = None
        self.state.cluster_directions = None
        self.state.clustered_solutions = None
        self.state.clustered_vectors = None
        self.state.cluster_columns = None

        self._invalidate_centroid()
        _, after_path, after_count = self._get_current_centroid()

        version_id = len(self.state.versions)
        desc = f"Selected cluster: {direction_label}"
        self.state.save_version(
            self.filter_state,
            solution_path=after_path,
            solution_count=after_count,
            description=desc,
        )

        text = f"v{version_id}: Cluster {cluster_id} selected -- {direction_label} ({after_count} solutions)"

        # Record in history so the LLM has context for subsequent turns
        self.history.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"I selected the {direction_label} cluster.")],
        ))
        self.history.append(types.Content(
            role="model",
            parts=[types.Part.from_text(text=text)],
        ))

        return {
            "text": text,
            "response_type": "solution_update",
            "solution_path": after_path,
            "description": desc,
        }

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
        handler = TOOL_HANDLERS.get(tool_name)
        if handler is not None:
            return handler(self, arguments)
        return ToolResult(f"Unknown tool: {tool_name}")

    def reset_filters(self):
        """Reset all filters to allow all solutions."""
        self.filter_state = FilterState()
        self._invalidate_centroid()

    def _build_feedback_context(self) -> str:
        """Return accumulated feedback summary from state."""
        return self.state.get_feedback_summary()

    def _compress_history(self):
        """Strip tool call/response entries from past turns, keeping only user and model text."""
        compressed = []
        for content in self.history:
            text_parts = [p for p in content.parts if p.text is not None]
            if text_parts:
                compressed.append(types.Content(role=content.role, parts=text_parts))
        self.history = compressed

    def chat(self, user_message: str, mode: str = "feedback") -> dict:
        """Process a user message and return structured response.

        Args:
            user_message: The user's message text
            mode: "feedback" (info tools only) or "generate" (filter tools available)

        Returns a dict with:
        - text: The LLM response text
        - response_type: "text" | "solution_update"
        - solution_path: Path to solution (if applicable)
        - description: Current version description
        """
        solution_path = None
        any_tool_called = False

        turn_prompt_tokens = 0
        turn_completion_tokens = 0
        turn_api_calls = 0

        self._defer_version_save = True
        self._pending_descriptions = []
        self._pending_solution_path = None

        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

        tools = self.generate_tools if mode == "generate" else self.feedback_tools
        feedback_summary = self._build_feedback_context()
        system_instruction = build_system_prompt(mode=mode, feedback_summary=feedback_summary)

        config = types.GenerateContentConfig(
            tools=[tools],
            system_instruction=system_instruction,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=self.history,
            config=config,
        )
        turn_api_calls += 1
        if response.usage_metadata:
            turn_prompt_tokens += response.usage_metadata.prompt_token_count
            turn_completion_tokens += response.usage_metadata.candidates_token_count
            logger.info("LLM call #%d: prompt=%d completion=%d total=%d",
                        turn_api_calls, response.usage_metadata.prompt_token_count,
                        response.usage_metadata.candidates_token_count,
                        response.usage_metadata.total_token_count)

        # Check for function calls in the response
        function_calls = response.function_calls

        while function_calls:
            # Append model's response (with function call parts) to history
            self.history.append(response.candidates[0].content)

            function_response_parts = []
            for fc in function_calls:
                tool_name = fc.name
                arguments = dict(fc.args) if fc.args else {}

                logger.info("Tool call: %s args=%s", tool_name, arguments)
                any_tool_called = True
                tool_result = self._execute_tool(tool_name, arguments)
                logger.info("Tool result: %d chars", len(tool_result.text))

                if tool_result.solution_path:
                    solution_path = tool_result.solution_path

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_result.text},
                    )
                )

            # Append all function responses as a single user turn
            self.history.append(types.Content(role="user", parts=function_response_parts))

            response = self.client.models.generate_content(
                model=self.model,
                contents=self.history,
                config=config,
            )
            turn_api_calls += 1
            if response.usage_metadata:
                turn_prompt_tokens += response.usage_metadata.prompt_token_count
                turn_completion_tokens += response.usage_metadata.candidates_token_count
                logger.info("LLM call #%d: prompt=%d completion=%d total=%d",
                            turn_api_calls, response.usage_metadata.prompt_token_count,
                            response.usage_metadata.candidates_token_count,
                            response.usage_metadata.total_token_count)
            function_calls = response.function_calls

        final_content = response.text or ""
        self.history.append(types.Content(role="model", parts=[types.Part.from_text(text=final_content)]))
        self._compress_history()

        # Accumulate session totals
        self.total_prompt_tokens += turn_prompt_tokens
        self.total_completion_tokens += turn_completion_tokens
        self.total_api_calls += turn_api_calls
        logger.info("Turn totals: prompt=%d completion=%d api_calls=%d | "
                     "Session totals: prompt=%d completion=%d api_calls=%d",
                     turn_prompt_tokens, turn_completion_tokens, turn_api_calls,
                     self.total_prompt_tokens, self.total_completion_tokens,
                     self.total_api_calls)

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
        if solution_path is None and any_tool_called:
            _, solution_path, _ = self._get_current_centroid()

        response_type = "solution_update" if solution_path is not None else "text"

        current_version = self.state.get_current_version()
        description = current_version.description if current_version else ""

        return {
            "text": final_content,
            "response_type": response_type,
            "solution_path": solution_path,
            "description": description,
            "usage": {
                "prompt_tokens": turn_prompt_tokens,
                "completion_tokens": turn_completion_tokens,
                "api_calls": turn_api_calls,
                "session_total_prompt_tokens": self.total_prompt_tokens,
                "session_total_completion_tokens": self.total_completion_tokens,
                "session_total_api_calls": self.total_api_calls,
            },
        }
