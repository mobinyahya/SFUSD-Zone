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
    handle_show_solution_clusters,
    handle_select_cluster,
    build_clusters_response,
)
from .clusters import (
    themed_cluster_solutions,
    vectorize_solutions,
    format_cluster_summary,
)

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
    "show_solution_clusters": handle_show_solution_clusters,
    "select_cluster": handle_select_cluster,
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

        self.tools = build_tools()
        self.system_instruction = build_system_prompt()
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

    def _compress_history(self):
        """Strip tool call/response entries from past turns, keeping only user and model text."""
        compressed = []
        for content in self.history:
            text_parts = [p for p in content.parts if p.text is not None]
            if text_parts:
                compressed.append(types.Content(role=content.role, parts=text_parts))
        self.history = compressed

    def chat(self, user_message: str, solution_context: dict = None) -> dict:
        """Process a user message and return structured response.

        Returns a dict with:
        - text: The LLM response text
        - response_type: "text" | "clusters" | "solution_update"
        - clusters: List of cluster info (if applicable)
        - solution_path: Path to solution (if applicable)
        - description: Current version description
        """
        # Return pre-computed themed clusters on first call (no LLM needed)
        if self._initial_clusters is not None:
            result = self._initial_clusters
            self._initial_clusters = None  # Only serve once

            # Record in history so the LLM knows clusters were already shown
            self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
            self.history.append(types.Content(role="model", parts=[types.Part.from_text(text=result["text"])]))

            result["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "api_calls": 0,
                "session_total_prompt_tokens": 0,
                "session_total_completion_tokens": 0,
                "session_total_api_calls": 0,
            }
            return result

        clusters_data = None
        solution_path = None
        any_tool_called = False

        turn_prompt_tokens = 0
        turn_completion_tokens = 0
        turn_api_calls = 0

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

        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=enhanced_message)]))

        config = types.GenerateContentConfig(
            tools=[self.tools],
            system_instruction=self.system_instruction,
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
                if tool_result.clusters:
                    clusters_data = tool_result.clusters

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
            "usage": {
                "prompt_tokens": turn_prompt_tokens,
                "completion_tokens": turn_completion_tokens,
                "api_calls": turn_api_calls,
                "session_total_prompt_tokens": self.total_prompt_tokens,
                "session_total_completion_tokens": self.total_completion_tokens,
                "session_total_api_calls": self.total_api_calls,
            },
        }
