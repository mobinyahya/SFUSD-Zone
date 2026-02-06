"""
School Zoning Exploration Agent.

An LLM-powered agent that helps users iteratively explore school zoning proposals
using adjustable filters on a Pareto frontier of solutions.
"""

import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

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
    METRIC_CONFIG,
    load_solutions,
    normalize_metrics,
    compute_pareto_frontier,
    get_centroid_solution,
    format_solution,
)
from .filters import (
    FilterState,
    FilterBounds,
    apply_filters,
    get_filter_summary,
    calculate_tightening,
    calculate_loosening,
    find_relaxation_needed,
)
from .clusters import (
    vectorize_solutions,
    cluster_solutions,
    compute_cluster_directions,
    get_cluster_bounds,
    format_cluster_summary,
)


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
                "name": "get_current_solution",
                "description": "Get the current 'balanced' centroid solution based on the current filters. Returns the solution metrics and overall statistics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "show_all_metrics": {
                            "type": "boolean",
                            "description": "If true, show all metrics including detailed ones. Default is false (core metrics only).",
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
                "description": "Tighten the constraint for a specific metric to improve it. For 'lower is better' metrics, this reduces the maximum allowed value. For 'higher is better' metrics, this increases the minimum allowed value. This will reduce the number of feasible solutions.",
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
                            "description": "How aggressively to tighten: 'mild' (~20% reduction), 'moderate' (~30%), or 'aggressive' (~50%)",
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
                "description": "Loosen the constraint for a specific metric to allow more diverse solutions. Use when the user is willing to accept worse values for a metric to improve others.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "type": "string",
                            "description": "The display name of the metric to loosen.",
                            "enum": all_metric_names,
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
    ]


def build_system_prompt():
    """Build system prompt with current metric information."""
    metric_summary = get_metric_summary()
    
    return f"""You are a helpful assistant that helps parents explore school zoning proposals for San Francisco Unified School District.

## Your Role
You help users find zoning solutions that match their priorities by iteratively adjusting filters on various metrics. Think of yourself as a friendly guide who translates high-level preferences ("I want more diverse schools") into concrete filter adjustments.

## How Metrics Work
- **Minimize metrics** (lower is better): Diversity deviations, distances, boundary cost
- **Maximize metrics** (higher is better): Program counts, school quality ratings

When tightening a filter:
- For minimize metrics → lower the max allowed value (keep only better solutions)
- For maximize metrics → raise the min allowed value (keep only better solutions)

{metric_summary}

## How to Help Users

1. **Start by presenting the current "balanced" solution** - Give a high level overview of the solution and ask the user what they would like to change. Do not list specific metrics unless asked.

2. **Use list_all_metrics or search_metrics** - When users ask about available metrics or specific programs.

3. **Listen to feedback** - Users will express preferences like "I want shorter commutes" or "economic diversity is most important".

4. **Translate to filter adjustments** - When a user wants to improve something, tighten that filter. Explain the trade-off.

5. **Handle impossible requests gracefully** - If filters become too tight (0 solutions), use find_feasible_relaxation.

6. **Always explain trade-offs** - Improving one metric often means accepting worse values for others.

## Communication Style
- Be friendly and accessible - users are parents, not optimization experts
- Use plain language, not technical jargon
- Be as concise as possible
- Proactively suggest trade-offs

## Clustering Feature
When users are exploring or want to see different types of solutions:
1. **show_solution_clusters** - Groups similar solutions and shows representatives
2. **select_cluster** - Narrows filters to solutions in that cluster
"""


class ZoningAgent:
    """Interactive agent for exploring school zoning solutions."""
    
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
        
        # Load and process solutions
        self.all_solutions = load_solutions(csv_path)
        
        # Drop solutions with NaN values in metric columns
        metric_cols = [m.column for m in ALL_METRICS if m.column in self.all_solutions.columns]
        before_count = len(self.all_solutions)
        self.all_solutions = self.all_solutions.dropna(subset=metric_cols)
        if before_count > len(self.all_solutions):
            print(f"Dropped {before_count - len(self.all_solutions)} solutions with missing metrics")
            
        # Drop duplicate solutions based on metric columns
        before_count = len(self.all_solutions)
        self.all_solutions = self.all_solutions.drop_duplicates(subset=metric_cols)
        if before_count > len(self.all_solutions):
            print(f"Dropped {before_count - len(self.all_solutions)} duplicate solutions")
            
        self.normalized_solutions = normalize_metrics(self.all_solutions)
        self.pareto_frontier = compute_pareto_frontier(self.normalized_solutions)
        
        # Get original (unnormalized) Pareto solutions
        pareto_indices = self.pareto_frontier.index
        self.pareto_original = self.all_solutions.loc[pareto_indices].copy()
        
        # Initialize filter state (no filters initially)
        self.filter_state = FilterState()
        
        # Clustering state
        self._cluster_labels = None
        self._cluster_centers = None
        self._cluster_directions = None
        self._clustered_solutions = None
        self._clustered_vectors = None
        
        # Build tools dynamically
        self.tools = build_tools()
        
        # Conversation history with dynamic system prompt
        self.history = [{"role": "system", "content": build_system_prompt()}]
        
        print(f"Loaded {len(self.all_solutions)} total solutions")
        print(f"Computed Pareto frontier with {len(self.pareto_frontier)} solutions")
        print(f"Available metrics: {len(ALL_METRICS)}")
    
    def _get_filtered_solutions(self):
        """Get currently filtered solutions from Pareto frontier."""
        return apply_filters(self.pareto_original, self.filter_state)
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result as a string."""
        
        if tool_name == "get_current_solution":
            filtered = self._get_filtered_solutions()
            
            if len(filtered) == 0:
                return "No solutions match the current filters. Use find_feasible_relaxation to see which constraints to relax."
            
            # Normalize filtered solutions
            normalized_filtered = normalize_metrics(filtered)
            
            # Get centroid
            solution, idx = get_centroid_solution(filtered, normalized_filtered)
            
            # Format response
            show_all = arguments.get("show_all_metrics", False)
            result = f"**Current Solution** (centroid of {len(filtered)} feasible solutions)\n\n"
            result += format_solution(solution, show_all=show_all)
            result += f"\n\n**Path:** {solution['path']}"
            
            return result
        
        elif tool_name == "list_all_metrics":
            return get_metric_summary()
        
        elif tool_name == "search_metrics":
            query = arguments.get("query", "")
            matches = search_metrics(query)
            
            if not matches:
                return f"No metrics found matching '{query}'. Use list_all_metrics to see all available metrics."
            
            lines = [f"**Metrics matching '{query}':**\n"]
            for m in matches:
                direction = "lower is better" if m.direction == "minimize" else "higher is better"
                lines.append(f"• **{m.display_name}** ({m.category}): {m.description} ({direction})")
            
            return "\n".join(lines)
        
        elif tool_name == "tighten_filter":
            metric_name = arguments["metric_name"]
            strength = arguments.get("strength", "moderate")
            
            # Map strength to reduction factor
            reduction_map = {"mild": 0.2, "moderate": 0.3, "aggressive": 0.5}
            reduction = reduction_map.get(strength, 0.3)
            
            filtered = self._get_filtered_solutions()
            
            if len(filtered) <= 1:
                return f"Cannot tighten: only {len(filtered)} solution(s) remaining. Consider loosening other filters first."
            
            try:
                metric = METRIC_BY_NAME[metric_name]
                new_bound, expected_remaining = calculate_tightening(
                    filtered, metric_name, reduction
                )
                
                # Apply the new bound based on direction
                if metric.direction == "minimize":
                    self.filter_state.bounds[metric_name].max_bound = new_bound
                    bound_type = "max"
                else:
                    self.filter_state.bounds[metric_name].min_bound = new_bound
                    bound_type = "min"
                
                actual_filtered = self._get_filtered_solutions()
                direction = "lower is better" if metric.direction == "minimize" else "higher is better"
                
                return f"Tightened '{metric_name}' ({direction}) to {bound_type} value {new_bound:.4f}. {len(actual_filtered)} solutions remaining (was {len(filtered)})."
            
            except Exception as e:
                return f"Error tightening filter: {str(e)}"
        
        elif tool_name == "loosen_filter":
            metric_name = arguments["metric_name"]
            
            try:
                metric = METRIC_BY_NAME[metric_name]
                new_bound, added_count = calculate_loosening(
                    self.pareto_original, self.filter_state, metric_name
                )
                
                if new_bound is None:
                    return f"'{metric_name}' is already unconstrained."
                
                before_count = len(self._get_filtered_solutions())
                
                if metric.direction == "minimize":
                    self.filter_state.bounds[metric_name].max_bound = new_bound
                else:
                    self.filter_state.bounds[metric_name].min_bound = new_bound
                
                after_count = len(self._get_filtered_solutions())
                
                return f"Loosened '{metric_name}' to value {new_bound:.4f}. {after_count} solutions now feasible (was {before_count})."
            
            except Exception as e:
                return f"Error loosening filter: {str(e)}"
        
        elif tool_name == "get_filter_bounds":
            filtered = self._get_filtered_solutions()
            category = arguments.get("category")
            return get_filter_summary(self.filter_state, self.pareto_original, filtered, show_category=category)
        
        elif tool_name == "find_feasible_relaxation":
            suggestions = find_relaxation_needed(
                self.pareto_original, self.filter_state
            )
            
            if not suggestions:
                return "Unable to find relaxations that would restore feasibility. Try loosening multiple filters manually."
            
            result = "**Suggested Relaxations** (relaxing any ONE of these could restore feasibility):\n\n"
            for metric_name, new_bound in suggestions.items():
                metric = METRIC_BY_NAME.get(metric_name)
                if metric and metric.direction == "minimize":
                    current = self.filter_state.bounds[metric_name].max_bound
                    result += f"• **{metric_name}**: relax max from {current:.4f} → {new_bound:.4f}\n"
                elif metric:
                    current = self.filter_state.bounds[metric_name].min_bound
                    result += f"• **{metric_name}**: relax min from {current:.4f} → {new_bound:.4f}\n"
            
            result += "\nAsk the user which metric they're willing to compromise on."
            return result
        
        elif tool_name == "show_solution_clusters":
            filtered = self._get_filtered_solutions()
            
            if len(filtered) < 3:
                return f"Not enough solutions to cluster (only {len(filtered)}). Need at least 3 solutions."
            
            # Determine number of clusters
            n_clusters = arguments.get("n_clusters")
            if n_clusters is None:
                n_clusters = min(max(2, len(filtered) // 3), 5)
            n_clusters = min(n_clusters, len(filtered) // 2)
            n_clusters = max(2, n_clusters)
            
            # Vectorize and cluster
            vectors = vectorize_solutions(filtered)
            labels, centers = cluster_solutions(vectors, n_clusters)
            directions = compute_cluster_directions(vectors, centers)
            
            # Store clustering state
            self._clustered_solutions = filtered
            self._clustered_vectors = vectors
            self._cluster_labels = labels
            self._cluster_centers = centers
            self._cluster_directions = directions
            
            return format_cluster_summary(filtered, vectors, labels, centers, directions)
        
        elif tool_name == "select_cluster":
            if self._cluster_labels is None:
                return "No clustering results available. Call show_solution_clusters first."
            
            cluster_id = arguments["cluster_id"] - 1  # Convert to 0-indexed
            
            n_clusters = len(self._cluster_centers)
            if cluster_id < 0 or cluster_id >= n_clusters:
                return f"Invalid cluster ID. Please choose between 1 and {n_clusters}."
            
            cluster_bounds = get_cluster_bounds(
                self._clustered_solutions,
                self._cluster_labels,
                cluster_id
            )
            
            # Apply bounds to filter state
            for metric_name, bounds in cluster_bounds.items():
                if metric_name in METRIC_BY_NAME:
                    metric = METRIC_BY_NAME[metric_name]
                    if metric.direction == "minimize":
                        self.filter_state.bounds[metric_name].max_bound = bounds.max_bound
                    else:
                        self.filter_state.bounds[metric_name].min_bound = bounds.min_bound
            
            # Clear clustering state
            cluster_size = (self._cluster_labels == cluster_id).sum()
            direction_label = self._cluster_directions[cluster_id]["direction_label"]
            self._cluster_labels = None
            self._cluster_centers = None
            self._cluster_directions = None
            self._clustered_solutions = None
            self._clustered_vectors = None
            
            actual_count = len(self._get_filtered_solutions())
            
            return f"Selected cluster {cluster_id + 1} ({direction_label}). Filters tightened to {actual_count} solutions from this cluster."
        
        else:
            return f"Unknown tool: {tool_name}"
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.
        
        Handles tool calling automatically.
        """
        self.history.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            tools=self.tools,
            tool_choice="auto",
        )
        
        assistant_message = response.choices[0].message
        
        # Handle tool calls
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
                
                result = self._execute_tool(tool_name, arguments)
                
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
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
        
        return final_content
    
    def reset_filters(self):
        """Reset all filters to allow all solutions."""
        self.filter_state = FilterState()
        print("Filters reset. All Pareto-optimal solutions are now feasible.")

    def chat_with_metadata(self, user_message: str) -> dict:
        """
        Process a user message and return the agent's response with metadata.

        Returns a dict with:
        - text: The LLM response text
        - response_type: "text" | "clusters" | "solution_update"
        - clusters: List of cluster info when show_solution_clusters was called
        - solution_path: Path to solution when select_cluster was called
        """
        tool_calls_made = []
        clusters_data = None
        solution_path = None

        self.history.append({"role": "user", "content": user_message})

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

                tool_calls_made.append(tool_name)

                if tool_name == "show_solution_clusters":
                    result = self._execute_tool(tool_name, arguments)
                    if self._cluster_labels is not None:
                        clusters_data = self._build_clusters_response()

                elif tool_name == "select_cluster":
                    cluster_id = arguments.get("cluster_id", 1) - 1
                    result = self._execute_tool(tool_name, arguments)
                    filtered = self._get_filtered_solutions()
                    if len(filtered) > 0:
                        normalized = normalize_metrics(filtered)
                        solution, idx = get_centroid_solution(filtered, normalized)
                        solution_path = solution["path"]

                elif tool_name in ["tighten_filter", "loosen_filter", "get_current_solution"]:
                    result = self._execute_tool(tool_name, arguments)
                    filtered = self._get_filtered_solutions()
                    if len(filtered) > 0:
                        normalized = normalize_metrics(filtered)
                        solution, idx = get_centroid_solution(filtered, normalized)
                        solution_path = solution["path"]

                else:
                    result = self._execute_tool(tool_name, arguments)

                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
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

        if solution_path is None and clusters_data is None and len(tool_calls_made) > 0:
            filtered = self._get_filtered_solutions()
            if len(filtered) > 0:
                normalized = normalize_metrics(filtered)
                solution, idx = get_centroid_solution(filtered, normalized)
                solution_path = solution["path"]

        if clusters_data is not None:
            response_type = "clusters"
        elif solution_path is not None:
            response_type = "solution_update"
        else:
            response_type = "text"

        return {
            "text": final_content,
            "response_type": response_type,
            "clusters": clusters_data,
            "solution_path": solution_path,
        }

    def _build_clusters_response(self) -> list:
        """Build cluster data for frontend response."""
        if self._cluster_labels is None:
            return []

        clusters = []
        n_clusters = len(self._cluster_centers)

        for cluster_id in range(n_clusters):
            mask = self._cluster_labels == cluster_id
            cluster_solutions = self._clustered_solutions[mask]

            if len(cluster_solutions) == 0:
                continue

            normalized = normalize_metrics(cluster_solutions)
            centroid_solution, _ = get_centroid_solution(cluster_solutions, normalized)

            direction_info = self._cluster_directions.get(cluster_id, {})

            clusters.append({
                "id": cluster_id + 1,
                "label": direction_info.get("direction_label", f"Cluster {cluster_id + 1}"),
                "count": int(mask.sum()),
                "path": centroid_solution["path"],
            })

        return clusters
