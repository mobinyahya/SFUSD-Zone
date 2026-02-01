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


# Tool definitions for the LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_solution",
            "description": "Get the current 'average' or centroid solution based on the current filters. Returns the solution metrics and overall statistics.",
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
            "name": "tighten_filter",
            "description": "Tighten the constraint for a specific metric to improve it (reduce its value since lower is better for all metrics). This will reduce the number of feasible solutions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "The name of the metric to tighten. Must be one of the available metrics.",
                        "enum": list(METRIC_CONFIG.keys()),
                    },
                    "strength": {
                        "type": "string",
                        "description": "How aggressively to tighten: 'mild' (~20% reduction in solutions), 'moderate' (~30%), or 'aggressive' (~50%)",
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
            "description": "Loosen the constraint for a specific metric to allow more diverse solutions. Use this when the user is willing to accept worse values for a metric to improve others.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "The name of the metric to loosen.",
                        "enum": list(METRIC_CONFIG.keys()),
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
            "description": "Get current filter bounds and statistics for all metrics. Shows the current constraints, ranges in all solutions, and ranges in filtered solutions.",
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
            "description": "Group the current feasible solutions into clusters and show a representative solution from each cluster with an interpretable direction label. Useful when there are many solutions and the user wants to see different 'types' of solutions available. Each cluster represents a different approach to trade-offs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of clusters to create. Default is automatically chosen based on solution count (typically 3-5).",
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
            "description": "Select a cluster from the previous show_solution_clusters results. This will tighten all metric filters to only include solutions within that cluster, effectively narrowing down to that type of solution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {
                        "type": "integer",
                        "description": "The cluster number to select (1 to N, as shown in show_solution_clusters results)",
                    },
                },
                "required": ["cluster_id"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a helpful assistant that helps parents explore school zoning proposals for San Francisco Unified School District.

## Your Role
You help users find zoning solutions that match their priorities by iteratively adjusting filters on various metrics. Think of yourself as a friendly guide who helps translate high-level preferences ("I want more diverse schools") into concrete adjustments to metric thresholds.

## Available Metrics
All metrics measure DEVIATION from ideal values, so LOWER IS BETTER for all of them:

1. **Free and Reduced Lunch Population % Deviation from district average** - Measures economic diversity. Lower = more balanced free lunch percentages across zones.

2. **Black Population % Deviation from district average** - Measures racial balance for Black students.

3. **Hispanic Population % Deviation from district average** - Measures racial balance for Hispanic/Latinx students.

4. **White Population % Deviation from district average** - Measures racial balance for White students.

5. **Asian Population % Deviation from district average** - Measures racial balance for Asian students.

6. **Total Population % Deviation from district average** - Measures seat availability balance across zones.

7. **Average distance to closest school** - Average distance students travel. Lower = shorter commutes.

8. **Compactness** - Measures how geographically compact the zones are. Lower = more compact, contiguous zones.

## How to Help Users

1. **Start by presenting the current "balanced" solution** - Show all metrics for the centroid solution.

2. **Listen to feedback** - Users will express preferences like "I want shorter commutes" or "economic diversity is most important to me".

3. **Translate to filter adjustments** - When a user wants to improve something, tighten that filter. Explain the trade-off (other metrics may get worse).

4. **Handle impossible requests gracefully** - If filters become too tight (0 solutions), use find_feasible_relaxation to suggest which constraints to relax. Ask the user which metrics they're willing to compromise on.

5. **Always explain trade-offs** - Help users understand that improving one metric often means accepting worse values for others.

## Communication Style
- Be friendly and accessible - users are parents, not optimization experts
- Use plain language, not technical jargon
- Always show the specific metric values when presenting solutions
- Proactively suggest trade-offs: "To get shorter commutes, you might need to accept less economic diversity"

## Clustering Feature
When users are exploring a large set of solutions or want to see what different types of solutions are available, use the clustering feature:

1. **show_solution_clusters** - Groups similar solutions together and shows a representative from each group with a label describing what trade-offs that cluster makes (e.g., "Better commute distance; accepts higher economic diversity deviation")

2. **select_cluster** - Once the user picks a cluster they like, this narrows the filters to only include solutions similar to that cluster

Use clustering when:
- The user seems overwhelmed by choices
- The user asks "what are my options?" or "what trade-offs can I make?"
- There are many (>10) feasible solutions and you want to help the user navigate
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
        self.normalized_solutions = normalize_metrics(self.all_solutions)
        self.pareto_frontier = compute_pareto_frontier(self.normalized_solutions)
        
        # Get original (unnormalized) Pareto solutions
        pareto_indices = self.pareto_frontier.index
        self.pareto_original = self.all_solutions.loc[pareto_indices].copy()
        
        # Initialize filter state (no filters initially)
        self.filter_state = FilterState()
        
        # Clustering state (populated when show_solution_clusters is called)
        self._cluster_labels = None  # np.ndarray of cluster assignments
        self._cluster_centers = None  # np.ndarray of cluster centers
        self._cluster_directions = None  # dict of cluster direction info
        self._clustered_solutions = None  # DataFrame of solutions used for clustering
        self._clustered_vectors = None  # np.ndarray of vectorized solutions
        
        # Conversation history
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        print(f"Loaded {len(self.all_solutions)} total solutions")
        print(f"Computed Pareto frontier with {len(self.pareto_frontier)} solutions")
    
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
            result = f"**Current Solution** (centroid of {len(filtered)} feasible solutions)\n\n"
            result += format_solution(solution)
            result += f"\n\n**Path:** {solution['path']}"
            
            return result
        
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
                new_bound, expected_remaining = calculate_tightening(
                    filtered, metric_name, reduction
                )
                
                # Apply the new bound
                self.filter_state.bounds[metric_name].max_bound = new_bound
                
                actual_filtered = self._get_filtered_solutions()
                
                return f"Tightened '{metric_name}' to max value {new_bound:.4f}. {len(actual_filtered)} solutions remaining (was {len(filtered)})."
            
            except Exception as e:
                return f"Error tightening filter: {str(e)}"
        
        elif tool_name == "loosen_filter":
            metric_name = arguments["metric_name"]
            
            try:
                new_bound, added_count = calculate_loosening(
                    self.pareto_original, self.filter_state, metric_name
                )
                
                if new_bound is None:
                    return f"'{metric_name}' is already unconstrained."
                
                before_count = len(self._get_filtered_solutions())
                self.filter_state.bounds[metric_name].max_bound = new_bound
                after_count = len(self._get_filtered_solutions())
                
                return f"Loosened '{metric_name}' to max value {new_bound:.4f}. {after_count} solutions now feasible (was {before_count})."
            
            except Exception as e:
                return f"Error loosening filter: {str(e)}"
        
        elif tool_name == "get_filter_bounds":
            filtered = self._get_filtered_solutions()
            return get_filter_summary(self.filter_state, self.pareto_original, filtered)
        
        elif tool_name == "find_feasible_relaxation":
            suggestions = find_relaxation_needed(
                self.pareto_original, self.filter_state
            )
            
            if not suggestions:
                return "Unable to find relaxations that would restore feasibility. Try loosening multiple filters manually."
            
            result = "**Suggested Relaxations** (relaxing any ONE of these could restore feasibility):\n\n"
            for metric_name, new_bound in suggestions.items():
                current = self.filter_state.bounds[metric_name].max_bound
                result += f"• **{metric_name}**: relax from {current:.4f} → {new_bound:.4f}\n"
            
            result += "\nAsk the user which metric they're willing to compromise on."
            return result
        
        elif tool_name == "show_solution_clusters":
            filtered = self._get_filtered_solutions()
            
            if len(filtered) < 3:
                return f"Not enough solutions to cluster (only {len(filtered)}). Need at least 3 solutions."
            
            # Determine number of clusters
            n_clusters = arguments.get("n_clusters")
            if n_clusters is None:
                # Auto-select: aim for ~3-5 clusters, with at least 2 solutions per cluster
                n_clusters = min(max(2, len(filtered) // 3), 5)
            n_clusters = min(n_clusters, len(filtered) // 2)  # At least 2 solutions per cluster
            n_clusters = max(2, n_clusters)  # At least 2 clusters
            
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
            
            # Format summary
            return format_cluster_summary(filtered, vectors, labels, centers, directions)
        
        elif tool_name == "select_cluster":
            if self._cluster_labels is None:
                return "No clustering results available. Call show_solution_clusters first."
            
            # User provides 1-indexed cluster ID
            cluster_id = arguments["cluster_id"] - 1  # Convert to 0-indexed
            
            n_clusters = len(self._cluster_centers)
            if cluster_id < 0 or cluster_id >= n_clusters:
                return f"Invalid cluster ID. Please choose between 1 and {n_clusters}."
            
            # Get bounds for this cluster
            cluster_bounds = get_cluster_bounds(
                self._clustered_solutions,
                self._cluster_labels,
                cluster_id
            )
            
            # Apply bounds to filter state
            for metric_name, bounds in cluster_bounds.items():
                # Only set max_bound since all metrics are minimize
                self.filter_state.bounds[metric_name].max_bound = bounds.max_bound
            
            # Clear clustering state
            cluster_size = (self._cluster_labels == cluster_id).sum()
            direction_label = self._cluster_directions[cluster_id]["direction_label"]
            self._cluster_labels = None
            self._cluster_centers = None
            self._cluster_directions = None
            self._clustered_solutions = None
            self._clustered_vectors = None
            
            # Get actual filtered count after applying bounds
            actual_count = len(self._get_filtered_solutions())
            
            return f"Selected cluster {cluster_id + 1} ({direction_label}). Filters tightened to {actual_count} solutions from this cluster."
        
        else:
            return f"Unknown tool: {tool_name}"
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.
        
        Handles tool calling automatically.
        """
        # Add user message to history
        self.history.append({"role": "user", "content": user_message})
        
        # Call the model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            tools=TOOLS,
            tool_choice="auto",
        )
        
        assistant_message = response.choices[0].message
        
        # Handle tool calls
        while assistant_message.tool_calls:
            # Add assistant message with tool calls to history
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
            
            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                result = self._execute_tool(tool_name, arguments)
                
                # Add tool result to history
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            
            # Get next response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                tools=TOOLS,
                tool_choice="auto",
            )
            assistant_message = response.choices[0].message
        
        # Add final response to history
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
        # Track tool calls and their results
        tool_calls_made = []
        clusters_data = None
        solution_path = None

        # Add user message to history
        self.history.append({"role": "user", "content": user_message})

        # Call the model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        # Handle tool calls
        while assistant_message.tool_calls:
            # Add assistant message with tool calls to history
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

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                # Track the tool call
                tool_calls_made.append(tool_name)

                # Capture cluster data before executing show_solution_clusters
                if tool_name == "show_solution_clusters":
                    # Execute the tool
                    result = self._execute_tool(tool_name, arguments)

                    # After execution, capture the cluster data
                    if self._cluster_labels is not None:
                        clusters_data = self._build_clusters_response()

                elif tool_name == "select_cluster":
                    # Get cluster info before it's cleared
                    cluster_id = arguments.get("cluster_id", 1) - 1

                    # Execute the tool (this clears cluster state)
                    result = self._execute_tool(tool_name, arguments)

                    # Get the centroid solution path after selection
                    filtered = self._get_filtered_solutions()
                    if len(filtered) > 0:
                        normalized = normalize_metrics(filtered)
                        solution, idx = get_centroid_solution(filtered, normalized)
                        solution_path = solution["path"]

                elif tool_name in ["tighten_filter", "loosen_filter", "get_current_solution"]:
                    # Execute the tool
                    result = self._execute_tool(tool_name, arguments)

                    # Get the centroid solution path after filter changes
                    filtered = self._get_filtered_solutions()
                    if len(filtered) > 0:
                        normalized = normalize_metrics(filtered)
                        solution, idx = get_centroid_solution(filtered, normalized)
                        solution_path = solution["path"]

                else:
                    result = self._execute_tool(tool_name, arguments)

                # Add tool result to history
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Get next response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                tools=TOOLS,
                tool_choice="auto",
            )
            assistant_message = response.choices[0].message

        # Add final response to history
        final_content = assistant_message.content or ""
        self.history.append({"role": "assistant", "content": final_content})

        # If no solution path was set yet but tool calls were made (and we're not showing clusters),
        # get the current centroid solution to display
        if solution_path is None and clusters_data is None and len(tool_calls_made) > 0:
            filtered = self._get_filtered_solutions()
            if len(filtered) > 0:
                normalized = normalize_metrics(filtered)
                solution, idx = get_centroid_solution(filtered, normalized)
                solution_path = solution["path"]

        # Determine response type
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

            # Get centroid solution for this cluster
            normalized = normalize_metrics(cluster_solutions)
            centroid_solution, _ = get_centroid_solution(cluster_solutions, normalized)

            direction_info = self._cluster_directions.get(cluster_id, {})

            clusters.append({
                "id": cluster_id + 1,  # 1-indexed for user display
                "label": direction_info.get("direction_label", f"Cluster {cluster_id + 1}"),
                "count": int(mask.sum()),
                "path": centroid_solution["path"],
            })

        return clusters
