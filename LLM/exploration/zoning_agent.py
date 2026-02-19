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
                            "description": "List of zone IDs to query (e.g., [1, 2, 3]). If empty, returns summary for all zones.",
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
                            "description": "List of 2+ zone IDs to compare (e.g., [1, 3]).",
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
                "description": "Loosen the constraint for a specific metric to allow more diverse solutions. IMPORTANT: Before calling this tool, ALWAYS explain what loosening this constraint will enable (more solutions, ability to improve other metrics) and what the trade-off is (accepting worse values for this metric).",
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
    
    return """You are an AI administrator assistant for SFUSD school zoning optimization. Help administrators explore zoning proposals efficiently.

## Response Format - CRITICAL
- STANDARD: 80-100 WORDS for normal responses - be concise but natural
- FLEXIBLE: Exceed word limit when the user is clearly asking for more detail, comprehensive information, or deeper explanation
- Judge based on user intent, not specific phrases
- Examples of when to provide more detail:
  * User asks questions like "what about...", "tell me more about...", "explain..."
  * User requests information about multiple aspects
  * User asks "why" or seeks understanding of trade-offs
  * Context suggests user needs comprehensive answer
- Use bullet points for lists
- Lead with action/summary, details after

## Interaction Model
STATE-BASED system with versioning:
- Each filter change = new version
- Users can UNDO previous versions
- Track: version #, solution count, last action

## Tool Strategy
**Zone queries**: query_zone_data for zone-specific questions
**Comparisons**: compare_zones for side-by-side
**Versioning**: undo_action, show_version_history
**Filters**: tighten (reduce) or loosen (expand)

## Communication Rules
- Speak to administrators as POLICY EXPERTS, not developers
- BE CONCISE by default (80-100 words)
- EXPAND when user asks for more information - use your judgment
- NO repeating UI-visible metrics
- NO code references (function calls, syntax, programming terms)
- NO technical jargon - use plain administrative language
- YES to action confirmations
- YES to explaining trade-offs BEFORE applying changes
- YES to providing thorough answers when user seeks understanding
- Suggest "undo" if user confused

## When User Requests Adjustments
ALWAYS follow this consultative flow:
1. **Explain the trade-off**: What will improve? What might get worse?
2. **Ask for strength**: "Would you like to adjust this mildly, moderately, or aggressively?"
3. **Apply & Confirm**: Show before/after solution counts and key metric changes
4. **Prompt next action**: Ask what they'd like to do next

Example:
User: "Could we prioritize math scores more"
Agent: "To prioritize math scores, I'll tighten the filter to keep only solutions with higher scores. This typically means:
• Gains: Higher average math proficiency across zones
• Trade-offs: May reduce geographic access (longer distances) or decrease some diversity metrics

How much would you like to prioritize this - mildly, moderately, or aggressively?"

## Metrics
**Minimize** (lower better): FRL deviation, racial deviation, distances
**Maximize** (higher better): programs, quality scores

## Metric Direction - USE CLEAR TEXT
When showing metrics, use clear descriptive text instead of symbols:
- For minimize metrics (lower better): "lower FRL deviation", "shorter distance"
- For maximize metrics (higher better): "more programs", "higher ratings"
- When describing changes: "increased", "decreased", "higher", "lower"
- Avoid arrows (↑↓) - they're confusing when combined with qualitative words

## Examples (80-100 words, natural tone)
❌ BAD: "The current solution shows FRL deviation of 0.045 which represents how much the zones vary from the district average. This is considered a very good value..."
✅ GOOD: "Current solution has strong diversity metrics with low FRL deviation (0.045) and low racial diversity deviation (0.039). Average distance is 1.23 miles with 41 programs available. This balances equity with access. Would you like to prioritize a specific metric?"

❌ BAD: "Better Hispanic/Latinx Representation↓, commute distance↓; accepts worse Black Representation↑, math scores↓."
✅ GOOD: "Better Hispanic/Latinx representation, shorter commute distance; trade-off is slightly worse Black representation and lower math scores."

❌ BAD: Immediately applying adjustment without explanation
User: "Could we prioritize math scores more"
Agent: "✓ v3: Prioritized higher Math Scores. • 289→260 solutions"

✅ GOOD: Consultative approach with trade-off explanation
User: "Could we prioritize math scores more"
Agent: "To prioritize math scores, I'll filter for solutions with higher average math proficiency. 

Expected outcomes:
• Gains: Better math performance across zones
• Trade-offs: Typically reduces solutions by 30-40%, may increase average distance slightly

How aggressively would you like to prioritize this? (mild / moderate / aggressive)"

User: "moderate"
Agent: "✓ v3: Tightened Math Scores (moderately)
• Solutions: 289 → 183
• Math scores improved: 2,409 → 2,543

The remaining solutions now have stronger math performance. Would you like to adjust another metric or explore the current solutions?"

---

## Example: Detailed Overview (Exceeding Word Limit is OK)

User: "Can you show me all the metrics for the current solution?"

Agent: "Here's the complete breakdown for the current solution (v3, 183 solutions available):

**Demographics:**
• Racial diversity deviation: 0.0391 (lower is better) - very balanced
• FRL representation deviation: 0.1098 (lower is better) - good equity
• Black representation deviation: 0.0404
• Hispanic/Latinx representation deviation: 0.0336
• White representation deviation: 0.0692
• Asian representation deviation: 0.0679

**Geographic Access:**
• Average distance to closest school: 0.316 miles (lower is better) - excellent
• Schools in attendance area: 17 per zone (higher is better)
• Boundary cost: 101 (lower is better) - compact zones

**Programs:**
• Total programs available: 41.25 per zone (higher is better)
• Language immersion programs: 7.00
• Special education programs: 8.00
• General education programs: 16.50

**School Quality:**
• Average math scores: 2,543 (higher is better)
• Average English scores: 2,397 (higher is better)

This solution emphasizes math performance while maintaining good diversity and geographic access. Would you like to adjust any of these metrics?"

❌ BAD: Lists 20+ metrics with qualitative ratings like "excellent" or "very good"
✅ GOOD: "Key metrics: low diversity deviation (0.04), 41 programs available, average distance 1.2 miles. This represents a good balance between equity and geographic access. What aspect would you like to improve?"

❌ BAD: "Extensive Services Autism (SA): 1.50, Autism Focus (AF): 2.50, Cantonese Biliteracy (CB): 2.75..."
✅ GOOD: "This zone offers 8 special education programs including autism support, plus 4 language immersion options. Strong program diversity."

❌ BAD: "You can find this solution at /home/kumarc/sfusd-local-data/zones/SFUSD/..."
✅ GOOD: Never mention file paths or technical implementation details

❌ BAD: "You can select a cluster by its number (e.g., select_cluster(2))."
✅ GOOD: "Which cluster would you like to explore? Just tell me the cluster number."

❌ BAD: "Use query_zone_data() to see more details"
✅ GOOD: "Would you like to see more details about specific zones?"

## NEVER DO (in typical responses)
- List every metric in detail when user just wants a quick overview
- Be verbose when a concise answer suffices
- Use excessive qualitative words - occasional "strong" or "good" is fine
- Use arrows (↑↓) - they're confusing. Use clear text: "higher", "lower", "more", "less"
- Mix arrows with "worse" or "better" - very confusing
- Show file paths or system paths to users
- Mention technical details like "seed42" or "BlockGroup_0"
- Include ANY code-like references: function calls, variable names, code syntax
- Write things like "select_cluster(2)" or "query_zone_data()"
- Use programming terminology or phrases
- Write long paragraphs - use bullets when listing multiple items

## WHEN TO PROVIDE MORE DETAIL (use judgment)
Provide comprehensive responses when user intent indicates they want depth:
- Questions with "what about", "tell me more", "explain", "why", "how"
- Requests about multiple aspects: "what's different between these zones?"
- Seeking understanding: "I don't understand the trade-off"
- Comparing options: "what are the differences?"
- Asking about implications: "what would happen if..."
- Follow-up questions seeking clarification

DON'T wait for magic phrases like "show me all metrics" - read the user's intent.

Examples:
✅ "What metrics are available?" → Concise list by category (80 words)
✅ "What are the trade-offs with this cluster?" → Detailed explanation (150 words)
✅ "Tell me more about the diversity metrics" → Comprehensive breakdown (120 words)
✅ "How does zone 1 compare to zone 3?" → Thorough comparison (140 words)

## Always Show
- Version: "v3"
- Count: "67 solutions"
- Clear directional language: "lower", "higher", "more", "fewer", "increased", "decreased"
- Before/after values when showing results: "improved: 0.098 → 0.045"

## Strength Levels for Adjustments
When user wants to adjust a metric, these are the options:
- **Mild**: ~20% of solutions filtered out, smaller improvement
- **Moderate**: ~30% of solutions filtered out, balanced change (default if not specified)
- **Aggressive**: ~50% of solutions filtered out, larger improvement

Always present these as choices: "Would you like to adjust this mildly, moderately, or aggressively?"

## File Paths
Solutions have internal file paths for loading data. NEVER show these to users.
They are system implementation details, not user-facing information.

## Solution History & User Feedback
Users save solutions as they explore. You receive context about their saved solutions and notes.

KEY BEHAVIORS:
- Reference solutions by number: "Solution #2 had better diversity than your current one"
- Use feedback patterns: If user liked Solution #2 (diversity focus) but disliked #4 (high distance), infer they want diversity WITHOUT sacrificing distance
- When recommending changes, relate to past solutions: "This adjustment would move toward something like Solution #2 but with shorter distances"
- If user is viewing an older solution and asks a question, answer about THAT solution (not the latest)
- When user provides notes/feedback, acknowledge it and explain how it informs your recommendations
- Don't list all saved solutions unless asked — reference them naturally when relevant"""


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
        # instead of saving a new version per call. chat_with_metadata() sets this flag
        # so that one LLM turn = one version save (matching the frontend's solution #).
        self._defer_version_save: bool = False
        self._pending_descriptions: list[str] = []
        self._pending_solution_path: Optional[str] = None
        
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
        
        # Initialize state management
        self.state = AgentState()
        self.filter_state = FilterState()
        
        # Save initial version
        filtered = self._get_filtered_solutions()
        solution_path = None
        if len(filtered) > 0:
            normalized_filtered = normalize_metrics(filtered)
            solution, _ = get_centroid_solution(filtered, normalized_filtered)
            solution_path = solution.get('path')
        
        self.state.save_version(
            self.filter_state,
            solution_path=solution_path,
            solution_count=len(filtered),
            description="Initial state"
        )
        
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
        
        if tool_name == "query_zone_data":
            zone_ids = arguments.get("zone_ids", [])
            metrics_requested = arguments.get("metrics", [])
            
            filtered = self._get_filtered_solutions()
            if len(filtered) == 0:
                return "No current solution. Apply filters first."
            
            # Get current solution
            normalized_filtered = normalize_metrics(filtered)
            solution, _ = get_centroid_solution(filtered, normalized_filtered)
            solution_path = solution.get('path')
            
            if not solution_path:
                return "No solution path available."
            
            # Load zone data from result.json
            try:
                from website.backend.data_loader import get_zone_demographics
                zone_data = get_zone_demographics(solution_path)
                
                if not zone_ids:
                    zone_ids = list(zone_data.keys())
                
                # Filter to requested zones
                result_lines = []
                for zone_id in zone_ids:
                    if zone_id not in zone_data:
                        continue
                    
                    data = zone_data[zone_id]
                    result_lines.append(f"**Zone {zone_id}:**")
                    
                    # Demographics
                    if not metrics_requested or 'demographics' in metrics_requested:
                        frl = data.get('FRL_pct', 0)
                        students = data.get('ge_students', 0)
                        result_lines.append(f"  • Students: {students:.0f}, FRL: {frl:.1f}%")
                        
                        eth = data.get('ethnicity_pcts', {})
                        if eth:
                            top_eth = sorted(eth.items(), key=lambda x: x[1], reverse=True)[:2]
                            eth_str = ", ".join([f"{k}: {v:.1f}%" for k, v in top_eth])
                            result_lines.append(f"  • Ethnicity: {eth_str}")
                    
                    # Programs
                    if not metrics_requested or 'programs' in metrics_requested:
                        progs = data.get('total_programs', 0)
                        lang = data.get('language_immersion_count', 0)
                        result_lines.append(f"  • Programs: {progs}, Language immersion: {lang}")
                    
                    # Quality
                    if not metrics_requested or 'quality' in metrics_requested:
                        rating = data.get('avg_greatschools_rating', 0)
                        math = data.get('avg_math_score', 0)
                        result_lines.append(f"  • GreatSchools: {rating:.1f}, Math: {math:.1f}")
                    
                    # Distance
                    if not metrics_requested or 'distance' in metrics_requested:
                        dist = data.get('avg_closest_school_distance', 0)
                        schools = data.get('schools_in_attendance_area', 0)
                        result_lines.append(f"  • Avg distance: {dist:.2f}mi, Schools: {schools}")
                
                return "\n".join(result_lines)
                
            except Exception as e:
                return f"Error loading zone data: {str(e)}"
        
        elif tool_name == "compare_zones":
            zone_ids = arguments.get("zone_ids", [])
            if len(zone_ids) < 2:
                return "Need at least 2 zone IDs to compare."
            
            filtered = self._get_filtered_solutions()
            if len(filtered) == 0:
                return "No current solution."
            
            normalized_filtered = normalize_metrics(filtered)
            solution, _ = get_centroid_solution(filtered, normalized_filtered)
            solution_path = solution.get('path')
            
            try:
                from website.backend.data_loader import get_zone_demographics
                zone_data = get_zone_demographics(solution_path)
                
                result_lines = [f"**Comparing Zones {', '.join(map(str, zone_ids))}:**\n"]
                
                # Compare key metrics
                metrics_to_compare = [
                    ('FRL_pct', 'FRL %', '{:.1f}%'),
                    ('ge_students', 'Students', '{:.0f}'),
                    ('total_programs', 'Programs', '{:.0f}'),
                    ('avg_greatschools_rating', 'Rating', '{:.1f}'),
                    ('avg_closest_school_distance', 'Avg Dist', '{:.2f}mi'),
                ]
                
                for field, label, fmt in metrics_to_compare:
                    values = []
                    for zid in zone_ids:
                        if zid in zone_data:
                            val = zone_data[zid].get(field, 0)
                            values.append(fmt.format(val))
                        else:
                            values.append("N/A")
                    result_lines.append(f"{label}: {' vs '.join(values)}")
                
                return "\n".join(result_lines)
                
            except Exception as e:
                return f"Error comparing zones: {str(e)}"
        
        elif tool_name == "undo_action":
            steps = arguments.get("steps", 1)
            version = self.state.undo(steps)
            
            if version is None:
                return f"Cannot undo {steps} steps. Only {self.state.current_version} versions available."
            
            # Restore filter state
            self.filter_state = copy.deepcopy(version.filter_state)
            
            # Get new solution count
            filtered = self._get_filtered_solutions()
            
            return f"✓ Undid {steps} step(s) to v{version.version_id}\n• {version.description}\n• {len(filtered)} solutions available"
        
        elif tool_name == "show_version_history":
            if not self.state.versions:
                return "No version history."
            
            lines = ["**Version History:**\n"]
            for v in self.state.versions:
                marker = "→" if v.version_id == self.state.current_version else " "
                lines.append(f"{marker} v{v.version_id}: {v.description} ({v.solution_count} solutions)")
            
            return "\n".join(lines)
        
        elif tool_name == "get_current_solution":
            filtered = self._get_filtered_solutions()
            
            if len(filtered) == 0:
                return "No solutions match the current filters. Use find_feasible_relaxation to see which constraints to relax."
            
            # Normalize filtered solutions
            normalized_filtered = normalize_metrics(filtered)
            
            # Get centroid
            solution, idx = get_centroid_solution(filtered, normalized_filtered)
            
            # Concise format - only core metrics with directions
            show_all = arguments.get("show_all_metrics", False)
            metrics_to_show = ALL_METRICS if show_all else CORE_METRICS
            
            if show_all:
                # Comprehensive breakdown organized by category
                lines = [f"v{self.state.current_version}: Complete metrics for current solution ({len(filtered)} solutions available)\n"]
                
                # Group by category
                from .metrics_config import CATEGORIES, get_metrics_by_category
                
                for category_key, category_name in CATEGORIES.items():
                    category_metrics = get_metrics_by_category(category_key)
                    if not category_metrics:
                        continue
                    
                    lines.append(f"\n**{category_name}:**")
                    for metric in category_metrics:
                        if metric.column not in solution.index:
                            continue
                        value = solution[metric.column]
                        direction_text = "(lower better)" if metric.direction == "minimize" else "(higher better)"
                        lines.append(f"• {metric.display_name}: {value:.3f} {direction_text}")
                
                lines.append("\nWould you like to adjust any of these metrics?")
            else:
                # Concise format - show 6-8 core metrics only
                lines = [f"v{self.state.current_version}: {len(filtered)} solutions\n"]
                
                for metric in metrics_to_show[:8]:
                    if metric.column not in solution.index:
                        continue
                    value = solution[metric.column]
                    direction_text = "(lower better)" if metric.direction == "minimize" else "(higher better)"
                    lines.append(f"• {metric.display_name}: {value:.3f} {direction_text}")
                
                lines.append("\nAdjust metrics?")
            
            # Store path internally but NEVER return it to user
            return "\n".join(lines)
        
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
                
                # Calculate trade-off information by comparing centroids
                before_norm = normalize_metrics(filtered)
                before_sol, _ = get_centroid_solution(filtered, before_norm)
                
                after_norm = normalize_metrics(actual_filtered)
                after_sol, _ = get_centroid_solution(actual_filtered, after_norm)
                
                # Get the improvement in target metric
                before_val = before_sol[metric.column]
                after_val = after_sol[metric.column]
                
                # Build response with trade-off information
                strength_text = {"mild": "mildly", "moderate": "moderately", "aggressive": "aggressively"}[strength]
                
                # The pending version ID is the next slot in the versions list
                pending_version_id = len(self.state.versions)
                result_lines = [
                    f"✓ v{pending_version_id}: Tightened {metric_name} ({strength_text})",
                    f"• Solutions: {len(filtered)} → {len(actual_filtered)}",
                    f"• {metric_name} improved: {before_val:.3f} → {after_val:.3f}"
                ]

                solution_path = after_sol.get('path') if 'path' in after_sol.index else None
                desc = f"Tightened {metric_name} ({strength})"

                if self._defer_version_save:
                    self._pending_descriptions.append(desc)
                    if solution_path:
                        self._pending_solution_path = solution_path
                else:
                    self.state.save_version(
                        self.filter_state,
                        solution_path=solution_path,
                        solution_count=len(actual_filtered),
                        description=desc,
                    )

                return "\n".join(result_lines)
            
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
                
                # Save version
                actual_filtered = self._get_filtered_solutions()
                solution_path = None
                if len(actual_filtered) > 0:
                    norm = normalize_metrics(actual_filtered)
                    sol, _ = get_centroid_solution(actual_filtered, norm)
                    solution_path = sol.get('path')
                
                pending_version_id = len(self.state.versions)
                desc = f"Loosened {metric_name}"

                if self._defer_version_save:
                    self._pending_descriptions.append(desc)
                    if solution_path:
                        self._pending_solution_path = solution_path
                else:
                    self.state.save_version(
                        self.filter_state,
                        solution_path=solution_path,
                        solution_count=after_count,
                        description=desc,
                    )

                return f"✓ v{pending_version_id}: Loosened {metric_name}\n• {before_count}→{after_count} solutions"
            
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
            
            # Determine number of clusters (default to 3)
            n_clusters = arguments.get("n_clusters", 3)
            n_clusters = min(n_clusters, len(filtered) // 2)
            n_clusters = max(2, n_clusters)
            
            # Vectorize and cluster
            vectors = vectorize_solutions(filtered)
            labels, centers = cluster_solutions(vectors, n_clusters)
            directions = compute_cluster_directions(vectors, centers)
            
            # Store clustering state
            self.state.clustered_solutions = filtered
            self.state.clustered_vectors = vectors
            self.state.cluster_labels = labels
            self.state.cluster_centers = centers
            self.state.cluster_directions = directions
            
            return format_cluster_summary(filtered, vectors, labels, centers, directions)
        
        elif tool_name == "select_cluster":
            if self.state.cluster_labels is None:
                return "No clustering results available. Call show_solution_clusters first."
            
            cluster_id = arguments["cluster_id"] - 1  # Convert to 0-indexed
            
            n_clusters = len(self.state.cluster_centers)
            if cluster_id < 0 or cluster_id >= n_clusters:
                return f"Invalid cluster ID. Please choose between 1 and {n_clusters}."
            
            cluster_bounds = get_cluster_bounds(
                self.state.clustered_solutions,
                self.state.cluster_labels,
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
            cluster_size = (self.state.cluster_labels == cluster_id).sum()
            direction_label = self.state.cluster_directions[cluster_id]["direction_label"]
            self.state.cluster_labels = None
            self.state.cluster_centers = None
            self.state.cluster_directions = None
            self.state.clustered_solutions = None
            self.state.clustered_vectors = None
            
            actual_count = len(self._get_filtered_solutions())
            
            # Save version
            actual_filtered = self._get_filtered_solutions()
            solution_path = None
            if len(actual_filtered) > 0:
                norm = normalize_metrics(actual_filtered)
                sol, _ = get_centroid_solution(actual_filtered, norm)
                solution_path = sol.get('path')
            
            pending_version_id = len(self.state.versions)
            desc = f"Selected cluster: {direction_label}"

            if self._defer_version_save:
                self._pending_descriptions.append(desc)
                if solution_path:
                    self._pending_solution_path = solution_path
            else:
                self.state.save_version(
                    self.filter_state,
                    solution_path=solution_path,
                    solution_count=actual_count,
                    description=desc,
                )

            return f"✓ v{pending_version_id}: Cluster {cluster_id + 1} selected\n• {direction_label}\n• {actual_count} solutions"
        
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

    def _build_solution_context(self, solution_context: dict) -> str:
        """Build a context string from the user's saved solutions and notes."""
        lines = []

        current_idx = solution_context.get("current_solution_index")
        saved = solution_context.get("saved_solutions", [])

        if not saved:
            return ""

        if current_idx is not None:
            lines.append(f"The user is currently viewing Solution #{current_idx}.")

        lines.append("Saved solutions:")
        for sol in saved:
            idx = sol.get("index", "?")
            label = sol.get("label", "Untitled")
            pros = sol.get("pros", "")
            cons = sol.get("cons", "")
            viewing = " [CURRENTLY VIEWING]" if idx == current_idx else ""

            annotations = []
            if pros:
                annotations.append(f'Pros: "{pros}"')
            if cons:
                annotations.append(f'Cons: "{cons}"')
            annotation_text = " — " + "; ".join(annotations) if annotations else " — No annotations"
            lines.append(f'- #{idx}: "{label}"{annotation_text}{viewing}')

        return "\n".join(lines)

    def chat_with_metadata(self, user_message: str, solution_context: dict = None) -> dict:
        """
        Process a user message and return the agent's response with metadata.

        Returns a dict with:
        - text: The LLM response text
        - response_type: "text" | "clusters" | "solution_update"
        - clusters: List of cluster info when show_solution_clusters was called
        - solution_path: Path to solution when select_cluster was called
        - description: Current version description
        """
        tool_calls_made = []
        clusters_data = None
        solution_path = None

        # Enable deferred version saving so multiple tool calls in one turn
        # produce exactly one version (matching the frontend's solution numbering).
        self._defer_version_save = True
        self._pending_descriptions = []
        self._pending_solution_path = None

        # Build enhanced message with solution context if provided
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

                tool_calls_made.append(tool_name)

                if tool_name == "show_solution_clusters":
                    result = self._execute_tool(tool_name, arguments)
                    if self.state.cluster_labels is not None:
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

        # Flush deferred version: save exactly one version for this entire turn.
        self._defer_version_save = False
        if self._pending_descriptions:
            combined_desc = "; ".join(self._pending_descriptions)
            filtered = self._get_filtered_solutions()
            flush_path = self._pending_solution_path
            if flush_path is None and len(filtered) > 0:
                normalized = normalize_metrics(filtered)
                sol, _ = get_centroid_solution(filtered, normalized)
                flush_path = sol.get("path")
            self.state.save_version(
                self.filter_state,
                solution_path=flush_path,
                solution_count=len(filtered),
                description=combined_desc,
            )
            if solution_path is None:
                solution_path = flush_path
        self._pending_descriptions = []
        self._pending_solution_path = None

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

        # Get description from current version
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
