"""Tool JSON schema definitions for the zoning agent."""

from google.genai import types

from .metrics_config import ALL_METRICS, CATEGORIES


def build_tools():
    """Build tool definitions with current metric names.

    Returns a ``types.Tool`` containing all function declarations
    for the Google GenAI native SDK.
    """
    all_metric_names = [m.display_name for m in ALL_METRICS]
    filterable_metric_names = [m.display_name for m in ALL_METRICS if m.direction is not None]
    category_names = list(CATEGORIES.keys())

    declarations = [
        types.FunctionDeclaration(
            name="query_zone_data",
            description="Query detailed data for specific zones in the current mapping. Returns demographics, programs, quality metrics, and distances for requested zones.",
            parameters_json_schema={
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
                    },
                    "version": {
                        "type": "integer",
                        "description": "Version number to query (e.g. 0, 1, 2). Defaults to the current version.",
                    },
                },
                "required": [],
            },
        ),
        # types.FunctionDeclaration(
        #     name="compare_zones",
        #     description="Compare two or more zones side-by-side on key metrics. Useful for understanding differences between zones.",
        #     parameters_json_schema={
        #         "type": "object",
        #         "properties": {
        #             "zone_ids": {
        #                 "type": "array",
        #                 "items": {"type": "integer"},
        #                 "description": "List of 2+ zone display numbers as shown on the map (e.g., [1, 3]).",
        #                 "minItems": 2,
        #             }
        #         },
        #         "required": ["zone_ids"],
        #     },
        # ),
        types.FunctionDeclaration(
            name="show_version_history",
            description="Show the history of filter changes and mapping states in this session.",
            parameters_json_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.FunctionDeclaration(
            name="get_solution",
            description="Get the 'balanced' mapping for a given version. Defaults to the current version. Use show_all_metrics based on user intent: if they're asking for overview/details/depth about metrics, set to true. If they just want current status or quick check, set to false.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "show_all_metrics": {
                        "type": "boolean",
                        "description": "Set to true when user is asking for detailed information, comprehensive view, or wants to understand all available metrics. Set to false for quick status checks. Use your judgment based on user intent, not specific trigger phrases.",
                    },
                    "version": {
                        "type": "integer",
                        "description": "Version number to query (e.g. 0, 1, 2). Defaults to the current version.",
                    },
                },
                "required": [],
            },
        ),
        types.FunctionDeclaration(
            name="list_all_metrics",
            description="List all available metrics organized by category with their descriptions and directions (higher/lower is better). Use this to understand what metrics are available before filtering.",
            parameters_json_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.FunctionDeclaration(
            name="search_metrics",
            description="Search for metrics by keyword. Returns matching metrics with their details.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term to find in metric names or descriptions (e.g., 'spanish', 'diversity', 'math').",
                    }
                },
                "required": ["query"],
            },
        ),
        types.FunctionDeclaration(
            name="tighten_filter",
            description="Tighten the constraint for a specific metric to improve it. Uses moderate strength by default. Infer strength based on user tone and intent, or explicitly ask if unclear. Not available for 'Number of Zones' (use set_filter with min_value/max_value instead).",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "The display name of the metric to tighten.",
                        "enum": filterable_metric_names,
                    },
                    "strength": {
                        "type": "string",
                        "description": "How aggressively to tighten: 'mild' (~5% of range), 'moderate' (~10%, default), or 'aggressive' (~25%)",
                        "enum": ["mild", "moderate", "aggressive"],
                        "default": "moderate",
                    },
                },
                "required": ["metric_name"],
            },
        ),
        types.FunctionDeclaration(
            name="loosen_filter",
            description="Loosen the constraint for a specific metric to allow more diverse solutions. IMPORTANT: Before calling this tool, ALWAYS explain what loosening this constraint will enable (more solutions, ability to improve other metrics) and what the trade-off is (accepting worse values for this metric). Not available for 'Number of Zones' (use set_filter with min_value/max_value instead).",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "The display name of the metric to loosen.",
                        "enum": filterable_metric_names,
                    },
                    "strength": {
                        "type": "string",
                        "description": "How much to loosen: 'mild' (~10% toward unconstrained), 'moderate' (~25%), or 'aggressive' (~50%)",
                        "enum": ["mild", "moderate", "aggressive"],
                    },
                },
                "required": ["metric_name"],
            },
        ),
        types.FunctionDeclaration(
            name="set_filter",
            description="Set an explicit filter bound for a metric by raw value, percentile, or clear it. Use when the user has a specific target (e.g. 'FRL deviation under 0.05' or 'top 25% for distance'). If neither value nor percentile is provided, the filter is cleared (unconstrained). Prefer tighten_filter/loosen_filter for relative adjustments. SPECIAL CASE: 'Number of Zones' has no optimization direction. For this metric only, use min_value and max_value to set an inclusive range (e.g. min_value=5, max_value=5 for exactly 5 zones, or min_value=5, max_value=7 for 5-7 zones). The value/percentile parameters do not apply to 'Number of Zones'. All other metrics work the same as before.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "The display name of the metric.",
                        "enum": all_metric_names,
                    },
                    "value": {
                        "type": "number",
                        "description": "Exact raw metric value to use as the bound. For 'minimize' metrics this sets an upper bound; for 'maximize' metrics a lower bound. Not used for 'Number of Zones'.",
                    },
                    "percentile": {
                        "type": "number",
                        "description": "Quality percentile 0-100 (higher = better). E.g. 75 keeps only solutions better than 75% of all Pareto solutions for this metric. Not used for 'Number of Zones'.",
                    },
                    "min_value": {
                        "type": "number",
                        "description": "Minimum of the range (inclusive). Only used for 'Number of Zones'.",
                    },
                    "max_value": {
                        "type": "number",
                        "description": "Maximum of the range (inclusive). Only used for 'Number of Zones'.",
                    },
                },
                "required": ["metric_name"],
            },
        ),
        types.FunctionDeclaration(
            name="get_filter_bounds",
            description="Get current filter bounds and statistics for metrics. Shows the current constraints, ranges in all solutions, and ranges in filtered solutions.",
            parameters_json_schema={
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
        ),
        types.FunctionDeclaration(
            name="find_feasible_relaxation",
            description="When there are no feasible solutions with current filters, find which filters need to be relaxed and by how much to restore feasibility.",
            parameters_json_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.FunctionDeclaration(
            name="apply_feedback_filters",
            description="Reset all filters and apply multiple metric constraints in one batch based on analysis of user feedback. Use this when the user asks to generate a new solution from their feedback, or when you identify clear metric preferences from their pros/cons notes. Be aggressive - apply a constraint for every identifiable preference.",
            parameters_json_schema={
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
                                    "enum": filterable_metric_names,
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
        ),
    ]

    return types.Tool(function_declarations=declarations)
