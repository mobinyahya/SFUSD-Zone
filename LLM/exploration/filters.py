"""
Filter Management for School Zoning Solutions.

This module handles:
- Managing filter bounds for each metric
- Applying filters to solution sets
- Calculating filter adjustments (tightening/loosening)
- Finding minimal relaxations when filters are too tight
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .metrics_config import (
    ALL_METRICS,
    METRIC_BY_COLUMN,
    METRIC_BY_NAME,
    CATEGORIES,
    get_metrics_by_category,
    MetricSpec,
)


@dataclass
class FilterBounds:
    """Bounds for a single metric filter."""
    min_bound: Optional[float] = None  # None means no lower bound
    max_bound: Optional[float] = None  # None means no upper bound


@dataclass
class FilterState:
    """
    Complete filter state for all metrics.
    
    For "minimize" metrics, we use max_bound (require metric <= max_bound).
    For "maximize" metrics, we use min_bound (require metric >= min_bound).
    """
    bounds: dict[str, FilterBounds] = field(default_factory=dict)
    
    def __post_init__(self):
        for metric in ALL_METRICS:
            if metric.direction is None:
                continue
            if metric.display_name not in self.bounds:
                self.bounds[metric.display_name] = FilterBounds()
    
    def get_active_filters(self) -> list[tuple[str, FilterBounds]]:
        """Get list of metrics with active constraints."""
        active = []
        for name, bounds in self.bounds.items():
            if bounds.min_bound is not None or bounds.max_bound is not None:
                active.append((name, bounds))
        return active


def apply_filters(df: pd.DataFrame, filter_state: FilterState) -> pd.DataFrame:
    """
    Apply current filters to the solution set.
    
    Returns filtered DataFrame containing only solutions within all bounds.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    
    for metric in ALL_METRICS:
        if metric.direction is None or metric.column not in df.columns:
            continue
        
        bounds = filter_state.bounds.get(metric.display_name, FilterBounds())
        col = metric.column
        
        if bounds.min_bound is not None:
            mask &= df[col] >= bounds.min_bound
        if bounds.max_bound is not None:
            mask &= df[col] <= bounds.max_bound
    
    return df[mask].copy()


def get_filter_summary(
    filter_state: FilterState, 
    all_solutions_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    show_category: Optional[str] = None
) -> str:
    """
    Get a human-readable summary of current filter state.
    
    Args:
        filter_state: Current filter state
        all_solutions_df: All solutions (unfiltered)
        filtered_df: Currently filtered solutions
        show_category: If provided, only show metrics in this category
    
    Returns:
        Formatted string summarizing filter state
    """
    lines = [f"**Current Filters** ({len(filtered_df)} solutions remaining out of {len(all_solutions_df)} total)\n"]
    
    # Determine which metrics to show
    if show_category:
        metrics_to_show = get_metrics_by_category(show_category)
        lines.append(f"**Category: {CATEGORIES.get(show_category, show_category)}**\n")
    else:
        metrics_to_show = ALL_METRICS
    
    for metric in metrics_to_show:
        col = metric.column
        if col not in all_solutions_df.columns:
            continue
        
        bounds = filter_state.bounds.get(metric.display_name, FilterBounds())
        
        all_min = all_solutions_df[col].min()
        all_max = all_solutions_df[col].max()
        
        if len(filtered_df) > 0 and col in filtered_df.columns:
            filt_min = filtered_df[col].min()
            filt_max = filtered_df[col].max()
            range_str = f"current range: {filt_min:.4f} - {filt_max:.4f}"
        else:
            range_str = "no feasible solutions"
        
        # Build constraint string
        bound_strs = []
        if bounds.max_bound is not None:
            bound_strs.append(f"max ≤ {bounds.max_bound:.4f}")
        if bounds.min_bound is not None:
            bound_strs.append(f"min ≥ {bounds.min_bound:.4f}")
        
        constraint_str = ", ".join(bound_strs) if bound_strs else "no constraint"
        if metric.direction == "minimize":
            direction = "lower is better"
        elif metric.direction == "maximize":
            direction = "higher is better"
        else:
            direction = "informational"
        
        lines.append(f"• **{metric.display_name}** ({direction})")
        lines.append(f"    All solutions range: {all_min:.4f} - {all_max:.4f}")
        lines.append(f"    Constraint: {constraint_str}")
        lines.append(f"    {range_str}")
    
    return "\n".join(lines)


def adjust_filter_bound(
    all_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    filter_state: FilterState,
    metric_name: str,
    direction: str,
    pct: float = 0.10,
    current_value: Optional[float] = None,
) -> Optional[float]:
    """
    Calculate a new filter bound by moving it a percentage of the remaining range.

    For tightening, the reference point is the current solution's value (not the
    filter bound). The bound moves pct of the gap between that value and the
    best reachable value. This produces meaningful improvements even with very
    few solutions remaining.

    Args:
        all_df: All Pareto solutions (used as reference for loosening)
        filtered_df: Currently filtered solutions
        filter_state: Current filter state
        metric_name: Display name of metric
        direction: "tighten" or "loosen"
        pct: Fraction of the remaining range to move (0.10 = 10%)
        current_value: The current solution's value for this metric (used for
            tightening). If None, falls back to the filtered set's worst value.

    Returns:
        New bound value, or None if the metric is already unconstrained (loosen only).
    """
    if metric_name not in METRIC_BY_NAME:
        raise ValueError(f"Unknown metric: {metric_name}")

    metric = METRIC_BY_NAME[metric_name]
    if metric.direction is None:
        raise ValueError(f"Cannot filter informational metric: {metric_name}")
    col = metric.column
    bounds = filter_state.bounds.get(metric_name, FilterBounds())

    if direction == "tighten":
        if len(filtered_df) == 0:
            raise ValueError("No solutions to tighten")

        if metric.direction == "minimize":
            best = filtered_df[col].min()
            ref = current_value if current_value is not None else filtered_df[col].max()
            new_bound = ref - pct * (ref - best)
        else:
            best = filtered_df[col].max()
            ref = current_value if current_value is not None else filtered_df[col].min()
            new_bound = ref + pct * (best - ref)

        return float(new_bound)

    elif direction == "loosen":
        if metric.direction == "minimize":
            if bounds.max_bound is None:
                return None
            global_worst = all_df[col].max()
            new_bound = bounds.max_bound + pct * (global_worst - bounds.max_bound)
            return float(min(new_bound, global_worst))
        else:
            if bounds.min_bound is None:
                return None
            global_worst = all_df[col].min()
            new_bound = bounds.min_bound - pct * (bounds.min_bound - global_worst)
            return float(max(new_bound, global_worst))

    raise ValueError(f"Unknown direction: {direction}")


# --- Old solution-count-based tightening/loosening (replaced by adjust_filter_bound) ---
#
# def calculate_tightening(
#     df: pd.DataFrame,
#     metric_name: str,
#     reduction_factor: float = 0.3
# ) -> tuple[float, int]:
#     """
#     Calculate how much to tighten a filter to reduce solutions by ~reduction_factor.
#     Problem: when few solutions remain, eliminating 30% barely moves the bound.
#     """
#     if metric_name not in METRIC_BY_NAME:
#         raise ValueError(f"Unknown metric: {metric_name}")
#     metric = METRIC_BY_NAME[metric_name]
#     col = metric.column
#     if col not in df.columns:
#         raise ValueError(f"Metric column '{col}' not in DataFrame")
#     if len(df) == 0:
#         raise ValueError("No solutions to filter")
#     target_remaining = int(len(df) * (1 - reduction_factor))
#     target_remaining = max(1, target_remaining)
#     sorted_values = df[col].sort_values()
#     if metric.direction == "minimize":
#         if target_remaining >= len(sorted_values):
#             new_bound = float(sorted_values.iloc[-1])
#         else:
#             new_bound = float(sorted_values.iloc[target_remaining - 1])
#         expected_remaining = (df[col] <= new_bound).sum()
#     else:
#         sorted_desc = sorted_values.iloc[::-1]
#         if target_remaining >= len(sorted_desc):
#             new_bound = float(sorted_desc.iloc[-1])
#         else:
#             new_bound = float(sorted_desc.iloc[target_remaining - 1])
#         expected_remaining = (df[col] >= new_bound).sum()
#     return new_bound, expected_remaining
#
#
# def calculate_loosening(
#     all_df: pd.DataFrame,
#     filter_state: FilterState,
#     metric_name: str,
#     expansion_factor: float = 0.3
# ) -> tuple[Optional[float], int]:
#     """
#     Calculate how much to loosen a filter.
#     Problem: same percentage-of-solutions approach as tightening.
#     """
#     if metric_name not in METRIC_BY_NAME:
#         raise ValueError(f"Unknown metric: {metric_name}")
#     metric = METRIC_BY_NAME[metric_name]
#     col = metric.column
#     bounds = filter_state.bounds.get(metric_name, FilterBounds())
#     if metric.direction == "minimize":
#         if bounds.max_bound is None:
#             return None, len(all_df)
#         global_max = all_df[col].max()
#         current_max = bounds.max_bound
#         new_bound = current_max + (global_max - current_max) * expansion_factor
#         new_bound = min(new_bound, global_max)
#         currently_included = (all_df[col] <= current_max).sum()
#         new_included = (all_df[col] <= new_bound).sum()
#     else:
#         if bounds.min_bound is None:
#             return None, len(all_df)
#         global_min = all_df[col].min()
#         current_min = bounds.min_bound
#         new_bound = current_min - (current_min - global_min) * expansion_factor
#         new_bound = max(new_bound, global_min)
#         currently_included = (all_df[col] >= current_min).sum()
#         new_included = (all_df[col] >= new_bound).sum()
#     return new_bound, new_included - currently_included


def find_relaxation_needed(
    all_df: pd.DataFrame,
    filter_state: FilterState,
    target_solutions: int = 5
) -> dict[str, float]:
    """
    When filters are too tight (0 solutions), find minimal relaxations needed.
    
    Uses binary search to find the minimum relaxation for each metric
    that would restore at least target_solutions.
    
    Returns dict mapping metric display_name to suggested new bound.
    """
    suggestions = {}
    
    for metric in ALL_METRICS:
        if metric.direction is None:
            continue
        col = metric.column
        if col not in all_df.columns:
            continue
        
        bounds = filter_state.bounds.get(metric.display_name, FilterBounds())
        
        if metric.direction == "minimize":
            if bounds.max_bound is None:
                continue
            
            # Binary search for minimal relaxation of max_bound
            global_max = all_df[col].max()
            low, high = bounds.max_bound, global_max
        else:
            if bounds.min_bound is None:
                continue
            
            # Binary search for minimal relaxation of min_bound
            global_min = all_df[col].min()
            low, high = global_min, bounds.min_bound
        
        # Create test state
        test_state = FilterState(bounds={
            n: FilterBounds(
                min_bound=filter_state.bounds[n].min_bound,
                max_bound=filter_state.bounds[n].max_bound
            ) for n in filter_state.bounds
        })
        
        for _ in range(20):  # Binary search iterations
            mid = (low + high) / 2
            
            if metric.direction == "minimize":
                test_state.bounds[metric.display_name].max_bound = mid
            else:
                test_state.bounds[metric.display_name].min_bound = mid
            
            filtered = apply_filters(all_df, test_state)
            
            if len(filtered) >= target_solutions:
                if metric.direction == "minimize":
                    high = mid
                else:
                    low = mid
            else:
                if metric.direction == "minimize":
                    low = mid
                else:
                    high = mid
        
        # Use the converged value
        if metric.direction == "minimize":
            test_state.bounds[metric.display_name].max_bound = high
        else:
            test_state.bounds[metric.display_name].min_bound = low
        
        filtered = apply_filters(all_df, test_state)
        if len(filtered) > 0:
            suggestions[metric.display_name] = high if metric.direction == "minimize" else low
    
    return suggestions
