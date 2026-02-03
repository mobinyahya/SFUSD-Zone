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
        # Initialize bounds for all metrics if not provided
        for metric in ALL_METRICS:
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
        if metric.column not in df.columns:
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
        direction = "↓ lower is better" if metric.direction == "minimize" else "↑ higher is better"
        
        lines.append(f"• **{metric.display_name}** ({direction})")
        lines.append(f"    All solutions range: {all_min:.4f} - {all_max:.4f}")
        lines.append(f"    Constraint: {constraint_str}")
        lines.append(f"    {range_str}")
    
    return "\n".join(lines)


def calculate_tightening(
    df: pd.DataFrame,
    metric_name: str,
    reduction_factor: float = 0.3
) -> tuple[float, int]:
    """
    Calculate how much to tighten a filter to reduce solutions by ~reduction_factor.
    
    For "minimize" metrics, we lower the max_bound.
    For "maximize" metrics, we raise the min_bound.
    
    Args:
        df: Current filtered DataFrame
        metric_name: Display name of metric to tighten
        reduction_factor: Target fraction of solutions to eliminate (0.3 = 30%)
        
    Returns:
        Tuple of (new_bound, expected_remaining_solutions)
    """
    if metric_name not in METRIC_BY_NAME:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    metric = METRIC_BY_NAME[metric_name]
    col = metric.column
    
    if col not in df.columns:
        raise ValueError(f"Metric column '{col}' not in DataFrame")
    
    if len(df) == 0:
        raise ValueError("No solutions to filter")
    
    # Calculate target number of remaining solutions
    target_remaining = int(len(df) * (1 - reduction_factor))
    target_remaining = max(1, target_remaining)  # Keep at least 1
    
    sorted_values = df[col].sort_values()
    
    if metric.direction == "minimize":
        # Tighten by lowering max bound (keep the lowest values)
        if target_remaining >= len(sorted_values):
            new_bound = float(sorted_values.iloc[-1])
        else:
            new_bound = float(sorted_values.iloc[target_remaining - 1])
        expected_remaining = (df[col] <= new_bound).sum()
    else:
        # Maximize: tighten by raising min bound (keep the highest values)
        sorted_desc = sorted_values.iloc[::-1]
        if target_remaining >= len(sorted_desc):
            new_bound = float(sorted_desc.iloc[-1])
        else:
            new_bound = float(sorted_desc.iloc[target_remaining - 1])
        expected_remaining = (df[col] >= new_bound).sum()
    
    return new_bound, expected_remaining


def calculate_loosening(
    all_df: pd.DataFrame,
    filter_state: FilterState,
    metric_name: str,
    expansion_factor: float = 0.3
) -> tuple[Optional[float], int]:
    """
    Calculate how much to loosen a filter.
    
    For "minimize" metrics, we raise the max_bound.
    For "maximize" metrics, we lower the min_bound.
    
    Args:
        all_df: All solutions (unfiltered)
        filter_state: Current filter state
        metric_name: Display name of metric to loosen
        expansion_factor: How much to expand (0.3 = 30% toward global extreme)
        
    Returns:
        Tuple of (new_bound or None to remove, expected_new_solutions)
    """
    if metric_name not in METRIC_BY_NAME:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    metric = METRIC_BY_NAME[metric_name]
    col = metric.column
    bounds = filter_state.bounds.get(metric_name, FilterBounds())
    
    if metric.direction == "minimize":
        # Loosen by raising max_bound
        if bounds.max_bound is None:
            return None, len(all_df)  # Already unconstrained
        
        global_max = all_df[col].max()
        current_max = bounds.max_bound
        
        new_bound = current_max + (global_max - current_max) * expansion_factor
        new_bound = min(new_bound, global_max)
        
        currently_included = (all_df[col] <= current_max).sum()
        new_included = (all_df[col] <= new_bound).sum()
    else:
        # Maximize: loosen by lowering min_bound
        if bounds.min_bound is None:
            return None, len(all_df)  # Already unconstrained
        
        global_min = all_df[col].min()
        current_min = bounds.min_bound
        
        new_bound = current_min - (current_min - global_min) * expansion_factor
        new_bound = max(new_bound, global_min)
        
        currently_included = (all_df[col] >= current_min).sum()
        new_included = (all_df[col] >= new_bound).sum()
    
    return new_bound, new_included - currently_included


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
