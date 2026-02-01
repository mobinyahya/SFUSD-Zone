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

from .pareto import METRIC_CONFIG


@dataclass
class FilterBounds:
    """Bounds for a single metric filter."""
    min_bound: Optional[float] = None  # None means no lower bound
    max_bound: Optional[float] = None  # None means no upper bound


@dataclass
class FilterState:
    """
    Complete filter state for all metrics.
    
    Since all our metrics are "minimize" (lower = better), we mainly use max_bound
    to constrain solutions (require metric <= max_bound).
    """
    bounds: dict[str, FilterBounds] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize bounds for all metrics if not provided
        for name in METRIC_CONFIG:
            if name not in self.bounds:
                self.bounds[name] = FilterBounds()


def apply_filters(df: pd.DataFrame, filter_state: FilterState) -> pd.DataFrame:
    """
    Apply current filters to the solution set.
    
    Returns filtered DataFrame containing only solutions within all bounds.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    
    for name, config in METRIC_CONFIG.items():
        col = config["column"]
        bounds = filter_state.bounds.get(name, FilterBounds())
        
        if bounds.min_bound is not None:
            mask &= df[col] >= bounds.min_bound
        if bounds.max_bound is not None:
            mask &= df[col] <= bounds.max_bound
    
    return df[mask].copy()


def get_filter_summary(
    filter_state: FilterState, 
    all_solutions_df: pd.DataFrame,
    filtered_df: pd.DataFrame
) -> str:
    """
    Get a human-readable summary of current filter state.
    
    Shows:
    - Current bounds for each metric
    - Range of values in all solutions vs filtered solutions
    - Number of feasible solutions
    """
    lines = [f"**Current Filters** ({len(filtered_df)} solutions remaining out of {len(all_solutions_df)} total)\n"]
    
    for name, config in METRIC_CONFIG.items():
        col = config["column"]
        bounds = filter_state.bounds.get(name, FilterBounds())
        
        all_min = all_solutions_df[col].min()
        all_max = all_solutions_df[col].max()
        
        if len(filtered_df) > 0:
            filt_min = filtered_df[col].min()
            filt_max = filtered_df[col].max()
            range_str = f"current range: {filt_min:.4f} - {filt_max:.4f}"
        else:
            range_str = "no feasible solutions"
        
        bound_str = []
        if bounds.max_bound is not None:
            bound_str.append(f"max ≤ {bounds.max_bound:.4f}")
        if bounds.min_bound is not None:
            bound_str.append(f"min ≥ {bounds.min_bound:.4f}")
        
        constraint_str = ", ".join(bound_str) if bound_str else "no constraint"
        
        lines.append(f"• **{name}**")
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
    
    Args:
        df: Current filtered DataFrame
        filter_state: Current filter state
        metric_name: Metric to tighten
        reduction_factor: Target fraction of solutions to eliminate (0.3 = 30%)
        
    Returns:
        Tuple of (new_bound, expected_remaining_solutions)
    """
    if metric_name not in METRIC_CONFIG:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    config = METRIC_CONFIG[metric_name]
    col = config["column"]
    
    if len(df) == 0:
        raise ValueError("No solutions to filter")
    
    # Calculate target number of remaining solutions
    target_remaining = int(len(df) * (1 - reduction_factor))
    target_remaining = max(1, target_remaining)  # Keep at least 1
    
    # For minimize metrics, we tighten by lowering the max bound
    # Find the value at the target_remaining-th percentile
    sorted_values = df[col].sort_values()
    
    if target_remaining >= len(sorted_values):
        new_bound = float(sorted_values.iloc[-1])
    else:
        new_bound = float(sorted_values.iloc[target_remaining - 1])
    
    # Count actual remaining solutions
    expected_remaining = (df[col] <= new_bound).sum()
    
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
    
    Args:
        all_df: All solutions (unfiltered)
        filter_state: Current filter state
        metric_name: Metric to loosen
        expansion_factor: How much to expand (0.3 = 30% toward global max)
        
    Returns:
        Tuple of (new_bound or None to remove, expected_new_solutions)
    """
    if metric_name not in METRIC_CONFIG:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    config = METRIC_CONFIG[metric_name]
    col = config["column"]
    bounds = filter_state.bounds.get(metric_name, FilterBounds())
    
    if bounds.max_bound is None:
        return None, len(all_df)  # Already unconstrained
    
    global_max = all_df[col].max()
    current_max = bounds.max_bound
    
    # Expand toward global max
    new_bound = current_max + (global_max - current_max) * expansion_factor
    new_bound = min(new_bound, global_max)  # Don't exceed global max
    
    # Count solutions that would be added
    currently_included = (all_df[col] <= current_max).sum()
    new_included = (all_df[col] <= new_bound).sum()
    
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
    
    Returns dict mapping metric name to suggested new max_bound.
    """
    suggestions = {}
    
    for name, config in METRIC_CONFIG.items():
        col = config["column"]
        bounds = filter_state.bounds.get(name, FilterBounds())
        
        if bounds.max_bound is None:
            continue  # Not constrained
        
        # Binary search for minimal relaxation
        global_max = all_df[col].max()
        low, high = bounds.max_bound, global_max
        
        # Check if relaxing this one metric alone would help
        test_state = FilterState(bounds={
            n: FilterBounds(
                min_bound=filter_state.bounds[n].min_bound,
                max_bound=filter_state.bounds[n].max_bound
            ) for n in filter_state.bounds
        })
        
        for _ in range(20):  # Binary search iterations
            mid = (low + high) / 2
            test_state.bounds[name].max_bound = mid
            filtered = apply_filters(all_df, test_state)
            
            if len(filtered) >= target_solutions:
                high = mid
            else:
                low = mid
        
        # Only suggest if relaxation would help
        test_state.bounds[name].max_bound = high
        filtered = apply_filters(all_df, test_state)
        if len(filtered) > 0:
            suggestions[name] = high
    
    return suggestions
