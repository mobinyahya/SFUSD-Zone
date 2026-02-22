"""
Pareto Frontier Computation for School Zoning Solutions.

This module handles:
- Loading zoning solutions from CSV
- Computing the Pareto frontier
- Normalizing metrics and finding centroid solutions
"""

import pandas as pd
import numpy as np
from pathlib import Path

from .metrics_config import (
    ALL_METRICS,
    CORE_METRICS,
    METRIC_BY_COLUMN,
    get_metric_columns,
    get_core_metric_columns,
    MetricSpec,
)


# ============================================================================
# SOLUTION LOADING
# ============================================================================

def load_solutions(
    csv_path: str | Path,
    use_core_only: bool = False
) -> pd.DataFrame:
    """
    Load zoning solutions from CSV file.
    
    Args:
        csv_path: Path to CSV with zoning solutions
        use_core_only: If True, only load core metrics (faster for basic usage)
    
    Returns:
        DataFrame with metric columns plus 'path' for identification.
    """
    df = pd.read_csv(csv_path)
    
    # Determine which columns to keep
    if use_core_only:
        metric_cols = get_core_metric_columns()
    else:
        metric_cols = get_metric_columns()
    
    # Filter to columns that exist in the CSV
    available_cols = [c for c in metric_cols if c in df.columns]
    missing_cols = set(metric_cols) - set(available_cols)
    
    if missing_cols:
        print(f"Note: {len(missing_cols)} metrics not in CSV: {list(missing_cols)[:5]}...")
    
    cols_to_keep = available_cols + ["path"]
    
    return df[cols_to_keep].copy()


def get_available_metrics(df: pd.DataFrame) -> list[MetricSpec]:
    """Get list of MetricSpecs for columns present in the DataFrame."""
    return [
        METRIC_BY_COLUMN[col] 
        for col in df.columns 
        if col in METRIC_BY_COLUMN
    ]


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all metrics to [0, 1] range where 0 = best.
    
    For "minimize" metrics, 0 = lowest value (best).
    For "maximize" metrics, 0 = highest value (best), inverted.
    """
    normalized = df.copy()
    metric_cols = [c for c in df.columns if c in METRIC_BY_COLUMN]
    
    for col in metric_cols:
        metric = METRIC_BY_COLUMN[col]
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val - min_val > 0:
            # Normalize to [0, 1]
            col_vals = (df[col] - min_val) / (max_val - min_val)
            
            # For maximize metrics, invert so 0 is still best
            if metric.direction == "maximize":
                col_vals = 1 - col_vals
            normalized[col] = col_vals
        else:
            normalized[col] = 0.0  # All values are the same
    
    return normalized


# ============================================================================
# PARETO FRONTIER
# ============================================================================

def dominates(row_a: pd.Series, row_b: pd.Series, metric_cols: list[str]) -> bool:
    """
    Check if row_a dominates row_b.
    
    Uses normalized values where lower = better for all metrics.
    row_a dominates row_b if it's better or equal in all metrics,
    and strictly better in at least one.
    """
    better_or_equal = all(row_a[col] <= row_b[col] for col in metric_cols)
    strictly_better = any(row_a[col] < row_b[col] for col in metric_cols)
    return better_or_equal and strictly_better


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pareto frontier of solutions using a vectorized NumPy approach.
    
    A solution is Pareto-optimal if no other solution dominates it.
    Performance: O(N^2) worst case, but heavily pruned via vectorization.
    
    Args:
        df: DataFrame with normalized metrics (lower = better)
        
    Returns:
        DataFrame containing only Pareto-optimal solutions
    """
    metric_cols = [c for c in df.columns if c in METRIC_BY_COLUMN]
    if not metric_cols:
        return df.copy()
        
    # Extract values as NumPy array for speed
    costs = df[metric_cols].values
    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if not is_efficient[i]:
            continue
            
        # A point 'c' dominates others if it's better or equal in all, 
        # and strictly better in at least one.
        # But for pruning, we just need to find points that are BETTER than the current one.
        # Here we use the property: if any point j is better than i, i is not Pareto.
        
        # Vectorized check: which points are better than or equal to costs[i]?
        # (Where better means <= since we normalized to lower-is-better)
        current_cost = costs[i]
        
        # Points that are better than or equal to current in ALL dimensions
        better_or_equal = np.all(costs <= current_cost, axis=1)
        
        # Points that are strictly better in at least ONE dimension
        strictly_better = np.any(costs < current_cost, axis=1)
        
        # Points that dominate the current point i
        dominators = better_or_equal & strictly_better
        
        if np.any(dominators):
            is_efficient[i] = False
        else:
            # If current point i is NOT dominated, it might dominate others.
            # We can prune points that are dominated by current point i.
            
            # Points current_cost dominates: current_cost is <= in all, < in one
            better_than_others = np.all(current_cost <= costs, axis=1)
            strictly_better_than_others = np.any(current_cost < costs, axis=1)
            
            dominated_by_i = better_than_others & strictly_better_than_others
            is_efficient[dominated_by_i] = False
            
    return df[is_efficient].copy()


# ============================================================================
# CENTROID / BALANCED SOLUTION
# ============================================================================

def get_centroid_solution(
    df: pd.DataFrame,
    normalized_df: pd.DataFrame
) -> tuple[pd.Series, int]:
    """
    Find the solution closest to the centroid of the normalized space.
    
    This represents a "balanced" solution trading off all metrics equally.
    
    Args:
        df: Original DataFrame with actual metric values
        normalized_df: Normalized DataFrame (lower = better)
        
    Returns:
        Tuple of (solution row from original df, index)
    """
    metric_cols = [c for c in normalized_df.columns if c in METRIC_BY_COLUMN]
    
    # Compute centroid of normalized values
    centroid = normalized_df[metric_cols].mean()
    
    # Find solution with minimum Euclidean distance to centroid
    distances = np.sqrt(((normalized_df[metric_cols] - centroid) ** 2).sum(axis=1))
    min_idx = distances.idxmin()
    
    return df.loc[min_idx], min_idx


# ============================================================================
# STATISTICS AND FORMATTING
# ============================================================================

def get_metric_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Get min, max, mean for each metric in the solution set.
    
    Returns dict mapping metric display_name to stats dict.
    """
    stats = {}
    for col in df.columns:
        if col not in METRIC_BY_COLUMN:
            continue
        
        metric = METRIC_BY_COLUMN[col]
        stats[metric.display_name] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "direction": metric.direction,
            "column": col,
        }
    return stats


def format_solution(row: pd.Series, show_all: bool = False) -> str:
    """
    Format a solution row as human-readable string.
    
    Args:
        row: Solution row from DataFrame
        show_all: If True, show all metrics; otherwise only core metrics
    """
    lines = []
    metrics_to_show = ALL_METRICS if show_all else CORE_METRICS
    
    for metric in metrics_to_show:
        if metric.column not in row.index:
            continue
        
        value = row[metric.column]
        direction = "lower is better" if metric.direction == "minimize" else "higher is better"
        lines.append(f"  • {metric.display_name}: {value:.4f} ({direction})")
    
    return "\n".join(lines)
