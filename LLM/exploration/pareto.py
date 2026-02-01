"""
Pareto Frontier Computation for School Zoning Solutions.

This module handles:
- Loading zoning solutions from CSV
- Computing the Pareto frontier
- Normalizing metrics and finding centroid solutions
"""

import pandas as pd
import numpy as np
from typing import Literal
from pathlib import Path


# Metric configuration: maps user-friendly names to CSV columns and optimization direction
# "minimize" means lower is better, "maximize" means higher is better
METRIC_CONFIG: dict[str, dict[str, str | Literal["minimize", "maximize"]]] = {
    "Free and Reduced Lunch Population % Deviation from district average": {
        "column": "FRL",
        "direction": "minimize",
    },
    "Black Population % Deviation from district average": {
        "column": "Ethnicity_Black_or_African_American",
        "direction": "minimize",
    },
    "Hispanic Population % Deviation from district average": {
        "column": "Ethnicity_Hispanic/Latinx",
        "direction": "minimize",
    },
    "White Population % Deviation from district average": {
        "column": "Ethnicity_White",
        "direction": "minimize",
    },
    "Asian Population % Deviation from district average": {
        "column": "Ethnicity_Asian",
        "direction": "minimize",
    },
    "Total Population % Deviation from district average": {
        "column": "seat_disparity",
        "direction": "minimize",
    },
    "Average distance to closest school": {
        "column": "closest_school_distances",
        "direction": "minimize",
    },
    "Compactness": {
        "column": "boundary_cost",
        "direction": "minimize",  # Lower boundary cost = more compact
    },
}

# Reverse mapping: column name -> user-friendly name
COLUMN_TO_NAME = {v["column"]: k for k, v in METRIC_CONFIG.items()}


def get_metric_columns() -> list[str]:
    """Get list of CSV column names for all metrics."""
    return [config["column"] for config in METRIC_CONFIG.values()]


def load_solutions(csv_path: str | Path) -> pd.DataFrame:
    """
    Load zoning solutions from CSV file.
    
    Returns DataFrame with only the metric columns plus the 'path' column for identification.
    """
    df = pd.read_csv(csv_path)
    
    # Keep only metric columns and the path for identification
    metric_cols = get_metric_columns()
    cols_to_keep = metric_cols + ["path"]
    
    return df[cols_to_keep].copy()


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all metrics to [0, 1] range.
    
    For "minimize" metrics, 0 = best (lowest), 1 = worst (highest).
    This allows us to take a simple average to find "balanced" solutions.
    """
    normalized = df.copy()
    
    for name, config in METRIC_CONFIG.items():
        col = config["column"]
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val - min_val > 0:
            # Normalize to [0, 1] where 0 is best
            normalized[col] = (df[col] - min_val) / (max_val - min_val)
            
            # If maximizing, invert so 0 is still best
            if config["direction"] == "maximize":
                normalized[col] = 1 - normalized[col]
        else:
            normalized[col] = 0  # All values are the same
    
    return normalized


def dominates(row_a: pd.Series, row_b: pd.Series, metric_cols: list[str]) -> bool:
    """
    Check if row_a dominates row_b (row_a is better or equal in all metrics, 
    and strictly better in at least one).
    
    Uses normalized values where lower = better for all metrics.
    """
    better_or_equal = all(row_a[col] <= row_b[col] for col in metric_cols)
    strictly_better = any(row_a[col] < row_b[col] for col in metric_cols)
    return better_or_equal and strictly_better


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pareto frontier of solutions.
    
    A solution is Pareto-optimal if no other solution dominates it
    (i.e., no solution is better in all metrics).
    
    Args:
        df: DataFrame with normalized metrics (lower = better)
        
    Returns:
        DataFrame containing only Pareto-optimal solutions
    """
    metric_cols = get_metric_columns()
    n = len(df)
    is_pareto = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # Check if j dominates i
            if dominates(df.iloc[j], df.iloc[i], metric_cols):
                is_pareto[i] = False
                break
    
    return df[is_pareto].copy()


def get_centroid_solution(
    df: pd.DataFrame,
    normalized_df: pd.DataFrame
) -> tuple[pd.Series, int]:
    """
    Find the solution closest to the centroid of the normalized Pareto frontier.
    
    This represents a "balanced" solution that trades off all metrics equally.
    
    Args:
        df: Original DataFrame with actual metric values
        normalized_df: Normalized DataFrame (lower = better)
        
    Returns:
        Tuple of (solution row from original df, index)
    """
    metric_cols = get_metric_columns()
    
    # Compute centroid of normalized values
    centroid = normalized_df[metric_cols].mean()
    
    # Find solution with minimum Euclidean distance to centroid
    distances = np.sqrt(((normalized_df[metric_cols] - centroid) ** 2).sum(axis=1))
    min_idx = distances.idxmin()
    
    return df.loc[min_idx], min_idx


def get_metric_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Get min, max, mean for each metric in the current solution set.
    
    Returns dict mapping metric name to stats dict.
    """
    stats = {}
    for name, config in METRIC_CONFIG.items():
        col = config["column"]
        stats[name] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "direction": config["direction"],
        }
    return stats


def format_solution(row: pd.Series) -> str:
    """Format a solution row as human-readable string."""
    lines = []
    for name, config in METRIC_CONFIG.items():
        col = config["column"]
        value = row[col]
        direction = "lower is better" if config["direction"] == "minimize" else "higher is better"
        lines.append(f"  • {name}: {value:.4f} ({direction})")
    return "\n".join(lines)
