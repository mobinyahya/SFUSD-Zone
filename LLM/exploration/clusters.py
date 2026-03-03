"""
Solution Clustering for School Zoning Exploration.

This module handles:
- Vectorizing solutions as metric arrays
- Themed (category-based) clustering for initial display
- Extracting cluster bounds for filter tightening
"""

import numpy as np
import pandas as pd

from .metrics_config import (
    ALL_METRICS,
    METRIC_BY_COLUMN,
    get_metric_columns,
    get_metrics_by_category,
)
from .filters import FilterBounds


def vectorize_solutions(df: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    """
    Convert solution DataFrame to numpy array of metric values.
    
    Args:
        df: DataFrame with metric columns
        columns: Optional subset of metric column names to use.
                 If None, uses all metric columns.
        
    Returns:
        2D numpy array of shape (n_solutions, n_metrics)
    """
    all_cols = columns if columns else get_metric_columns()
    metric_cols = [col for col in all_cols if col in df.columns]
    return df[metric_cols].values.astype(np.float64)


CATEGORY_THEME = {
    "diversity": "Equity",
    "proximity": "Proximity",
    "programs": "Programs",
    "quality": "Quality",
}

def get_cluster_bounds(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    columns: list[str] | None = None,
) -> dict[str, FilterBounds]:
    """
    Calculate min/max bounds for each metric within a cluster.
    
    Args:
        df: DataFrame with solutions
        labels: Cluster assignments
        cluster_id: Which cluster to get bounds for
        columns: Optional subset of metric column names to restrict bounds to.
                 If None, computes bounds for all metrics.
        
    Returns:
        Dict mapping metric display_name to FilterBounds with min/max set
    """
    cluster_mask = labels == cluster_id
    cluster_df = df.iloc[np.where(cluster_mask)[0]]
    
    col_set = set(columns) if columns else None
    bounds = {}
    for metric in ALL_METRICS:
        col = metric.column
        if col not in cluster_df.columns:
            continue
        if col_set is not None and col not in col_set:
            continue
        
        bounds[metric.display_name] = FilterBounds(
            min_bound=float(cluster_df[col].min()),
            max_bound=float(cluster_df[col].max())
        )
    
    return bounds


# ============================================================================
# THEMED CLUSTERING (category-based assignment)
# ============================================================================

# Theme definitions: label -> list of metric category keys
CLUSTER_THEMES = {
    "Diversity & Equity": ["diversity"],
    "Proximity": ["proximity"],
    "School Performance": ["quality"],
}


def themed_cluster_solutions(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict], list[str] | None]:
    """
    Assign solutions to predefined thematic clusters based on category scores.

    Each solution is assigned to the theme (diversity, distance, school performance)
    where it performs best relative to other solutions.

    Args:
        df: DataFrame with metric columns.

    Returns:
        Tuple of (labels, centers, directions, columns) where:
        - labels: array of cluster assignments (0 to n_themes-1)
        - centers: array of shape (n_themes, n_metrics) in original scale
        - directions: dict of direction info per cluster
        - columns: None (uses all metric columns)
    """
    theme_names = list(CLUSTER_THEMES.keys())
    n_themes = len(theme_names)

    # Compute a composite percentile score per theme for each solution
    # Higher score = better performance in that theme
    theme_scores = np.zeros((len(df), n_themes))

    for theme_idx, (theme_label, categories) in enumerate(CLUSTER_THEMES.items()):
        theme_cols = []
        for cat in categories:
            theme_cols.extend(
                m.column for m in get_metrics_by_category(cat)
                if m.column in df.columns and m.direction is not None
            )

        if not theme_cols:
            continue

        for col in theme_cols:
            metric = METRIC_BY_COLUMN[col]
            values = df[col].values
            # Rank: higher rank = better for that metric
            from scipy.stats import rankdata
            ranks = rankdata(values, method="average")
            if metric.direction == "minimize":
                ranks = len(values) + 1 - ranks  # Invert: lower raw value = higher rank
            # Normalize ranks to [0, 1]
            normalized_ranks = (ranks - 1) / max(len(values) - 1, 1)
            theme_scores[:, theme_idx] += normalized_ranks

        # Average across metrics in this theme
        if theme_cols:
            theme_scores[:, theme_idx] /= len(theme_cols)

    # Assign each solution to the theme where it scores highest
    labels = np.argmax(theme_scores, axis=1)

    # Handle edge case: if a theme has no solutions, reassign from largest cluster
    for theme_idx in range(n_themes):
        if np.sum(labels == theme_idx) == 0:
            # Find solutions in the largest cluster
            largest = np.argmax([np.sum(labels == i) for i in range(n_themes)])
            largest_mask = np.where(labels == largest)[0]
            # Move the solution with highest score for the empty theme
            best_for_empty = largest_mask[np.argmax(theme_scores[largest_mask, theme_idx])]
            labels[best_for_empty] = theme_idx

    # Vectorize all solutions for center computation
    vectors = vectorize_solutions(df)
    centers = np.array([
        vectors[labels == i].mean(axis=0) for i in range(n_themes)
    ])

    # Build direction info with predefined labels
    directions = {}
    for theme_idx, theme_label in enumerate(theme_names):
        categories = CLUSTER_THEMES[theme_label]
        # Strengths are the theme's own categories
        strengths = []
        for cat in categories:
            cat_theme = CATEGORY_THEME.get(cat, cat.capitalize())
            strengths.append(cat_theme)

        # Weaknesses: find which other themes this cluster is weakest on
        weaknesses = []
        for other_idx, other_label in enumerate(theme_names):
            if other_idx == theme_idx:
                continue
            other_cats = CLUSTER_THEMES[other_label]
            for cat in other_cats:
                cat_theme = CATEGORY_THEME.get(cat, cat.capitalize())
                weaknesses.append(cat_theme)

        directions[theme_idx] = {
            "direction_label": theme_label,
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3],
        }

    return labels, centers, directions, None


def format_cluster_summary(
    df: pd.DataFrame,
    vectors: np.ndarray,
    labels: np.ndarray,
    cluster_centers: np.ndarray,
    directions: dict[int, dict]
) -> str:
    """Format a compact, scannable summary of all clusters."""
    n_clusters = len(cluster_centers)
    total = len(df)
    lines = [f"**{n_clusters} groups** found across {total} solutions:\n"]

    for cluster_id in range(n_clusters):
        cluster_size = (labels == cluster_id).sum()
        info = directions[cluster_id]
        strengths = info.get("strengths", [])
        weaknesses = info.get("weaknesses", [])

        lines.append(f"**{cluster_id + 1}. {info['direction_label']}** ({cluster_size} solutions)")
        if strengths:
            lines.append(f"  Stronger: {', '.join(strengths[:2])}")
        if weaknesses:
            lines.append(f"  Weaker: {', '.join(weaknesses[:2])}")
        lines.append("")

    lines.append(f"Which group would you like to explore? (1-{n_clusters})")

    return "\n".join(lines)
