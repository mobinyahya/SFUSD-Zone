"""Solution clustering for school zoning exploration."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .metrics_config import (
    ALL_METRICS,
    METRIC_BY_COLUMN,
    get_metric_columns,
    get_metrics_by_category,
)
from .filters import FilterBounds

CATEGORY_THEME = {
    "diversity": "Equity",
    "proximity": "Proximity",
    "programs": "Programs",
    "quality": "Quality",
}

CLUSTER_THEMES = {
    "Balanced Approach": ["diversity", "proximity", "quality", "programs"],
    "Diversity": ["diversity"],
    "Proximity": ["proximity"],
}


def vectorize_solutions(df: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    """Convert solution DataFrame to numpy array of metric values."""
    all_cols = columns if columns else get_metric_columns()
    metric_cols = [col for col in all_cols if col in df.columns]
    return df[metric_cols].values.astype(np.float64)


def get_cluster_bounds(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
) -> dict[str, FilterBounds]:
    """Calculate min/max bounds for each metric within a cluster."""
    cluster_df = df.iloc[np.where(labels == cluster_id)[0]]

    bounds = {}
    for metric in ALL_METRICS:
        col = metric.column
        if col not in cluster_df.columns:
            continue
        bounds[metric.display_name] = FilterBounds(
            min_bound=float(cluster_df[col].min()),
            max_bound=float(cluster_df[col].max()),
        )
    return bounds


def themed_cluster_solutions(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict]]:
    """Assign solutions to predefined thematic clusters based on category scores.

    Each solution is assigned to the theme where it performs best relative to
    other solutions (by normalized rank across the theme's metrics).

    Returns (labels, centers, directions).
    """
    theme_names = list(CLUSTER_THEMES.keys())
    n_themes = len(theme_names)

    theme_scores = np.zeros((len(df), n_themes))

    for theme_idx, categories in enumerate(CLUSTER_THEMES.values()):
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
            ranks = rankdata(values, method="average")
            if metric.direction == "minimize":
                ranks = len(values) + 1 - ranks
            theme_scores[:, theme_idx] += (ranks - 1) / max(len(values) - 1, 1)

        theme_scores[:, theme_idx] /= len(theme_cols)

    labels = np.argmax(theme_scores, axis=1)

    # If a theme has no solutions, steal one from the largest cluster
    for theme_idx in range(n_themes):
        if np.sum(labels == theme_idx) == 0:
            largest = np.argmax([np.sum(labels == i) for i in range(n_themes)])
            largest_mask = np.where(labels == largest)[0]
            best_for_empty = largest_mask[np.argmax(theme_scores[largest_mask, theme_idx])]
            labels[best_for_empty] = theme_idx

    vectors = vectorize_solutions(df)
    centers = np.array([
        vectors[labels == i].mean(axis=0) for i in range(n_themes)
    ])

    directions = {}
    for theme_idx, theme_label in enumerate(theme_names):
        cats = CLUSTER_THEMES[theme_label]
        strengths = [CATEGORY_THEME.get(c, c.capitalize()) for c in cats]
        weaknesses = [
            CATEGORY_THEME.get(c, c.capitalize())
            for other_idx, other_label in enumerate(theme_names) if other_idx != theme_idx
            for c in CLUSTER_THEMES[other_label]
        ]
        directions[theme_idx] = {
            "direction_label": theme_label,
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3],
        }

    return labels, centers, directions


def format_cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_centers: np.ndarray,
    directions: dict[int, dict],
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
