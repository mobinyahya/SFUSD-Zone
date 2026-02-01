"""
Solution Clustering for School Zoning Exploration.

This module handles:
- Vectorizing solutions as metric arrays
- Clustering solutions using K-means
- Computing interpretable direction labels for each cluster
- Extracting cluster bounds for filter tightening
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .pareto import METRIC_CONFIG, get_metric_columns
from .filters import FilterBounds


def vectorize_solutions(df: pd.DataFrame) -> np.ndarray:
    """
    Convert solution DataFrame to numpy array of metric values.
    
    Args:
        df: DataFrame with metric columns
        
    Returns:
        2D numpy array of shape (n_solutions, n_metrics)
    """
    metric_cols = get_metric_columns()
    return df[metric_cols].values.astype(np.float64)


def normalize_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize vectors to [0, 1] range for clustering.
    
    Returns:
        Tuple of (normalized_vectors, min_values, max_values)
    """
    min_vals = vectors.min(axis=0)
    max_vals = vectors.max(axis=0)
    
    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    
    normalized = (vectors - min_vals) / ranges
    return normalized, min_vals, max_vals


def cluster_solutions(
    vectors: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run K-means clustering on solution vectors.
    
    Args:
        vectors: 2D array of shape (n_solutions, n_metrics)
        n_clusters: Number of clusters to create
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (labels, cluster_centers) where:
        - labels: array of cluster assignments (0 to n_clusters-1)
        - cluster_centers: array of shape (n_clusters, n_metrics)
    """
    # Normalize for clustering
    normalized, min_vals, max_vals = normalize_vectors(vectors)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(normalized)
    
    # Denormalize centers back to original scale
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    centers = kmeans.cluster_centers_ * ranges + min_vals
    
    return labels, centers


def compute_cluster_directions(
    vectors: np.ndarray,
    cluster_centers: np.ndarray
) -> dict[int, dict]:
    """
    Compute interpretable direction labels for each cluster.
    
    The direction is computed as the difference between the cluster center
    and the overall centroid of all solutions. This shows what each cluster
    "emphasizes" relative to the average solution.
    
    Args:
        vectors: All solution vectors
        cluster_centers: Centers of each cluster
        
    Returns:
        Dict mapping cluster_id to {
            "direction_vector": np.ndarray,
            "direction_label": str,
            "normalized_direction": np.ndarray (for comparison)
        }
    """
    metric_cols = get_metric_columns()
    metric_names = list(METRIC_CONFIG.keys())
    
    # Compute overall centroid
    overall_centroid = vectors.mean(axis=0)
    
    # Normalize for direction comparison
    _, min_vals, max_vals = normalize_vectors(vectors)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    
    directions = {}
    
    for cluster_id, center in enumerate(cluster_centers):
        # Direction from overall centroid to cluster center
        direction = center - overall_centroid
        
        # Normalize direction for interpretation
        normalized_direction = direction / ranges
        
        # Generate interpretable label
        label_parts = []
        
        # Find metrics with significant deviations
        # Lower values are better for all metrics, so:
        # - Negative direction = lower than avg = BETTER at this metric
        # - Positive direction = higher than avg = WORSE at this metric
        
        emphasized = []  # Metrics this cluster is BETTER at (lower values)
        compromised = []  # Metrics this cluster is WORSE at (higher values)
        
        threshold = 0.1  # 10% of range is significant
        
        for i, (name, norm_dir) in enumerate(zip(metric_names, normalized_direction)):
            # Get a shortened name for the label
            short_name = _get_short_metric_name(name)
            
            if norm_dir < -threshold:
                emphasized.append(short_name)
            elif norm_dir > threshold:
                compromised.append(short_name)
        
        # Build label
        if emphasized and compromised:
            label = f"Better {', '.join(emphasized[:2])}; accepts higher {', '.join(compromised[:2])}"
        elif emphasized:
            label = f"Optimizes for {', '.join(emphasized[:3])}"
        elif compromised:
            label = f"Allows higher {', '.join(compromised[:3])}"
        else:
            label = "Balanced trade-offs"
        
        directions[cluster_id] = {
            "direction_vector": direction,
            "normalized_direction": normalized_direction,
            "direction_label": label,
        }
    
    return directions


def _get_short_metric_name(full_name: str) -> str:
    """Convert full metric names to shorter labels."""
    name_map = {
        "Free and Reduced Lunch Population % Deviation from district average": "economic diversity",
        "Black Population % Deviation from district average": "Black population balance",
        "Hispanic Population % Deviation from district average": "Hispanic population balance",
        "White Population % Deviation from district average": "White population balance",
        "Asian Population % Deviation from district average": "Asian population balance",
        "Total Population % Deviation from district average": "seat availability",
        "Average distance to closest school": "commute distance",
        "Compactness": "compactness",
    }
    return name_map.get(full_name, full_name)


def get_representative_solution(
    df: pd.DataFrame,
    vectors: np.ndarray,
    cluster_centers: np.ndarray,
    labels: np.ndarray,
    cluster_id: int
) -> tuple[pd.Series, int]:
    """
    Find the solution closest to a cluster's center.
    
    Args:
        df: Original DataFrame with solutions
        vectors: Vectorized solutions
        cluster_centers: Cluster centers from K-means
        labels: Cluster assignments
        cluster_id: Which cluster to get representative for
        
    Returns:
        Tuple of (solution_row, index_in_df)
    """
    # Get indices of solutions in this cluster
    cluster_mask = labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    cluster_vectors = vectors[cluster_mask]
    
    # Find closest to cluster center
    center = cluster_centers[cluster_id]
    distances = np.linalg.norm(cluster_vectors - center, axis=1)
    closest_idx_in_cluster = np.argmin(distances)
    
    # Map back to original indices
    original_idx = cluster_indices[closest_idx_in_cluster]
    df_idx = df.index[original_idx]
    
    return df.loc[df_idx], df_idx


def get_cluster_bounds(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int
) -> dict[str, FilterBounds]:
    """
    Calculate min/max bounds for each metric within a cluster.
    
    These bounds can be used to tighten filters to only include
    solutions similar to the selected cluster.
    
    Args:
        df: DataFrame with solutions
        labels: Cluster assignments
        cluster_id: Which cluster to get bounds for
        
    Returns:
        Dict mapping metric_name to FilterBounds with min/max set
    """
    metric_cols = get_metric_columns()
    metric_names = list(METRIC_CONFIG.keys())
    
    # Get solutions in this cluster
    cluster_mask = labels == cluster_id
    cluster_df = df.iloc[np.where(cluster_mask)[0]]
    
    bounds = {}
    for name, col in zip(metric_names, metric_cols):
        bounds[name] = FilterBounds(
            min_bound=float(cluster_df[col].min()),
            max_bound=float(cluster_df[col].max())
        )
    
    return bounds


def format_cluster_summary(
    df: pd.DataFrame,
    vectors: np.ndarray,
    labels: np.ndarray,
    cluster_centers: np.ndarray,
    directions: dict[int, dict]
) -> str:
    """
    Format a human-readable summary of all clusters.
    
    Shows each cluster's size, direction label, and representative solution.
    """
    from .pareto import format_solution
    
    n_clusters = len(cluster_centers)
    lines = [f"**Solution Clusters** ({n_clusters} groups found)\n"]
    lines.append("Each cluster represents a different approach to balancing trade-offs:\n")
    
    for cluster_id in range(n_clusters):
        cluster_size = (labels == cluster_id).sum()
        direction_info = directions[cluster_id]
        
        lines.append(f"---\n")
        lines.append(f"### Cluster {cluster_id + 1}: {direction_info['direction_label']}")
        lines.append(f"*({cluster_size} solutions)*\n")
        
        # Get representative solution
        rep_solution, _ = get_representative_solution(
            df, vectors, cluster_centers, labels, cluster_id
        )
        
        lines.append("**Representative Solution:**")
        lines.append(format_solution(rep_solution))
        lines.append("")
    
    lines.append("---\n")
    lines.append("Use `select_cluster` with the cluster number (1-{}) to narrow down to that group of solutions.".format(n_clusters))
    
    return "\n".join(lines)
