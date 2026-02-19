"""
Solution Clustering for School Zoning Exploration.

This module handles:
- Vectorizing solutions as metric arrays
- Clustering solutions using HDBSCAN (primary) or K-means (fallback)
- Computing interpretable direction labels for each cluster
- Extracting cluster bounds for filter tightening
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering

from .metrics_config import (
    ALL_METRICS,
    METRIC_BY_COLUMN,
    get_metric_columns,
)
from .pareto import format_solution
from .filters import FilterBounds


# ============================================================================
# CLUSTERING METHOD ABSTRACTION
# ============================================================================


class ClusteringMethod(ABC):
    """Abstract base class for clustering methods."""

    def __init__(self, n_clusters: int = 3):
        """
        Initialize the clustering method.

        Args:
            n_clusters: Desired number of clusters.
        """
        self.n_clusters = n_clusters

    @abstractmethod
    def fit_predict(
        self, vectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cluster the vectors and return labels and centers.

        Args:
            vectors: 2D array of shape (n_samples, n_features), already normalized.

        Returns:
            Tuple of (labels, centers) where:
            - labels: array of cluster assignments (0 to n_clusters-1)
            - centers: array of shape (n_clusters, n_features)
        """
        raise NotImplementedError("Subclasses must implement fit_predict")


class HDBSCANClusterer(ClusteringMethod):
    """HDBSCAN clustering with n largest clusters and noise assignment."""

    def __init__(self, n_clusters: int = 3, min_cluster_size: int = 3):
        """
        Initialize HDBSCAN clusterer.

        Args:
            n_clusters: Number of clusters to keep (takes n largest).
            min_cluster_size: Minimum cluster size for HDBSCAN.
        """
        super().__init__(n_clusters)
        self.min_cluster_size = min_cluster_size

    def fit_predict(
        self, vectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run HDBSCAN, keep n largest clusters, assign noise to nearest.

        Returns:
            Tuple of (labels, centers) with contiguous labels 0 to n-1.
        """
        # Run HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1,
        )
        raw_labels = clusterer.fit_predict(vectors)

        # Count cluster sizes (excluding noise label -1)
        unique_labels = set(raw_labels) - {-1}

        if len(unique_labels) < 2:
            # Not enough clusters found, return None to trigger fallback
            raise ValueError("HDBSCAN found fewer than 2 clusters")

        # Get n largest clusters by size
        cluster_sizes = {
            label: np.sum(raw_labels == label) for label in unique_labels
        }
        sorted_clusters = sorted(
            cluster_sizes.keys(), key=lambda x: cluster_sizes[x], reverse=True
        )
        kept_clusters = sorted_clusters[: self.n_clusters]

        # Compute centers for kept clusters
        kept_centers = np.array(
            [vectors[raw_labels == label].mean(axis=0) for label in kept_clusters]
        )

        # Create label mapping: kept clusters -> 0, 1, 2, ...
        label_map = {old: new for new, old in enumerate(kept_clusters)}

        # Assign new labels; noise and small clusters get -1 temporarily
        new_labels = np.array(
            [label_map.get(label, -1) for label in raw_labels]
        )

        # Assign orphaned points (noise + small clusters) to nearest cluster
        orphan_mask = new_labels == -1
        if orphan_mask.any():
            orphan_vectors = vectors[orphan_mask]
            # Compute distances to each kept cluster center
            distances = np.linalg.norm(
                orphan_vectors[:, np.newaxis, :] - kept_centers[np.newaxis, :, :],
                axis=2,
            )
            nearest_cluster = np.argmin(distances, axis=1)
            new_labels[orphan_mask] = nearest_cluster

        # Recompute final centers with all assigned points
        final_centers = np.array(
            [
                vectors[new_labels == i].mean(axis=0)
                for i in range(len(kept_clusters))
            ]
        )

        return new_labels, final_centers


class KMeansClusterer(ClusteringMethod):
    """K-means clustering (used as fallback)."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Initialize K-means clusterer.

        Args:
            n_clusters: Number of clusters.
            random_state: Random seed for reproducibility.
        """
        super().__init__(n_clusters)
        self.random_state = random_state

    def fit_predict(
        self, vectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run K-means clustering.

        Returns:
            Tuple of (labels, centers).
        """
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(vectors)
        centers = kmeans.cluster_centers_
        return labels, centers


class SpectralClusterer(ClusteringMethod):
    """Spectral clustering using graph-based similarity."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Initialize spectral clusterer.

        Args:
            n_clusters: Number of clusters.
            random_state: Random seed for reproducibility.
        """
        super().__init__(n_clusters)
        self.random_state = random_state

    def fit_predict(
        self, vectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run spectral clustering.

        Returns:
            Tuple of (labels, centers).
        """
        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            affinity="rbf",
            assign_labels="kmeans",
        )
        labels = spectral.fit_predict(vectors)

        # Compute cluster centers (spectral doesn't provide them directly)
        centers = np.array(
            [vectors[labels == i].mean(axis=0) for i in range(self.n_clusters)]
        )
        return labels, centers


# Default clustering method
DEFAULT_CLUSTERING_METHOD = "spectral"


def vectorize_solutions(df: pd.DataFrame) -> np.ndarray:
    """
    Convert solution DataFrame to numpy array of metric values.
    
    Args:
        df: DataFrame with metric columns
        
    Returns:
        2D numpy array of shape (n_solutions, n_metrics)
    """
    # Only use columns that exist in both df and metric config
    metric_cols = [col for col in get_metric_columns() if col in df.columns]
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
    n_clusters: int = 3,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster solutions using HDBSCAN (primary) with K-means fallback.

    Args:
        vectors: 2D array of shape (n_solutions, n_metrics)
        n_clusters: Number of clusters to create (default 3)
        random_state: Random seed for reproducibility (K-means only)

    Returns:
        Tuple of (labels, cluster_centers) where:
        - labels: array of cluster assignments (0 to n_clusters-1)
        - cluster_centers: array of shape (n_clusters, n_metrics)
    """
    # Normalize for clustering
    normalized, min_vals, max_vals = normalize_vectors(vectors)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    # Try the default clustering method first
    try:
        if DEFAULT_CLUSTERING_METHOD == "hdbscan":
            clusterer = HDBSCANClusterer(n_clusters=n_clusters)
        elif DEFAULT_CLUSTERING_METHOD == "spectral":
            clusterer = SpectralClusterer(n_clusters=n_clusters)
        elif DEFAULT_CLUSTERING_METHOD == "kmeans":
            clusterer = KMeansClusterer(n_clusters=n_clusters, random_state=random_state)
        else:
            raise ValueError(f"Unknown clustering method: {DEFAULT_CLUSTERING_METHOD}")
        labels, normalized_centers = clusterer.fit_predict(normalized)
    except ValueError:
        # Fall back to K-means if the default clustering method fails
        clusterer = KMeansClusterer(n_clusters=n_clusters, random_state=random_state)
        labels, normalized_centers = clusterer.fit_predict(normalized)

    # Denormalize centers back to original scale
    centers = normalized_centers * ranges + min_vals

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
            "normalized_direction": np.ndarray
        }
    """
    metric_cols = get_metric_columns()
    
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
        emphasized = []  # Metrics this cluster is BETTER at
        compromised = []  # Metrics this cluster is WORSE at
        
        threshold = 0.1  # 10% of range is significant
        
        for i, col in enumerate(metric_cols):
            if col not in METRIC_BY_COLUMN:
                continue
            
            metric = METRIC_BY_COLUMN[col]
            norm_dir = normalized_direction[i]
            short_name = _get_short_metric_name(metric.display_name)
            
            # For MINIMIZE metrics: negative = better, positive = worse
            # For MAXIMIZE metrics: positive = better, negative = worse
            if metric.direction == "minimize":
                if norm_dir < -threshold:
                    emphasized.append(short_name)
                elif norm_dir > threshold:
                    compromised.append(short_name)
            else:  # maximize
                if norm_dir > threshold:
                    emphasized.append(short_name)
                elif norm_dir < -threshold:
                    compromised.append(short_name)
        
        # Build label
        if emphasized and compromised:
            label = f"Better {', '.join(emphasized[:2])}; accepts worse {', '.join(compromised[:2])}"
        elif emphasized:
            label = f"Optimizes for {', '.join(emphasized[:3])}"
        elif compromised:
            label = f"Accepts worse {', '.join(compromised[:3])}"
        else:
            label = "Balanced trade-offs"
        
        directions[cluster_id] = {
            "direction_vector": direction,
            "normalized_direction": normalized_direction,
            "direction_label": label,
        }
    
    return directions


def _get_short_metric_name(display_name: str) -> str:
    """Convert display names to shorter labels for cluster descriptions."""
    name_map = {
        # Diversity
        "FRL Deviation": "economic diversity",
        "Black Population Deviation": "Black population balance",
        "Hispanic Population Deviation": "Hispanic population balance",
        "White Population Deviation": "White population balance",
        "Asian Population Deviation": "Asian population balance",
        "Seat Disparity": "seat availability",
        # Distance
        "Avg Distance to Closest School": "commute distance",
        "Schools in Attendance Area": "local school access",
        "Boundary Cost (Compactness)": "compactness",
        # Programs
        "Total Programs": "program variety",
        "Language Immersion Programs": "language programs",
        "Special Education Programs": "special ed access",
        "General Education Programs": "GE programs",
        # Quality
        "GreatSchools Rating": "school ratings",
        "Math Scores": "math scores",
        "English Scores": "English scores",
        "Suspension Index": "discipline climate",
    }
    return name_map.get(display_name, display_name)


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
        Dict mapping metric display_name to FilterBounds with min/max set
    """
    # Get solutions in this cluster
    cluster_mask = labels == cluster_id
    cluster_df = df.iloc[np.where(cluster_mask)[0]]
    
    bounds = {}
    for metric in ALL_METRICS:
        col = metric.column
        if col not in cluster_df.columns:
            continue
        
        bounds[metric.display_name] = FilterBounds(
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
    lines.append("To explore a specific cluster, please tell me which cluster number you'd like (1-{}).".format(n_clusters))
    
    return "\n".join(lines)
