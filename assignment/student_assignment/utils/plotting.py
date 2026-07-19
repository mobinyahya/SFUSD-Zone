"""Centralized matplotlib/seaborn plotting configuration for professional plots.

This module provides a unified styling configuration for all plots in the project,
ensuring consistent, publication-quality visualizations.

Usage:
    from student_assignment.utils.plotting import apply_plot_style, save_figure

    apply_plot_style()  # Call once before creating any plots
    # ... create your plot ...
    save_figure(output_path)  # Saves with consistent settings
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Centralized rcParams Configuration
# =============================================================================

PLOT_RC_PARAMS = {
    # Font settings
    "font.family": "sans-serif",
    "font.sans-serif": [
        "TeX Gyre Heros",
        "Nimbus Sans",
        "Latin Modern Sans",
        "DejaVu Sans",
    ],
    "font.size": 11,
    # Title and label sizes
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "medium",
    # Tick sizes
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    # Legend
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    # Figure settings
    "figure.autolayout": True,
    "figure.dpi": 100,
    "figure.figsize": (10, 6),
    "figure.facecolor": "white",
    # Save settings
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    # Axes styling
    "axes.facecolor": "white",
    "axes.edgecolor": "0.2",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Grid styling
    "grid.color": "0.9",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    # Lines and markers
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "lines.markeredgewidth": 1.0,
    # Error bars
    "errorbar.capsize": 5,
}

# Color palette with 20+ distinct colors for multi-label charts
# Combines tab20 and additional colors for maximum distinguishability
DEFAULT_PALETTE_NAME = "tab20"
EXTENDED_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#c49c94",  # light brown
    "#f7b6d2",  # light pink
    "#c7c7c7",  # light gray
    "#dbdb8d",  # light olive
    "#9edae5",  # light cyan
    "#393b79",  # dark blue
    "#637939",  # dark olive
    "#8c6d31",  # dark tan
    "#843c39",  # dark red
]


def apply_plot_style(seaborn_style: str = "whitegrid") -> None:
    """Apply the centralized plotting style to matplotlib and seaborn.

    This function should be called once at the start of any script that
    produces plots. It sets up both seaborn's theme and matplotlib's rcParams.

    Args:
        seaborn_style: The seaborn style to use. Defaults to "whitegrid".
            Options: "whitegrid", "darkgrid", "white", "dark", "ticks".
    """
    # Apply seaborn theme first
    sns.set_theme(style=seaborn_style)

    # Override with our custom rcParams
    plt.rcParams.update(PLOT_RC_PARAMS)


def save_figure(
    filepath: str | Path,
    fig: plt.Figure | None = None,
    dpi: int = 150,
    close: bool = True,
) -> None:
    """Save a figure with consistent, high-quality settings.

    Args:
        filepath: Path where the figure will be saved.
        fig: The matplotlib figure to save. If None, uses current figure.
        dpi: Resolution in dots per inch. Defaults to 150.
        close: Whether to close the figure after saving. Defaults to True.
    """
    if fig is None:
        fig = plt.gcf()

    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")

    if close:
        plt.close(fig)


def get_color_palette(n_colors: int = 20) -> list[str]:
    """Get a color palette with support for many distinct labels.

    Returns a list of colors suitable for distinguishing up to 24 categories.
    For fewer than 20 colors, uses seaborn's tab20 palette.
    For more colors, uses an extended custom palette.

    Args:
        n_colors: Number of distinct colors needed. Defaults to 20.

    Returns:
        List of hex color strings.
    """
    if n_colors <= 20:
        return sns.color_palette(DEFAULT_PALETTE_NAME, n_colors).as_hex()
    elif n_colors <= len(EXTENDED_COLORS):
        # For more than 20 colors, use our extended palette
        return EXTENDED_COLORS[:n_colors]
    else:
        # For very large palettes, fall back to husl which handles any count
        return sns.color_palette("husl", n_colors).as_hex()


def get_categorical_palette(n_colors: int = 10) -> list[str]:
    """Get a categorical color palette optimized for readability.

    Uses seaborn's "deep" palette for smaller numbers of categories,
    switching to tab20 for larger sets.

    Args:
        n_colors: Number of distinct colors needed. Defaults to 10.

    Returns:
        List of hex color strings.
    """
    if n_colors <= 10:
        return sns.color_palette("deep", n_colors).as_hex()
    else:
        return get_color_palette(n_colors)
