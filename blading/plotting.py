"""
Plotting utilities for blading module.

This module contains shared plotting functions that provide multi-view layouts
and zoomed views for airfoil analysis and visualization.
"""

import matplotlib.pyplot as plt


def plot_zoomed_view(
    ax,
    plot_functions,
    camber_line,
    zoom_type="LE",
    title=None,
    chord_fraction=0.02,
    bias=0.5,
):
    """
    Plot a zoomed-in view of leading or trailing edge on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    plot_functions : list of callable
        List of functions to call for plotting. Each function should accept an ax parameter.
    camber_line : PlaneCurve
        Camber line used to determine LE and TE positions for close-up views.
    zoom_type : str, optional
        Type of zoom view: "LE" for leading edge or "TE" for trailing edge. Default is "LE".
    title : str, optional
        Title for the zoom view. If None, uses "Leading Edge" or "Trailing Edge".
    chord_fraction : float, optional
        Fraction of the chord length to use as the zoom window size. Default is 0.02.
    bias : float, optional
        Bias factor for positioning zoom windows. Default is 0.5.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object that was plotted on.
    """
    # Get camber line points for determining leading and trailing edges
    camber_coords = camber_line.coords

    if zoom_type.upper() == "LE":
        # Leading edge is at the start of the camber line
        center_x, center_y = camber_coords[0]
        default_title = "Leading Edge"
    elif zoom_type.upper() == "TE":
        # Trailing edge is at the end of the camber line
        center_x, center_y = camber_coords[-1]
        default_title = "Trailing Edge"
    else:
        raise ValueError(f"Invalid zoom_type '{zoom_type}'. Must be 'LE' or 'TE'.")

    # Calculate chord length and zoom window size
    chord_length = camber_line.length()
    zoom_size = chord_length * chord_fraction

    # Calculate bias offsets for directional positioning
    bias_offset = zoom_size * bias

    # Plot all functions on the axis
    for plot_func in plot_functions:
        plot_func(ax)

    # Set title
    ax.set_title(title if title is not None else default_title)
    ax.axis("equal")

    # Set zoom limits based on type
    if zoom_type.upper() == "LE":
        # Bias LE towards lower-left: move left (reduce x) and down (reduce y)
        ax.set_xlim(
            center_x - zoom_size + bias_offset, center_x + zoom_size + bias_offset
        )
        ax.set_ylim(
            center_y - zoom_size + bias_offset, center_y + zoom_size + bias_offset
        )
    else:  # TE
        # Bias TE towards upper-right: move right (increase x) and up (increase y)
        ax.set_xlim(
            center_x - zoom_size - bias_offset, center_x + zoom_size - bias_offset
        )
        ax.set_ylim(
            center_y - zoom_size - bias_offset, center_y + zoom_size - bias_offset
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)

    return ax


def create_multi_view_plot(
    plot_functions,
    camber_line,
    title="Multi-view Plot",
    show_closeups=True,
    main_figsize=(12, 6),
    single_figsize=(8, 6),
    chord_fraction=0.02,
    bias=0.5,
):
    """
    Create a plot with main view and optional close-up views of leading and trailing edges.

    Parameters
    ----------
    plot_functions : list of callable
        List of functions to call for plotting. Each function should accept an ax parameter
        and plot on the provided axes. Functions will be called for main, LE, and TE axes.
    camber_line : PlaneCurve
        Camber line used to determine LE and TE positions for close-up views.
    title : str, optional
        Main plot title. Default is "Multi-view Plot".
    show_closeups : bool, optional
        Whether to show close-up views of leading and trailing edges. Default is True.
    main_figsize : tuple, optional
        Figure size when showing close-ups. Default is (12, 6).
    single_figsize : tuple, optional
        Figure size when not showing close-ups. Default is (8, 6).
    chord_fraction : float, optional
        Fraction of the chord length to use as the zoom window size for close-up views.
        Default is 0.02 (2% of chord length).
    bias : float, optional
        Bias factor for positioning zoom windows. 0.0 centers the zoom on the LE/TE points.
        Positive values bias LE towards lower-left and TE towards upper-right.
        Default is 0.5.

    Returns
    -------
    tuple
        If show_closeups is True: (fig, ax_main, ax_le, ax_te)
        If show_closeups is False: (fig, ax_main)
    """
    if show_closeups:
        fig = plt.figure(figsize=main_figsize)
        # Create main axis on the left (70% width)
        ax_main = fig.add_subplot(1, 3, (1, 2))

        # Create two smaller axes on the right (50% height each)
        ax_le = fig.add_subplot(2, 3, 3)  # Leading edge (top right)
        ax_te = fig.add_subplot(2, 3, 6)  # Trailing edge (bottom right)
    else:
        fig, ax_main = plt.subplots(figsize=single_figsize)
        ax_le = None
        ax_te = None

    # Plot on main axis
    for plot_func in plot_functions:
        plot_func(ax_main)

    ax_main.set_title(title)
    ax_main.legend()
    ax_main.axis("equal")

    # Plot close-up views if requested
    if show_closeups and ax_le is not None and ax_te is not None:
        # Use the extracted zoomed view function for both LE and TE
        plot_zoomed_view(
            ax_le,
            plot_functions,
            camber_line,
            zoom_type="LE",
            chord_fraction=chord_fraction,
            bias=bias,
        )

        plot_zoomed_view(
            ax_te,
            plot_functions,
            camber_line,
            zoom_type="TE",
            chord_fraction=chord_fraction,
            bias=bias,
        )

        fig.tight_layout()
        return fig, ax_main, ax_le, ax_te

    return fig, ax_main