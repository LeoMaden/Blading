from typing import Protocol
import matplotlib.pyplot as plt
from geometry.curves import plot_plane_curve
from numpy.typing import NDArray
import numpy as np
from geometry.curves import PlaneCurve
from dataclasses import dataclass, field
from .camber import Camber
from .section import Section
from .section_perimiter import SectionPerimiter
from .thickness import Thickness
from .plotting import plot_zoomed_view, create_multi_view_plot
from scipy.signal import find_peaks


################################################################################
########################## Section splitting ###################################
################################################################################


@dataclass
class SplitConfig:
    """
    Configuration parameters for airfoil section splitting based on curvature analysis.

    Controls the thresholds and parameters used to identify low-curvature regions
    along an airfoil section perimeter for extracting upper and lower surface curves.

    Parameters
    ----------
    curvature_threshold_factor : float, optional
        Multiplier applied to the smallest detected curvature peak to determine
        the threshold for masking high curvature regions. Higher values result
        in more restrictive thresholds. Default is 0.9.
    curvature_prominence_factor : float, optional
        Minimum prominence of curvature peaks to consider, specified as a fraction
        of the maximum curvature value. Used to filter noise and identify significant
        curvature features. Default is 0.1.
    min_segment_length : int, optional
        Minimum length (in number of points) required for a continuous low-curvature
        segment to be considered valid. Shorter segments are discarded. Default is 10.

    Notes
    -----
    These parameters work together to control the sensitivity of the splitting algorithm:
    - Lower curvature_threshold_factor values make the algorithm more permissive
    - Higher curvature_prominence_factor values require more pronounced curvature peaks
    - Higher min_segment_length values require longer continuous low-curvature regions
    """

    curvature_threshold_factor: float = 0.9
    curvature_prominence_factor: float = 0.1
    min_segment_length: int = 10


@dataclass
class SplitSectionResult:
    """
    Result of splitting an airfoil section perimeter into upper and lower curves.

    Contains the extracted curves along with diagnostic information about the
    splitting process, including curvature analysis data and success status.

    Parameters
    ----------
    upper_curve : PlaneCurve or None
        The upper surface curve of the airfoil, or None if splitting failed.
    lower_curve : PlaneCurve or None
        The lower surface curve of the airfoil, or None if splitting failed.
    success : bool
        True if exactly two segments were found and curves extracted successfully.
    section : SectionPerimiter
        The original section perimeter that was split.
    segments : list of tuple
        List of (start, end) index tuples representing the detected low-curvature
        segments along the perimeter.
    curvature_threshold : float
        The threshold value used for curvature-based masking during splitting.
    s_data : NDArray
        Normalized arc length parameter values along the section perimeter.
    curvature_data : NDArray
        Curvature values computed along the section perimeter for diagnostic purposes.
    error_message : str, optional
        Description of any errors or warnings encountered during splitting. Default is "".

    Methods
    -------
    plot_curvature(ax=None)
        Plot the curvature data with threshold and detected segments.
    plot_section(ax=None)
        Plot the original section and extracted upper/lower curves.
    plot_summary()
        Create a summary plot showing both curvature analysis and section curves.
    """

    upper_curve: PlaneCurve | None
    lower_curve: PlaneCurve | None
    success: bool

    section: SectionPerimiter
    segments: list[tuple[int, int]]
    curvature_threshold: float
    s_data: NDArray
    curvature_data: NDArray
    error_message: str = ""

    def plot_curvature(self, ax=None):
        """
        Plot the curvature data with threshold and detected segments.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure and axes.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the curvature plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.set_title("Curvature of Section")
        ax.semilogy(self.s_data, self.curvature_data, label="Curvature", color="k")
        ax.axhline(
            self.curvature_threshold, color="red", linestyle="--", label="Threshold"
        )
        ax.set_xlabel("Normalised Arc Length (s)")
        for segment in self.segments:
            ax.semilogy(
                self.s_data[segment[0] : segment[1]],
                self.curvature_data[segment[0] : segment[1]],
                label=f"Segment {segment[0]}-{segment[1]}",
            )
        ax.legend()
        return ax

    def plot_section(self, ax=None):
        """
        Plot the original section and extracted upper/lower curves.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure and axes.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the section plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        self.section.curve.plot(ax=ax, label="Section", color="k", linestyle="--")

        if not self.upper_curve or not self.lower_curve:
            ax.set_title("No valid upper/lower curves found")
        else:
            self.upper_curve.plot(ax=ax, label="Upper Curve", linewidth=2)
            self.lower_curve.plot(ax=ax, label="Lower Curve", linewidth=2)

        ax.legend()
        ax.axis("equal")
        return ax

    def plot_summary(self):
        """
        Create a summary plot showing both curvature analysis and section curves.

        Creates a two-panel figure with curvature analysis on the left and
        the section with extracted curves on the right.

        Returns
        -------
        tuple
            (figure, (ax1, ax2)) where ax1 contains curvature plot and ax2 contains section plot.
        """
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))
        self.plot_curvature(ax1)
        self.plot_section(ax2)
        return fig, (ax1, ax2)


def split_section(
    section: SectionPerimiter,
    config: SplitConfig | None = None,
):
    """
    Split an airfoil section perimeter into upper and lower curves based on curvature analysis.

    Analyzes the curvature along the section perimeter to identify regions of low curvature,
    which typically correspond to the upper and lower surfaces of an airfoil. Uses threshold
    masking and segment detection to extract exactly two continuous segments representing
    the upper and lower curves.

    Parameters
    ----------
    section : SectionPerimiter
        The airfoil section perimeter to split, containing a closed curve representing
        the airfoil boundary.
    config : SplitConfig or None, optional
        Configuration parameters for the splitting algorithm. If None, uses default
        SplitConfig values. Default is None.

    Returns
    -------
    SplitSectionResult
        Result object containing the extracted upper and lower curves (if successful),
        diagnostic information, and success status. See SplitSectionResult documentation
        for detailed field descriptions.

    Notes
    -----
    The algorithm works by:
    1. Computing curvature along the normalized arc length parameter
    2. Applying threshold masking to identify low curvature regions
    3. Finding continuous segments that meet minimum length requirements
    4. Extracting upper and lower curves if exactly 2 segments are found

    The lower curve coordinates are reversed to maintain consistent orientation.
    Success requires exactly two segments to be detected; any other number results
    in failure with appropriate error messaging.
    """
    if config is None:
        config = SplitConfig()

    # Normalised arc length parameter, enforced by SectionPerimiter
    s = section.curve.param

    # Calculate curvature
    curvature = section.curve.curvature()

    # Apply threshold masking
    low_curvature_mask, threshold = _threshold_masking(
        curvature=curvature,
        threshold_factor=config.curvature_threshold_factor,
        prominence_factor=config.curvature_prominence_factor,
    )

    # Find continuous segments of low curvature
    segments = _find_continuous_segments(
        mask=low_curvature_mask,
        min_length=config.min_segment_length,
    )

    # Early return with failure if anything other than 2 segments found
    if len(segments) != 2:
        return SplitSectionResult(
            upper_curve=None,
            lower_curve=None,
            success=False,
            section=section,
            segments=segments,
            curvature_threshold=threshold,
            s_data=s,
            curvature_data=curvature,
            error_message=(
                "Failed to split section into exactly two segments. "
                f"Found {len(segments)} segments."
            ),
        )

    # Exactly two segments found, extract upper and lower curves
    upper_coords = section.curve.coords[segments[0][0] : segments[0][1]]
    lower_coords = section.curve.coords[segments[1][0] : segments[1][1]][::-1]

    upper_curve = PlaneCurve.new_unit_speed(upper_coords)
    lower_curve = PlaneCurve.new_unit_speed(lower_coords)

    return SplitSectionResult(
        upper_curve=upper_curve,
        lower_curve=lower_curve,
        success=True,
        section=section,
        segments=segments,
        curvature_threshold=threshold,
        s_data=s,
        curvature_data=curvature,
        error_message="",
    )


def _threshold_masking(
    curvature, threshold_factor: float = 3, prominence_factor: float = 0.1
) -> tuple[NDArray, float]:
    """
    Mask high curvature regions based on a threshold.

    Identifies peaks in the curvature array using a specified prominence, then masks out regions where the curvature exceeds a threshold determined by the smallest detected peak and a threshold factor.

    Parameters
    ----------
    curvature : array_like
        Array of curvature values.
    threshold_factor : float, optional
        Multiplier applied to the smallest detected peak to determine the curvature threshold. Default is 3.
    prominence_factor : float, optional
        Minimum prominence of peaks to consider, specified as a fraction of the maximum curvature. Default is 0.1.

    Returns
    -------
    low_curvature_mask : ndarray of bool
        Boolean mask array where True indicates low curvature regions (below threshold).
    threshold : float
        The curvature threshold used for masking.

    Raises
    ------
    ValueError
        If no peaks are found in the curvature data.
    """
    # Find the smallest peak in curvature with a given prominence
    prominence = np.max(curvature) * prominence_factor
    peaks, _ = find_peaks(curvature, prominence=prominence)

    if len(peaks) == 0:
        raise ValueError("No peaks found in curvature data")

    # Create mask: True for low curvature, False for high curvature
    threshold = np.min(curvature[peaks]) * threshold_factor

    low_curvature_mask = curvature < threshold

    return low_curvature_mask, threshold


def _find_continuous_segments(mask, min_length=10):
    """
    Find continuous segments of True values in a boolean mask.

    Identifies consecutive True values in the input boolean array and returns
    only those segments that meet the minimum length requirement.

    Parameters
    ----------
    mask : array_like of bool
        Boolean array where True values indicate regions of interest.
    min_length : int, optional
        Minimum length of segments to include in the output. Default is 10.

    Returns
    -------
    list of tuple
        List of (start, end) tuples representing the start and end indices
        of continuous segments that meet the minimum length requirement.
        End indices are exclusive (i.e., segment spans mask[start:end]).

    Examples
    --------
    >>> mask = np.array([False, True, True, True, False, True, True, False])
    >>> find_continuous_segments(mask, min_length=2)
    [(1, 4), (5, 7)]

    >>> find_continuous_segments(mask, min_length=5)
    []
    """
    # Find where mask changes
    diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    segments = []
    for start, end in zip(starts, ends):
        if end - start >= min_length:
            segments.append((start, end))

    return segments


################################################################################
###################### Initial guess of camber and thickness ###################
################################################################################


def _initial_guess(
    upper_curve: PlaneCurve, lower_curve: PlaneCurve
) -> tuple[PlaneCurve, NDArray]:
    """
    Create initial guesses for camber and thickness based on upper and lower curves.

    Parameters
    ----------
    upper_curve : PlaneCurve
        The upper surface curve of the airfoil.
    lower_curve : PlaneCurve
        The lower surface curve of the airfoil.

    Returns
    -------
    tuple[NDArray, NDArray]
        A tuple containing the initial camber line and thickness values.
    """
    # Ensure both curves have normalised arc length parameterisation
    upper_curve = upper_curve.reparameterise_unit().normalise()
    lower_curve = lower_curve.reparameterise_unit().normalise()

    # Interpolate the upper and lower curves to have equal number of points
    # using the spacing from the curve with more points
    if len(upper_curve.param) > len(lower_curve.param):
        lower_curve = lower_curve.interpolate(upper_curve.param, k=2)
    else:
        upper_curve = upper_curve.interpolate(lower_curve.param, k=2)

    # Camber is the average the coordinates of upper and lower curves
    camber_coords = (upper_curve.coords + lower_curve.coords) / 2
    camber_line = PlaneCurve.new_unit_speed(camber_coords)

    # Thickness is the distance between upper and lower curves
    thickness_values = np.linalg.norm(upper_curve.coords - lower_curve.coords, axis=1)

    return camber_line, thickness_values


################################################################################
##################### Extend camber line to LE and TE ##########################
################################################################################


class ExtendCamberError(Exception):
    """Custom exception for errors during camber line extension."""

    pass


@dataclass
class ExtendCamberResult:
    original_line: PlaneCurve
    extended_line: PlaneCurve | None
    success: bool
    section: SectionPerimiter
    LE_line: NDArray
    TE_line: NDArray
    error_message: str = ""

    def unwrap(self) -> PlaneCurve:
        """
        Unwrap the extended camber line if successful.

        Raises an exception if the extension was not successful.
        """
        if not self.success or self.extended_line is None:
            raise ExtendCamberError(self.error_message)
        return self.extended_line

    def plot(self, ax=None, show_closeups=True):
        """
        Plot the camber extension result showing original line, section, and extension lines.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure and axes or multi-view plot.
        show_closeups : bool, optional
            Whether to show close-up views of leading and trailing edges. Default is True.
            Ignored if ax is provided.

        Returns
        -------
        ...
        """
        # If ax is provided, use single axis plotting (legacy behavior)
        if ax is not None:
            self._plot_on_single_axis(ax)
            return ax

        # Use multi-view plotting utility
        plot_functions = [
            lambda ax: self.section.curve.plot(
                ax=ax, label="Section Curve", color="k", alpha=0.8
            ),
            lambda ax: self.original_line.plot(
                ax=ax, label="Original Camber", color="b", linewidth=2
            ),
            lambda ax: ax.plot(
                self.LE_line[:, 0],
                self.LE_line[:, 1],
                "r:",
                label="LE Extension Line",
                alpha=0.8,
            ),
            lambda ax: ax.plot(
                self.TE_line[:, 0],
                self.TE_line[:, 1],
                "g:",
                label="TE Extension Line",
                alpha=0.8,
            ),
        ]

        # Add extended camber plotting if successful
        if self.success and self.extended_line is not None:
            plot_functions.extend(
                [
                    lambda ax: self.extended_line.plot(
                        ax=ax,
                        label="Extended Camber",
                        color="m",
                        linestyle="--",
                        alpha=0.9,
                    ),
                    lambda ax: ax.plot(
                        *self.extended_line.start(),
                        "co",
                        markeredgecolor="k",
                        label="New LE Point",
                    ),
                    lambda ax: ax.plot(
                        *self.extended_line.end(),
                        "cs",
                        markeredgecolor="k",
                        label="New TE Point",
                    ),
                ]
            )

        # Set title based on success
        title = (
            "Camber Extension Result (Success)"
            if self.success
            else f"Camber Extension Result (Failed - {self.error_message})"
        )

        return create_multi_view_plot(
            plot_functions=plot_functions,
            camber_line=self.extended_line or self.original_line,
            title=title,
            show_closeups=show_closeups,
        )

    def _plot_on_single_axis(self, ax):
        """Helper method for plotting on a single provided axis."""
        # Plot original camber line
        self.original_line.plot(ax=ax, label="Original Camber", color="b", linewidth=2)

        # Plot section curve
        self.section.curve.plot(ax=ax, label="Section Curve", color="k", alpha=0.8)

        # Plot extension lines
        ax.plot(
            self.LE_line[:, 0],
            self.LE_line[:, 1],
            "r--",
            label="LE Extension Line",
            linewidth=2,
            alpha=0.8,
        )
        ax.plot(
            self.TE_line[:, 0],
            self.TE_line[:, 1],
            "g--",
            label="TE Extension Line",
            linewidth=2,
            alpha=0.8,
        )

        # Plot extended camber if successful
        if self.success and self.extended_line is not None:
            self.extended_line.plot(
                ax=ax,
                label="Extended Camber",
                color="m",
                linestyle="--",
                linewidth=3,
                alpha=0.9,
            )
            # Mark the new endpoints
            ax.plot(
                *self.extended_line.start(),
                "co",
                markersize=10,
                markeredgecolor="k",
                markeredgewidth=1,
                label="New LE Point",
            )
            ax.plot(
                *self.extended_line.end(),
                "cs",
                markersize=10,
                markeredgecolor="k",
                markeredgewidth=1,
                label="New TE Point",
            )
            ax.set_title("Camber Extension Result (Success)")
        else:
            ax.set_title(f"Camber Extension Result (Failed - {self.error_message})")

        ax.legend()
        ax.axis("equal")
        ax.grid(True, alpha=0.3)


def _extend_camber_line(
    camber_line: PlaneCurve, section: SectionPerimiter, ext_factor: float
) -> ExtendCamberResult:
    chord = camber_line.length()
    tangent = camber_line.tangent()

    # Create lines extending from the first and last points of the camber line
    LE_line = np.c_[
        camber_line.start() - tangent.coords[0] * ext_factor * chord,
        camber_line.start(),
    ].T
    TE_line = np.c_[
        camber_line.end() + tangent.coords[-1] * ext_factor * chord, camber_line.end()
    ].T

    # Intersect lines with the section curve to find new leading and trailing edge points
    LE_point = section.curve.intersect_coords(LE_line)
    TE_point = section.curve.intersect_coords(TE_line)

    # Ensure there is only one intersection point for each end
    if LE_point.shape[0] != 1 or TE_point.shape[0] != 1:
        return ExtendCamberResult(
            original_line=camber_line,
            extended_line=None,
            success=False,
            section=section,
            LE_line=LE_line,
            TE_line=TE_line,
            error_message="Failed to find unique intersection points for LE or TE lines.",
        )

    # Only one intersection point found for each end
    new_coords = np.r_[LE_point, camber_line.coords, TE_point]
    extended_line = PlaneCurve.new_unit_speed(new_coords)

    return ExtendCamberResult(
        original_line=camber_line,
        extended_line=extended_line,
        success=True,
        section=section,
        LE_line=LE_line,
        TE_line=TE_line,
    )


################################################################################
####################### Camber line improvement ################################
################################################################################


@dataclass
class ApproximateCamberConfig:
    tolerance: float = 1e-7
    max_iterations: int = 100
    relax_factor: float = 0.2
    thickness_scale: float = 1.2
    extension_factor: float = 0.02  # Factor of chord length for LE/TE extension
    num_points_le: int = 50
    num_points_te: int = 50
    split_config: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class CamberIterationResult:
    prev_camber: Camber
    prev_thickness: Thickness
    section: SectionPerimiter
    new_camber: Camber | None = None
    new_thickness: Thickness | None = None
    upper_targets: NDArray | None = None
    lower_targets: NDArray | None = None
    extend_result: ExtendCamberResult | None = None
    delta: float = float("inf")
    success: bool = False
    error_message: str = ""

    def plot_zoomed(
        self, ax=None, zoom_type="LE", chord_fraction=0.02, bias=0.5, title=None
    ):
        """
        Plot a zoomed-in view of leading or trailing edge on the provided or new axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure and axes.
        zoom_type : str, optional
            Type of zoom view: "LE" for leading edge or "TE" for trailing edge. Default is "LE".
        chord_fraction : float, optional
            Fraction of the chord length to use as the zoom window size. Default is 0.02.
        bias : float, optional
            Bias factor for positioning zoom windows. Default is 0.5.
        title : str, optional
            Title for the zoom view. If None, uses default based on zoom_type.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object that was plotted on.
        """
        if ax is None:
            fig, ax = plt.subplots()

        plot_functions = self._get_plot_functions()
        camber_line = self._get_camber_line_for_zoom()

        return plot_zoomed_view(
            ax=ax,
            plot_functions=plot_functions,
            camber_line=camber_line,
            zoom_type=zoom_type,
            title=title,
            chord_fraction=chord_fraction,
            bias=bias,
        )

    def plot(self, show_closeups=True):
        """
        Plot the camber line, section, and target connections using multi-view layout.

        If success is True, also plots the new camber line.
        Shows lines connecting corresponding points on upper and lower targets.

        Parameters
        ----------
        show_closeups : bool, optional
            Whether to show LE/TE closeup views.

        Returns
        -------
        matplotlib.figure.Figure
            The figure returned by create_multi_view_plot.
        """
        plot_functions = self._get_plot_functions()
        camber_line = self._get_camber_line_for_zoom()
        title = self._get_plot_title()

        return create_multi_view_plot(
            plot_functions=plot_functions,
            camber_line=camber_line,
            title=title,
            show_closeups=show_closeups,
        )

    def _get_plot_functions(self):
        """Get the plot functions for this iteration result."""
        plot_functions = [
            lambda ax: self.section.curve.plot(
                ax=ax, label="Section", color="k", linestyle="-", alpha=0.7
            ),
            lambda ax: self.prev_camber.line.plot(
                ax=ax, label="Previous Camber", color="b"
            ),
        ]

        # Add target points and connecting lines
        if (self.upper_targets is not None) and (self.lower_targets is not None):
            # Capture references to avoid None type issues in lambdas
            upper_targets = self.upper_targets
            lower_targets = self.lower_targets

            plot_functions.extend(
                [
                    lambda ax: ax.scatter(
                        upper_targets[:, 0],
                        upper_targets[:, 1],
                        c="r",
                        s=20,
                        alpha=0.6,
                        label="Upper Targets",
                        marker="^",
                    ),
                    lambda ax: ax.scatter(
                        lower_targets[:, 0],
                        lower_targets[:, 1],
                        c="g",
                        s=20,
                        alpha=0.6,
                        label="Lower Targets",
                        marker="v",
                    ),
                ]
            )

            # Add connecting lines function
            def plot_connecting_lines(ax):
                for i in range(len(upper_targets)):
                    ax.plot(
                        [upper_targets[i, 0], lower_targets[i, 0]],
                        [upper_targets[i, 1], lower_targets[i, 1]],
                        color="gray",
                        alpha=0.3,
                    )

            plot_functions.append(plot_connecting_lines)

        # Add new camber if successful
        if self.success and self.new_camber is not None:
            new_camber_line = self.new_camber.line
            plot_functions.append(
                lambda ax: new_camber_line.plot(
                    ax=ax, label="New Camber", color="m", alpha=0.8
                )
            )

        # Add extend result if available
        if self.extend_result is not None:
            extend_result = self.extend_result
            plot_functions.extend(
                [
                    lambda ax: ax.plot(
                        extend_result.LE_line[:, 0],
                        extend_result.LE_line[:, 1],
                        "r:",
                        label="LE Extension Line",
                        alpha=0.7,
                    ),
                    lambda ax: ax.plot(
                        extend_result.TE_line[:, 0],
                        extend_result.TE_line[:, 1],
                        "g:",
                        label="TE Extension Line",
                        alpha=0.7,
                    ),
                ]
            )

        return plot_functions

    def _get_camber_line_for_zoom(self):
        """Get the appropriate camber line for zoom reference."""
        return (
            self.new_camber.line
            if (self.success and self.new_camber)
            else self.prev_camber.line
        )

    def _get_plot_title(self):
        """Get the plot title based on success status."""
        if self.success:
            title = f"Camber Iteration Result (Success - δ={self.delta:.2e})"
            if self.extend_result is not None:
                extend_status = (
                    "Extended" if self.extend_result.success else "Extension Failed"
                )
                title += f" - {extend_status}"
        else:
            title = f"Camber Iteration Result (Failed - {self.error_message})"
        return title


def _find_best_intersection(intersections: NDArray, target_point: NDArray) -> NDArray:
    """Find the closest intersection point to the target."""
    if intersections.shape[0] == 0:
        raise ValueError("No intersections found")
    if intersections.shape[0] == 1:
        return intersections[0]

    distances = np.linalg.norm(intersections - target_point, axis=1)
    return intersections[np.argmin(distances)]


def _compute_intersection_coords(
    camber_point: NDArray,
    upper_target: NDArray,
    lower_target: NDArray,
    normal: NDArray,
    thickness_value: float,
    section: SectionPerimiter,
) -> tuple[NDArray, NDArray, bool]:
    """Compute upper and lower intersection coordinates for a single point."""
    try:
        # Find intersections
        int_u = section.curve.intersect_coords(np.c_[camber_point, upper_target].T)
        int_l = section.curve.intersect_coords(np.c_[lower_target, camber_point].T)

        # Get best intersections. Will throw if no intersections found
        upper_coord = _find_best_intersection(int_u, upper_target)
        lower_coord = _find_best_intersection(int_l, lower_target)

        return upper_coord, lower_coord, True

    except Exception:
        # Fallback to normal projection
        upper_coord = camber_point + thickness_value * normal
        lower_coord = camber_point - thickness_value * normal
        return upper_coord, lower_coord, False


def _calc_new_camber_thickness(
    upper_coords: NDArray, lower_coords: NDArray
) -> tuple[NDArray, NDArray]:
    """Create new camber and thickness from computed coordinates."""
    new_camber_coords = (upper_coords + lower_coords) / 2
    new_thickness_values = np.linalg.norm(upper_coords - lower_coords, axis=1)
    return new_camber_coords, new_thickness_values


def _improve_camber_robust(
    camber: Camber,
    thickness: Thickness,
    section: SectionPerimiter,
    config: ApproximateCamberConfig,
) -> CamberIterationResult:
    """Robust version of camber improvement with proper error handling."""
    try:
        N = len(thickness.t)
        normal = camber.line.signed_normal()
        scaled_thickness = config.thickness_scale * thickness.t[:, np.newaxis]
        camber_coords = camber.line.coords
        upper_targets = camber_coords + 0.5 * scaled_thickness * normal.coords
        lower_targets = camber_coords - 0.5 * scaled_thickness * normal.coords

        upper_intersections = np.zeros((N - 2, 2))
        lower_intersections = np.zeros((N - 2, 2))
        upper_targets = upper_targets[1:-1]  # Exclude LE and TE
        lower_targets = lower_targets[1:-1]
        successful_intersections = 0

        # Process each point along the camber line
        # i is the index for the input arrays (1 to N-1, inclusive)
        # j is the index for the output arrays (0 to N-2, inclusive)
        for i, j in zip(range(1, N - 1), range(N - 2)):
            c = camber_coords[i]
            u = upper_targets[j]
            l = lower_targets[j]

            upper_intersections[j], lower_intersections[j], success = (
                _compute_intersection_coords(
                    c,
                    u,
                    l,
                    normal.coords[i],
                    thickness.t[i],
                    section,
                )
            )

            if success:
                successful_intersections += 1

        # Require all intersections to be successful
        min_required = N - 2
        if successful_intersections < min_required:
            return CamberIterationResult(
                prev_camber=camber,
                prev_thickness=thickness,
                section=section,
                upper_targets=upper_targets,
                lower_targets=lower_targets,
                error_message=f"Only {successful_intersections}/{min_required} intersections found",
            )

        # Create new camber and thickness exclusing leading and trailing edges
        short_camber_coords, short_thickness_values = _calc_new_camber_thickness(
            upper_intersections, lower_intersections
        )

        # Apply relaxation
        camber_coords_short = camber.line.coords[1:-1]  # Exclude LE and TE
        delta_coords = (short_camber_coords - camber_coords_short) * config.relax_factor
        relaxed_coords_short = camber_coords_short + delta_coords

        # Use the relaxed coordinates to create a new short camber line
        short_camber_line = PlaneCurve(relaxed_coords_short, camber.line.param[1:-1])

        # Extend to leading and trailing edges
        new_thickness_values = np.r_[0, short_thickness_values, 0]
        extend_result = _extend_camber_line(
            short_camber_line, section, config.extension_factor
        )
        if not extend_result.success:
            return CamberIterationResult(
                prev_camber=camber,
                prev_thickness=thickness,
                section=section,
                upper_targets=upper_targets,
                lower_targets=lower_targets,
                extend_result=extend_result,
                error_message=extend_result.error_message,
            )
        new_camber_line = extend_result.unwrap()

        # Create final camber and thickness objects
        new_thickness = Thickness(thickness.s, new_thickness_values, thickness.chord)
        final_camber = Camber(new_camber_line)

        # Calculate delta as the sum of distance moved by each camber point
        delta = np.sum(
            np.linalg.norm(new_camber_line.coords - camber.line.coords, axis=1)
        )

        return CamberIterationResult(
            prev_camber=camber,
            new_camber=final_camber,
            prev_thickness=thickness,
            new_thickness=new_thickness,
            section=section,
            upper_targets=upper_targets,
            lower_targets=lower_targets,
            extend_result=extend_result,
            delta=delta,
            success=True,
        )

    except Exception as e:
        return CamberIterationResult(
            prev_camber=camber,
            prev_thickness=thickness,
            section=section,
            error_message=str(e),
        )


@dataclass
class IterationLoopResult:
    initial_camber: Camber
    initial_thickness: Thickness

    success: bool = False
    final_camber: Camber | None = None
    final_thickness: Thickness | None = None
    final_iteration: CamberIterationResult | None = None

    final_delta: float = float("inf")
    convergence_history: list[float] = field(default_factory=list)
    iterations: int = 0
    error_message: str = ""


@dataclass
class CamberProgress:
    """Progress callback data for camber approximation iterations."""

    iteration: int
    iteration_result: CamberIterationResult


class ApproximateCamberCallback(Protocol):
    def __call__(self, progress: CamberProgress) -> bool:
        """Return False to abort iteration."""
        ...


def _run_iteration_loop(
    camber: Camber,
    thickness: Thickness,
    section: SectionPerimiter,
    config: ApproximateCamberConfig,
    callback: ApproximateCamberCallback | None,
) -> IterationLoopResult:
    """Run the iterative camber improvement loop."""
    convergence_history = []
    iterations = 0
    new_camber = None
    new_thickness = None
    iteration_result = None

    new_camber = camber
    new_thickness = thickness

    while iterations < config.max_iterations:
        try:
            iteration_result = _improve_camber_robust(
                new_camber, new_thickness, section, config
            )

            if not iteration_result.success:
                return IterationLoopResult(
                    success=False,
                    initial_camber=camber,
                    initial_thickness=thickness,
                    final_camber=new_camber,
                    final_thickness=new_thickness,
                    final_iteration=iteration_result,
                    final_delta=float("inf"),
                    iterations=iterations,
                    convergence_history=convergence_history,
                    error_message=f"Iteration {iterations} failed: {iteration_result.error_message}",
                )

            new_camber = iteration_result.new_camber
            new_thickness = iteration_result.new_thickness

            # Iteration is successful, expect new camber and thickness to be set
            assert new_camber is not None, "Expected new camber to be set"
            assert new_thickness is not None, "Expected new thickness to be set"

            delta = iteration_result.delta
            convergence_history.append(delta)
            iterations += 1

            # Check convergence
            if delta <= config.tolerance:
                return IterationLoopResult(
                    success=True,
                    initial_camber=camber,
                    initial_thickness=thickness,
                    final_camber=new_camber,
                    final_thickness=new_thickness,
                    final_iteration=iteration_result,
                    final_delta=delta,
                    iterations=iterations,
                    convergence_history=convergence_history,
                )

            # Handle callback
            if callback is not None:
                progress = CamberProgress(
                    iteration=iterations,
                    iteration_result=iteration_result,
                )
                if not callback(progress):
                    return IterationLoopResult(
                        success=False,
                        initial_camber=camber,
                        initial_thickness=thickness,
                        final_camber=new_camber,
                        final_thickness=new_thickness,
                        final_iteration=iteration_result,
                        final_delta=delta,
                        iterations=iterations,
                        convergence_history=convergence_history,
                        error_message="Cancelled by user callback",
                    )

        except Exception as e:
            return IterationLoopResult(
                success=False,
                initial_camber=camber,
                initial_thickness=thickness,
                final_camber=new_camber,
                final_thickness=new_thickness,
                final_iteration=iteration_result,
                final_delta=(
                    convergence_history[-1] if convergence_history else float("inf")
                ),
                iterations=iterations,
                convergence_history=convergence_history,
                error_message=f"Unexpected error at iteration {iterations}: {str(e)}",
            )

    # Max iterations reached
    final_delta = convergence_history[-1] if convergence_history else float("inf")
    return IterationLoopResult(
        success=False,
        initial_camber=camber,
        initial_thickness=thickness,
        final_camber=new_camber,
        final_thickness=new_thickness,
        final_iteration=iteration_result,
        final_delta=final_delta,
        iterations=iterations,
        convergence_history=convergence_history,
        error_message=(
            f"Failed to converge after {config.max_iterations} iterations "
            f"(final delta: {delta:.2e}, tolerance: {config.tolerance:.2e})"
        ),
    )


################################################################################
####################### Final section creation #################################
################################################################################


def create_final_section(
    camber: Camber,
    thickness: Thickness,
    section: SectionPerimiter,
    config: ApproximateCamberConfig,
) -> Section:
    """Create final section with extended camber and refined point distribution."""
    camber_line = camber.line
    param = camber.line.param

    # Add refined point distribution near LE and TE
    # First use cosine spacing to cluster points near LE and TE
    # Then map this spacing to between the first two points for LE,
    # and between the last two points for TE.
    spacing_le = np.cos(np.linspace(0, np.pi / 2, config.num_points_le))
    new_param_LE = param[1] - (param[1] - param[0]) * spacing_le

    spacing_te = np.cos(np.linspace(np.pi / 2, 0, config.num_points_te))
    new_param_TE = param[-2] + (param[-1] - param[-2]) * spacing_te

    # Create new camber line parameter
    new_param = np.r_[new_param_LE, param[2:-2], new_param_TE]
    new_camber_line = camber_line.interpolate(new_param, k=2)
    new_camber = Camber(new_camber_line)

    # Extend thickness array to match new parameterization
    extended_thickness_values = np.r_[
        np.repeat(thickness.t[1], config.num_points_le),
        thickness.t[2:-2],
        np.repeat(thickness.t[-2], config.num_points_te),
    ]
    extended_thickness = Thickness(
        new_camber.line.param, extended_thickness_values, camber.chord
    )

    improve_result = _improve_camber_robust(
        new_camber, extended_thickness, section, config
    )
    if not improve_result.success:
        improve_result.plot()
        plt.show()
        raise CamberApproximationError(
            f"Failed to calculate final thickness: {improve_result.error_message}"
        )

    new_thickness = improve_result.new_thickness
    assert new_thickness is not None, "Expected new thickness to be set"

    return Section(
        camber=new_camber,
        thickness=new_thickness,
        stream_line=section.stream_line,
    )


################################################################################
###################### Camber line approximation ###############################
################################################################################


@dataclass
class ApproximateCamberResult:
    section_perimiter: SectionPerimiter
    split_result: SplitSectionResult

    initial_extension_result: ExtendCamberResult | None = None
    initial_camber: Camber | None = None
    initial_thickness: Thickness | None = None

    iterations: int = 0
    iteration_loop_result: IterationLoopResult | None = None

    section: Section | None = None
    success: bool = False

    error_message: str = ""

    def unwrap(self) -> Section:
        if not self.success or self.section is None:
            raise CamberApproximationError(self.error_message)
        return self.section

    def plot_split_summary(self):
        return self.split_result.plot_summary()

    def plot_initial_extension(self):
        if self.initial_extension_result is None:
            raise ValueError("Initial extension result must be set for plotting.")
        self.initial_extension_result.plot()

    def plot_initial_camber(self, show_closeups=True):
        """
        Plot the initial camber line with optional close-up views of leading and trailing edges

        Parameters
        ----------
        show_closeups : bool, optional
            Whether to show close-up views of leading and trailing edges. Default is True.

        Returns
        -------
        ...
        """
        if self.initial_camber is None:
            raise ValueError("Initial camber must be set for plotting.")

        assert (
            self.split_result.upper_curve is not None
        ), "Expected upper curve to be valid"
        assert (
            self.split_result.lower_curve is not None
        ), "Expected lower curve to be valid"
        assert (
            self.initial_thickness is not None
        ), "Expected initial thickness to be valid"

        # Define plot functions
        plot_functions = [
            lambda ax: self.section_perimiter.curve.plot(
                ax=ax, label="Section Curve", color="k"
            ),
            lambda ax: self.split_result.upper_curve.plot(
                ax=ax, label="Upper Curve", color="r"
            ),
            lambda ax: self.split_result.lower_curve.plot(
                ax=ax, label="Lower Curve", color="g"
            ),
            lambda ax: self.initial_camber.line.plot(
                ax=ax, label="Initial Camber", color="b"
            ),
        ]

        # Use the multi-view plotting utility
        return create_multi_view_plot(
            plot_functions=plot_functions,
            camber_line=self.initial_camber.line,
            title="Initial camber line",
            show_closeups=show_closeups,
        )

    def plot_summary(self):
        """
        Plot summary based on the furthest successful stage of the approximation process.

        Returns
        -------
        Any
            The plot result from the most advanced available stage.
        """
        # If success, plot the final section
        if self.success and self.section is not None:
            return self.section.plot_comparison(
                self.section_perimiter, this_name="New", other_name="Original"
            )

        # If not, plot final iteration result, if iteration loop result is available
        if (
            self.iteration_loop_result is not None
            and self.iteration_loop_result.final_iteration is not None
        ):
            return self.iteration_loop_result.final_iteration.plot()

        # If not, plot initial camber and thickness, if available
        if self.initial_camber is not None:
            return self.plot_initial_camber()

        # If not, plot initial extension result, if available
        if self.initial_extension_result is not None:
            return self.initial_extension_result.plot()

        # If not, plot split result summary
        return self.plot_split_summary()

    def get_summary(self, concise: bool = False) -> str:
        """
        Return a human-readable summary of the approximation result.

        Parameters
        ----------
        concise : bool, optional
            If True, return only status and error message. Default is False.

        Returns
        -------
        str
            Formatted summary string showing the status and key metrics.
        """
        lines = ["ApproximateCamberResult Summary:"]
        lines.append("=" * 35)

        # Overall status
        status = "SUCCESS" if self.success else "FAILED"
        lines.append(f"Status: {status}")

        if not self.success and self.error_message:
            lines.append(f"Error: {self.error_message}")

        # Return early if concise output requested
        if concise:
            return "\n".join(lines)

        # Section splitting stage
        split_status = "SUCCESS" if self.split_result.success else "FAILED"
        lines.append(f"Section splitting: {split_status}")
        if not self.split_result.success:
            lines.append(f"  - Error: {self.split_result.error_message}")
        else:
            lines.append(f"  - Found {len(self.split_result.segments)} segments")

        # Initial extension stage
        if self.initial_extension_result is not None:
            ext_status = (
                "SUCCESS" if self.initial_extension_result.success else "FAILED"
            )
            lines.append(f"Initial camber extension: {ext_status}")
            if not self.initial_extension_result.success:
                lines.append(
                    f"  - Error: {self.initial_extension_result.error_message}"
                )

        # Initial camber/thickness stage
        if self.initial_camber is not None and self.initial_thickness is not None:
            chord = self.initial_camber.chord
            max_thickness = max(self.initial_thickness.t)
            lines.append(
                f"Initial camber created: chord={chord:.4f}, max_thickness={max_thickness:.4f}"
            )

        # Iteration loop stage
        if self.iteration_loop_result is not None:
            loop_result = self.iteration_loop_result
            loop_status = "CONVERGED" if loop_result.success else "FAILED"
            lines.append(f"Iteration loop: {loop_status}")
            lines.append(f"  - Iterations: {loop_result.iterations}")
            lines.append(f"  - Final delta: {loop_result.final_delta:.2e}")
            if not loop_result.success:
                lines.append(f"  - Error: {loop_result.error_message}")

        # Final section stage
        if self.section is not None:
            final_chord = self.section.camber.chord
            final_max_thickness = max(self.section.thickness.t)
            lines.append(
                f"Final section: chord={final_chord:.4f}, max_thickness={final_max_thickness:.4f}"
            )

        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Return a string representation of the ApproximateCamberResult.

        This will call get_summary() to provide a detailed overview.
        """
        return self.get_summary()


class CamberApproximationError(Exception):
    """Custom exception for errors during camber approximation."""

    pass


def approximate_camber(
    section: SectionPerimiter,
    config: ApproximateCamberConfig | None = None,
    callback: ApproximateCamberCallback | None = None,
) -> ApproximateCamberResult:
    # Set defaults
    config = config or ApproximateCamberConfig()
    split_config = config.split_config

    # Split the section into upper and lower curves
    split_result = split_section(section, split_config)
    if not split_result.success:
        return ApproximateCamberResult(
            section_perimiter=section,
            split_result=split_result,
            error_message=split_result.error_message,
        )

    # Initialize camber and thickness
    assert split_result.upper_curve is not None, "Expected upper curve to be valid"
    assert split_result.lower_curve is not None, "Expected lower curve to be valid"
    short_camber_line, short_thickness_values = _initial_guess(
        split_result.upper_curve, split_result.lower_curve
    )

    # Extend camber line
    initial_extend_result = _extend_camber_line(
        short_camber_line, section, config.extension_factor
    )
    if not initial_extend_result.success:
        return ApproximateCamberResult(
            section_perimiter=section,
            split_result=split_result,
            initial_extension_result=initial_extend_result,
            error_message=initial_extend_result.error_message,
        )
    initial_camber = Camber(initial_extend_result.unwrap())

    # Extend thickness
    thickness_values = np.r_[0, short_thickness_values, 0]
    initial_thickness = Thickness(
        initial_camber.s, thickness_values, initial_camber.chord
    )

    # Run iterative improvement
    loop_result = _run_iteration_loop(
        initial_camber, initial_thickness, section, config, callback
    )

    if not loop_result.success:
        return ApproximateCamberResult(
            section_perimiter=section,
            split_result=split_result,
            initial_extension_result=initial_extend_result,
            initial_camber=initial_camber,
            initial_thickness=initial_thickness,
            iteration_loop_result=loop_result,
            iterations=loop_result.iterations,
            error_message=loop_result.error_message,
        )

    final_camber = loop_result.final_camber
    final_thickness = loop_result.final_thickness
    assert final_camber is not None, "Expected final camber to be set"
    assert final_thickness is not None, "Expected final thickness to be set"
    final_section = create_final_section(final_camber, final_thickness, section, config)
    # final_section = None

    return ApproximateCamberResult(
        section=final_section,
        section_perimiter=section,
        split_result=split_result,
        initial_extension_result=initial_extend_result,
        initial_camber=initial_camber,
        initial_thickness=initial_thickness,
        iteration_loop_result=loop_result,
        iterations=loop_result.iterations,
        success=True,
    )
