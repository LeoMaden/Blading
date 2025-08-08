from typing import Protocol
import matplotlib.pyplot as plt
from geometry.curves import plot_plane_curve
from numpy.typing import NDArray
import numpy as np
from geometry.curves import PlaneCurve
from dataclasses import dataclass
from .camber import Camber
from .section import Section
from .section_perimiter import SectionPerimiter
from .thickness import Thickness
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
) -> tuple[Camber, Thickness]:
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
    tuple[Camber, Thickness]
        A tuple containing the initial camber and thickness objects.
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
    camber = Camber(PlaneCurve.new_unit_speed(camber_coords))

    # Thickness is the distance between upper and lower curves
    thickness_values = np.linalg.norm(upper_curve.coords - lower_curve.coords, axis=1)
    thickness = Thickness(upper_curve.param, thickness_values, None)

    return camber, thickness


################# Extend camber line to LE and TE #################


@dataclass
class ExtendCamberResult:
    extended_camber: Camber | None
    success: bool
    original_camber: Camber
    section: SectionPerimiter
    LE_line: NDArray
    TE_line: NDArray
    error_message: str = ""

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        self.original_camber.line.plot(ax=ax, label="Original Camber", color="b")
        self.section.curve.plot(ax=ax, label="Section Curve", color="k")
        ax.plot(self.LE_line[:, 0], self.LE_line[:, 1], "r--", label="LE Line")
        ax.plot(self.TE_line[:, 0], self.TE_line[:, 1], "g--", label="TE Line")

        if self.extended_camber:
            self.extended_camber.line.plot(
                ax=ax, label="Extended Camber", color="m", linestyle="--"
            )
            plt.plot(*self.extended_camber.line.start(), "mo")
            plt.plot(*self.extended_camber.line.end(), "mo")

        ax.legend()
        ax.axis("equal")
        return ax


def extend_camber_line(
    camber: Camber, section: SectionPerimiter, ext_factor: float
) -> ExtendCamberResult:
    chord = camber.line.length()
    tangent = camber.line.tangent()

    # Create lines extending from the first and last points of the camber line
    LE_line = np.c_[
        camber.line.start() - tangent.coords[0] * ext_factor * chord,
        camber.line.start(),
    ].T
    TE_line = np.c_[
        camber.line.end() + tangent.coords[-1] * ext_factor * chord, camber.line.end()
    ].T

    # Intersect lines with the section curve to find new leading and trailing edge points
    LE_point = section.curve.intersect_coords(LE_line)
    TE_point = section.curve.intersect_coords(TE_line)

    # Ensure there is only one intersection point for each end
    if LE_point.shape[0] != 1 or TE_point.shape[0] != 1:
        extended_camber = None
        success = False
        error_message = "Failed to find unique intersection points for LE or TE lines."
    else:
        new_coords = np.r_[LE_point, camber.line.coords, TE_point]
        new_line = PlaneCurve.new_unit_speed(new_coords)
        extended_camber = Camber(new_line)
        success = True
        error_message = ""

    return ExtendCamberResult(
        extended_camber,
        success,
        camber,
        section,
        LE_line,
        TE_line,
        error_message,
    )


################################################################################
###################### Camber line approximation ###############################
################################################################################


@dataclass
class ApproximateCamberConfig:
    tolerance: float = 1e-7
    max_iterations: int = 100
    relax_factor: float = 0.2
    thickness_scale: float = 1.2
    extension_factor: float = 0.1  # Factor of chord length for LE/TE extension
    num_points_le: int = 50
    num_points_te: int = 50


@dataclass
class CamberProgress:
    iteration: int
    delta: float
    camber: Camber
    converged: bool
    error: str | None = None


@dataclass
class ApproximateCamberResult:
    section: Section | None  # Set if the approximation was successful
    success: bool
    section_perimiter: SectionPerimiter
    split_result: SplitSectionResult
    initial_camber: Camber | None  # Set if the split was successful
    initial_thickness: Thickness | None  # Set if the split was successful
    iterations: int
    final_delta: float
    convergence_history: list[float]
    error_message: str = ""

    def unwrap(self) -> Section:
        if not self.success or self.section is None:
            raise CamberApproximationError(self.error_message)
        return self.section

    def plot_split_summary(self):
        return self.split_result.plot_summary()

    def plot_initial_camber(self, show_closeups=True):
        """
        Plot the initial camber line with optional close-up views of leading and trailing edges

        Parameters
        ----------
        show_closeups : bool, optional
            Whether to show close-up views of leading and trailing edges. Default is True.

        Returns
        -------
        matplotlib.axes.Axes
            The main axes object containing the initial camber plot.
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

        if show_closeups:
            fig = plt.figure(figsize=(12, 6))
            # Create main axis on the left (70% width)
            ax = fig.add_subplot(1, 3, (1, 2))

            # Create two smaller axes on the right (50% height each)
            ax_le = fig.add_subplot(2, 3, 3)  # Leading edge (top right)
            ax_te = fig.add_subplot(2, 3, 6)  # Trailing edge (bottom right)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax_le = None
            ax_te = None

        # Plot on main axis
        self.section_perimiter.curve.plot(ax=ax, label="Section Curve", color="k")
        self.split_result.upper_curve.plot(ax=ax, label="Upper Curve", color="r")
        self.split_result.lower_curve.plot(ax=ax, label="Lower Curve", color="g")
        self.initial_camber.line.plot(ax=ax, label="Initial Camber", color="b")

        ax.set_title("Initial camber line")
        ax.legend()
        ax.axis("equal")

        # Plot close-up views if requested
        if show_closeups and (ax_le is not None) and (ax_te is not None):
            # Get camber line points for determining leading and trailing edges
            camber_coords = self.initial_camber.line.coords

            # Leading edge is at the start of the camber line
            le_x, le_y = camber_coords[0]
            # Trailing edge is at the end of the camber line
            te_x, te_y = camber_coords[-1]

            # Define zoom window size based on local thickness
            le_thickness = self.initial_thickness.t[0]  # Leading edge thickness
            te_thickness = self.initial_thickness.t[-1]  # Trailing edge thickness
            le_zoom_size = le_thickness
            te_zoom_size = te_thickness

            # Plot leading edge close-up
            for curve, label, color in [
                (self.section_perimiter.curve, "Section Curve", "k"),
                (self.split_result.upper_curve, "Upper Curve", "r"),
                (self.split_result.lower_curve, "Lower Curve", "g"),
                (self.initial_camber.line, "Initial Camber", "b"),
            ]:
                curve.plot(ax=ax_le, color=color)

            ax_le.set_title("Leading Edge")
            ax_le.axis("equal")
            ax_le.set_xlim(le_x - le_zoom_size, le_x + le_zoom_size)
            ax_le.set_ylim(le_y - le_zoom_size, le_y + le_zoom_size)
            ax_le.set_xticks([])
            ax_le.set_yticks([])
            ax_le.grid(True, alpha=0.3)

            # Plot trailing edge close-up
            for curve, label, color in [
                (self.section_perimiter.curve, "Section Curve", "k"),
                (self.split_result.upper_curve, "Upper Curve", "r"),
                (self.split_result.lower_curve, "Lower Curve", "g"),
                (self.initial_camber.line, "Initial Camber", "b"),
            ]:
                curve.plot(ax=ax_te, color=color)

            ax_te.set_title("Trailing Edge")
            ax_te.axis("equal")
            ax_te.set_xlim(te_x - te_zoom_size, te_x + te_zoom_size)
            ax_te.set_ylim(te_y - te_zoom_size, te_y + te_zoom_size)
            ax_te.set_xticks([])
            ax_te.set_yticks([])
            ax_te.grid(True, alpha=0.3)

        fig.tight_layout()

        return ax


class ApproximateCamberCallback(Protocol):
    def __call__(self, progress: CamberProgress) -> bool:
        """Return False to abort iteration."""
        ...


class CamberApproximationError(Exception):
    """Custom exception for errors during camber approximation."""

    pass


def approximate_camber(
    section: SectionPerimiter,
    split_config: SplitConfig | None = None,
    approx_camber_config: ApproximateCamberConfig | None = None,
    callback: ApproximateCamberCallback | None = None,
) -> ApproximateCamberResult:
    # Set defaults
    split_config = split_config or SplitConfig()
    approx_camber_config = approx_camber_config or ApproximateCamberConfig()

    # Split the section into upper and lower curves
    split_result = split_section(section, split_config)
    if not split_result.success:
        return ApproximateCamberResult(
            section=None,
            split_result=split_result,
            initial_camber=None,
            initial_thickness=None,
            success=False,
            iterations=0,
            final_delta=0.0,
            convergence_history=[],
            error_message=split_result.error_message,
            section_perimiter=section,
        )

    # Initialize camber and thickness
    assert split_result.upper_curve is not None, "Expected upper curve to be valid"
    assert split_result.lower_curve is not None, "Expected lower curve to be valid"
    initial_camber, initial_thickness = _initial_guess(
        split_result.upper_curve, split_result.lower_curve
    )

    # TESTING
    return ApproximateCamberResult(
        section=None,  # Set to None until we create the final section
        split_result=split_result,
        initial_camber=initial_camber,
        initial_thickness=initial_thickness,
        success=False,  # Set to False until we run the iteration loop
        iterations=0,
        final_delta=0.0,
        convergence_history=[],
        error_message="",
        section_perimiter=section,
    )

    # # Run iterative improvement
    # loop_result = _run_iteration_loop(
    #     initial_camber, initial_thickness, section, approx_camber_config, callback
    # )

    # # Create final section if successful
    # new_section = None
    # error_message = loop_result.error_message

    # if loop_result.success:
    #     new_section, section_error = _create_final_section_safe(
    #         loop_result.camber, loop_result.thickness, section, approx_camber_config
    #     )
    #     if new_section is None:
    #         loop_result.success = False
    #         error_message = section_error
    #     elif section_error:
    #         error_message = section_error  # Warning message

    # # Final callback notification
    # if callback is not None:
    #     final_progress = CamberProgress(
    #         iteration=loop_result.iterations,
    #         delta=loop_result.final_delta,
    #         camber=loop_result.camber,
    #         converged=loop_result.success,
    #         error=error_message if not loop_result.success else None,
    #     )
    #     callback(final_progress)

    # return ApproximateCamberResult(
    #     section=new_section,
    #     split_result=split_result,
    #     initial_camber=initial_camber,
    #     initial_thickness=initial_thickness,
    #     success=loop_result.success,
    #     iterations=loop_result.iterations,
    #     final_delta=loop_result.final_delta,
    #     convergence_history=loop_result.convergence_history,
    #     error_message=error_message,
    #     section_perimiter=section,
    # )


@dataclass
class CamberIterationResult:
    camber: Camber
    thickness: Thickness
    delta: float
    success: bool
    error_message: str = ""


#################


@dataclass
class IterationLoopResult:
    success: bool
    camber: Camber
    thickness: Thickness
    final_delta: float
    iterations: int
    convergence_history: list[float]
    error_message: str = ""


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

    while iterations < config.max_iterations:
        try:
            iteration_result = improve_camber_robust(camber, thickness, section, config)

            if not iteration_result.success:
                return IterationLoopResult(
                    success=False,
                    camber=camber,
                    thickness=thickness,
                    final_delta=float("inf"),
                    iterations=iterations,
                    convergence_history=convergence_history,
                    error_message=f"Iteration {iterations} failed: {iteration_result.error_message}",
                )

            camber = iteration_result.camber
            thickness = iteration_result.thickness
            final_delta = iteration_result.delta
            convergence_history.append(final_delta)
            iterations += 1

            # Check convergence
            if final_delta <= config.tolerance:
                return IterationLoopResult(
                    success=True,
                    camber=camber,
                    thickness=thickness,
                    final_delta=final_delta,
                    iterations=iterations,
                    convergence_history=convergence_history,
                )

            # Handle callback
            if callback is not None:
                progress = CamberProgress(iterations, final_delta, camber, False, None)
                if not callback(progress):
                    return IterationLoopResult(
                        success=False,
                        camber=camber,
                        thickness=thickness,
                        final_delta=final_delta,
                        iterations=iterations,
                        convergence_history=convergence_history,
                        error_message="Cancelled by user callback",
                    )

        except Exception as e:
            return IterationLoopResult(
                success=False,
                camber=camber,
                thickness=thickness,
                final_delta=(
                    convergence_history[-1] if convergence_history else float("inf")
                ),
                iterations=iterations,
                convergence_history=convergence_history,
                error_message=f"Unexpected error at iteration {iterations}: {str(e)}",
            )

    # Max iterations reached
    return IterationLoopResult(
        success=False,
        camber=camber,
        thickness=thickness,
        final_delta=convergence_history[-1] if convergence_history else float("inf"),
        iterations=iterations,
        convergence_history=convergence_history,
        error_message=f"Failed to converge after {config.max_iterations} iterations",
    )


def _create_final_section_safe(
    camber: Camber,
    thickness: Thickness,
    section: SectionPerimiter,
    config: ApproximateCamberConfig,
) -> tuple[Section | None, str]:
    """Create final section with safe error handling."""
    try:
        extend_result = extend_camber_line(camber, section, config.extension_factor)
        if not extend_result.success or not extend_result.extended_camber:
            return None, f"Failed to extend camber line: {extend_result.error_message}"

        final_section_result = create_final_section(
            extend_result.extended_camber, thickness, section, config
        )

        if not final_section_result.success:
            return (
                None,
                f"Failed to create final section: {final_section_result.error_message}",
            )

        return final_section_result.section, final_section_result.error_message

    except Exception as e:
        return None, f"Unexpected error in final section creation: {str(e)}"


def improve_camber_robust(
    camber: Camber,
    thickness: Thickness,
    section: SectionPerimiter,
    config: ApproximateCamberConfig,
) -> CamberIterationResult:
    """Robust version of camber improvement with proper error handling."""
    try:
        N = len(thickness.t)
        upper_coords = np.zeros((N, 2))
        lower_coords = np.zeros((N, 2))

        normal = camber.line.signed_normal()

        successful_intersections = 0
        for i in range(N):
            c = camber.line.coords[i]
            u = c + config.thickness_scale * thickness.t[i] * normal.coords[i]
            l = c - config.thickness_scale * thickness.t[i] * normal.coords[i]

            # Try to find intersections with error handling
            int_u = section.curve.intersect_coords(np.c_[c, u].T)
            int_l = section.curve.intersect_coords(np.c_[l, c].T)

            # Handle multiple or no intersections gracefully
            if int_u.shape[0] == 1 and int_l.shape[0] == 1:
                upper_coords[i] = int_u[0]
                lower_coords[i] = int_l[0]
                successful_intersections += 1
            elif int_u.shape[0] > 1 or int_l.shape[0] > 1:
                # Take closest intersection
                if int_u.shape[0] > 1:
                    distances = np.linalg.norm(int_u - u, axis=1)
                    upper_coords[i] = int_u[np.argmin(distances)]
                else:
                    upper_coords[i] = int_u[0]

                if int_l.shape[0] > 1:
                    distances = np.linalg.norm(int_l - l, axis=1)
                    lower_coords[i] = int_l[np.argmin(distances)]
                else:
                    lower_coords[i] = int_l[0]
                successful_intersections += 1
            else:
                # No intersection found - keep previous position
                upper_coords[i] = (
                    camber.line.coords[i] + thickness.t[i] * normal.coords[i]
                )
                lower_coords[i] = (
                    camber.line.coords[i] - thickness.t[i] * normal.coords[i]
                )

        if successful_intersections < N * 0.8:  # Less than 80% success rate
            return CamberIterationResult(
                camber=camber,
                thickness=thickness,
                delta=float("inf"),
                success=False,
                error_message=f"Only {successful_intersections}/{N} intersections found",
            )

        # Create new camber line
        camber_coords = (upper_coords + lower_coords) / 2
        new_camber = Camber(PlaneCurve.new_unit_speed(camber_coords))
        new_thickness_values = np.linalg.norm(upper_coords - lower_coords, axis=1)
        new_thickness = Thickness(thickness.s, new_thickness_values, new_camber.chord)

        # Apply relaxation
        delta_coords = (
            new_camber.line.coords - camber.line.coords
        ) * config.relax_factor
        relaxed_coords = camber.line.coords + delta_coords
        final_camber = Camber(PlaneCurve(relaxed_coords, camber.line.param))

        delta = np.sum(np.linalg.norm(delta_coords, axis=1))

        return CamberIterationResult(
            camber=final_camber, thickness=new_thickness, delta=delta, success=True
        )

    except Exception as e:
        return CamberIterationResult(
            camber=camber,
            thickness=thickness,
            delta=float("inf"),
            success=False,
            error_message=str(e),
        )


@dataclass
class FinalSectionResult:
    section: Section | None
    success: bool
    error_message: str = ""


def create_final_section(
    extended_camber: Camber,
    thickness: Thickness,
    section: SectionPerimiter,
    config: ApproximateCamberConfig,
) -> FinalSectionResult:
    """Create final section with extended camber and refined point distribution."""
    try:
        camber_line = extended_camber.line

        # Add refined point distribution near LE and TE
        t = camber_line.param
        if len(t) < 3:
            return FinalSectionResult(
                section=None,
                success=False,
                error_message="Camber line has insufficient points for refinement",
            )

        spacing_le = np.cos(np.linspace(0, np.pi / 2, config.num_points_le))
        t_new_le = t[1] - (t[1] - t[0]) * spacing_le

        spacing_te = np.cos(np.linspace(np.pi / 2, 0, config.num_points_te))
        t_new_te = t[-2] + (t[-1] - t[-2]) * spacing_te

        t_new = np.r_[t_new_le, t[2:-2], t_new_te]

        # Validate parameter range
        if np.any(t_new < 0) or np.any(t_new > 1):
            return FinalSectionResult(
                section=None,
                success=False,
                error_message="Refined parameter values outside valid range [0,1]",
            )

        refined_camber_line = camber_line.interpolate(t_new, k=2)
        refined_camber = Camber(refined_camber_line)
        final_chord = refined_camber.chord

        # Extend thickness array to match new parameterization
        if len(thickness.t) == 0:
            return FinalSectionResult(
                section=None,
                success=False,
                error_message="Empty thickness array provided",
            )

        extended_thickness_values = np.r_[
            np.repeat(thickness.t[0], config.num_points_le),
            thickness.t[1:-1] if len(thickness.t) > 2 else [thickness.t[0]],
            np.repeat(thickness.t[-1], config.num_points_te),
        ]

        # Re-calculate thicknesses with robust error handling
        if len(refined_camber_line.coords) < 3:
            return FinalSectionResult(
                section=None,
                success=False,
                error_message="Refined camber has insufficient points",
            )

        camber_no_ends = Camber(refined_camber_line[1:-1])
        extended_thickness_no_ends = Thickness(
            t_new[1:-1], extended_thickness_values[1:-1], None
        )
        iter_result = improve_camber_robust(
            camber_no_ends, extended_thickness_no_ends, section, config
        )

        if not iter_result.success:
            # Fallback: use original thickness distribution
            final_thickness_values = np.r_[0, extended_thickness_values[1:-1], 0]
            final_thickness = Thickness(t_new, final_thickness_values, final_chord)
            return FinalSectionResult(
                section=Section(
                    thickness=final_thickness,
                    camber=Camber(refined_camber_line),
                    stream_line=section.stream_line,
                ),
                success=False,
                error_message=f"Warning: Used fallback thickness due to: {iter_result.error_message}",
            )

        # Success case: use recalculated thickness
        final_thickness_values = np.r_[0, iter_result.thickness.t, 0]
        final_thickness = Thickness(t_new, final_thickness_values, final_chord)

        return FinalSectionResult(
            section=Section(
                thickness=final_thickness,
                camber=Camber(refined_camber_line),
                stream_line=section.stream_line,
            ),
            success=True,
        )

    except Exception as e:
        return FinalSectionResult(
            section=None,
            success=False,
            error_message=f"Unexpected error in create_final_section: {str(e)}",
        )
