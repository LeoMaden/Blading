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


################# LE and TE detection #################


@dataclass
class SplitConfig:
    curvature_threshold_factor: float = 5
    min_segment_length: int = 10


@dataclass
class SplitSectionResult:
    upper_curve: PlaneCurve | None
    lower_curve: PlaneCurve | None
    success: bool

    section: SectionPerimiter
    segments: list[tuple[int, int]]  # List of (start, end) indices for segments
    curvature_threshold: float
    s_data: NDArray
    curvature_data: NDArray  # For diagnostics
    error_message: str = ""

    def plot_curvature(self, ax=None):
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
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))
        self.plot_curvature(ax1)
        self.plot_section(ax2)


def threshold_masking(curvature, threshold_factor: float = 3):
    """
    Mask high curvature regions based on a threshold

    Parameters:
    curvature: array of curvature values
    threshold_factor: multiplier for median-based threshold
    """
    # Use median-based threshold to handle log-scale data
    median_curvature = np.median(curvature)
    threshold = median_curvature * threshold_factor

    # Create mask: True for low curvature, False for high curvature
    low_curvature_mask = curvature < threshold

    return low_curvature_mask, threshold


def find_continuous_segments(mask, min_length=10):
    """
    Find continuous segments of True values in the mask

    Parameters:
    mask: boolean array
    min_length: minimum length of segments to keep
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


def split_section(
    section: SectionPerimiter,
    config: SplitConfig | None = None,
):
    if config is None:
        config = SplitConfig()

    # Ensure perimiter is going clockwise and starting from (near) the leading edge
    # !todo

    # Calculate normalised arc length and curvature
    s = section.curve.reparameterise_unit().normalise().param
    curvature = section.curve.curvature()

    # Apply threshold masking
    low_curvature_mask, threshold = threshold_masking(
        curvature,
        config.curvature_threshold_factor,
    )

    # Find continuous segments of low curvature
    segments = find_continuous_segments(
        low_curvature_mask,
        config.min_segment_length,
    )

    if len(segments) == 2:
        success = True
        error_message = ""

        upper_coords = section.curve.coords[segments[0][0] : segments[0][1]]
        lower_coords = section.curve.coords[segments[1][0] : segments[1][1]][::-1]

        upper_curve = PlaneCurve.new_unit_speed(upper_coords)
        lower_curve = PlaneCurve.new_unit_speed(lower_coords)
    else:
        success = False
        error_message = (
            "Failed to split section into exactly two segments. "
            f"Found {len(segments)} segments."
        )

        upper_curve = None
        lower_curve = None

    return SplitSectionResult(
        upper_curve=upper_curve,
        lower_curve=lower_curve,
        success=success,
        section=section,
        segments=segments,
        curvature_threshold=threshold,
        s_data=s,
        curvature_data=curvature,
        error_message=error_message,
    )


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


################# Camber Approximation #################


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
    section: Section | None
    split_result: SplitSectionResult
    success: bool
    iterations: int
    final_delta: float
    convergence_history: list[float]
    error_message: str = ""

    def unwrap(self) -> Section:
        if not self.success or self.section is None:
            raise CamberApproximationError(self.error_message)
        return self.section


class ApproximateCamberCallback(Protocol):
    def __call__(self, progress: CamberProgress) -> bool:
        """Return False to abort iteration."""
        ...


class CamberApproximationError(Exception):
    """Custom exception for errors during camber approximation."""

    pass


@dataclass
class IterationLoopResult:
    success: bool
    camber: Camber
    thickness: NDArray
    final_delta: float
    iterations: int
    convergence_history: list[float]
    error_message: str = ""


def _run_iteration_loop(
    camber: Camber,
    thickness: NDArray,
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
    thickness: NDArray,
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
            success=False,
            iterations=0,
            final_delta=0.0,
            convergence_history=[],
            error_message=split_result.error_message,
        )

    # Initialize camber and thickness
    camber = initial_camber_guess(split_result.upper_curve, split_result.lower_curve)  # type: ignore
    thickness = initial_thickness_guess(split_result.upper_curve, split_result.lower_curve)  # type: ignore

    # Run iterative improvement
    loop_result = _run_iteration_loop(
        camber, thickness, section, approx_camber_config, callback
    )

    # Create final section if successful
    new_section = None
    error_message = loop_result.error_message

    if loop_result.success:
        new_section, section_error = _create_final_section_safe(
            loop_result.camber, loop_result.thickness, section, approx_camber_config
        )
        if new_section is None:
            loop_result.success = False
            error_message = section_error
        elif section_error:
            error_message = section_error  # Warning message

    # Final callback notification
    if callback is not None:
        final_progress = CamberProgress(
            iteration=loop_result.iterations,
            delta=loop_result.final_delta,
            camber=loop_result.camber,
            converged=loop_result.success,
            error=error_message if not loop_result.success else None,
        )
        callback(final_progress)

    return ApproximateCamberResult(
        section=new_section,
        split_result=split_result,
        success=loop_result.success,
        iterations=loop_result.iterations,
        final_delta=loop_result.final_delta,
        convergence_history=loop_result.convergence_history,
        error_message=error_message,
    )


@dataclass
class CamberIterationResult:
    camber: Camber
    thickness: NDArray
    delta: float
    success: bool
    error_message: str = ""


def initial_thickness_guess(
    upper_curve: PlaneCurve, lower_curve: PlaneCurve
) -> NDArray:
    """Calculate initial thickness as distance between upper and lower curves."""
    upper_curve = upper_curve.reparameterise_unit().normalise()
    lower_curve = lower_curve.reparameterise_unit().normalise()

    if len(upper_curve.param) > len(lower_curve.param):
        lower_curve = lower_curve.interpolate(upper_curve.param, k=2)
    else:
        upper_curve = upper_curve.interpolate(lower_curve.param, k=2)

    thickness = np.linalg.norm(upper_curve.coords - lower_curve.coords, axis=1)
    return thickness


def improve_camber_robust(
    camber: Camber,
    thickness: NDArray,
    section: SectionPerimiter,
    config: ApproximateCamberConfig,
) -> CamberIterationResult:
    """Robust version of camber improvement with proper error handling."""
    try:
        N = len(thickness)
        upper_coords = np.zeros((N, 2))
        lower_coords = np.zeros((N, 2))

        normal = camber.line.signed_normal()

        successful_intersections = 0
        for i in range(N):
            c = camber.line.coords[i]
            u = c + config.thickness_scale * thickness[i] * normal.coords[i]
            l = c - config.thickness_scale * thickness[i] * normal.coords[i]

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
                    camber.line.coords[i] + thickness[i] * normal.coords[i]
                )
                lower_coords[i] = (
                    camber.line.coords[i] - thickness[i] * normal.coords[i]
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
        new_thickness = np.linalg.norm(upper_coords - lower_coords, axis=1)

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
    thickness: NDArray,
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

        refined_camber = camber_line.interpolate(t_new, k=2)

        # Extend thickness array to match new parameterization
        if len(thickness) == 0:
            return FinalSectionResult(
                section=None,
                success=False,
                error_message="Empty thickness array provided",
            )

        extended_thickness = np.r_[
            np.repeat(thickness[0], config.num_points_le),
            thickness[1:-1] if len(thickness) > 2 else [thickness[0]],
            np.repeat(thickness[-1], config.num_points_te),
        ]

        # Re-calculate thicknesses with robust error handling
        if len(refined_camber.coords) < 3:
            return FinalSectionResult(
                section=None,
                success=False,
                error_message="Refined camber has insufficient points",
            )

        camber_no_ends = Camber(refined_camber[1:-1])
        iter_result = improve_camber_robust(
            camber_no_ends, extended_thickness[1:-1], section, config
        )

        if not iter_result.success:
            # Fallback: use original thickness distribution
            final_thickness = np.r_[0, extended_thickness[1:-1], 0]
            return FinalSectionResult(
                section=Section(
                    Camber(refined_camber), final_thickness, section.stream_line
                ),
                success=True,
                error_message=f"Warning: Used fallback thickness due to: {iter_result.error_message}",
            )

        # Success case: use recalculated thickness
        final_thickness = np.r_[0, iter_result.thickness, 0]

        return FinalSectionResult(
            section=Section(
                Camber(refined_camber), final_thickness, section.stream_line
            ),
            success=True,
        )

    except Exception as e:
        return FinalSectionResult(
            section=None,
            success=False,
            error_message=f"Unexpected error in create_final_section: {str(e)}",
        )


def initial_camber_guess(upper_curve: PlaneCurve, lower_curve: PlaneCurve) -> Camber:
    """
    Create an initial guess for the camber line as the average of upper and lower curves.
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

    # Average the coordinates of upper and lower curves
    camber_coords = (upper_curve.coords + lower_curve.coords) / 2
    return Camber(PlaneCurve.new_unit_speed(camber_coords))
