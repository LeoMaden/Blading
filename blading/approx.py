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
    curvature_threshold_factor: float = 3
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


################# Camber Approximation #################


@dataclass
class ApproximateCamberConfig:
    pass


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
        pass


class CamberApproximationError(Exception):
    """Custom exception for errors during camber approximation."""

    pass


def approximate_camber(
    section: SectionPerimiter,
    split_config: SplitConfig | None = None,
    approx_camber_config: ApproximateCamberConfig | None = None,
    callback: ApproximateCamberCallback | None = None,
) -> ApproximateCamberResult:
    pass


###############################################


Mask = NDArray[np.bool]


@dataclass(frozen=True)
class CamberIteration:
    lines: NDArray
    upper_intersections: NDArray
    lower_intersections: NDArray
    camber: PlaneCurve
    thickness: NDArray
    delta: float


@dataclass(frozen=True)
class ApproxCamberResult:
    success: bool
    mask_LE: Mask
    mask_TE: Mask
    upper_split: PlaneCurve
    lower_split: PlaneCurve
    camber_iterations: list[CamberIteration]
    n_iter: int
    section: Section | None
    err_msg: str = ""

    def plot(self, orig_sec: SectionPerimiter):
        fig, ax = plt.subplots()
        plot_plane_curve(self.upper_split, ax)
        plot_plane_curve(self.lower_split, ax)
        plot_plane_curve(orig_sec.curve, ax, "k:")
        plot_plane_curve(orig_sec.curve[self.mask_LE], ax, "r.")
        plot_plane_curve(orig_sec.curve[self.mask_TE], ax, "b.")
        plot_plane_curve(self.camber_iterations[-1].camber, ax, "m.-")
        plt.axis("equal")
        plt.show()


def find_LE_TE(
    section: SectionPerimiter,
    thresh_LE: float = 0.5,
    thresh_TE: float = 1.0,
    mask_front: Mask | None = None,
) -> tuple[Mask, Mask]:
    """Split a blade section according to curvature.

    Parameters
    ----------
    section : FlatSection
        The section to split.
    thresh_LE : float, optional
        The numbers of orders of magnitude below the maximum leading edge
        curvature that will be included in the leading edge, by default 0.5
    thresh_TE : float, optional
        The same as `thresh_LE` for the trailing edge, by default 1.0
    mask_front : NDArray[np.bool] | None, optional
        Boolean array marking where the front of the blade is, by default None.
        If this parameter is None, the first and last 25% of the curve defining
        `section` is assumed to be the "front" of the blade.

    Returns
    -------
    tuple[Mask, Mask] = (mask_LE, mask_TE)
        Boolean arrays marking the location of the leading and trailing edges.
    """
    curve = section.curve.reparameterise_unit().normalise()
    k = curve.curvature()

    if mask_front is None:
        x_mid = (min(curve.x) + max(curve.x)) / 2
        mask_front = curve.x < x_mid

    mask_rear = ~mask_front  # type: ignore

    max_k_LE = np.max(k[mask_front])
    max_k_TE = np.max(k[mask_rear])

    mask_LE_k = k > max_k_LE / 10**thresh_LE  # Orders of magnitude below peak
    mask_TE_k = k > max_k_TE / 10**thresh_TE

    mask_LE = mask_front & mask_LE_k
    mask_TE = mask_rear & mask_TE_k

    return mask_LE, mask_TE


def improve_camber(
    camber: PlaneCurve,
    thickness: NDArray,
    section: SectionPerimiter,
    relax_factor: float = 0.2,
) -> CamberIteration:
    N = len(thickness)
    upper_coords = np.zeros((N, 2))
    lower_coords = np.zeros((N, 2))

    lines = np.zeros((N, 3, 2))

    normal = camber.signed_normal()
    _THICKNESS_SCALE = 1.2

    for i in range(N):
        c = camber.coords[i]
        u = c + _THICKNESS_SCALE * thickness[i] * normal.coords[i]
        l = c - _THICKNESS_SCALE * thickness[i] * normal.coords[i]
        int_u = section.curve.intersect_coords(np.c_[c, u].T)
        int_l = section.curve.intersect_coords(np.c_[l, c].T)
        assert int_u.shape[0] == 1, "Too many intersections"
        assert int_l.shape[0] == 1, "Too many intersections"
        upper_coords[i] = int_u[0]
        lower_coords[i] = int_l[0]
        lines[i] = np.c_[l, c, u].T

    camber_new = PlaneCurve((upper_coords + lower_coords) / 2, camber.param)
    thickness_new = np.linalg.norm(upper_coords - lower_coords, axis=1)

    # Relaxation
    delta = (camber_new - camber) * relax_factor
    camber_new = camber + delta

    return CamberIteration(
        lines=lines,
        upper_intersections=upper_coords,
        lower_intersections=lower_coords,
        camber=camber_new,
        thickness=thickness_new,
        delta=np.sum(delta.norm()),
    )


def extend(camber_line: PlaneCurve, section: SectionPerimiter) -> PlaneCurve:
    # Extend leading and trailing edges
    chord = camber_line.length()
    tangent = camber_line.tangent()
    LE_line = np.c_[
        camber_line.start() - tangent.coords[0] * 0.1 * chord, camber_line.start()
    ].T
    LE_point = section.curve.intersect_coords(LE_line)
    TE_line = np.c_[
        camber_line.end() + tangent.coords[-1] * 0.1 * chord, camber_line.end()
    ].T
    TE_point = section.curve.intersect_coords(TE_line)
    assert LE_point.shape[0] == 1, "Too many intersections"
    assert TE_point.shape[0] == 1, "Too many intersections"
    new_coords = np.r_[LE_point, camber_line.coords, TE_point]
    camber_line = PlaneCurve.new_unit_speed(new_coords).normalise()
    return camber_line


def approx_camber_line(
    section: SectionPerimiter,
    tol: float = 1e-7,
    thresh_LE: float = 0.5,
    thresh_TE: float = 1.0,
    mask_front: Mask | None = None,
    relax_factor: float = 0.2,
    num_points_LE: int = 25,
    num_points_TE: int = 25,
) -> ApproxCamberResult:
    mask_LE, mask_TE = find_LE_TE(section, thresh_LE, thresh_TE, mask_front)
    not_LE_TE = ~mask_LE & ~mask_TE
    diff = np.diff(not_LE_TE.astype(int))
    starts = np.where(diff == 1)[0] + 1  # start of 1s
    ends = np.where(diff == -1)[0] + 1  # end of 1s

    upper = section.curve[starts[0] : ends[0]]
    lower = section.curve[starts[1] : ends[1]][::-1]

    # Constant speed normalised with same parameter
    upper = upper.reparameterise_unit().normalise()
    lower = lower.reparameterise_unit().normalise().interpolate(upper.param)
    camber_guess = (upper + lower) / 2
    thickness_guess = (upper - lower).norm()

    camber_iters: list[CamberIteration] = []
    while True:
        try:
            iter = improve_camber(
                camber_guess,
                thickness_guess,
                section,
                relax_factor,
            )
        except Exception as e:
            return ApproxCamberResult(
                success=False,
                mask_LE=mask_LE,
                mask_TE=mask_TE,
                upper_split=upper,
                lower_split=lower,
                camber_iterations=camber_iters,
                n_iter=len(camber_iters),
                section=None,
                err_msg=str(e),
            )

        camber_iters.append(iter)
        if iter.delta <= tol:
            break
        camber_guess = iter.camber
        thickness_guess = iter.thickness

    camber_new = extend(iter.camber, section)
    thickness_new = iter.thickness

    # Add points near leading edge
    t = camber_new.param
    spacing_LE = np.cos(np.linspace(0, np.pi / 2, num_points_LE))
    t_new_LE = t[1] - (t[1] - t[0]) * spacing_LE

    spacing_TE = np.cos(np.linspace(np.pi / 2, 0, num_points_TE))
    t_new_TE = t[-2] + (t[-1] - t[-2]) * spacing_TE

    t_new = np.r_[t_new_LE, t[2:-2], t_new_TE]
    camber_new = camber_new.interpolate(t_new, k=2)

    # Ensure thickness matches camber
    thickness_new = np.r_[
        np.repeat(thickness_new[0], num_points_LE),
        thickness_new[1:-1],
        np.repeat(thickness_new[-1], num_points_TE),
    ]
    iter = improve_camber(camber_new[1:-1], thickness_new[1:-1], section)
    thickness_new = np.r_[0, iter.thickness, 0]

    section_new = Section(camber_new, thickness_new, section.stream_line)

    return ApproxCamberResult(
        success=True,
        mask_LE=mask_LE,
        mask_TE=mask_TE,
        upper_split=upper,
        lower_split=lower,
        camber_iterations=camber_iters,
        n_iter=len(camber_iters),
        section=section_new,
    )
