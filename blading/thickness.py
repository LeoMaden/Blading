import logging
from icecream import ic
from dataclasses import dataclass
from functools import cached_property
from numpy.typing import NDArray, ArrayLike
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
from . import shape_space
import matplotlib.pyplot as plt
from circle_fit import taubinSVD
from numpy.polynomial import Polynomial
from typing import Callable, Optional
from scipy.optimize import minimize, fmin, fsolve

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thickness:
    s: NDArray  # Normalised arc length
    t: NDArray  # Thickness distribution
    chord: Optional[float]

    def __post_init__(self):
        if len(self.s) != len(self.t):
            raise ValueError("s and t arrays must have the same length")

        if self.s[0] != 0 or self.s[-1] != 1:
            raise ValueError(
                "s must start at 0 and end at 1. "
                f"Actual values: start={self.s[0]}, end={self.s[-1]}"
            )

    @property
    def t_over_c(self) -> NDArray:
        """Thickness to chord ratio."""
        if self.chord is None:
            raise ValueError("Chord must be defined to calculate t_over_c")
        return self.t / self.chord

    @cached_property
    def t_spline(self) -> BSpline:
        """Cubic BSpline representation of the thickness distribution."""
        return make_interp_spline(self.s, self.t, k=3)

    @cached_property
    def max_t(self) -> float:
        """Maximum thickness value."""
        return float(self.t_spline(self.s_max_t))

    @cached_property
    def s_max_t(self) -> float:
        """Normalised arc length position of maximum thickness."""
        (s_max_t,) = fmin(lambda s: -self.t_spline(s), x0=0.5, disp=False)  # type: ignore
        return s_max_t

    @property
    def t_TE(self) -> float:
        """Thickness at leading edge."""
        return self.t[-1]

    @cached_property
    def shape_space(self) -> NDArray:
        """Shape space representation of the thickness distribution."""
        if not self.has_blunt_TE:
            raise ValueError(
                "Trailing edge must be blunt for shape space representation"
            )
        return shape_space.value(self.s, self.t, self.t_TE)

    @cached_property
    def ss_spline(self) -> BSpline:
        """Cubic BSpline representation of the shape space."""
        return make_interp_spline(self.s, self.shape_space, k=3)

    @cached_property
    def wedge_TE(self) -> float:
        """Wedge angle at trailing edge in radians."""
        grad_TE = (self.t[-1] - self.t[-2]) / (self.s[-1] - self.s[-2])
        return -np.arctan(grad_TE)

    @property
    def has_round_TE(self) -> bool:
        """Check if the trailing edge is round."""
        return bool(np.isclose(self.t_TE, 0).all())

    @property
    def has_blunt_TE(self) -> bool:
        """Check if the trailing edge is blunt (non-zero thickness)."""
        return not self.has_round_TE

    def interpolate(self, s_new: NDArray) -> "Thickness":
        """
        Return a new Thickness object with interpolated values at new normalised arc lengths.

        Parameters
        ----------
        s_new : NDArray
            New normalised arc length array.

        Returns
        -------
        Thickness
            New Thickness object with interpolated values.
        """
        t_new = self.t_spline(s_new)
        return Thickness(s_new, t_new, self.chord)

    def fit_LE_circle(self, s_LE_fit: float = 0.0001) -> tuple[float, float]:
        """
        Fit a circle to the leading edge of the thickness distribution.

        Parameters
        ----------
        s_LE_fit : float
            Normalised arc length up to which to fit the leading edge circle.
            Default is 0.0001 (0.01% of chord).

        Returns
        -------
        tuple[float, float]
            (centre x-coordinate, radius)
        """
        if self.chord is None:
            raise ValueError("Chord must be defined for LE circle fitting")

        # Mask points in the first s_LE_fit
        mask = self.s < s_LE_fit
        if len(mask) < 10:
            msg = f"Fitting leading edge circle with less than 10 points ({len(mask)})."
            logger.warning(msg)

        s = self.s[mask]
        t_over_c = self.t_over_c[mask]

        upper = np.c_[s, 0.5 * t_over_c]
        lower = np.c_[s, -0.5 * t_over_c]
        points = np.r_[upper, lower[::-1]]

        s_centre, t_over_c_centre, rad_over_c, _ = taubinSVD(points)

        # Check if the leading edge circle is centred within 0.01%
        if not np.allclose(t_over_c_centre, 0, atol=1e-4):
            raise RuntimeError(
                "Leading edge circle is not centred. "
                f"Centre at ({s_centre:.4f}, {t_over_c_centre:.4f}) (non-dimensional)"
            )

        rad = rad_over_c * self.chord  # type: ignore
        return s_centre, rad

    def plot(self, ax=None, normalise: bool = False, *plot_args, **plot_kwargs):
        """Plot thickness distribution."""
        if ax is None:
            _, ax = plt.subplots()
        if normalise:
            ax.plot(self.s, self.t_over_c, *plot_args, **plot_kwargs)
            ax.set_ylabel("Thickness-to-chord ratio")
        else:
            ax.plot(self.s, self.t, *plot_args, **plot_kwargs)
            ax.set_ylabel("Thickness")
        ax.set_xlabel("Normalised arc length")
        ax.grid(True)
        return ax

    def plot_shape_space(self, ax=None, *plot_args, **plot_kwargs):
        """Plot shape space representation."""
        if ax is None:
            _, ax = plt.subplots()
        ss = self.shape_space
        ax.plot(self.s, ss, *plot_args, **plot_kwargs)
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Shape space value")
        ax.grid(True)
        return ax

    def with_blunt_TE(self) -> "Thickness":
        """Create a new Thickness with a blunt trailing edge."""
        if self.has_blunt_TE:
            raise ValueError(
                f"Trailing edge is already blunt: thickness is {self.t_TE}"
            )

        # Calculate thickness gradient
        grad = np.gradient(self.t, self.s)
        grad_magnitude = np.abs(grad)

        # Find gradient magnitude at s=0.9 as reference
        ref_grad_magnitude = np.interp(0.9, self.s, grad_magnitude)

        # Create mask for high gradient points (>1.1x reference gradient)
        mask_high_grad = grad_magnitude > 1.1 * ref_grad_magnitude

        # Create mask for fitting region (s > 0.85) excluding high gradient points
        mask_fit_region = (self.s > 0.85) & ~mask_high_grad

        if not np.any(mask_fit_region):
            # No suitable points for fitting, return original
            return Thickness(self.s.copy(), self.t.copy(), self.chord)

        # Fit straight line through fitting region
        poly = Polynomial.fit(self.s[mask_fit_region], self.t[mask_fit_region], deg=1)

        # Find boundary between linear and high gradient regions for continuity
        fit_indices = np.where(mask_fit_region)[0]
        if len(fit_indices) > 0:
            boundary_idx = fit_indices[-1]
            # Translate line to be coincident with remaining curve
            y_curve = self.t[boundary_idx]
            y_line = poly(self.s[boundary_idx])
            poly -= y_line - y_curve

        # Create new thickness array
        new_t = self.t.copy()
        # Replace high gradient points with straight line approximation
        mask_replace = (self.s > 0.85) & mask_high_grad
        new_t[mask_replace] = poly(self.s[mask_replace])

        return Thickness(self.s.copy(), new_t, self.chord)

    def with_round_TE(self) -> "Thickness":
        """Create a new Thickness with a round trailing edge."""
        if self.chord is None:
            raise ValueError("Chord must be defined for round TE calculation")
        if self.has_round_TE:
            raise ValueError("Trailing edge must be blunt to create a round TE.")

        # Thickness relative to chord.
        t_over_c = self.t / self.chord

        # Find trailing edge angle.
        x1, x2 = self.s[[-2, -1]]
        y1, y2 = t_over_c[[-2, -1]]
        tan_angle_TE = (y1 - y2) / (x2 - x1)

        # Trailing edge thickness.
        t_TE = t_over_c[-1]

        # Analytical formula for round TE tangent to linear section.
        a = np.arctan(0.5 * tan_angle_TE)
        b = 2 * np.cos(a) - (1 - np.sin(a)) * tan_angle_TE
        rad_TE = t_TE / b

        # Create points on round TE and interpolate given `s` distribution.
        pts = np.linspace(a, np.pi / 2, 70)
        x_te = 1 - rad_TE + rad_TE * np.sin(pts)
        y_te = 2 * rad_TE * np.cos(pts)
        mask_round_TE = self.s > min(x_te)

        # Create new thickness array
        new_t_over_c = t_over_c.copy()
        new_t_over_c[mask_round_TE] = np.interp(self.s[mask_round_TE], x_te, y_te)
        new_t = new_t_over_c * self.chord

        return Thickness(self.s.copy(), new_t, self.chord)


################################################################################
##################### Thickness parameters definitions #########################
################################################################################


@dataclass
class MeasuredThicknessParams:
    """Parameters that can be directly measured from a blade section."""

    radius_LE: float
    s_max_t: float
    max_t: float
    curv_max_t: float
    thickness_TE: float
    wedge_TE: float
    ss_stretch_join: float
    ss_grad_TE: float
    chord: Optional[float]
    round_TE: bool


@dataclass
class FittedThicknessParams:
    """Parameters determined during thickness fitting process."""

    stretch_amount: float
    stretch_join: float
    knot_positions: tuple[float, float, float, float]
    ss_grad_LE: float


@dataclass
class ThicknessParams:
    """Complete set of thickness parameters needed to create a thickness distribution."""

    measured: MeasuredThicknessParams
    fitted: FittedThicknessParams


def measure_thickness(
    thickness: Thickness, stretch_join: float
) -> MeasuredThicknessParams:
    """Measure thickness parameters that can be directly extracted from a blade section."""

    # Assume trailing edge is round if t_TE is zero.
    round_TE = thickness.has_round_TE

    # Remove round trailing edge if it exists.
    if round_TE:
        thickness = thickness.with_blunt_TE()

    s, t = thickness.s, thickness.t

    # Thickness curvature at maximum thickness.
    grad = np.gradient(t, s)
    curv = np.gradient(grad, s)
    curv_spline = make_interp_spline(s, curv)
    curv_max_t = curv_spline(thickness.s_max_t)

    # Fit circle to leading edge to get radius.
    _, radius_LE = thickness.fit_LE_circle()

    # Shape space vlaue at stretch join.
    ss_spline = thickness.ss_spline
    ss_stretch_join = float(ss_spline(stretch_join))

    # Shape space gradient at trailing edge.
    ss = thickness.shape_space
    ss_grad = np.gradient(ss, s)
    ss_grad_spline = make_interp_spline(s, ss_grad)
    ss_grad_TE = ss_grad_spline(1)

    return MeasuredThicknessParams(
        radius_LE=radius_LE,
        s_max_t=thickness.s_max_t,
        max_t=thickness.max_t,
        curv_max_t=curv_max_t,
        thickness_TE=thickness.t_TE,
        wedge_TE=thickness.wedge_TE,
        ss_stretch_join=ss_stretch_join,
        ss_grad_TE=ss_grad_TE,
        chord=thickness.chord,
        round_TE=round_TE,
    )


################################################################################
####################### Create thickness from parameters #######################
################################################################################


@dataclass(frozen=True)
class ThicknessResult:
    """Result of creating a thickness distribution from a set of parameters."""

    ss_spline: BSpline
    stretch_func: Callable[[NDArray], NDArray]
    ss_control_points: NDArray
    params: ThicknessParams  # Parameters used to create this thickness distribution.

    def eval(self, s: ArrayLike) -> Thickness:
        """Evaluate thickness distribution at the given normalised arc length `s`."""
        s = np.asarray(s)

        s_stretch = self.stretch_func(s)
        ss = self.ss_spline(s_stretch)
        t = shape_space.inverse(s, ss, self.params.measured.thickness_TE)
        chord = self.params.measured.chord

        thickness = Thickness(s, t, chord)

        # Add back round trailing edge if it was specified.
        if self.params.measured.round_TE:
            thickness = thickness.with_round_TE()

        return thickness

    # def create_thickness_distribution(self, s: NDArray) -> NDArray:
    #     """Create thickness distribution for given arc length distribution."""
    #     s_stretch = self.stretch_func(s)
    #     ss_fit = self.ss_spline(s_stretch)
    #     return shape_space.inverse(s, ss_fit, self.t_TE)

    # def create_thickness(self, s: NDArray) -> Thickness:
    #     """Create Thickness dataclass for given arc length distribution."""
    #     t = self.create_thickness_distribution(s)
    #     return Thickness(s, t)

    # def create_section(self, camber_line, stream_line=None) -> Section:
    #     """Create a Section with this thickness distribution."""
    #     s = camber_line.param
    #     thickness = self.create_thickness_distribution(s)
    #     return Section(camber_line, thickness, stream_line)

    # def plot_shape_space(self, original_sec: Section = None, ax=None):
    #     """Plot the shape space representation with fit."""
    #     if ax is None:
    #         _, ax = plt.subplots()

    #     # Plot the fitted spline
    #     s_plot = np.linspace(self.s_LE, self.s_TE, 200)
    #     ss_plot = self.ss_spline(s_plot)
    #     ax.plot(s_plot, ss_plot, "r-", label="Fitted spline", linewidth=2)

    #     # Plot original if provided
    #     if original_sec is not None:
    #         s_orig = original_sec.normalised_arc_length
    #         t_orig = original_sec.thickness
    #         # Calculate original shape space values for plotting
    #         ss_orig = shape_space.value(s_orig, t_orig, t_orig[-1])
    #         s_stretch_orig = self.stretch_func(s_orig)
    #         ax.plot(s_stretch_orig, ss_orig, "k-", label="Original", alpha=0.7)

    #     # Plot key points
    #     ax.plot(self.s_LE, self.ss_LE, "ro", markersize=8, label="Leading edge")
    #     ax.plot(self.s_TE, self.ss_TE, "ro", markersize=8, label="Trailing edge")
    #     ax.plot(self.s_max_t, self.ss_max_t, "go", markersize=8, label="Max thickness")
    #     ax.plot(
    #         self.params.fitted.stretch_join,
    #         self.ss_join,
    #         "bo",
    #         markersize=8,
    #         label="Stretch join",
    #     )

    #     # Plot knots
    #     degree = self.ss_spline.k
    #     n_coef = len(self.ss_spline.c)
    #     knot_avg = np.array(
    #         [self.ss_spline.t[i : i + degree + 1].mean() for i in range(n_coef)]
    #     )
    #     ax.plot(
    #         knot_avg,
    #         self.ss_spline.c,
    #         "bx",
    #         markersize=10,
    #         markeredgewidth=2,
    #         label="Control points",
    #     )

    #     ax.set_xlabel("Stretched arc length")
    #     ax.set_ylabel("Shape space value")
    #     ax.set_title("Thickness Shape Space Representation")
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)
    #     return ax

    # def plot_thickness_comparison(self, original_sec: Section, ax=None):
    #     """Plot comparison between original section and fitted thickness."""
    #     if ax is None:
    #         _, ax = plt.subplots()

    #     s_orig = original_sec.normalised_arc_length
    #     t_orig = original_sec.thickness
    #     t_fit = self.create_thickness_distribution(s_orig)

    #     ax.plot(s_orig, t_orig, "k-", label="Original", linewidth=2)
    #     ax.plot(s_orig, t_fit, "r--", label="Fitted", linewidth=2)

    #     self._add_thickness_markers(ax)
    #     self._format_thickness_plot(ax, "Thickness Distribution Comparison")
    #     return ax

    # def plot_thickness_comparison_from_thickness(
    #     self, original_thickness: Thickness, ax=None
    # ):
    #     """Plot comparison between original and fitted thickness using Thickness object."""
    #     if ax is None:
    #         _, ax = plt.subplots()

    #     original_thickness.plot(ax, "k-", label="Original", linewidth=2)
    #     fitted_thickness = self.create_thickness(original_thickness.s)
    #     fitted_thickness.plot(ax, "r--", label="Fitted", linewidth=2)

    #     self._add_thickness_markers(ax)
    #     self._format_thickness_plot(ax, "Thickness Distribution Comparison")
    #     return ax

    # def _add_thickness_markers(self, ax):
    #     """Add key markers to thickness plots."""
    #     ax.axvline(
    #         self.params.measured.s_max_t,
    #         color="g",
    #         linestyle=":",
    #         alpha=0.7,
    #         label="Max thickness",
    #     )
    #     ax.axvline(
    #         self.params.fitted.stretch_join,
    #         color="b",
    #         linestyle=":",
    #         alpha=0.7,
    #         label="Stretch join",
    #     )

    # def _format_thickness_plot(self, ax, title: str):
    #     """Format thickness plot with labels and styling."""
    #     ax.set_xlabel("Normalised arc length")
    #     ax.set_ylabel("Thickness")
    #     ax.set_title(title)
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)

    # def plot_section_comparison(self, original_sec: Section, ax=None):
    #     """Plot comparison between original and fitted blade sections."""
    #     if ax is None:
    #         _, ax = plt.subplots()

    #     # Plot original section
    #     plot_plane_curve(original_sec.upper_and_lower(), ax, "k-", label="Original")

    #     # Create and plot fitted section
    #     fitted_sec = self.create_section(
    #         original_sec.camber_line, original_sec.stream_line
    #     )
    #     plot_plane_curve(fitted_sec.upper_and_lower(), ax, "r--", label="Fitted")

    #     ax.set_title("Blade Section Comparison")
    #     ax.legend()
    #     ax.axis("equal")
    #     ax.grid(True, alpha=0.3)
    #     return ax

    # def plot_fit_summary(self, original_sec: Section):
    #     """Create a comprehensive summary plot of the thickness fit."""
    #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    #     self.plot_shape_space(original_sec, ax1)
    #     self.plot_thickness_comparison(original_sec, ax2)
    #     self.plot_section_comparison(original_sec, ax3)

    #     # Plot LE circle fit
    #     le_circle = fit_LE_circle(original_sec)
    #     self.plot_section_comparison(original_sec, ax4)
    #     le_circle.plot(ax4)
    #     ax4.set_title(f"Leading Edge Circle (R={le_circle.radius:.4f})")

    #     plt.tight_layout()
    #     plt.show()
    #     return fig


def create_thickness(params: ThicknessParams) -> ThicknessResult:
    """Create thickness distribution from unified parameters."""
    stretch_join = params.fitted.stretch_join
    stretch_amount = params.fitted.stretch_amount

    # Leading edge.
    s_LE = -stretch_amount
    ss_LE = np.sqrt(params.measured.radius_LE)

    # Trailing edge.
    s_TE = 1
    ss_TE = np.tan(params.measured.wedge_TE) + params.measured.thickness_TE

    # Maximum thickness.
    s_max_t = params.measured.s_max_t
    max_t = params.measured.max_t
    t_TE = params.measured.thickness_TE
    curv_max_t = params.measured.curv_max_t
    ss_max_t = shape_space.value(s_max_t, max_t, t_TE)
    ss_grad_max_t = shape_space.deriv1(s_max_t, max_t, t_TE, 0)
    ss_curv_max_t = shape_space.deriv2(s_max_t, max_t, t_TE, 0, curv_max_t)

    # Calculate stretching function.
    stretch_func = _get_stretch_func(stretch_amount, stretch_join)

    # Calculate knot positions based on position of maximum thickness and stretch.
    s_transform = make_interp_spline([0, 1, 2], [s_LE, s_max_t, s_TE], k=1)
    s_knots = s_transform(np.asarray(params.fitted.knot_positions))

    # Calculate knot vector.
    knots = np.r_[np.repeat(s_LE, 4), s_knots, np.repeat(s_TE, 4)]
    degree = 3
    n_coef = len(knots) - degree - 1

    # Calculate BSpline basis functions.
    basis: list[BSpline] = []
    identity = np.eye(n_coef)
    for coef_row in identity:
        basis.append(BSpline(knots, coef_row, degree))

    # Functions to apply constraints.
    value_at = lambda s: np.array([b(s) for b in basis])
    grad_at = lambda s: np.array([b.derivative()(s) for b in basis])
    curv_at = lambda s: np.array([b.derivative(2)(s) for b in basis])

    # Define constraints.
    constraints = [
        (value_at(s_LE), ss_LE),
        (value_at(s_TE), ss_TE),
        (value_at(s_max_t), ss_max_t),
        (grad_at(s_max_t), ss_grad_max_t),
        (curv_at(s_max_t), ss_curv_max_t),
        (value_at(stretch_join), params.measured.ss_stretch_join),
        (grad_at(s_TE), params.measured.ss_grad_TE),
        (grad_at(s_LE), params.fitted.ss_grad_LE),
    ]
    assert len(constraints) == n_coef

    # Create matrices and solve for coefficients.
    A = np.c_[*[c[0] for c in constraints]].T
    b = np.r_[*[c[1] for c in constraints]]
    coefs = np.linalg.solve(A, b)

    ss_spline = BSpline(knots, coefs, degree)
    ss_control_points = np.array(
        [
            [s_LE, ss_LE],
            [stretch_join, params.measured.ss_stretch_join],
            [s_max_t, ss_max_t],
            [s_TE, ss_TE],
        ]
    )

    return ThicknessResult(
        ss_spline=ss_spline,
        stretch_func=stretch_func,
        params=params,
        ss_control_points=ss_control_points,
    )


####################################################################
################## Fit thickness parameterisation ##################
####################################################################


@dataclass
class FitThicknessResult:
    """Result of fitting a thickness parameterisation to a given thickness distribution."""

    result: ThicknessResult
    original: Thickness

    def compare_shape_space(self, ax=None):
        """Plot comparison of fitted spline and original shape space representation."""
        result = self.result

        if ax is None:
            _, ax = plt.subplots()

        # Plot fitted spline
        # If the original thickness has a round TE, remove it for shape space comparison
        thickness_fit = self.result.eval(self.original.s)
        if self.result.params.measured.round_TE:
            thickness_fit = thickness_fit.with_blunt_TE()
        thickness_fit.plot_shape_space(ax, "r-", label="Fitted spline")

        # Plot original shape space representation
        # If the original thickness has a round TE, remove it for shape space comparison
        original_thickness = self.original
        if self.original.has_round_TE:
            original_thickness = original_thickness.with_blunt_TE()
        original_thickness.plot_shape_space(ax, "k-", alpha=0.5, label="Original")

        # Un-stretch and plot control points
        s_control_stretched = result.ss_control_points[:, 0]
        stretch_func = self.result.stretch_func
        # Solve for unstretched values of s
        s_control: NDArray = fsolve(
            lambda s: stretch_func(s) - s_control_stretched, s_control_stretched
        )  # type: ignore

        ax.plot(s_control, result.ss_control_points[:, 1], "bo", label="Control points")

        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Shape space value")
        title = "Shape Space Comparison"
        if self.original.has_round_TE:
            title += " (Round TE removed for shape space)"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def compare_thickness(self, ax=None, normalise: bool = True):
        """Plot comparison of fitted and original thickness distributions."""
        if ax is None:
            _, ax = plt.subplots()

        # Plot fitted thickness
        thickness_fit = self.result.eval(self.original.s)
        thickness_fit.plot(ax, normalise, "r-", label="Fitted")

        # Plot original thickness
        self.original.plot(ax, normalise, "k-", label="Original", alpha=0.5)

        ax.set_title("Thickness Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def compare_summary(self, figsize=(12, 5), normalise: bool = True):
        """Create side-by-side comparison with shape space on left and thickness on right."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Shape space comparison on the left
        self.compare_shape_space(ax=ax1)

        # Thickness comparison on the right
        self.compare_thickness(ax=ax2, normalise=normalise)

        plt.tight_layout()
        return fig, (ax1, ax2)


def fit_thickness(thickness: Thickness) -> FitThicknessResult:
    """Fit thickness parameterisation to a blade section."""

    # Initial guess of fitting parameters.
    x0 = [0.1, 0.15, -0.5, 0.05, 0.65, 1, 1.2]

    # Function to calculate parameters from optimisation vector.
    def create_params(x) -> ThicknessParams:
        measured = measure_thickness(thickness, x[1])  # x[1] is stretch_join
        return ThicknessParams(
            measured=measured,
            fitted=FittedThicknessParams(
                stretch_amount=x[0],
                stretch_join=x[1],
                ss_grad_LE=x[2],
                knot_positions=tuple(x[3:]),
            ),
        )

    # Function to create thickness from optimisation vector.
    def create(x) -> ThicknessResult:
        params = create_params(x)
        return create_thickness(params)

    # Function to calculate error between fit and actual in shape space.
    s, t = thickness.s, thickness.t
    # Calculate original shape space values for comparison
    ss = shape_space.value(s, t, t[-1])

    def err(x):
        res = create(x)
        s_stretch = res.stretch_func(s)
        ss_fit = res.ss_spline(s_stretch)
        return np.trapezoid((ss - ss_fit) ** 2, s)

    # Create constraints with validation
    constraints = _create_thickness_fitting_constraints()
    _validate_initial_guess(x0, constraints)
    e0 = err(x0)
    tol = 1e-8 * e0
    opt = minimize(err, x0, tol=tol, constraints=constraints)
    if not opt.success:
        raise RuntimeError(repr(opt))

    x = opt.x
    result = create(x)

    return FitThicknessResult(
        result=result,
        original=thickness,
    )


# Private functions


def _get_stretch_func(amount, join):
    b = [amount, 0, 0, 0]
    A = np.zeros((4, 4))
    x1 = 0
    x2 = join
    A[0, :] = [x1**3, x1**2, x1, 1]
    A[1, :] = [x2**3, x2**2, x2, 1]
    A[2, :] = [3 * x2**2, 2 * x2, 1, 0]
    A[3, :] = [6 * x2, 2, 0, 0]
    coefs = np.linalg.solve(A, b)
    poly = Polynomial(coefs[::-1])

    def stretch(s):
        s_front = s[s <= join]
        s_rear = s[s > join]
        new_s_front = s_front - poly(s_front)
        return np.r_[new_s_front, s_rear]

    return stretch


def _create_thickness_fitting_constraints():
    """Create constraints for thickness fitting optimization."""

    def gt_zero(constraint_func):
        return {"type": "ineq", "fun": constraint_func}

    return [
        gt_zero(lambda x: x[0]),  # stretch_amount > 0
        gt_zero(lambda x: 0.5 - x[0]),  # stretch_amount < 0.5
        gt_zero(lambda x: x[1] - 0.01),  # stretch_join > 0.01
        gt_zero(lambda x: 0.4 - x[1]),  # stretch_join < 0.4
        gt_zero(lambda x: -x[2]),  # ss_grad_LE <= 0
        gt_zero(lambda x: x[4] - x[3] - 0.1),  # knots spaced >= 0.1
        gt_zero(lambda x: x[5] - x[4] - 0.1),
        gt_zero(lambda x: x[6] - x[5] - 0.1),
        gt_zero(lambda x: x[3] - 0.01),  # knot_positions[0] > 0.01
        gt_zero(lambda x: 1.99 - x[6]),  # knot_positions[3] < 1.99
    ]


def _validate_initial_guess(x0, constraints):
    """Validate initial guess against constraints."""
    constraint_names = [
        "stretch_amount > 0",
        "stretch_amount < 0.5",
        "stretch_join > 0.01",
        "stretch_join < 0.4",
        "ss_grad_LE <= 0",
        "knot[1] - knot[0] > 0.1",
        "knot[2] - knot[1] > 0.1",
        "knot[3] - knot[2] > 0.1",
        "knot[0] > 0.01",
        "knot[3] < 1.99",
    ]

    violated_constraints = []
    for i, constraint in enumerate(constraints):
        if constraint["fun"](x0) <= 0:
            violated_constraints.append(constraint_names[i])

    if violated_constraints:
        raise ValueError(
            f"Initial guess violates constraints: {', '.join(violated_constraints)}"
        )
