from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# from .section import Section
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
from . import shape_space
import matplotlib.pyplot as plt
from circle_fit import taubinSVD
from numpy.polynomial import Polynomial
from typing import Callable
from scipy.optimize import minimize, fmin
from geometry.curves import plot_plane_curve

Section = NDArray  # Placeholder for Section type


@dataclass
class Thickness:
    s: NDArray  # Normalised arc length
    t: NDArray  # Thickness distribution

    def __post_init__(self):
        if len(self.s) != len(self.t):
            raise ValueError("s and t arrays must have the same length")

    @property
    def max_thickness(self) -> float:
        """Maximum thickness value."""
        return np.max(self.t)

    @property
    def max_thickness_position(self) -> float:
        """Arc length position of maximum thickness."""
        max_idx = np.argmax(self.t)
        return self.s[max_idx]

    @property
    def thickness_at_TE(self) -> float:
        """Thickness at trailing edge."""
        return self.t[-1]

    @property
    def thickness_at_LE(self) -> float:
        """Thickness at leading edge."""
        return self.t[0]

    def plot(self, ax=None, *plot_args, **plot_kwargs):
        """Plot thickness distribution."""
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.s, self.t, *plot_args, **plot_kwargs)
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Thickness")
        ax.grid(True)
        return ax

    def plot_with_markers(self, ax=None):
        """Plot thickness distribution with key markers."""
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.s, self.t, "b-", linewidth=2, label="Thickness")

        # Mark maximum thickness
        max_pos = self.max_thickness_position
        max_thick = self.max_thickness
        ax.plot(
            max_pos,
            max_thick,
            "ro",
            markersize=8,
            label=f"Max thickness: {max_thick:.4f}",
        )

        # Mark leading and trailing edges
        ax.plot(
            0,
            self.thickness_at_LE,
            "go",
            markersize=6,
            label=f"LE thickness: {self.thickness_at_LE:.4f}",
        )
        ax.plot(
            1,
            self.thickness_at_TE,
            "mo",
            markersize=6,
            label=f"TE thickness: {self.thickness_at_TE:.4f}",
        )

        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Thickness")
        ax.legend()
        ax.grid(True)
        return ax

    def with_blunt_TE(self) -> "Thickness":
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
            return Thickness(self.s.copy(), self.t.copy())

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

        return Thickness(self.s.copy(), new_t)

    def with_round_TE(self, chord: float = 1) -> "Thickness":
        # Thickness relative to chord.
        t_over_c = self.t / chord

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
        new_t = new_t_over_c * chord

        return Thickness(self.s.copy(), new_t)


@dataclass
class LECircle:
    xc: float
    yc: float
    radius: float
    rms_err: float
    fit_points: NDArray

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        theta = np.linspace(0, 2 * np.pi, 180)
        x = self.xc + self.radius * np.cos(theta)
        y = self.yc + self.radius * np.sin(theta)

        ax.plot(x, y, "b--", label="LE circle")
        ax.plot(*self.fit_points.T, "b.", label="LE points")

        ax.legend()
        plt.axis("equal")

        # Zoom into leading edge region
        x_range = self.radius * 3
        y_range = self.radius * 3
        ax.set_xlim(self.xc - x_range, self.xc + x_range)
        ax.set_ylim(self.yc - y_range, self.yc + y_range)


def fit_LE_circle(sec: Section, num_LE: int = 11):
    i = num_LE // 2 + 1
    LE_points = np.r_[sec.lower_curve().coords[:i][::-1], sec.upper_curve().coords[1:i]]
    xc, yc, radius, rms_err = taubinSVD(LE_points)
    return LECircle(xc, yc, radius, rms_err, LE_points)


def stretch_func(amount, join):
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
    ss_grad_TE: float  # Measured from original thickness


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


def measure_thickness(sec: Section, stretch_join: float) -> MeasuredThicknessParams:
    """Measure thickness parameters that can be directly extracted from a blade section."""
    s = sec.normalised_arc_length

    ##### Thickness #####
    t = sec.thickness

    # Find point of maximum thickness.
    t_spline = make_interp_spline(s, t)
    (s_max_t,) = fmin(lambda s: -t_spline(s), x0=0.5, disp=False)  # type: ignore
    max_t = t_spline(s_max_t)

    # Trailing edge thickness (assuming blunt TE).
    t_TE = t[-1]

    # Wedge angle at TE.
    grad = np.gradient(t, s)
    wedge_TE = -np.arctan(grad[-1])

    # Thickness curvature at maximum thickness.
    curv = np.gradient(grad, s)
    curv_spline = make_interp_spline(s, curv)
    curv_max_t = curv_spline(s_max_t)

    # Leading edge radius.
    LE_circle = fit_LE_circle(sec)
    radius_LE = LE_circle.radius

    ##### Shape space #####

    # Extrapolate LE and TE singularities.
    ss = shape_space.value(s, t, t[-1])
    ss_spline = make_interp_spline(s, ss)
    ss_stretch_join = ss_spline(stretch_join)
    ss_grad = np.gradient(ss, s)
    ss_grad_spline = make_interp_spline(s, ss_grad)
    ss_grad_TE = ss_grad_spline(1)

    return MeasuredThicknessParams(
        radius_LE=radius_LE,
        s_max_t=s_max_t,
        max_t=max_t,
        curv_max_t=curv_max_t,
        thickness_TE=t_TE,
        wedge_TE=wedge_TE,
        ss_stretch_join=ss_stretch_join,
        ss_grad_TE=ss_grad_TE,
    )


@dataclass
class ThicknessResult:
    """Result of thickness parameterisation with improved usability and plotting."""

    spline: BSpline
    stretch: Callable[[NDArray], NDArray]
    params: ThicknessParams
    t_TE: float
    s_LE: float
    ss_LE: float
    s_TE: float
    ss_TE: float
    s_max_t: float
    ss_max_t: float
    ss_join: float

    def create_thickness_distribution(self, s: NDArray) -> NDArray:
        """Create thickness distribution for given arc length distribution."""
        s_stretch = self.stretch(s)
        ss_fit = self.spline(s_stretch)
        return shape_space.inverse(s, ss_fit, self.t_TE)

    def create_thickness(self, s: NDArray) -> Thickness:
        """Create Thickness dataclass for given arc length distribution."""
        t = self.create_thickness_distribution(s)
        return Thickness(s, t)

    def create_section(self, camber_line, stream_line=None) -> Section:
        """Create a Section with this thickness distribution."""
        s = camber_line.param
        thickness = self.create_thickness_distribution(s)
        return Section(camber_line, thickness, stream_line)

    def plot_shape_space(self, original_sec: Section = None, ax=None):
        """Plot the shape space representation with fit."""
        if ax is None:
            _, ax = plt.subplots()

        # Plot the fitted spline
        s_plot = np.linspace(self.s_LE, self.s_TE, 200)
        ss_plot = self.spline(s_plot)
        ax.plot(s_plot, ss_plot, "r-", label="Fitted spline", linewidth=2)

        # Plot original if provided
        if original_sec is not None:
            s_orig = original_sec.normalised_arc_length
            t_orig = original_sec.thickness
            # Calculate original shape space values for plotting
            ss_orig = shape_space.value(s_orig, t_orig, t_orig[-1])
            s_stretch_orig = self.stretch(s_orig)
            ax.plot(s_stretch_orig, ss_orig, "k-", label="Original", alpha=0.7)

        # Plot key points
        ax.plot(self.s_LE, self.ss_LE, "ro", markersize=8, label="Leading edge")
        ax.plot(self.s_TE, self.ss_TE, "ro", markersize=8, label="Trailing edge")
        ax.plot(self.s_max_t, self.ss_max_t, "go", markersize=8, label="Max thickness")
        ax.plot(
            self.params.fitted.stretch_join,
            self.ss_join,
            "bo",
            markersize=8,
            label="Stretch join",
        )

        # Plot knots
        degree = self.spline.k
        n_coef = len(self.spline.c)
        knot_avg = np.array(
            [self.spline.t[i : i + degree + 1].mean() for i in range(n_coef)]
        )
        ax.plot(
            knot_avg,
            self.spline.c,
            "bx",
            markersize=10,
            markeredgewidth=2,
            label="Control points",
        )

        ax.set_xlabel("Stretched arc length")
        ax.set_ylabel("Shape space value")
        ax.set_title("Thickness Shape Space Representation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_thickness_comparison(self, original_sec: Section, ax=None):
        """Plot comparison between original section and fitted thickness."""
        if ax is None:
            _, ax = plt.subplots()

        s_orig = original_sec.normalised_arc_length
        t_orig = original_sec.thickness
        t_fit = self.create_thickness_distribution(s_orig)

        ax.plot(s_orig, t_orig, "k-", label="Original", linewidth=2)
        ax.plot(s_orig, t_fit, "r--", label="Fitted", linewidth=2)

        self._add_thickness_markers(ax)
        self._format_thickness_plot(ax, "Thickness Distribution Comparison")
        return ax

    def plot_thickness_comparison_from_thickness(
        self, original_thickness: Thickness, ax=None
    ):
        """Plot comparison between original and fitted thickness using Thickness object."""
        if ax is None:
            _, ax = plt.subplots()

        original_thickness.plot(ax, "k-", label="Original", linewidth=2)
        fitted_thickness = self.create_thickness(original_thickness.s)
        fitted_thickness.plot(ax, "r--", label="Fitted", linewidth=2)

        self._add_thickness_markers(ax)
        self._format_thickness_plot(ax, "Thickness Distribution Comparison")
        return ax

    def _add_thickness_markers(self, ax):
        """Add key markers to thickness plots."""
        ax.axvline(
            self.params.measured.s_max_t,
            color="g",
            linestyle=":",
            alpha=0.7,
            label="Max thickness",
        )
        ax.axvline(
            self.params.fitted.stretch_join,
            color="b",
            linestyle=":",
            alpha=0.7,
            label="Stretch join",
        )

    def _format_thickness_plot(self, ax, title: str):
        """Format thickness plot with labels and styling."""
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Thickness")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_section_comparison(self, original_sec: Section, ax=None):
        """Plot comparison between original and fitted blade sections."""
        if ax is None:
            _, ax = plt.subplots()

        # Plot original section
        plot_plane_curve(original_sec.upper_and_lower(), ax, "k-", label="Original")

        # Create and plot fitted section
        fitted_sec = self.create_section(
            original_sec.camber_line, original_sec.stream_line
        )
        plot_plane_curve(fitted_sec.upper_and_lower(), ax, "r--", label="Fitted")

        ax.set_title("Blade Section Comparison")
        ax.legend()
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_fit_summary(self, original_sec: Section):
        """Create a comprehensive summary plot of the thickness fit."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        self.plot_shape_space(original_sec, ax1)
        self.plot_thickness_comparison(original_sec, ax2)
        self.plot_section_comparison(original_sec, ax3)

        # Plot LE circle fit
        le_circle = fit_LE_circle(original_sec)
        self.plot_section_comparison(original_sec, ax4)
        le_circle.plot(ax4)
        ax4.set_title(f"Leading Edge Circle (R={le_circle.radius:.4f})")

        plt.tight_layout()
        plt.show()
        return fig


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
    stretch = stretch_func(stretch_amount, stretch_join)

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

    spline = BSpline(knots, coefs, degree)

    return ThicknessResult(
        spline=spline,
        stretch=stretch,
        params=params,
        t_TE=t_TE,
        s_LE=s_LE,
        ss_LE=ss_LE,
        s_TE=s_TE,
        ss_TE=ss_TE,
        s_max_t=s_max_t,
        ss_max_t=ss_max_t,
        ss_join=params.measured.ss_stretch_join,
    )


def fit_thickness(sec: Section, plot_intermediate: bool = False) -> ThicknessResult:
    """Fit thickness parameterisation to a blade section."""

    # Initial guess of fitting parameters.
    x0 = [0.1, 0.15, -0.5, 0.05, 0.65, 1, 1.2]

    # Function to calculate parameters from optimisation vector.
    def create_params(x) -> ThicknessParams:
        measured = measure_thickness(sec, x[1])  # x[1] is stretch_join
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
    s_sec = sec.normalised_arc_length
    t_sec = sec.thickness
    # Calculate original shape space values for comparison
    ss = shape_space.value(s_sec, t_sec, t_sec[-1])

    def err(x):
        res = create(x)
        s_stretch = res.stretch(s_sec)
        ss_fit = res.spline(s_stretch)
        return np.trapezoid((ss - ss_fit) ** 2, s_sec)

    # Create constraints with validation
    constraints = _create_thickness_constraints()
    _validate_initial_guess(x0, constraints)
    e0 = err(x0)
    tol = 1e-4 * e0
    opt = minimize(err, x0, tol=tol, constraints=constraints)
    if not opt.success:
        raise RuntimeError(repr(opt))

    x = opt.x
    result = create(x)

    if plot_intermediate:
        result.plot_fit_summary(sec)

    return result


def _create_thickness_constraints():
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
