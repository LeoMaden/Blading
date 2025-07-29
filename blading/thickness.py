from dataclasses import dataclass
from numpy.typing import NDArray
from .section import Section
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
from . import shape_space
import matplotlib.pyplot as plt
from circle_fit import taubinSVD
from numpy.polynomial import Polynomial
from typing import Callable
from scipy.optimize import minimize, fmin
from geometry.curves import plot_plane_curve


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
    ss_grad_LE: float  # Measured but typically not used due to steepness


@dataclass
class ThicknessParams:
    """Complete set of thickness parameters needed to create a thickness distribution."""

    radius_LE: float
    s_max_t: float
    max_t: float
    curv_max_t: float
    thickness_TE: float
    wedge_TE: float
    ss_stretch_join: float
    ss_grad_TE: float
    stretch_amount: float
    stretch_join: float
    knot_positions: tuple[float, float, float, float]
    ss_grad_LE: float

    @classmethod
    def from_measured_and_fit(
        cls,
        measured: MeasuredThicknessParams,
        stretch_amount: float,
        stretch_join: float,
        knot_positions: tuple[float, float, float, float],
        ss_grad_LE: float,
    ) -> "ThicknessParams":
        """Create ThicknessParams from measured parameters and fit parameters."""
        return cls(
            radius_LE=measured.radius_LE,
            s_max_t=measured.s_max_t,
            max_t=measured.max_t,
            curv_max_t=measured.curv_max_t,
            thickness_TE=measured.thickness_TE,
            wedge_TE=measured.wedge_TE,
            ss_stretch_join=measured.ss_stretch_join,
            ss_grad_TE=measured.ss_grad_TE,
            stretch_amount=stretch_amount,
            stretch_join=stretch_join,
            knot_positions=knot_positions,
            ss_grad_LE=ss_grad_LE,
        )


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
    ss_grad_LE = ss_grad_spline(0)

    return MeasuredThicknessParams(
        radius_LE=radius_LE,
        s_max_t=s_max_t,
        max_t=max_t,
        curv_max_t=curv_max_t,
        thickness_TE=t_TE,
        wedge_TE=wedge_TE,
        ss_stretch_join=ss_stretch_join,
        ss_grad_TE=ss_grad_TE,
        ss_grad_LE=ss_grad_LE,
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
            self.params.stretch_join,
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
        """Plot comparison between original and fitted thickness."""
        if ax is None:
            _, ax = plt.subplots()

        s_orig = original_sec.normalised_arc_length
        t_orig = original_sec.thickness
        t_fit = self.create_thickness_distribution(s_orig)

        ax.plot(s_orig, t_orig, "k-", label="Original", linewidth=2)
        ax.plot(s_orig, t_fit, "r--", label="Fitted", linewidth=2)

        # Highlight key points
        ax.axvline(
            self.params.s_max_t,
            color="g",
            linestyle=":",
            alpha=0.7,
            label="Max thickness",
        )
        ax.axvline(
            self.params.stretch_join,
            color="b",
            linestyle=":",
            alpha=0.7,
            label="Stretch join",
        )

        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Thickness")
        ax.set_title("Thickness Distribution Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

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
    stretch_join = params.stretch_join
    stretch_amount = params.stretch_amount

    # Leading edge.
    s_LE = -stretch_amount
    ss_LE = np.sqrt(params.radius_LE)

    # Trailing edge.
    s_TE = 1
    ss_TE = np.tan(params.wedge_TE) + params.thickness_TE

    # Maximum thickness.
    s_max_t = params.s_max_t
    max_t = params.max_t
    t_TE = params.thickness_TE
    curv_max_t = params.curv_max_t
    ss_max_t = shape_space.value(s_max_t, max_t, t_TE)
    ss_grad_max_t = shape_space.deriv1(s_max_t, max_t, t_TE, 0)
    ss_curv_max_t = shape_space.deriv2(s_max_t, max_t, t_TE, 0, curv_max_t)

    # Calculate stretching function.
    stretch = stretch_func(stretch_amount, stretch_join)

    # Calculate knot positions based on position of maximum thickness and stretch.
    s_transform = make_interp_spline([0, 1, 2], [s_LE, s_max_t, s_TE], k=1)
    s_knots = s_transform(np.asarray(params.knot_positions))

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
        (value_at(stretch_join), params.ss_stretch_join),
        (grad_at(s_TE), params.ss_grad_TE),
        (grad_at(s_LE), params.ss_grad_LE),
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
        ss_join=params.ss_stretch_join,
    )


def fit_thickness(sec: Section, plot_intermediate: bool = False) -> ThicknessResult:
    """Fit thickness parameterisation to a blade section."""

    # Initial guess of fitting parameters.
    x0 = [0.1, 0.15, -0.5, 0.05, 0.65, 1, 1.2]

    # Function to calculate parameters from optimisation vector.
    def create_params(x) -> ThicknessParams:
        measured = measure_thickness(sec, x[1])  # x[1] is stretch_join
        return ThicknessParams.from_measured_and_fit(
            measured=measured,
            stretch_amount=x[0],
            stretch_join=x[1],
            knot_positions=tuple(x[3:]),
            ss_grad_LE=x[2],  # Use fitted value, not measured
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

    def gt_zero(constraint_func):
        """Helper function to create scipy constraint dict from lambda"""
        return {"type": "ineq", "fun": constraint_func}

    # Constraints on optimisation variables.
    constraints = [
        gt_zero(lambda x: x[0]),  # stretch_amount must be > 0
        gt_zero(lambda x: 0.5 - x[0]),  # stretch_amount must be < 0.5
        gt_zero(lambda x: x[1] - 0.01),  # stretch_join must be > 0.01
        gt_zero(lambda x: 0.4 - x[1]),  # stretch_join must be < 0.4
        gt_zero(lambda x: -x[2]),  # ss_grad_LE must be <= 0
        gt_zero(lambda x: x[4] - x[3] - 0.1),  # knots must be at least 0.1 apart
        gt_zero(lambda x: x[5] - x[4] - 0.1),  # knots must be at least 0.1 apart
        gt_zero(lambda x: x[6] - x[5] - 0.1),  # knots must be at least 0.1 apart
        gt_zero(lambda x: x[3] - 0.01),  # knot_positions[0] must be > 0.01
        gt_zero(lambda x: 1.99 - x[6]),  # knot_positions[3] must be < 1.99
    ]

    # Validate initial guess against constraints
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
        value = constraint["fun"](x0)
        if value <= 0:
            violated_constraints.append(f"{constraint_names[i]} (value: {value:.3f})")

    if violated_constraints:
        raise ValueError(
            f"Initial guess x0={x0} violates constraints: {', '.join(violated_constraints)}"
        )
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


def plot_fitted_thickness(sec: Section, res: ThicknessResult):
    """Legacy plotting function - use ThicknessResult.plot_fit_summary() instead."""
    print(
        "Warning: plot_fitted_thickness is deprecated. Use ThicknessResult.plot_fit_summary() instead."
    )
    res.plot_fit_summary(sec)
