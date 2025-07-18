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

        ax.plot(x, y, "k-", label="LE circle")
        ax.plot(*self.fit_points.T, "b.", label="LE points")
        ax.legend()
        plt.axis("equal")


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
class Params:
    radius_LE: float
    s_max_t: float
    max_t: float
    curv_max_t: float
    thickness_TE: float
    wedge_TE: float
    ss_stretch_join: float
    ss_grad_TE: float
    ss: NDArray


def measure_thickness(sec: Section, stretch_join: float):
    s = sec.normalised_arc_length

    ##### Thickness #####
    t = sec.thickness

    # Find point of maximum thickness.
    t_spline = make_interp_spline(s, t)
    (s_max_t,) = fmin(lambda s: -t_spline(s), x0=0.5)  # type: ignore
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
    ss_spline = make_interp_spline(s[1:-1], ss[1:-1], k=1)
    ss = ss_spline(s)
    ss_stretch_join = ss_spline(stretch_join)
    ss_grad = np.gradient(ss_spline(s), s)
    ss_grad_spline = make_interp_spline(s, ss_grad)
    ss_grad_TE = ss_grad_spline(1)

    return Params(
        radius_LE=radius_LE,
        s_max_t=s_max_t,
        max_t=max_t,
        curv_max_t=curv_max_t,
        thickness_TE=t_TE,
        wedge_TE=wedge_TE,
        ss_stretch_join=ss_stretch_join,
        ss_grad_TE=ss_grad_TE,
        ss=ss,
    )


@dataclass
class FitParams:
    stretch_amount: float
    stretch_join: float
    knot_positions: tuple[float, float, float, float]
    ss_grad_LE: float


@dataclass
class Result:
    spline: BSpline
    stretch: Callable[[NDArray], NDArray]
    p: Params
    p_fit: FitParams
    ss: NDArray
    t_TE: float
    s_LE: float
    ss_LE: float
    s_TE: float
    ss_TE: float
    s_max_t: float
    ss_max_t: float
    ss_join: float


def create_thickness(p: Params, p_fit: FitParams):
    stretch_join = p_fit.stretch_join
    stretch_amount = p_fit.stretch_amount

    # Leading edge.
    s_LE = -stretch_amount
    ss_LE = np.sqrt(p.radius_LE)

    # Trailing edge.
    s_TE = 1
    ss_TE = np.tan(p.wedge_TE) + p.thickness_TE

    # Maximum thickness.
    s_max_t = p.s_max_t
    max_t = p.max_t
    t_TE = p.thickness_TE
    curv_max_t = p.curv_max_t
    ss_max_t = shape_space.value(s_max_t, max_t, t_TE)
    ss_grad_max_t = shape_space.deriv1(s_max_t, max_t, t_TE, 0)
    ss_curv_max_t = shape_space.deriv2(s_max_t, max_t, t_TE, 0, curv_max_t)

    # Calculate stretching function.
    stretch = stretch_func(stretch_amount, stretch_join)

    # Calculate knot positions based on position of maximum thickness and stretch.
    s_transform = make_interp_spline([0, 1, 2], [s_LE, s_max_t, s_TE], k=1)
    s_knots = s_transform(np.asarray(p_fit.knot_positions))

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
        (value_at(stretch_join), p.ss_stretch_join),
        (grad_at(s_TE), p.ss_grad_TE),
        (grad_at(s_LE), p_fit.ss_grad_LE),
    ]
    assert len(constraints) == n_coef

    # Create matrices and solve for coefficients.
    A = np.c_[*[c[0] for c in constraints]].T
    b = np.r_[*[c[1] for c in constraints]]
    coefs = np.linalg.solve(A, b)

    spline = BSpline(knots, coefs, degree)

    return Result(
        spline=spline,
        stretch=stretch,
        p=p,
        p_fit=p_fit,
        ss=p.ss,
        t_TE=t_TE,
        s_LE=s_LE,
        ss_LE=ss_LE,
        s_TE=s_TE,
        ss_TE=ss_TE,
        s_max_t=s_max_t,
        ss_max_t=ss_max_t,
        ss_join=p.ss_stretch_join,
    )


def fit_thickness(sec: Section) -> tuple[Params, FitParams]:

    # Initial guess of fitting parameters.
    x0 = [0.1, 0.15, -0.5, 0.05, 0.65, 1, 1.2]

    # Function to calculate parameters from optimisation vector.
    def create_params(x):
        p_fit = FitParams(
            stretch_amount=x[0],
            stretch_join=x[1],
            ss_grad_LE=x[2],
            knot_positions=tuple(x[3:]),
        )
        p = measure_thickness(sec, p_fit.stretch_join)
        return p, p_fit

    # Function to create thickness from optimisation vector.
    def create(x):
        p, p_fit = create_params(x)
        return create_thickness(p, p_fit)

    # Function to calculate error between fit and actual in shape space.
    def err(x):
        res = create(x)
        s = sec.normalised_arc_length
        s_stretch = res.stretch(s)
        ss_fit = res.spline(s_stretch)
        ss = res.ss
        return np.trapezoid((ss - ss_fit) ** 2, s)

    # Constraints on optimisation variables.
    constraints = [
        {"type": "ineq", "fun": lambda x: x[0]},
        {"type": "ineq", "fun": lambda x: x[1]},
        {"type": "ineq", "fun": lambda x: 0.5 - x[0]},
        {"type": "ineq", "fun": lambda x: 0.4 - x[1]},
        {"type": "ineq", "fun": lambda x: -x[2]},
        {"type": "ineq", "fun": lambda x: x[4] - x[3] - 0.1},
        {"type": "ineq", "fun": lambda x: x[5] - x[4] - 0.1},
        {"type": "ineq", "fun": lambda x: x[6] - x[5] - 0.1},
        {"type": "ineq", "fun": lambda x: x[3] - 0.01},
        {"type": "ineq", "fun": lambda x: 1.99 - x[6]},
    ]
    e0 = err(x0)
    tol = 1e-8 * e0
    opt = minimize(err, x0, tol=tol, constraints=constraints)
    assert opt.success, opt.message

    x = opt.x
    return create_params(x)


def plot_fitted_thickness(sec: Section, res: Result):
    s = sec.normalised_arc_length
    s_stretch = res.stretch(s)
    ss_fit = res.spline(s_stretch)

    # Calculate avg. knot vector.
    degree = res.spline.k
    n_coef = len(res.spline.c)
    knot_avg = np.array(
        [res.spline.t[i : i + degree + 1].mean() for i in range(n_coef)]
    )

    t_fit = shape_space.inverse(s, ss_fit, res.t_TE)
    sec_fit = Section(sec.camber_line, t_fit, sec.stream_line)

    # Plotting
    plt.figure()
    plt.title("Shape space")
    plt.plot(s_stretch, res.ss, "k-")
    plt.plot(s_stretch, ss_fit, "r--")
    plt.plot(res.s_LE, res.ss_LE, "r.")
    plt.plot(res.s_TE, res.ss_TE, "r.")
    plt.plot(res.s_max_t, res.ss_max_t, "r.")
    plt.plot(res.p_fit.stretch_join, res.ss_join, "r.")
    plt.plot(knot_avg, res.spline.c, "bx")

    fig, ax = plt.subplots()
    plot_plane_curve(sec.upper_and_lower(), ax, "k.-")
    plot_plane_curve(sec_fit.upper_and_lower(), ax, "r--")
    plt.axis("equal")

    fig, ax = plt.subplots()
    sec.plot_thickness(ax, "k-")
    sec_fit.plot_thickness(ax, "r--")

    plt.show()
