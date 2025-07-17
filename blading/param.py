from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable
from geometry.curves import PlaneCurve, plot_plane_curve
from numpy.typing import NDArray, ArrayLike
import numpy as np
from scipy.interpolate import (
    make_interp_spline,
    BSpline,
    CubicHermiteSpline,
    CubicSpline,
)
from blading import shape_space
from scipy.optimize import fmin, least_squares, minimize
from circle_fit import taubinSVD, plot_data_circle
from .section import Section


CamberDistribution = Callable[[NDArray], NDArray]


def create_camber(
    distribution: CamberDistribution,
    inlet_angle: float,
    outlet_angle: float,
    arc_length: NDArray,
    offset: ArrayLike,
):
    assert arc_length[0] == 0
    s = arc_length / arc_length[-1]
    distr = distribution(s)
    angle = distr * (outlet_angle - inlet_angle) + inlet_angle
    curve = PlaneCurve.from_angle(np.radians(angle), arc_length)
    return curve + np.asarray(offset).reshape((1, 2))


@dataclass
class DoubleCircularArcCamber:
    ss_turning_ratio: float
    ss_chord_ratio: float

    def make(self) -> BSpline:
        return make_interp_spline(
            [0, self.ss_chord_ratio, 1],
            [0, self.ss_turning_ratio, 1],
            k=1,
        )


@dataclass
class TransonicCamber:
    ss_turning_ratio: float
    ss_chord_ratio: float
    alpha: float = 0.8
    beta: float = 0.3
    ss_scale: tuple[float, float] = (1, 1)
    sb_scale: tuple[float, float] = (1, 1)

    def __post_init__(self) -> None:
        pass

    def make(self) -> BSpline:
        # Set position of interior knots.
        xj = self.ss_chord_ratio
        t6 = (3 * xj - self.beta) / (2 + self.alpha - self.beta)
        t5 = self.alpha * t6
        t7 = t6 + self.beta * (1 - t6)

        int_knots = np.r_[t5, t6, t7]
        knots = np.r_[np.zeros(4), int_knots, np.ones(4)]

        # Calculate chordwise positions.
        low_tri = np.tri(3) / 3
        x_ss1, x_ss2, _ = low_tri @ int_knots
        up_tri = np.tri(3).T / 3
        _, x_sb1, x_sb2 = up_tri @ int_knots + np.r_[0, 1, 2] / 3

        # Averaged knot vector.
        t_star = np.r_[0, x_ss1, x_ss2, xj, x_sb1, x_sb2, 1]

        # DCA distribution.
        dca_spline = DoubleCircularArcCamber(
            self.ss_turning_ratio,
            self.ss_chord_ratio,
        ).make()
        coefs = dca_spline(t_star)

        # Adjust coefficients.
        coefs[1] *= self.ss_scale[0]
        coefs[2] *= self.ss_scale[1]
        coefs[4] *= self.sb_scale[0]
        coefs[5] *= self.sb_scale[1]

        self.t_star = t_star
        self.coefs = coefs
        return BSpline(knots, coefs, k=3)


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
class ContinuousCurvThickness:
    radius_LE: float
    thickness_TE: float
    wedge_angle: float
    max_thickness: float
    pos_max_t: float
    curv_max_t: float
    pos_knot: float
    stretch_amount: float
    stretch_join: float

    @staticmethod
    def fit(sec: Section):
        s = sec.normalised_arc_length
        t = sec.thickness

        s_stretch = stretch_func(0.5, 0.2)(s)

        # Maximum thickness.
        t_spl = make_interp_spline(s, t)
        func = lambda s: -t_spl(s)
        (s_max_t,) = fmin(func, x0=0.5)  # type: ignore
        max_t = t_spl(s_max_t)

        t_TE = t[-1]

        # Calculate shape space values, extrapolating LE and TE singularities.
        ss = shape_space.value(s, t, t[-1])[1:-1]
        ss_spl = make_interp_spline(s[1:-1], ss, bc_type="natural")
        ss = ss_spl(s)

        # LE radius.
        LE_circle = fit_LE_circle(sec)

        ss_LE = np.sqrt(2 * LE_circle.radius)
        ss_TE = ss[-1]
        ss_max_t = ss_spl(s_max_t)

        wedge_angle = np.arctan(ss_TE - t_TE)

        def create(x):
            return ContinuousCurvThickness(
                radius_LE=LE_circle.radius,
                thickness_TE=t_TE,
                wedge_angle=wedge_angle,
                max_thickness=max_t,
                pos_max_t=s_max_t,
                curv_max_t=x[0],
                pos_knot=x[1],
                stretch_amount=0.3,
                stretch_join=0.2,
            )

        def err(x):
            ss_fit = create(x).eval_ss(s)
            return np.trapezoid((ss_fit - ss) ** 2, s)

        res = minimize(err, x0=[-5, 0.5], bounds=[(None, None), (0, 1)])
        assert res.success
        print(res.x)
        fitted = create(res.x)

        plt.plot(s, ss, "k-")
        plt.plot(s, fitted.eval_ss(s), "r--")
        # plt.plot(s_stretch, ss, "b--")
        # plt.plot(0, ss_LE, "r.")
        # plt.plot(1, ss_TE, "r.")
        # plt.plot(s_max_t, ss_max_t, "r.")
        plt.show()

    def make(self):
        knots = np.r_[-np.ones(4) * self.stretch_amount, self.pos_knot, 1, 1, 1, 1]
        m = len(knots) - 1
        p = 3
        n = m - p - 1

        basis: list = [0] * (n + 1)
        for i in range(n + 1):  # number of basis functions
            coef = np.astype(np.arange(n + 1) == i, np.floating)
            basis[i] = BSpline(knots, coef, p)

        # Constraints on function in shape-space.
        x0 = -self.stretch_amount
        y0 = np.sqrt(2 * self.radius_LE)

        x1 = self.pos_max_t
        y1 = shape_space.value(x1, self.max_thickness, self.thickness_TE)
        dy1 = shape_space.deriv1(x1, self.max_thickness, self.thickness_TE, 0)
        ddy1 = shape_space.deriv2(
            x1, self.max_thickness, self.thickness_TE, 0, self.curv_max_t
        )

        x2 = 1
        y2 = np.tan(self.wedge_angle) + self.thickness_TE

        # Create system of equations
        A = np.zeros((n + 1, n + 1))
        A[0, :] = [b(x0) for b in basis]
        A[1, :] = [b(x1) for b in basis]
        A[2, :] = [b.derivative()(x1) for b in basis]
        A[3, :] = [b.derivative().derivative()(x1) for b in basis]
        A[4, :] = [b(x2) for b in basis]

        b = np.r_[y0, y1, dy1, ddy1, y2]

        c = np.linalg.solve(A, b)

        spl = BSpline(knots, c, p)

        def eval(x):
            ss = spl(x)
            return shape_space.inverse(x, ss, self.thickness_TE)

        return eval, spl

    def eval_ss(self, s):
        _, spl = self.make()
        s_stretch = stretch_func(self.stretch_amount, self.stretch_join)(s)
        return spl(s_stretch)


###################################################


@dataclass
class MeasuredParams:
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

    return MeasuredParams(
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
    ss: NDArray
    t_TE: float
    s_LE: float
    ss_LE: float
    s_TE: float
    ss_TE: float
    s_max_t: float
    ss_max_t: float
    ss_join: float


def test_thickness(sec: Section, fit_params: FitParams):
    s = sec.normalised_arc_length

    stretch_join = fit_params.stretch_join
    stretch_amount = fit_params.stretch_amount

    measured = measure_thickness(sec, stretch_join)

    # Leading edge.
    s_LE = -stretch_amount
    ss_LE = np.sqrt(measured.radius_LE)

    # Trailing edge.
    s_TE = 1
    ss_TE = np.tan(measured.wedge_TE) + measured.thickness_TE

    # Maximum thickness.
    s_max_t = measured.s_max_t
    max_t = measured.max_t
    t_TE = measured.thickness_TE
    curv_max_t = measured.curv_max_t
    ss_max_t = shape_space.value(s_max_t, max_t, t_TE)
    ss_grad_max_t = shape_space.deriv1(s_max_t, max_t, t_TE, 0)
    ss_curv_max_t = shape_space.deriv2(s_max_t, max_t, t_TE, 0, curv_max_t)

    # Calculate stretching function.
    stretch = stretch_func(stretch_amount, stretch_join)

    # Calculate knot positions based on position of maximum thickness and stretch.
    s_transform = make_interp_spline([0, 1, 2], [s_LE, s_max_t, s_TE], k=1)
    s_knots = s_transform(np.asarray(fit_params.knot_positions))

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
        (value_at(stretch_join), measured.ss_stretch_join),
        (grad_at(s_TE), measured.ss_grad_TE),
        (grad_at(s_LE), fit_params.ss_grad_LE),
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
        ss=measured.ss,
        t_TE=t_TE,
        s_LE=s_LE,
        ss_LE=ss_LE,
        s_TE=s_TE,
        ss_TE=ss_TE,
        s_max_t=s_max_t,
        ss_max_t=ss_max_t,
        ss_join=measured.ss_stretch_join,
    )
