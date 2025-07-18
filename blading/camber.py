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
from pprint import pprint


@dataclass
class Camber:
    angle_LE: float
    angle_TE: float
    camber: NDArray
    s: NDArray


def to_camber_line(c: Camber):
    angle = c.camber * (c.angle_TE - c.angle_LE) + c.angle_LE
    curve = PlaneCurve.from_angle(angle, c.s)
    return curve


def from_camber_line(camber_line: PlaneCurve):
    s = camber_line.param
    angle = camber_line.turning_angle()
    angle_LE = angle[0]
    angle_TE = angle[-1]
    camber = (angle - angle_LE) / (angle_TE - angle_LE)
    return Camber(angle_LE, angle_TE, camber, s)


@dataclass
class DCAParams:
    ss_turning: float
    ss_chord: float


def create_dca_camber(p: DCAParams) -> BSpline:
    return make_interp_spline(
        [0, p.ss_chord, 1],
        [0, p.ss_turning, 1],
        k=1,
    )


@dataclass
class TransonicParams:
    ss_turning: float
    ss_chord: float
    alpha: float = 0.8
    beta: float = 0.3
    ss_scale: tuple[float, float] = (1, 1)
    sb_scale: tuple[float, float] = (1, 1)


def create_transonic_camber(p: TransonicParams) -> BSpline:
    # Set position of interior knots.
    xj = p.ss_chord
    t6 = (3 * xj - p.beta) / (2 + p.alpha - p.beta)
    t5 = p.alpha * t6
    t7 = t6 + p.beta * (1 - t6)

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
    dca_spline = create_dca_camber(DCAParams(p.ss_turning, p.ss_chord))
    coefs = dca_spline(t_star)

    # Adjust coefficients.
    coefs[1] *= p.ss_scale[0]
    coefs[2] *= p.ss_scale[1]
    coefs[4] *= p.sb_scale[0]
    coefs[5] *= p.sb_scale[1]

    return BSpline(knots, coefs, k=3)


@dataclass
class Result:
    spline: BSpline
    s: NDArray
    p: TransonicParams


def fit_transonic_camber(sec: Section) -> Result:
    camber = from_camber_line(sec.camber_line)
    s = camber.s
    distr = camber.camber

    x0 = [0.5, 0.5, 1, 1, 1, 1]

    def params(x):
        return TransonicParams(
            ss_turning=x[0],
            ss_chord=x[1],
            ss_scale=(x[2], x[3]),
            sb_scale=(x[4], x[5]),
        )

    def create(x):
        p = params(x)
        return create_transonic_camber(p)(s)

    def err(x):
        fit = create(x)
        return np.trapezoid((distr - fit) ** 2, s)

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
    bounds = [
        (0.1, 0.9),
        (0.1, 0.9),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
    ]
    opt = minimize(err, x0, tol=1e-16, bounds=bounds)
    assert opt.success, opt.message
    x = opt.x
    # x = [0.5, 0.5]
    fit = create(x)
    p = params(x)

    pprint(p)

    plt.figure()
    plt.plot(s, distr, "k-")
    plt.plot(s, fit, "r--")
    plt.show()
