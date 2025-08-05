import matplotlib.pyplot as plt
from dataclasses import dataclass
from geometry.curves import PlaneCurve
from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import (
    make_interp_spline,
    BSpline,
)
from scipy.optimize import minimize
from .section import Section


@dataclass
class Camber:
    line: PlaneCurve

    def __post_init__(self):
        line = self.line

        # Ensure the curve has constant speed (arc length) parameterisation and
        # is normalised.
        if not line.is_unit:
            line = line.reparameterise_unit()
        if not line.is_normalised:
            line = line.normalise()

        self.line = line

    @property
    def s(self) -> NDArray:
        return self.line.param  # This is correct since the curve is normalised.


@dataclass
class CamberStackingParams:
    chord: float  # Chord length of the section
    x_offset: float  # Offset in the meridional or x direction
    y_offset: float  # Offset in the circumferential or y direction


@dataclass
class CamberMetalAngles:
    LE: float  # Leading edge angle
    TE: float  # Trailing edge angle


##################################################


@dataclass
class CamberParams:
    angle_LE: float
    angle_TE: float
    chord: float

    ss_turning: float
    ss_chord: float
    alpha: float = 0.8
    beta: float = 0.3
    ss_scale: tuple[float, float] = (1, 1)
    sb_scale: tuple[float, float] = (1, 1)


@dataclass
class _Camber:
    s: NDArray
    non_dim: NDArray
    angle: NDArray
    curve: PlaneCurve

    @staticmethod
    def from_camber_line(curve: PlaneCurve):
        s = curve.param
        angle = curve.turning_angle()
        non_dim = (angle - angle[0]) / (angle[-1] - angle[0])
        return Camber(s, non_dim, angle, curve)

    def plot_non_dimensional(self, ax=None, *plot_args, **plot_kwargs):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.s, self.non_dim, *plot_args, **plot_kwargs)
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Non-dimensional camber")
        ax.grid(True)
        return ax

    def plot_angle_distribution(self, ax=None, *plot_args, **plot_kwargs):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.s, np.degrees(self.angle), *plot_args, **plot_kwargs)
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Turning angle (degrees)")
        ax.grid(True)
        return ax


def _dca_camber(ss_turning: float, ss_chord: float) -> BSpline:
    return make_interp_spline(
        [0, ss_chord, 1],
        [0, ss_turning, 1],
        k=1,
    )


@dataclass
class CamberResult:
    non_dim_spline: BSpline
    params: CamberParams

    def create_camber_line(self, s: NDArray) -> PlaneCurve:
        non_dim = self.non_dim_spline(s)
        return create_camber(non_dim, s, self.params)

    def plot_non_dim_spline(
        self, s: NDArray = None, ax=None, *plot_args, **plot_kwargs
    ):
        if s is None:
            s = np.linspace(0, 1, 100)
        if ax is None:
            _, ax = plt.subplots()

        non_dim = self.non_dim_spline(s)
        ax.plot(s, non_dim, *plot_args, **plot_kwargs)
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Non-dimensional camber")
        ax.grid(True)
        return ax

    def plot_camber_comparison(self, original_camber: Camber, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        original_camber.plot_non_dimensional(ax, "k-", label="Original")
        self.plot_non_dim_spline(original_camber.s, ax, "r--", label="Fitted")
        ax.legend()
        return ax


def create_non_dim_camber(p: CamberParams) -> BSpline:
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
    dca_spline = _dca_camber(p.ss_turning, p.ss_chord)
    coefs = dca_spline(t_star)

    # Adjust coefficients.
    coefs[1] *= p.ss_scale[0]
    coefs[2] *= p.ss_scale[1]
    coefs[4] *= p.sb_scale[0]
    coefs[5] *= p.sb_scale[1]

    spline = BSpline(knots, coefs, k=3)

    return spline


def create_camber(non_dim: NDArray, s: NDArray, p: CamberParams):
    angle = non_dim * (p.angle_TE - p.angle_LE) + p.angle_LE
    camber_line = PlaneCurve.from_angle(angle, s)
    camber_line *= p.chord
    return camber_line


def fit_camber(sec: Section, plot_intermediate=False) -> CamberResult:
    camber = Camber.from_camber_line(sec.camber_line)
    s = camber.s
    non_dim = camber.non_dim
    chord = camber.curve.length()
    angle = camber.angle

    x0 = [0.5, 0.5, 1, 1, 1, 1]

    def params(x):
        return CamberParams(
            angle_LE=angle[0],
            angle_TE=angle[-1],
            chord=chord,
            ss_turning=x[0],
            ss_chord=x[1],
            ss_scale=(x[2], x[3]),
            sb_scale=(x[4], x[5]),
        )

    def create(x):
        p = params(x)
        return create_non_dim_camber(p)(s)

    def err(x):
        non_dim_fit = create(x)
        return np.trapezoid((non_dim - non_dim_fit) ** 2, s)

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

    final_params = params(x)
    spline = create_non_dim_camber(final_params)
    result = CamberResult(spline, final_params)

    if plot_intermediate:
        result.plot_camber_comparison(camber)
        plt.title("Camber Fitting Results")
        plt.show()

    return result
