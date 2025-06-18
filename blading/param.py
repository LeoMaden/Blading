from dataclasses import dataclass
from typing import Callable
from geometry.curves import PlaneCurve
from numpy.typing import NDArray, ArrayLike
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


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
