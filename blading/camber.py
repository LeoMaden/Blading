from typing import Literal
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


@dataclass
class CamberStackingParams:
    chord: float  # Chord length of the section
    x_offset: float  # Offset in the meridional or x direction
    y_offset: float  # Offset in the circumferential or y direction


@dataclass
class CamberMetalAngles:
    LE: float  # Leading edge angle
    TE: float  # Trailing edge angle


@dataclass
class CamberNonDimParams:
    ss_turning_frac: float
    ss_chord_frac: float
    alpha: float = 0.8
    beta: float = 0.3
    ss_scale: tuple[float, float] = (1, 1)
    sb_scale: tuple[float, float] = (1, 1)


@dataclass
class CamberParams:
    angles: CamberMetalAngles
    stacking: CamberStackingParams
    non_dim: CamberNonDimParams


@dataclass
class NonDimCamberSplineResult:
    spline: BSpline
    t_star: NDArray
    coefs: NDArray


@dataclass
class Camber:
    line: PlaneCurve

    def __post_init__(self):
        # Ensure the curve has constant speed (arc length) parameterisation and
        # is normalised.
        line = self.line
        if not line.is_unit():
            line = line.reparameterise_unit()
        if not line.is_normalised():
            line = line.normalise()

        self.line = line

    @property
    def s(self) -> NDArray:
        return self.line.param  # This is correct since the curve is normalised.

    @property
    def non_dim(self) -> NDArray:
        """Non-dimensional camber line."""
        angle = self.line.turning_angle()
        return (angle - angle[0]) / (angle[-1] - angle[0])

    @property
    def angle(self) -> NDArray:
        """Turning angle of the camber line."""
        return self.line.turning_angle()

    @property
    def chord(self) -> float:
        """Chord length of the camber line."""
        return self.line.length()

    def plot_non_dim(self, ax=None, *plot_args, **plot_kwargs):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.s, self.non_dim, *plot_args, **plot_kwargs)
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Non-dimensional camber")
        ax.grid(True)
        return ax

    def plot_angle(self, ax=None, *plot_args, **plot_kwargs):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.s, np.degrees(self.angle), *plot_args, **plot_kwargs)
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Turning angle (degrees)")
        ax.grid(True)
        return ax


def _dca_camber_spline(ss_turning: float, ss_chord: float) -> BSpline:
    return make_interp_spline(
        [0, ss_chord, 1],
        [0, ss_turning, 1],
        k=1,
    )


##################################################


@dataclass
class FitCamberResult:
    non_dim_spline: BSpline
    params: CamberParams
    t_star: NDArray
    coefs: NDArray

    def create_camber_line(self, s: NDArray) -> PlaneCurve:
        non_dim = self.non_dim_spline(s)
        return create_camber(non_dim, s, self.params.angles, self.params.stacking)

    def plot_non_dim_spline(
        self,
        s: NDArray | None = None,
        ax=None,
        show_control_points: bool = True,
        *plot_args,
        **plot_kwargs,
    ):
        if s is None:
            s = np.linspace(0, 1, 100)
        if ax is None:
            _, ax = plt.subplots()

        non_dim = self.non_dim_spline(s)
        ax.plot(s, non_dim, *plot_args, **plot_kwargs)

        if show_control_points:
            ax.plot(
                self.t_star[[0, 3, 6]],
                self.coefs[[0, 3, 6]],
                "go--",
                markersize=8,
                label="DCA Control points",
            )
            ax.plot(
                self.t_star[[1, 2, 4, 5]],
                self.coefs[[1, 2, 4, 5]],
                "bo",
                markersize=4,
                label="Camber style control points",
            )

        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Non-dimensional camber")
        ax.grid(True)
        if show_control_points:
            ax.legend()
        return ax

    def plot_camber_comparison(self, original_camber: Camber, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        original_camber.plot_non_dim(ax, "k-", label="Original")
        self.plot_non_dim_spline(
            original_camber.s,
            ax,
            show_control_points=True,
            label="Fitted",
            linestyle="--",
            color="r",
        )
        ax.legend()
        return ax


def create_non_dim_camber_spline(p: CamberNonDimParams) -> NonDimCamberSplineResult:
    # Set position of interior knots.
    xj = p.ss_chord_frac
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
    dca_spline = _dca_camber_spline(p.ss_turning_frac, p.ss_chord_frac)
    coefs = dca_spline(t_star)

    # Adjust coefficients.
    coefs[1] *= p.ss_scale[0]
    coefs[2] *= p.ss_scale[1]
    coefs[4] *= p.sb_scale[0]
    coefs[5] *= p.sb_scale[1]

    spline = BSpline(knots, coefs, k=3)

    return NonDimCamberSplineResult(spline, t_star, coefs)


def create_camber(
    non_dim: NDArray,
    s: NDArray,
    angles: CamberMetalAngles,
    stacking: CamberStackingParams,
) -> PlaneCurve:
    angle = non_dim * (angles.TE - angles.LE) + angles.LE
    camber_line = PlaneCurve.from_angle(angle, s)
    camber_line *= stacking.chord
    # Apply offsets by translating the curve
    translated_points = camber_line.coords.copy()
    translated_points[0] += stacking.x_offset
    translated_points[1] += stacking.y_offset
    return PlaneCurve(translated_points, camber_line.param)


def fit_camber(camber: Camber, plot_intermediate: bool = False) -> FitCamberResult:
    s = camber.s
    non_dim = camber.non_dim
    chord = camber.line.length()
    angle = camber.angle
    LE_coords = camber.line.start()

    x0 = [0.5, 0.5, 1, 1, 1, 1]

    def create_params(x) -> CamberParams:
        return CamberParams(
            angles=CamberMetalAngles(LE=angle[0], TE=angle[-1]),
            stacking=CamberStackingParams(
                chord=chord, x_offset=LE_coords[0], y_offset=LE_coords[1]
            ),
            non_dim=CamberNonDimParams(
                ss_turning_frac=x[0],
                ss_chord_frac=x[1],
                ss_scale=(x[2], x[3]),
                sb_scale=(x[4], x[5]),
            ),
        )

    def create_spline(x) -> NDArray:
        p = create_params(x)
        return create_non_dim_camber_spline(p.non_dim).spline(s)

    def objective(x) -> float:
        non_dim_fit = create_spline(x)
        return float(np.trapezoid((non_dim - non_dim_fit) ** 2, s))

    result = minimize(objective, x0)
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    final_params = create_params(result.x)
    spline_result = create_non_dim_camber_spline(final_params.non_dim)
    camber_result = FitCamberResult(
        spline_result.spline, final_params, spline_result.t_star, spline_result.coefs
    )

    if plot_intermediate:
        camber_result.plot_camber_comparison(camber)
        plt.title("Camber Fitting Results")
        plt.show()

    return camber_result
