from functools import cached_property
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


################################################################################
######################### Main camber data type ################################
################################################################################


@dataclass(frozen=True)
class Camber:
    line: PlaneCurve

    def __post_init__(self):
        line = self.line

        # Ensure the curve has constant speed (arc length) parameterisation and
        # is normalised.
        if not line.is_unit():
            line = line.reparameterise_unit()
        if not line.is_normalised():
            line = line.normalise()

        object.__setattr__(self, "line", line)

    @property
    def s(self) -> NDArray:
        return self.line.param  # This is correct since the curve is normalised.

    @property
    def non_dim(self) -> NDArray:
        """Non-dimensional camber line."""
        angle = self.line.turning_angle()
        return (angle - angle[0]) / (angle[-1] - angle[0])

    @cached_property
    def angle(self) -> NDArray:
        """Turning angle of the camber line."""
        return self.line.turning_angle()

    @property
    def angle_LE(self) -> float:
        """Leading edge angle of the camber line."""
        return self.angle[0]

    @property
    def angle_TE(self) -> float:
        """Trailing edge angle of the camber line."""
        return self.angle[-1]

    @property
    def chord(self) -> float:
        """Chord length of the camber line."""
        return self.line.length()

    def interpolate(self, s: NDArray) -> "Camber":
        """Interpolate the camber line at given s values."""
        new_line = self.line.interpolate(s)
        return Camber(new_line)

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


################################################################################
###################### Parameters defining camber line #########################
################################################################################


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


################################################################################
################## Non-dimensional camber creation from params #################
################################################################################


@dataclass
class NonDimCamberResult:
    spline: BSpline
    t_star: NDArray
    coefs: NDArray

    def eval(self, s: NDArray) -> NDArray:
        """Evaluate the non-dimensional camber spline at given s values."""
        return self.spline(s)


def _dca_camber_spline(ss_turning: float, ss_chord: float) -> BSpline:
    return make_interp_spline(
        [0, ss_chord, 1],
        [0, ss_turning, 1],
        k=1,
    )


def create_non_dim_camber_spline(p: CamberNonDimParams) -> NonDimCamberResult:
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

    return NonDimCamberResult(spline, t_star, coefs)


################################################################################
###################### Dimensional camber creation #############################
################################################################################


@dataclass
class CamberResult:
    non_dim_result: NonDimCamberResult
    params: CamberParams

    def eval(self, s: NDArray) -> Camber:
        """Evaluate the camber line at given s values."""
        non_dim = self.non_dim_result.eval(s)
        angle = _angle_from_non_dim(non_dim, self.params.angles)
        return _camber_from_angle(angle, s, self.params.stacking)


def _angle_from_non_dim(non_dim: NDArray, angles: CamberMetalAngles) -> NDArray:
    """Convert non-dimensional camber to angle."""
    return non_dim * (angles.TE - angles.LE) + angles.LE


def _camber_from_angle(
    angle: NDArray, s: NDArray, stacking: CamberStackingParams
) -> Camber:
    """Create a camber line from angle and stacking parameters."""
    camber_line = PlaneCurve.from_angle(angle, s)
    camber_line *= stacking.chord
    # Apply offsets by translating the curve
    translated_points = camber_line.coords.copy()
    translated_points[:, 0] += stacking.x_offset
    translated_points[:, 1] += stacking.y_offset
    translated_line = PlaneCurve(translated_points, camber_line.param)
    return Camber(translated_line)


def create_camber(params: CamberParams) -> CamberResult:
    """Create a camber line from given parameters."""
    non_dim_result = create_non_dim_camber_spline(params.non_dim)
    return CamberResult(non_dim_result, params)


@dataclass
class FitCamberResult:
    result: CamberResult
    original: Camber

    def compare_non_dim(self, ax=None):
        """Compare the original and fitted non-dimensional camber distributions."""
        if ax is None:
            _, ax = plt.subplots()

        # Plot fitted non-dimensional camber directly from BSpline
        fitted_camber = self.result.eval(self.original.s)
        fitted_camber.plot_non_dim(ax, "r-", label="Fitted")

        # Plot original using Camber's plotting method
        self.original.plot_non_dim(ax, "k-", alpha=0.5, label="Original")

        # Plot control points
        x_control = self.result.non_dim_result.t_star
        y_control = self.result.non_dim_result.coefs
        i_dca = [0, 3, 6]
        i_interior = [1, 2, 4, 5]
        ax.plot(x_control[i_dca], y_control[i_dca], "bo", label="DCA control")
        ax.plot(x_control[i_dca], y_control[i_dca], "b--", alpha=0.5)
        ax.plot(
            x_control[i_interior],
            y_control[i_interior],
            "bx",
            label="Camber style control",
        )

        ax.legend()
        ax.set_title("Non-dimensional Camber Comparison")

        return ax

    def compare_lines(self, ax=None):
        """Compare the original and fitted camber lines."""
        if ax is None:
            _, ax = plt.subplots()

        # Get and plot fitted camber line
        fitted_camber = self.result.eval(self.original.s)
        fitted_camber.line.plot(ax, "r-", label="Fitted")

        # Plot original camber line
        self.original.line.plot(ax, "k-", alpha=0.5, label="Original")

        ax.axis("equal")
        ax.legend()

        # Get LE and TE angles for title
        le_angle = np.degrees(self.result.params.angles.LE)
        te_angle = np.degrees(self.result.params.angles.TE)
        turning_angle = le_angle - te_angle
        ax.set_title(
            f"Camber Line Comparison (LE: {le_angle:.1f}°, TE: {te_angle:.1f}°, Δ: {turning_angle:.1f}°)"
        )
        ax.grid(True, alpha=0.3)

        return ax

    def compare_summary(self):
        """Compare the original and fitted non-dimensional camber and camber lines
        side-by-side.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Compare non-dimensional camber
        self.compare_non_dim(ax=ax1)

        # Compare camber lines
        self.compare_lines(ax=ax2)

        plt.tight_layout()
        return fig, (ax1, ax2)


def fit_camber(camber: Camber) -> FitCamberResult:
    s = camber.s
    non_dim = camber.non_dim
    LE_coords = camber.line.start()

    x0 = [0.5, 0.5, 1, 1, 1, 1]

    def create_params(x) -> CamberParams:
        return CamberParams(
            angles=CamberMetalAngles(LE=camber.angle_LE, TE=camber.angle_TE),
            stacking=CamberStackingParams(
                chord=camber.chord, x_offset=LE_coords[0], y_offset=LE_coords[1]
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

    # Constrain the optimisation to improve reliability
    def gt_zero(constraint_func):
        return {"type": "ineq", "fun": constraint_func}

    constraints = [
        gt_zero(lambda x: x[0]),  # ss_turning_frac > 0
        gt_zero(lambda x: 1 - x[0]),  # ss_turning_frac < 1
        gt_zero(lambda x: x[1] - 0.2),  # ss_chord_frac > 0.2
        gt_zero(lambda x: 0.8 - x[1]),  # ss_chord_frac < 0.8
    ]

    result = minimize(objective, x0, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    final_params = create_params(result.x)
    spline_result = create_non_dim_camber_spline(final_params.non_dim)
    camber_result = CamberResult(spline_result, final_params)

    return FitCamberResult(camber_result, camber)
