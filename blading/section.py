from dataclasses import dataclass
from typing import Callable, Generic, MutableMapping, Optional, TypeVar
from geometry.curves import PlaneCurve
from numpy.typing import NDArray
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Section:
    camber_line: PlaneCurve
    thickness: NDArray
    stream_line: Optional[PlaneCurve]

    def __post_init__(self) -> None:
        assert self.normalised_arc_length[0] == 0
        assert self.normalised_arc_length[-1] == 1

    @property
    def chord(self) -> float:
        return self.camber_line.length()

    @property
    def normalised_arc_length(self) -> NDArray:
        return self.camber_line.param

    def upper_curve(self) -> PlaneCurve:
        cl = self.camber_line.coords
        thick = self._signed_thickness()
        upper_coords = cl + 0.5 * thick
        upper_param = self.camber_line.normalise().param
        return PlaneCurve(upper_coords, upper_param)

    def lower_curve(self) -> PlaneCurve:
        cl = self.camber_line.coords
        thick = self._signed_thickness()
        upper_coords = cl - 0.5 * thick
        upper_param = self.camber_line.normalise().param
        return PlaneCurve(upper_coords, upper_param)

    def upper_and_lower(self) -> PlaneCurve:
        cl = self.camber_line.coords
        thick = self._signed_thickness()

        upper_coords = cl + 0.5 * thick
        lower_coords = cl - 0.5 * thick
        coords = np.r_[lower_coords[1:][::-1], upper_coords]

        # Parameter 0 to -1 on lower and 0 to 1 on upper
        camber_param = self.camber_line.normalise().param
        param = np.r_[-camber_param[1:][::-1], camber_param]

        return PlaneCurve(coords, param)

    def _signed_thickness(self) -> NDArray:
        normal = self.camber_line.signed_normal().coords
        thickness = self.thickness[:, np.newaxis]
        return normal * thickness

    def remove_round_TE(
        self,
        mask_rear: NDArray[np.bool] | None = None,
        mask_high_curv: NDArray[np.bool] | None = None,
    ) -> "Section":
        s = self.normalised_arc_length
        # Mask for the rear section of the blade.
        if mask_rear is None:
            mask_rear = s >= 0.8

        # Mask for areas of high curvature on the blade
        if mask_high_curv is None:
            grad = np.gradient(self.thickness, s)
            curv = np.abs(np.gradient(grad, s))
            avg_curv = np.median(curv)
            mask_high_curv = curv > 4 * avg_curv
            assert mask_high_curv is not None

        # Fit straight line through linear part of thickness
        mask_linear = mask_rear & ~mask_high_curv  # linear part
        mask_round_TE = mask_rear & mask_high_curv  # round part
        poly = Polynomial.fit(s[mask_linear], self.thickness[mask_linear], deg=1)

        # Ensure new TE has continuous value.
        y1 = self.thickness[mask_linear][-1]
        y2 = poly(s[mask_linear][-1])
        poly -= y2 - y1

        # Extrapolate straight line to TE
        thickness = self.thickness.copy()
        thickness[mask_round_TE] = poly(s[mask_round_TE])

        return Section(self.camber_line, thickness, self.stream_line)

    def add_round_TE(self) -> "Section":
        s = self.normalised_arc_length

        # Thickness relative to chord.
        t_over_c = self.thickness.copy() / self.chord

        # Find trailing edge angle.
        x1, x2 = s[[-2, -1]]
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
        mask_round_TE = s > min(x_te)
        t_over_c[mask_round_TE] = np.interp(s[mask_round_TE], x_te, y_te)

        return Section(self.camber_line, t_over_c * self.chord, self.stream_line)

    ###### Plotting ######

    def plot_thickness(self, ax=None, *plot_args, **plot_kwargs):
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.normalised_arc_length, self.thickness, *plot_args, **plot_kwargs)
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Thickness")

    def plot_thickness_to_chord(self, ax=None, *plot_args, **plot_kwargs):
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(
            self.normalised_arc_length,
            self.thickness / self.chord,
            *plot_args,
            **plot_kwargs,
        )
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Thickness / chord")

    def plot_nondim_camber(self, ax=None, *plot_args, **plot_kwargs):
        if ax is None:
            _, ax = plt.subplots()

        angle = self.camber_line.turning_angle()
        nondim_camber = (angle - angle[0]) / (angle[-1] - angle[0])
        ax.plot(self.normalised_arc_length, nondim_camber, *plot_args, **plot_kwargs)
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Non-dimensional camber")


@dataclass(frozen=True)
class FlatSection:
    curve: PlaneCurve
    stream_line: Optional[PlaneCurve]


@dataclass(frozen=True)
class FlatBlade:
    sections: list[FlatSection]


T = TypeVar("T", bound=MutableMapping)


@dataclass(frozen=True)
class PSection(Generic[T]):
    params: T
    create: Callable[[T], Section]

    def section(self) -> Section:
        return self.create(self.params)

    def update(self, params: T) -> "PSection":
        updated = type(params)()
        updated.update(**self.params)  # Fill with old parameters
        updated.update(params)  # Update new parameters
        return PSection(updated, self.create)
