from dataclasses import dataclass
from typing import Callable, Generic, MutableMapping, Optional, TypeVar
from geometry.curves import PlaneCurve
from numpy.typing import NDArray
import numpy as np


@dataclass(frozen=True)
class Section:
    camber_line: PlaneCurve
    thickness: NDArray
    stream_line: Optional[PlaneCurve]

    def chord(self) -> float:
        return self.camber_line.length()

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


@dataclass(frozen=True)
class PBlade:
    sections: list[PSection]


def approximate_camber_line(section: FlatSection) -> Section:
    raise NotImplementedError()
