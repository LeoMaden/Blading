from dataclasses import dataclass, field
from typing import Callable, Generic, MutableMapping, Optional, TypeVar
from geometry.curves import PlaneCurve
from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import griddata


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


@dataclass(frozen=True)
class Annulus:
    hub: PlaneCurve
    cas: PlaneCurve
    point_distr: Optional[NDArray] = None

    _interp_stream: NDArray = field(init=False)
    _interp_physical: NDArray = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hub", self.hub.reparameterise_unit().normalise())
        object.__setattr__(self, "cas", self.cas.reparameterise_unit().normalise())
        self._create_interp_grid()

    def _calc_point(self, coord):
        xi, eta = coord
        hub = self.hub.interpolate([xi])
        cas = self.cas.interpolate([xi])
        return hub * (1 - eta) + cas * eta

    def physical_coords(self, stream_coords: NDArray) -> NDArray:
        streamwise, spanwise = stream_coords.T
        hub = self.hub.interpolate(streamwise)
        cas = self.cas.interpolate(streamwise)
        k = spanwise[:, np.newaxis]
        return hub.coords * (1 - k) + cas.coords * k

    def stream_coords(self, physical_coords: NDArray) -> NDArray:
        streamwise_grid = self._interp_stream[:, 0]
        spanwise_grid = self._interp_stream[:, 1]

        streamwise = griddata(self._interp_physical, streamwise_grid, physical_coords)
        spanwise = griddata(self._interp_physical, spanwise_grid, physical_coords)

        return np.c_[streamwise, spanwise]

    def _create_interp_grid(self) -> None:
        N = 100
        pts = np.linspace(-0.1, 1.1, N)
        streamwise_grid, spanwise_grid = np.meshgrid(pts, pts, indexing="ij")
        stream_grid = np.stack((streamwise_grid, spanwise_grid), axis=2)
        grid_coords = np.zeros((N, N, 2))
        for i in range(N):
            grid_coords[i] = self.physical_coords(stream_grid[i])

        object.__setattr__(self, "_interp_physical", grid_coords.reshape((-1, 2)))
        object.__setattr__(self, "_interp_stream", stream_grid.reshape((-1, 2)))


def approximate_camber_line(section: FlatSection) -> Section:
    raise NotImplementedError()
