from geometry.curves import PlaneCurve
from dataclasses import dataclass, field
from typing import Optional
from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import griddata


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
