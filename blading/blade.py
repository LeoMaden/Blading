from dataclasses import dataclass, field
from pprint import pprint
from typing import (
    Callable,
    Self,
)
from numpy.typing import NDArray
import numpy as np
from .geom import gradient, integrate_from_zero, sum_from_zero, length, cum_length


"""
Section definitions
- Just a set of coordinates around the section
- Coordinates of upper and lower surfaces
- Non-dimensional camber and thickness distributions + chord
- Parameterised camber, spline thickness + chord
- Parameterised thickness, spline camber + chord
- Parameterised thickness & camber + chord



"""


# -------------------------------------------
# Coordinate systems
# -------------------------------------------


class Polar(NDArray):
    def __new__(cls, object) -> Self:
        obj = np.asarray(object).view(cls)
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    @classmethod
    def new_2d(cls, r, t) -> "Polar":
        return Polar(np.stack((r, t), axis=-1))

    @classmethod
    def new_3d(cls, x, r, t) -> "Polar":
        return Polar(np.stack((r, t, x), axis=-1))

    @property
    def is_2d(self) -> bool:
        return self.shape[-1] == 2

    @property
    def is_3d(self) -> bool:
        return self.shape[-1] == 3

    @property
    def x(self) -> "Polar":
        if not self.is_3d:
            raise Exception("Array must be 3D")
        return Polar(self[..., 2])

    @property
    def r(self) -> "Polar":
        return Polar(self[..., 0])

    @property
    def t(self) -> "Polar":
        return Polar(self[..., 1])


class Cartesian(NDArray):
    def __new__(cls, object) -> Self:
        obj = np.asarray(object).view(cls)
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    @classmethod
    def new_2d(cls, x, y) -> "Cartesian":
        return Cartesian(np.stack((x, y), axis=-1))

    @classmethod
    def new_3d(cls, x, y, z) -> "Cartesian":
        return Cartesian(np.stack((x, y, z), axis=-1))

    @property
    def is_2d(self) -> bool:
        return self.shape[-1] == 2

    @property
    def is_3d(self) -> bool:
        return self.shape[-1] == 3

    @property
    def x(self) -> "Cartesian":
        return Cartesian(self[..., 0])

    @property
    def y(self) -> "Cartesian":
        return Cartesian(self[..., 1])

    @property
    def z(self) -> "Cartesian":
        if not self.is_3d:
            raise Exception("Array must be 3D")
        return Cartesian(self[..., 2])


Coords = Polar | Cartesian


# --------------------------------------
# Conversions between coordinate systems
# --------------------------------------


def pol_to_cart(pol: Polar) -> Cartesian:
    if pol.is_2d:
        x = pol.r * np.cos(pol.t)
        y = pol.r * np.sin(pol.t)
        return Cartesian.new_2d(x, y)
    elif pol.is_3d:
        pass
    else:
        raise ValueError("`pol` must be 2 or 3D")


def cart_to_pol(cart: Cartesian) -> Polar:
    if cart.is_2d:
        r = np.sqrt(cart.x**2 + cart.y**2)
        t = np.arctan2(cart.y, cart.x)
        return Polar.new_2d(r, t)
    elif cart.is_3d:
        pass
    else:
        raise ValueError("`cart` must be 2 or 3D")


# -----
# Types
# -----


@dataclass
class Camber:
    s: NDArray  # Position along camberline from 0 to 1
    angle: NDArray  # Angle in degrees
    chord: float = 1.0
    offset: NDArray = field(default_factory=lambda: np.zeros(2))

    @classmethod
    def from_coords(cls, xy: NDArray) -> "Camber":
        s = cum_length(xy)
        dydx = gradient(xy)
        angle_rad = np.arctan(dydx)
        angle_deg = np.degrees(angle_rad)
        return Camber(s, angle_deg)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        assert self.s.ndim == 1, "`s` must be one-dimensional"
        assert self.angle.ndim == 1, "`angle` must be one-dimensional"
        assert self.s.shape == self.angle.shape, "`s` and `angle` must have same shape"
        assert self.s[0] == 0.0, "`s` must start from 0"
        assert self.s[-1] == 1.0, "`s` must finish at 1"
        assert self.chord > 0, "`chord` must be positive"
        assert self.offset.shape == (2,), "`offset` must have shape (2,)"

    @property
    def angle_deg(self) -> NDArray:
        return self.angle

    @property
    def angle_rad(self) -> NDArray:
        return np.radians(self.angle)

    @property
    def coords(self) -> NDArray:
        x = integrate_from_zero(np.cos(self.angle_rad), self.s)
        y = integrate_from_zero(np.sin(self.angle_rad), self.s)
        xy = np.c_[x, y]  # This curve should have length 1
        return (
            self.chord * xy + self.offset[np.newaxis, :]
        )  # (1,) * (N, 2) + (1, 2) -> (N, 2)


@dataclass
class Thickness:
    s: NDArray  # Position along camberline from 0 to 1
    rel_thick: NDArray  # Thickness as a proportion of maximum thickness
    maxt: float = 1.0  # Maximum thickness

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        assert self.s.ndim == 1, "`s` must be one-dimensional"
        assert self.rel_thick.ndim == 1, "`rel_thick` must be one-dimensional"
        assert (
            self.s.shape == self.rel_thick.shape
        ), "`s` and `rel_thick` must have same shape"
        assert self.s[0] == 0.0, "`s` must start from 0"
        assert self.s[-1] == 1.0, "`s` must finish at 1"
        assert self.maxt > 0, "`maxt` must be positive"

    @property
    def abs_thick(self) -> NDArray:
        return self.maxt * self.rel_thick


CamberParams = dict[str, float]
ThicknessParams = dict[str, float]

CamberParamFunc = Callable[[CamberParams], Camber]
ThicknessParamFunc = Callable[[ThicknessParams], Thickness]


def eval_upper(camber: Camber, thickness: Thickness) -> Coords:
    raise NotImplementedError()


def eval_lower(camber: Camber, thickness: Thickness) -> Coords:
    raise NotImplementedError()


@dataclass(init=False)
class FlatBlade:
    coords: Coords

    def __init__(self, coords: Coords) -> None:
        self.coords = coords


@dataclass(init=False)
class SplitBlade:
    _upper_coords: Coords
    _lower_coords: Coords

    def __init__(self, upper: Coords, lower: Coords) -> None:
        self._upper_coords = upper
        self._lower_coords = lower

    @property
    def upper_coords(self) -> Coords:
        return self._upper_coords

    @property
    def lower_coords(self) -> Coords:
        return self._lower_coords


@dataclass(init=False)
class DistrBlade(SplitBlade):
    _camber: Camber
    _thickness: Thickness

    def __init__(self, camber: Camber, thickness: Thickness) -> None:
        self._camber = camber
        self._thickness = thickness

    @property
    def camber(self) -> Camber:
        return self._camber

    @property
    def thickness(self) -> Thickness:
        return self._thickness

    @property
    def upper_coords(self) -> Coords:
        return eval_upper(self._camber, self._thickness)

    @property
    def lower_coords(self) -> Coords:
        return eval_lower(self._camber, self._thickness)


@dataclass(init=False)
class ParamCamberBlade(DistrBlade):
    c_param: CamberParams
    c_func: CamberParamFunc

    def __init__(
        self, c_param: CamberParams, c_func: CamberParamFunc, thickness: Thickness
    ) -> None:
        self._thickness = thickness
        self.c_param = c_param
        self.c_func = c_func

    @property
    def camber(self) -> Camber:
        return self.c_func(self.c_param)


@dataclass(init=False)
class ParamThickBlade(DistrBlade):
    t_param: ThicknessParams
    t_func: ThicknessParamFunc

    def __init__(
        self, t_param: ThicknessParams, t_func: ThicknessParamFunc, camber: Camber
    ) -> None:
        self._camber = camber
        self.t_param = t_param
        self.t_func = t_func

    @property
    def thickness(self) -> Thickness:
        return self.t_func(self.t_param)


@dataclass(init=False)
class FullParamBlade(ParamThickBlade, ParamCamberBlade):
    def __init__(
        self,
        c_param: CamberParams,
        c_func: CamberParamFunc,
        t_param: ThicknessParams,
        t_func: ThicknessParamFunc,
    ) -> None:
        self.c_param = c_param
        self.c_func = c_func
        self.t_param = t_param
        self.t_func = t_func


coords = Cartesian(np.arange(12).reshape(6, 2))
coords.x
b = FlatBlade(coords)
pprint(b)

upper = Polar(np.arange(12).reshape((4, 3)))
lower = Polar(np.arange(12)[::-1].reshape((4, 3)))
b = SplitBlade(upper, lower)
pprint(b)
