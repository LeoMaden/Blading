from dataclasses import dataclass, field
from enum import Enum, auto
from pprint import pprint
from typing import (
    Callable,
    Generic,
    Iterable,
    Self,
    TypeVar,
)
from numpy.typing import NDArray
import numpy as np
from . import geom1d, geom2d


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
    def empty(cls) -> "Polar":
        return Polar([])

    @classmethod
    def new_2d(cls, x, t) -> "Polar":
        return Polar(np.stack((x, t), axis=-1))

    @classmethod
    def new_3d(cls, x, r, t) -> "Polar":
        return Polar(np.stack((x, t, r), axis=-1))

    @property
    def is_2d(self) -> bool:
        return self.shape[-1] == 2

    @property
    def is_3d(self) -> bool:
        return self.shape[-1] == 3

    @property
    def x(self) -> "Polar":
        return Polar(self[..., 0])

    @property
    def t(self) -> "Polar":
        return Polar(self[..., 1])

    @property
    def r(self) -> "Polar":
        if not self.is_3d:
            raise Exception("Array must be 3D")
        return Polar(self[..., 2])


class Cartesian(NDArray):
    def __new__(cls, object) -> Self:
        obj = np.asarray(object).view(cls)
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    @classmethod
    def empty(cls) -> "Cartesian":
        return Cartesian([])

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


class Meridional(NDArray):
    def __new__(cls, object) -> Self:
        obj = np.asarray(object).view(cls)
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    @classmethod
    def empty(cls) -> "Meridional":
        return Meridional([])

    @classmethod
    def new(cls, x, r) -> "Meridional":
        return Meridional(np.stack((x, r), axis=-1))

    @property
    def x(self) -> "Cartesian":
        return Cartesian(self[..., 0])

    @property
    def r(self) -> "Cartesian":
        return Cartesian(self[..., 1])


T = TypeVar("T", bound=Polar | Cartesian)
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
        raise NotImplementedError()
    else:
        raise ValueError("`pol` must be 2D or 3D")


def cart_to_pol(cart: Cartesian) -> Polar:
    if cart.is_2d:
        r = np.sqrt(cart.x**2 + cart.y**2)
        t = np.arctan2(cart.y, cart.x)
        return Polar.new_2d(r, t)
    elif cart.is_3d:
        raise NotImplementedError()
    else:
        raise ValueError("`cart` must be 2D or 3D")


# ----------------------------------
# Camber and thickness distributions
# ----------------------------------


@dataclass
class Camber(Generic[T]):
    s: NDArray  # Position along camberline from 0 to 1
    angle: NDArray  # Angle in degrees
    t: T
    chord: float = 1.0
    offset: NDArray = field(default_factory=lambda: np.zeros(2))

    @classmethod
    def from_coords(cls, xy: T) -> "Camber":
        s = geom2d.cum_length(xy)
        dydx = geom2d.gradient(xy)
        angle_rad = np.arctan(dydx)
        angle_deg = np.degrees(angle_rad)
        return Camber(s, angle_deg, type(xy).empty())

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
    def coords(self) -> T:
        x = geom1d.integrate_from_zero(np.cos(self.angle_rad), self.s)
        y = geom1d.integrate_from_zero(np.sin(self.angle_rad), self.s)
        xy = np.c_[x, y]  # This curve should have length 1
        return type(self.t)(
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


def eval_upper_2d(camber: Camber[T], thickness: Thickness) -> T:
    xy = camber.coords
    norm = geom2d.normal(xy)
    t = thickness.abs_thick
    return type(camber.coords)(
        xy + 0.5 * norm * t[:, np.newaxis]
    )  # (N, 2) + (1,) * (N, 2) * (N, 1) -> (N, 2)


def eval_lower_2d(camber: Camber[T], thickness: Thickness) -> T:
    xy = camber.coords
    norm = geom2d.normal(xy)
    t = thickness.abs_thick
    return type(camber.coords)(
        xy - 0.5 * norm * t[:, np.newaxis]
    )  # (N, 2) - (1,) * (N, 2) * (N, 1) -> (N, 2)


def eval_upper_3d(
    cambers: Iterable[Camber[T]], thicknesses: Iterable[Thickness[T]]
) -> T:
    raise NotImplementedError()


# -------------------------
# Blade section definitions
# -------------------------


@dataclass(init=False)
class FlatSection(Generic[T]):
    coords: T

    def __init__(self, coords: T) -> None:
        self.coords = coords


@dataclass(init=False)
class SplitSection(Generic[T]):
    _upper_coords: T
    _lower_coords: T

    def __init__(self, upper: T, lower: T) -> None:
        self._upper_coords = upper
        self._lower_coords = lower

    @property
    def upper_coords(self) -> T:
        return self._upper_coords

    @property
    def lower_coords(self) -> T:
        return self._lower_coords


@dataclass(init=False)
class DistrSection(SplitSection, Generic[T]):
    _camber: Camber
    _thickness: Thickness
    _type: T

    def __init__(self, camber: Camber[T], thickness: Thickness) -> None:
        self._camber = camber
        self._thickness = thickness

    def __post_init__(self):
        self._type = self.camber.coords.empty()

    @property
    def camber(self) -> Camber:
        return self._camber

    @property
    def thickness(self) -> Thickness:
        return self._thickness

    @property
    def upper_coords(self) -> T:
        upper = eval_upper_2d(self._camber, self._thickness)
        return type(self.camber.coords)(upper)

    @property
    def lower_coords(self) -> T:
        lower = eval_lower_2d(self._camber, self._thickness)
        return type(self.camber.coords)(lower)


@dataclass(init=False)
class ParamCamberSection(DistrSection, Generic[T]):
    c_param: CamberParams
    c_func: CamberParamFunc

    def __init__(
        self,
        c_param: CamberParams,
        c_func: CamberParamFunc,
        thickness: Thickness,
        t: T,
    ) -> None:
        self._thickness = thickness
        self.c_param = c_param
        self.c_func = c_func
        self._type = t

    @property
    def camber(self) -> Camber:
        return self.c_func(self.c_param)


@dataclass(init=False)
class ParamThickSection(DistrSection, Generic[T]):
    t_param: ThicknessParams
    t_func: ThicknessParamFunc

    def __init__(
        self,
        t_param: ThicknessParams,
        t_func: ThicknessParamFunc,
        camber: Camber[T],
    ) -> None:
        self._camber = camber
        self.t_param = t_param
        self.t_func = t_func

    @property
    def thickness(self) -> Thickness:
        return self.t_func(self.t_param)


@dataclass(init=False)
class ParamSection(ParamThickSection, ParamCamberSection, Generic[T]):
    def __init__(
        self,
        c_param: CamberParams,
        c_func: CamberParamFunc,
        t_param: ThicknessParams,
        t_func: ThicknessParamFunc,
        t: T,
    ) -> None:
        self.c_param = c_param
        self.c_func = c_func
        self.t_param = t_param
        self.t_func = t_func
        self._type = t


AnySection = (
    FlatSection[T]
    | SplitSection[T]
    | DistrSection[T]
    | ParamCamberSection[T]
    | ParamThickSection[T]
    | ParamSection[T]
)


# -----------------
# Blade definitions
# -----------------

SecT = TypeVar("SecT", bound=AnySection)


class CamberNormal(Enum):
    Line = auto()  # Thickness is added normal to the camberline in 2D
    Surface = auto()  # Thickness is added normal to the camber surface in 3D


@dataclass
class FlatBlade(Generic[T]):
    sections: list[FlatSection[T]]
    streamtubes: list[Meridional]


@dataclass
class SplitBlade(Generic[T]):
    sections: list[SplitSection[T]]
    streamtubes: list[Meridional]

    _type: T = field(init=False)

    def __post_init__(self) -> None:
        self._type = self.sections[0].upper_coords[0]

    @property
    def upper_coords(self) -> T:
        upper_list = [s.upper_coords for s in self.sections]
        upper_arr = np.stack(upper_list, axis=1)
        return type(self._type)(upper_arr)

    @property
    def lower_coords(self) -> T:
        lower_list = [s.lower_coords for s in self.sections]
        lower_arr = np.stack(lower_list, axis=1)
        return type(self._type)(lower_arr)


@dataclass
class Blade(Generic[SecT]):
    sections: list[SecT]
    streamtubes: list[Meridional]
