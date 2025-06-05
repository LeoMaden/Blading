from enum import Enum, auto
from typing import Callable, Self
from numpy.typing import NDArray
import numpy as np


"""
In the most general sense, blade sections do not know about thickness or camber,
they are just a set of coordinates for the upper and lower surfaces.

sec_2d = Section2D.cart(xy_upper, xy_lower)  # 2D flat blade
sec_2d = Section2D.pol(xrt_upper, xrt_lower)  # 2D blade around annulus
sec_2d = Section2D.from_dist(camber, thickness)

sec_3d = Section3D(xyz_upper, xy_lower)




"""


# -------------------------------------------
# Coordinate systems
# -------------------------------------------


class Polar(NDArray):
    def __new__(cls, input_array) -> Self:
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

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
    def __new__(cls, input_array) -> Self:
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

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


# --------------------------------------
# Conversions between coordinate systems
# --------------------------------------


def pol_to_cart(pol: Polar) -> Cartesian:
    raise NotImplementedError()


def cart_to_pol(cart: Cartesian) -> Polar:
    raise NotImplementedError()


# -----
# Types
# -----


class Blade:
    coords: Polar | Cartesian
    pass


class Section2D:
    pass


class Section:
    pass


class Camber:
    pass


class Thickness:
    pass


CamberParamFunc = Callable[[dict[str, float]], Camber]
ThicknessParamFunc = Callable[[dict[str, float]], Thickness]

p = Polar(np.arange(12).reshape((4, 3)))
print(f"{p.shape = }")
print(f"{p.is_3d = }")
print(f"{p = }")
# print(f"{p.x = }")
print(f"{p.r = }")
print(f"{p.t = }")
