import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid


# ------------------
# Geometry functions
# ------------------


def integrate_from_zero(y: ArrayLike, x: ArrayLike) -> NDArray:
    return np.r_[0, cumulative_trapezoid(y, x)]


def sum_from_zero(a: ArrayLike):
    return np.r_[0, np.cumsum(a)]


def length(xy: ArrayLike) -> NDArray:
    dxy = np.diff(xy, axis=0)
    ds = np.linalg.norm(dxy, axis=1)
    return np.sum(ds)


def cum_length(xy: ArrayLike) -> NDArray:
    dxy = np.diff(xy, axis=0)
    ds = np.linalg.norm(dxy, axis=1)
    return sum_from_zero(ds)


def gradient(xy: ArrayLike) -> NDArray:
    xy = np.asarray(xy)
    return np.gradient(xy[..., 1], xy[..., 0])
