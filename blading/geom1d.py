from numpy.typing import ArrayLike, NDArray
import numpy as np
from scipy.integrate import cumulative_trapezoid


def integrate_from_zero(y: ArrayLike, x: ArrayLike) -> NDArray:
    return np.r_[0, cumulative_trapezoid(y, x)]


def sum_from_zero(a: ArrayLike):
    return np.r_[0, np.cumsum(a)]
