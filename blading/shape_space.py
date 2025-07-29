import numpy as np


def value(x, z, zT):
    """
    Calculate the value of the state-space function

    Parameters
    ----------
    x : arraylike
        x coordinate (from 0 to 1)
    z : arraylike
        thickness at each x coordinate
    zT : arraylike
        thickness at the trailing edge (x=1)
    """
    x = np.asarray(x)
    z = np.asarray(z)

    # Calculate denominator and find valid points
    denom = x**0.5 * (1 - x)
    eps = 1e-12
    valid = np.abs(denom) > eps

    # Initialize result array
    result = np.full_like(x, np.nan, dtype=float)

    # Calculate result only for valid points
    if np.any(valid):
        result[valid] = (z[valid] - x[valid] * zT) / denom[valid]

    # Use interpolation to fill invalid points
    # np.interp uses nearest neighbour extrapolation
    if np.any(~valid) and np.sum(valid) >= 2:
        result[~valid] = np.interp(x[~valid], x[valid], result[valid])

    return result


def deriv1(x, z, zT, zp):
    """
    Calculate the first derivative of the state-space function

    Parameters
    ----------
    x : arraylike
        x coordinate (from 0 to 1)
    z : arraylike
        thickness at each x coordinate
    zT : arraylike
        thickness at the trailing edge (x=1)
    zp : arraylike
        thickness gradient dz/dx at each x coordinate
    """
    a = (3 * x - 1) * z
    b = -2 * x * (x - 1) * zp
    c = -x * (x + 1) * zT
    d = 2 * (1 - x) ** 2 * x**1.5
    return (a + b + c) / d


def deriv2(x, z, zT, zp, zpp):
    """
    Calculate the second derivative of the state-space function

    Parameters
    ----------
    x : arraylike
        x coordinate (from 0 to 1)
    z : arraylike
        thickness at each x coordinate
    zT : arraylike
        thickness at the trailing edge (x=1)
    zp : arraylike
        thickness gradient dz/dx at each x coordinate
    zpp : arraylike
        thickness curvature d^2z/dx^2 at each x coordinate
    """
    a = -(15 * x**2 - 10 * x + 3) * z
    b = 4 * x * (x - 1) * (3 * x - 1) * zp
    c = -4 * x**2 * (x - 1) ** 2 * zpp
    d = x * (3 * x**2 + 6 * x - 1) * zT
    e = 4 * (x - 1) ** 3 * x**2.5
    return (a + b + c + d) / e


def inverse(x, ss, zT):
    """
    Calculate the inverse of the shape space function

    x : arraylike
        x coordinate (from 0 to 1)
    ss : arraylike
        value of the state space function at each x coordinate
    zT : arraylike
        thickness at the trailing edge (x=1)
    """
    return ss * (np.sqrt(x) * (1 - x)) + x * zT
