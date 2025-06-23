from numpy.typing import NDArray
import numpy as np
from numpy.polynomial import Polynomial


def remove_round_TE(
    s_norm: NDArray,
    thickness: NDArray,
    mask_rear: NDArray[np.bool] | None = None,
    mask_high_curv: NDArray[np.bool] | None = None,
) -> NDArray:
    assert s_norm[0] == 0, "Must start at 0"
    assert s_norm[-1] == 1, "Must end at 1"

    # Mask for the rear section of the blade.
    if mask_rear is None:
        mask_rear = s_norm >= 0.8

    # Mask for areas of high curvature on the blade
    if mask_high_curv is None:
        grad = np.gradient(thickness, s_norm)
        curv = np.abs(np.gradient(grad, s_norm))
        avg_curv = np.median(curv)
        mask_high_curv = curv > 4 * avg_curv
        assert mask_high_curv is not None

    # Fit straight line through linear part of thickness
    mask_linear = mask_rear & ~mask_high_curv  # linear part
    mask_round_TE = mask_rear & mask_high_curv  # round part
    poly = Polynomial.fit(s_norm[mask_linear], thickness[mask_linear], deg=1)

    # Ensure new TE has continuous value.
    y1 = thickness[mask_linear][-1]
    y2 = poly(s_norm[mask_linear][-1])
    poly -= y2 - y1

    # Extrapolate straight line to TE
    thickness = thickness.copy()
    thickness[mask_round_TE] = poly(s_norm[mask_round_TE])

    return thickness


def add_round_TE(
    s_norm: NDArray,
    thickness: NDArray,
    chord: float,
):
    assert s_norm[0] == 0, "Must start at 0"
    assert s_norm[-1] == 1, "Must end at 1"

    # Thickness relative to chord.
    thickness = thickness.copy() / chord

    # Find trailing edge angle.
    x1, x2 = s_norm[[-2, -1]]
    y1, y2 = thickness[[-2, -1]]
    tan_angle_TE = (y1 - y2) / (x2 - x1)

    # Trailing edge thickness.
    t_TE = thickness[-1]

    # Analytical formula for round TE tangent to linear section.
    a = np.arctan(0.5 * tan_angle_TE)
    b = 2 * np.cos(a) - (1 - np.sin(a)) * tan_angle_TE
    rad_TE = t_TE / b

    # Create points on round TE and interpolate given `s` distribution.
    pts = np.linspace(a, np.pi / 2, 70)
    x_te = 1 - rad_TE + rad_TE * np.sin(pts)
    y_te = 2 * rad_TE * np.cos(pts)
    mask_round_TE = s_norm > min(x_te)
    thickness[mask_round_TE] = np.interp(s_norm[mask_round_TE], x_te, y_te)

    return thickness * chord, rad_TE * chord
