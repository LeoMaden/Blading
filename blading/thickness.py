from numpy.typing import NDArray
import numpy as np
from numpy.polynomial import Polynomial


def remove_round_TE(
    s: NDArray,
    thickness: NDArray,
    mask_rear: NDArray[np.bool] | None = None,
    mask_high_curv: NDArray[np.bool] | None = None,
) -> NDArray:
    # Mask for the rear section of the blade.
    if mask_rear is None:
        mask_rear = s >= 0.8

    # Mask for areas of high curvature on the blade
    if mask_high_curv is None:
        grad = np.gradient(thickness, s)
        curv = np.abs(np.gradient(grad, s))
        avg_curv = np.median(curv)
        mask_high_curv = curv > 4 * avg_curv
        assert mask_high_curv is not None

    # Fit straight line through linear part of thickness
    mask_linear = mask_rear & ~mask_high_curv  # linear part
    mask_round_TE = mask_rear & mask_high_curv  # round part
    poly = Polynomial.fit(s[mask_linear], thickness[mask_linear], deg=1)

    # Ensure new TE has continuous value.
    y1 = thickness[mask_linear][-1]
    y2 = poly(s[mask_linear][-1])
    poly -= y2 - y1

    # Extrapolate straight line to TE
    thickness = thickness.copy()
    thickness[mask_round_TE] = poly(s[mask_round_TE])

    return thickness
