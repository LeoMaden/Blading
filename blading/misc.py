import numpy as np
from numpy.typing import NDArray
from numpy.polynomial import Polynomial


# Interpolation between hub and casing lines to give pseudo-streamsurfaces
def interp_streamsurfs(spanwise_pos: NDArray, xr_hub: NDArray, xr_cas: NDArray):
    xr_streamsurf = np.tensordot(xr_hub, 1 - spanwise_pos, axes=0) + np.tensordot(
        xr_cas, spanwise_pos, axes=0
    )
    xr_streamsurf = np.swapaxes(xr_streamsurf, 1, 2)
    return xr_streamsurf


# Find centroid of thickness distribution
def find_centroid(chordwise_pos, thickness):
    centroid_pos = np.trapz(thickness * chordwise_pos, chordwise_pos) / np.trapz(
        thickness, chordwise_pos
    )
    return centroid_pos


def fit_te_poly(thickness):
    s, t = thickness.s, thickness.t.copy()

    # Threshold off gradient of thickness
    dtds = np.abs(np.gradient(t, s))
    thresh = np.interp(0.8, s, dtds)  # dt/dx at 80% chord
    try:
        i_te = np.min(np.where((dtds >= 1.2 * thresh) & (s > 0.8))[0])
    except ValueError:
        i_te = -1

    # Linearly extend trailing edge wedge
    i_wedge = (s <= s[i_te]) & (s >= 0.9)
    te_poly = Polynomial.fit(s[i_wedge], t[i_wedge], 1)
    return te_poly, i_te


def calc_te_t_param(thickness):
    te_poly, i_te = fit_te_poly(thickness)
    t_te = te_poly(1)
    angle_te = np.degrees(np.arctan(-te_poly.deriv(1)(1)))

    return t_te, angle_te
