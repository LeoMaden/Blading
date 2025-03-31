from dataclasses import dataclass
from itertools import count, product
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from blade import Camber, Section2D, Thickness
from intersect import intersection
from scipy.spatial import Delaunay


import geometry
import geometry.surfaces
from geometry.distributions import double_sided, cluster_left


@dataclass
class FitSectionResult:
    section: Section2D
    conv: NDArray
    n_arr: NDArray


@dataclass
class ImproveCamberConvergenceError(Exception):
    xy_camber: NDArray
    xy_camber_new: NDArray
    xy_section: NDArray
    delta: NDArray

    def __post_init__(self):
        super().__init__("Iterative camber improvement started to diverge")

    def plot(self):
        plt.figure()
        plt.plot(self.xy_section[:, 0], self.xy_section[:, 1], "k-")
        plt.plot(self.xy_camber[:, 0], self.xy_camber[:, 1], "g.-")
        plt.plot(self.xy_camber_new[:, 0], self.xy_camber_new[:, 1], "r.--")
        plt.axis("equal")

        plt.figure()
        plt.semilogy(self.delta)

        plt.show()


@dataclass
class NormalIntersectionsError(Exception):
    xy_section: NDArray
    xy_camber: NDArray
    line: NDArray

    def __post_init__(self):
        super().__init__(
            "Error calculating intersections between camber line normals and section"
        )

    def plot(self):
        plt.figure()
        plt.plot(self.xy_section[:, 0], self.xy_section[:, 1], "k-")
        plt.plot(self.xy_camber[:, 0], self.xy_camber[:, 1], "b.-")
        plt.plot(self.line[:, 0], self.line[:, 1], "rx-")
        plt.axis("equal")
        plt.show()


def split_section(section: NDArray) -> tuple[NDArray, NDArray]:
    """
    Split an aerofoil section into upper and lower surfaces using the minimum and maximum
    x coordinates as the leading and trailing edge points respectively.

    Parameters
    ----------
    section : ndarray, shape (N, 2)
        Array of the x and y coordinates of the section, points should be ordered going
        clockwise.

    Returns
    -------
    (upper, lower) : (ndarray, ndarray)
        Returns the coordinates of the upper and lower surfaces, ordered from leading edge
        to trailing edge.
    """
    ile = np.argmin(section[:, 0])
    ite = np.argmax(section[:, 0])

    if ile > ite:
        upper = np.r_[section[ile - len(section) :], section[: ite + 1]]
        lower = section[ite : ile + 1][::-1]
    elif ite > ile:
        upper = section[ile : ite + 1]
        lower = np.r_[section[ite:], section[: ile + 1]][::-1]
    else:
        raise Exception("Indices of minimum and maximum x coordinate are the same")

    return upper, lower


def extend_le_te(xy_camber, xy_section):
    c = xy_camber
    chord = geometry.total_length(c)
    ext_dist = 0.5 * chord

    # Extend to LE
    d1 = c[[2]] - c[[1]]
    d1 /= np.linalg.norm(d1, axis=1)
    c1 = np.r_[c[[1]] - ext_dist * d1, c[[1]]]
    xi, yi = intersection(*xy_section.T, *c1.T)
    xy_camber[0] = np.c_[xi[0], yi[0]]

    d2 = c[[-2]] - c[[-3]]
    d2 /= np.linalg.norm(d2, axis=1)
    c2 = np.r_[c[[-2]], c[[-2]] + ext_dist * d2]
    xi, yi = intersection(*xy_section.T, *c2.T)
    xy_camber[-1] = np.c_[xi[0], yi[0]]

    return c


def calc_normal_intersections(xy_camber, xy_section, t_ref):
    normals = geometry.calc_normals(xy_camber)

    intersections = []
    # Bisect normal intersections
    for i in range(1, len(xy_camber) - 1):
        top = xy_camber[i] + 0.5 * t_ref * normals[i]
        bot = xy_camber[i] - 0.5 * t_ref * normals[i]
        line = np.c_[bot, top].T

        xi, yi = intersection(
            xy_section[:, 0], xy_section[:, 1], line[:, 0], line[:, 1]
        )
        if len(xi) != 2:
            raise NormalIntersectionsError(xy_section, xy_camber, line)

        intersections.append(np.c_[xi, yi])

    return intersections


def improve_camber(xy_camber, xy_section, t_ref) -> NDArray:
    xy_camber_new = np.zeros_like(xy_camber)
    intersections = calc_normal_intersections(xy_camber, xy_section, t_ref)

    # Bisect normal intersections
    for i, coords in enumerate(intersections, 1):
        xy_camber_new[i] = np.mean(coords, axis=0)

    xy_camber_new = extend_le_te(xy_camber_new, xy_section)

    return xy_camber_new


def improve_camber_iter(xy_camber, xy_section, t_ref, tol=1e-9):
    delta = []
    for i in count():
        xy_camber_new = improve_camber(xy_camber, xy_section, t_ref)
        delta.append(np.linalg.norm(xy_camber - xy_camber_new))

        if i < 1:
            xy_camber = xy_camber_new
            continue
        elif delta[i] > delta[i - 1]:
            raise ImproveCamberConvergenceError(
                xy_camber=xy_camber,
                xy_camber_new=xy_camber_new,
                xy_section=xy_section,
                delta=np.array(delta),
            )

        xy_camber = xy_camber_new

        if delta[i] < tol:
            break

    return xy_camber, delta


def calc_thickness(xy_camber, xy_section, t_ref) -> NDArray:
    thickness = np.zeros(len(xy_camber))
    intersections = calc_normal_intersections(xy_camber, xy_section, t_ref)

    for i, coords in enumerate(intersections, 1):
        thickness[i] = np.linalg.norm(np.diff(coords, axis=0))

    return thickness


def fit_section(xy_section, n_fit: int = 200, n_fine: int = 250):
    upper, lower = split_section(xy_section)

    s = np.linspace(0, 1, 50)
    a = geometry.curve_interp(upper, s, deg=3, normalise=True)
    b = geometry.curve_interp(lower, s, deg=3, normalise=True)
    xy_camber = (a + b) / 2

    # Calculate thickness
    t_ref = 1.5 * np.max(np.linalg.norm(a - b, axis=1))

    # Close section
    xy_section = np.r_[xy_section, xy_section[[0]]]

    # Linear spacing
    n_arr = [*np.arange(50, n_fit, 50), n_fit, n_fit]
    conv = []
    for n in n_arr:
        s = np.linspace(0, 1, n)
        xy_camber = geometry.curve_interp(xy_camber, s, deg=3, normalise=True)
        xy_camber_new, delta = improve_camber_iter(
            xy_camber, xy_section, t_ref, tol=1e-12
        )
        conv += delta
        xy_camber = xy_camber_new

    # Reinterpolate with extra fine spacing
    s = double_sided(0, 1, 2 * n_fine, 0.7, r=1.2)
    xy_camber = geometry.curve_interp(xy_camber, s, deg=3, normalise=True)
    t = calc_thickness(xy_camber, xy_section, t_ref)

    # Reinterpolate with equal fine spacing along s-t curve
    c = np.c_[s, t / max(t)]
    sc = np.linspace(0, 1, n_fine)
    c = geometry.curve_interp(c, sc, deg=3, normalise=True)
    s = c[:, 0]
    xy_camber = geometry.curve_interp(xy_camber, s, deg=3, normalise=True)
    t = calc_thickness(xy_camber, xy_section, t_ref)

    # Construct section
    camber = Camber.from_xy(xy_camber)
    thickness = Thickness(s, t / camber.chord)
    section = Section2D.new(camber, thickness)

    return FitSectionResult(section=section, conv=np.array(conv), n_arr=np.array(n_arr))


def fit_blade(sections: list[NDArray]):
    #     num_pts, num_sections, _ = blade.shape
    #     x, r, rt = np.moveaxis(blade, -1, 0)

    #     camber = np.zeros((50, num_sections, 3))
    #     thickness = np.zeros((50, num_sections))

    #     for j in range(num_sections):
    #         xy_section = np.c_[x[:, j], rt[:, j]]
    #         xr_section = np.c_[x[:, j], r[:, j]]

    #         res = fit_section(xy_section)
    #         if not res.success:
    #             raise Exception("Did not fit section within tolerance")
    #         xy_camber = res.camber

    #         # Interp section camberline onto section mid xr coordinates
    #         xr_section, _ = section_camber_guess(xr_section)
    #         r_camber = np.interp(xy_camber[:, 0], xr_section[:, 0], xr_section[:, 1])

    #         camber[:, j, :] = np.c_[xy_camber[:, 0], r_camber, xy_camber[:, 1]]
    #         # thickness[:, j] = res.thickness

    #     return camber
    raise NotImplementedError()
