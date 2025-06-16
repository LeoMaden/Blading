from numpy.typing import NDArray
import numpy as np
from geometry.curves import PlaneCurve
from dataclasses import dataclass
from .blade import Section, FlatSection


Mask = NDArray[np.bool]


@dataclass(frozen=True)
class CamberIteration:
    lines: NDArray
    upper_intersections: NDArray
    lower_intersections: NDArray
    camber: PlaneCurve
    thickness: NDArray
    delta: float


@dataclass(frozen=True)
class ApproxCamberResult:
    mask_LE: Mask
    mask_TE: Mask
    upper_split: PlaneCurve
    lower_split: PlaneCurve
    camber_iterations: list[CamberIteration]
    n_iter: int
    section: Section


def find_LE_TE(
    section: FlatSection,
    thresh_LE: float = 0.5,
    thresh_TE: float = 1.0,
    mask_front: Mask | None = None,
) -> tuple[Mask, Mask]:
    """Split a blade section according to curvature.

    Parameters
    ----------
    section : FlatSection
        The section to split.
    thresh_LE : float, optional
        The numbers of orders of magnitude below the maximum leading edge
        curvature that will be included in the leading edge, by default 0.5
    thresh_TE : float, optional
        The same as `thresh_LE` for the trailing edge, by default 1.0
    mask_front : NDArray[np.bool] | None, optional
        Boolean array marking where the front of the blade is, by default None.
        If this parameter is None, the first and last 25% of the curve defining
        `section` is assumed to be the "front" of the blade.

    Returns
    -------
    tuple[Mask, Mask] = (mask_LE, mask_TE)
        Boolean arrays marking the location of the leading and trailing edges.
    """
    curve = section.curve.reparameterise_unit().normalise()
    k = curve.curvature()

    if mask_front is None:
        x_mid = (min(curve.x) + max(curve.x)) / 2
        mask_front = curve.x < x_mid

    mask_rear = ~mask_front  # type: ignore

    max_k_LE = np.max(k[mask_front])
    max_k_TE = np.max(k[mask_rear])

    mask_LE_k = k > max_k_LE / 10**thresh_LE  # Orders of magnitude below peak
    mask_TE_k = k > max_k_TE / 10**thresh_TE

    mask_LE = mask_front & mask_LE_k
    mask_TE = mask_rear & mask_TE_k

    return mask_LE, mask_TE


def improve_camber(
    camber: PlaneCurve,
    thickness: NDArray,
    section: FlatSection,
    relax_factor: float = 0.2,
) -> CamberIteration:
    N = len(thickness)
    upper_coords = np.zeros((N, 2))
    lower_coords = np.zeros((N, 2))

    lines = np.zeros((N, 3, 2))

    normal = camber.signed_normal()
    _THICKNESS_SCALE = 1.2

    for i in range(N):
        c = camber.coords[i]
        u = c + _THICKNESS_SCALE * thickness[i] * normal.coords[i]
        l = c - _THICKNESS_SCALE * thickness[i] * normal.coords[i]
        int_u = section.curve.intersect_coords(np.c_[c, u].T)
        int_l = section.curve.intersect_coords(np.c_[l, c].T)
        assert int_u.shape[0] == 1, "Too many intersections"
        assert int_l.shape[0] == 1, "Too many intersections"
        upper_coords[i] = int_u[0]
        lower_coords[i] = int_l[0]
        lines[i] = np.c_[l, c, u].T

    camber_new = PlaneCurve((upper_coords + lower_coords) / 2, camber.param)
    thickness_new = np.linalg.norm(upper_coords - lower_coords, axis=1)

    # Relaxation
    delta = (camber_new - camber) * relax_factor
    camber_new = camber + delta

    return CamberIteration(
        lines=lines,
        upper_intersections=upper_coords,
        lower_intersections=lower_coords,
        camber=camber_new,
        thickness=thickness_new,
        delta=np.sum(delta.norm()),
    )


def extend(camber_line: PlaneCurve, section: FlatSection) -> PlaneCurve:
    # Extend leading and trailing edges
    chord = camber_line.length()
    tangent = camber_line.tangent()
    LE_line = np.c_[
        camber_line.start() - tangent.coords[0] * 0.1 * chord, camber_line.start()
    ].T
    LE_point = section.curve.intersect_coords(LE_line)
    TE_line = np.c_[
        camber_line.end() + tangent.coords[-1] * 0.1 * chord, camber_line.end()
    ].T
    TE_point = section.curve.intersect_coords(TE_line)
    assert LE_point.shape[0] == 1, "Too many intersections"
    assert TE_point.shape[0] == 1, "Too many intersections"
    new_coords = np.r_[LE_point, camber_line.coords, TE_point]
    camber_line = PlaneCurve.new_unit_speed(new_coords).normalise()
    return camber_line


def approx_camber_line(
    section: FlatSection,
    tol: float = 1e-7,
    thresh_LE: float = 0.5,
    thresh_TE: float = 1.0,
    mask_front: Mask | None = None,
    relax_factor: float = 0.2,
    num_points_LE: int = 25,
    num_points_TE: int = 25,
) -> ApproxCamberResult:
    mask_LE, mask_TE = find_LE_TE(section, thresh_LE, thresh_TE, mask_front)
    not_LE_TE = ~mask_LE & ~mask_TE
    diff = np.diff(not_LE_TE.astype(int))
    starts = np.where(diff == 1)[0] + 1  # start of 1s
    ends = np.where(diff == -1)[0] + 1  # end of 1s

    upper = section.curve[starts[0] : ends[0]]
    lower = section.curve[starts[1] : ends[1]][::-1]

    # Constant speed normalised with same parameter
    upper = upper.reparameterise_unit().normalise()
    lower = lower.reparameterise_unit().normalise().interpolate(upper.param)
    camber_guess = (upper + lower) / 2
    thickness_guess = (upper - lower).norm()

    camber_iters: list[CamberIteration] = []
    while True:
        # plot_plane_curve(camber_guess, ax)
        iter = improve_camber(
            camber_guess,
            thickness_guess,
            section,
            relax_factor,
        )
        camber_iters.append(iter)
        if iter.delta <= tol:
            break
        camber_guess = iter.camber
        thickness_guess = iter.thickness

    camber_new = extend(iter.camber, section)
    thickness_new = iter.thickness

    # Add points near leading edge
    t = camber_new.param
    spacing_LE = np.cos(np.linspace(0, np.pi / 2, num_points_LE))
    t_new_LE = t[1] - (t[1] - t[0]) * spacing_LE

    spacing_TE = np.cos(np.linspace(np.pi / 2, 0, num_points_TE))
    t_new_TE = t[-2] + (t[-1] - t[-2]) * spacing_TE

    t_new = np.r_[t_new_LE, t[2:-2], t_new_TE]
    camber_new = camber_new.interpolate(t_new, k=2)

    # Ensure thickness matches camber
    thickness_new = np.r_[
        np.repeat(thickness_new[0], num_points_LE),
        thickness_new[1:-1],
        np.repeat(thickness_new[-1], num_points_TE),
    ]
    iter = improve_camber(camber_new[1:-1], thickness_new[1:-1], section)
    thickness_new = np.r_[0, iter.thickness, 0]

    section_new = Section(camber_new, thickness_new, section.stream_line)

    return ApproxCamberResult(
        mask_LE=mask_LE,
        mask_TE=mask_TE,
        upper_split=upper,
        lower_split=lower,
        camber_iterations=camber_iters,
        n_iter=len(camber_iters),
        section=section_new,
    )
