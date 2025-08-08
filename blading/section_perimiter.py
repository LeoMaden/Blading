from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional
from geometry.curves import PlaneCurve
import numpy as np


@dataclass(frozen=True)
class SectionPerimiter:
    """A class representing the perimeter of a blade section."""

    curve: PlaneCurve
    stream_line: Optional[PlaneCurve]

    def __post_init__(self):
        # Reorient the curve to start from the leading edge and go clockwise
        curve = _reorient_curve(self.curve)

        # Ensure normalised arc length parameterisation
        curve = curve.reparameterise_unit().normalise()

        object.__setattr__(self, "curve", curve)

    def with_stream_line(self, stream_line: PlaneCurve) -> "SectionPerimiter":
        """
        Create a new SectionPerimiter with the specified stream line.

        Parameters
        ----------
        stream_line : PlaneCurve
            The stream line to associate with this section.

        Returns
        -------
        SectionPerimiter
            A new SectionPerimiter instance with the specified stream line.
        """
        return SectionPerimiter(curve=self.curve, stream_line=stream_line)


def _find_leading_edge_index(curve: PlaneCurve) -> int:
    """
    Find the index of the leading edge (minimum x-coordinate) of the curve.

    Parameters
    ----------
    curve : PlaneCurve
        The curve to analyze.

    Returns
    -------
    int
        Index of the point with minimum x-coordinate.
    """
    return int(np.argmin(curve.x))


def _is_clockwise(curve_coords: NDArray) -> bool:
    """
    Determine if a closed curve is oriented clockwise using the shoelace formula.

    Parameters
    ----------
    curve_coords : NDArray, shape (n_points, 2)
        Coordinates of the curve as a 2D array. Must not be closed.

    Returns
    -------
    bool
        True if the curve is oriented clockwise, False otherwise.
    """
    x, y = curve_coords[:, 0], curve_coords[:, 1]
    # Shoelace formula for signed area
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    # Negative signed area indicates clockwise orientation
    return signed_area < 0


def _reorient_curve(curve: PlaneCurve) -> PlaneCurve:
    """
    Reorient section to start from leading edge and go clockwise.

    Parameters
    ----------
    curve : PlaneCurve
        The curve to reorient.

    Returns
    -------
    PlaneCurve
        Reoriented curve starting from leading edge and going clockwise.
    """
    # Find leading edge index
    le_index = _find_leading_edge_index(curve)

    # Remove duplicate point if curve is closed
    is_closed = curve.is_closed
    curve_coords = curve.coords
    if is_closed:
        curve_coords = curve_coords[:-1]

    # Reorder to start from leading edge
    coords_reordered = np.roll(curve_coords, -le_index, axis=0)

    if not _is_clockwise(coords_reordered):
        # Reverse the curve (except the first point to maintain starting point)
        coords_reordered = np.r_[coords_reordered[0], coords_reordered[1:][::-1]]

    # Ensure the curve is closed if it was originally closed
    if is_closed:
        coords_reordered = np.r_[coords_reordered, coords_reordered[[0]]]

    reordered_curve = PlaneCurve(coords_reordered, curve.param)
    return reordered_curve
