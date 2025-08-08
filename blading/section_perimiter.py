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


def _is_clockwise(curve: PlaneCurve) -> bool:
    """
    Determine if a closed curve is oriented clockwise using the shoelace formula.

    Parameters
    ----------
    curve : PlaneCurve
        The curve to analyze.

    Returns
    -------
    bool
        True if the curve is oriented clockwise, False otherwise.
    """
    x, y = curve.x, curve.y
    # Shoelace formula for signed area
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    # For a closed curve, also include the wraparound term
    if not np.allclose(curve.start(), curve.end()):
        signed_area += 0.5 * (x[-1] * y[0] - x[0] * y[-1])

    # Negative signed area indicates clockwise orientation
    return signed_area < 0


def _reorient_curve(curve: PlaneCurve) -> PlaneCurve:
    """
    Reorient section to start from leading edge and go clockwise.

    Parameters
    ----------
    section : SectionPerimiter
        The section to reorient.

    Returns
    -------
    SectionPerimiter
        Reoriented section starting from leading edge and going clockwise.
    """
    # Find leading edge index
    le_index = _find_leading_edge_index(curve)

    # Reorder to start from leading edge
    coords_reordered = np.roll(curve.coords, -le_index, axis=0)
    param_reordered = np.roll(curve.param, -le_index, axis=0)

    # Create new curve starting from leading edge
    reordered_curve = PlaneCurve(coords_reordered, param_reordered)

    # Check if we need to reverse for clockwise orientation
    if not _is_clockwise(reordered_curve):
        # Reverse the curve (except the first point to maintain starting point)
        coords_reversed = np.concatenate(
            [
                coords_reordered[:1],  # Keep first point (leading edge)
                coords_reordered[1:][::-1],  # Reverse the rest
            ]
        )
        param_reversed = np.concatenate(
            [param_reordered[:1], param_reordered[1:][::-1]]
        )
        reordered_curve = PlaneCurve(coords_reversed, param_reversed)

    return reordered_curve
