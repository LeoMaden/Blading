from dataclasses import dataclass, field
from enum import Enum
from .thickness import Thickness, ThicknessParams, create_thickness, fit_LE_circle
from .camber import Camber, CamberParams
from geometry.curves import PlaneCurve
import numpy as np
from numpy.typing import NDArray
from matplotlib.patches import Circle as mplCircle


class ReferencePoint(Enum):
    """Enumeration of valid reference points for blade sections."""

    LeadingEdge = "leading_edge"
    TrailingEdge = "trailing_edge"
    Centroid = "centroid"

    def __str__(self) -> str:
        return self.value

    @property
    def display_name(self) -> str:
        """Return a human-readable display name."""
        return self.value.replace("_", " ").title()


@dataclass
class Section:
    thickness: Thickness  # Thickness of the section along the camber line
    camber: Camber  # Camber line of this section on stream surface (m, c) coordinates
    stream_line: PlaneCurve | None  # Stream line curve in (x, r) meridional coordinates

    thickness_params: ThicknessParams | None = None
    camber_params: CamberParams | None = None

    reference_point: ReferencePoint = ReferencePoint.Centroid

    def with_thickness(self, thickness: Thickness | ThicknessParams) -> "Section":
        """Create a new Section with the specified thickness."""
        if isinstance(thickness, Thickness):
            new_thickness = thickness
        elif isinstance(thickness, ThicknessParams):
            new_thickness = create_thickness(thickness).create_thickness(self.camber.s)
        else:
            raise TypeError(
                f"thickness must be Thickness or ThicknessParams not {type(thickness).__name__}"
            )

        return Section(
            thickness=new_thickness,
            camber=self.camber,
            stream_line=self.stream_line,
            thickness_params=self.thickness_params,
            camber_params=self.camber_params,
            reference_point=self.reference_point,
        )

    def upper_curve(self) -> PlaneCurve:
        cl = self.camber.line.coords
        thick = self._calc_signed_thickness()
        upper_coords = cl + 0.5 * thick
        upper_param = self.camber.line.param
        return PlaneCurve(upper_coords, upper_param)

    def lower_curve(self) -> PlaneCurve:
        cl = self.camber.line.coords
        thick = self._calc_signed_thickness()
        lower_coords = cl - 0.5 * thick
        lower_param = self.camber.line.param
        return PlaneCurve(lower_coords, lower_param)

    def curves(self) -> tuple[PlaneCurve, PlaneCurve]:
        cl = self.camber.line.coords
        thick = self._calc_signed_thickness()
        lower_coords = cl - 0.5 * thick
        upper_coords = cl + 0.5 * thick
        param = self.camber.line.param
        upper_curve = PlaneCurve(upper_coords, param)
        lower_curve = PlaneCurve(lower_coords, param)
        return upper_curve, lower_curve

    def get_reference_point_coords(self) -> NDArray:
        """Get the coordinates of the reference point."""
        match self.reference_point:
            case ReferencePoint.LeadingEdge:
                return self.camber.line.start()
            case ReferencePoint.TrailingEdge:
                return self.camber.line.end()
            case ReferencePoint.Centroid:
                return self._calc_centroid_coords()

    def _calc_signed_thickness(self) -> NDArray:
        normal = self.camber.line.signed_normal().coords
        thickness = self.thickness.t[:, np.newaxis]
        return normal * thickness

    def _calc_centroid_coords(self) -> NDArray:
        """Calculate the centroid coordinates of the area between upper and lower curves.

        Uses the shoelace formula for area and first moments to compute the centroid
        of the polygon formed by the upper curve, trailing edge, reversed lower curve,
        and leading edge.

        Returns:
            NDArray: [x_centroid, y_centroid] coordinates
        """
        upper_curve, lower_curve = self.curves()

        # Create closed polygon: upper curve + lower curve (reversed)
        # The curves are coincident at start and end points
        upper_coords = upper_curve.coords  # Shape: (n_points, 2)
        lower_coords = lower_curve.coords[::-1]  # Reverse and shape: (n_points, 2)

        # Combine into closed polygon (excluding duplicate endpoints)
        polygon_coords = np.r_[upper_coords, lower_coords[1:-1]]

        n = len(polygon_coords)
        if n < 3:
            raise ValueError(
                "Insufficient points to form a polygon for centroid calculation"
            )

        # Shoelace formula for area and centroid using vectorized operations
        x = polygon_coords[:, 0]
        y = polygon_coords[:, 1]

        # Shift arrays to get next points (wrapping around)
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)

        # Cross products for all points at once
        cross_products = x * y_next - x_next * y

        # Calculate area using vectorized shoelace formula
        area = 0.5 * np.sum(cross_products)

        if abs(area) < 1e-12:
            raise ValueError(
                "Section area is too small for reliable centroid calculation"
            )

        # Calculate centroid coordinates using vectorized operations
        cx = np.sum((x + x_next) * cross_products) / (6.0 * area)
        cy = np.sum((y + y_next) * cross_products) / (6.0 * area)

        return np.array([cx, cy])

    def plot(
        self,
        ax=None,
        show_reference_point: bool = True,
        show_camber_line: bool = True,
        show_LE_circle: bool = True,
    ):
        """Plot the blade section showing upper/lower curves, camber line, and reference point.

        Args:
            ax: Matplotlib axes object (optional)
            show_reference_point: Whether to show the reference point marker
            show_camber_line: Whether to show the camber line

        Returns:
            Matplotlib axes object
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        upper_curve, lower_curve = self.curves()

        # Plot upper and lower curves using PlaneCurve's plot method
        upper_curve.plot(ax=ax, color="b", linewidth=2, label="Upper surface")
        lower_curve.plot(ax=ax, color="r", linewidth=2, label="Lower surface")

        # Plot camber line if requested
        if show_camber_line:
            self.camber.line.plot(
                ax=ax,
                color="k",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="Camber line",
            )

        # Plot reference point if requested
        if show_reference_point:
            ref_coords = self.get_reference_point_coords()
            ref_name = self.reference_point.display_name
            ax.plot(
                ref_coords[0],
                ref_coords[1],
                "go",
                markersize=8,
                label=f"Reference point ({ref_name})",
            )

        # Plot leading edge circle if requested
        if show_LE_circle:
            sc, r = fit_LE_circle(self.thickness)
            centre = self.camber.line.interpolate([sc]).coords
            xc, yc = centre[0, 0], centre[0, 1]
            circle = mplCircle(
                (xc, yc),
                r,
                color="orange",
                fill=False,
                linestyle="--",
                linewidth=1.5,
                label="Leading edge circle",
            )
            ax.add_artist(circle)

        # Formatting
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title("Blade Section")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        return ax

    def plot_comparison(self, other, ax=None, show_reference_points: bool = True):
        """Plot comparison between this section and another Section or SectionPerimiter.

        Args:
            other: Another Section or SectionPerimiter object to compare with
            ax: Matplotlib axes object (optional)
            show_reference_points: Whether to show reference points for Section objects

        Returns:
            Matplotlib axes object
        """
        import matplotlib.pyplot as plt
        from .section_perimiter import SectionPerimiter

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 8))

        # Plot this section
        self.plot(
            ax=ax, show_reference_point=show_reference_points, show_camber_line=True
        )

        # Update labels to distinguish from comparison
        for line in ax.get_lines():
            if line.get_label() and not line.get_label().startswith("_"):
                line.set_label(f"Original {line.get_label()}")

        # Plot the comparison object
        if isinstance(other, Section):
            # Plot another Section
            upper_curve, lower_curve = other.curves()
            upper_curve.plot(
                ax=ax,
                color="orange",
                linewidth=2,
                linestyle=":",
                label="Comparison Upper surface",
            )
            lower_curve.plot(
                ax=ax,
                color="purple",
                linewidth=2,
                linestyle=":",
                label="Comparison Lower surface",
            )

            # Plot comparison camber line
            other.camber.line.plot(
                ax=ax,
                color="gray",
                linestyle="-.",
                linewidth=1,
                alpha=0.7,
                label="Comparison Camber line",
            )

            # Plot comparison reference point
            if show_reference_points:
                ref_coords = other.get_reference_point_coords()
                ref_name = other.reference_point.display_name
                ax.plot(
                    ref_coords[0],
                    ref_coords[1],
                    "mo",
                    markersize=8,
                    label=f"Comparison Reference point ({ref_name})",
                )

        elif isinstance(other, SectionPerimiter):
            # Plot SectionPerimiter
            other.curve.plot(
                ax=ax,
                color="cyan",
                linewidth=2,
                linestyle=":",
                label="Comparison Perimeter",
            )

        else:
            raise TypeError(
                f"Cannot compare with object of type {type(other)}. "
                "Expected Section or SectionPerimiter."
            )

        # Update plot formatting
        ax.set_title("Section Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        return ax
