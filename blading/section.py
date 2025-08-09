from dataclasses import dataclass, field
from enum import Enum
from .thickness import Thickness, ThicknessParams, create_thickness
from .camber import Camber, CamberParams
from .plotting import create_multi_view_plot
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
            new_thickness = create_thickness(thickness).eval(self.camber.s)
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
        show_reference_point: bool = True,
        show_camber_line: bool = True,
        show_LE_circle: bool = True,
        show_closeups: bool = True,
    ):
        """Plot the blade section showing upper/lower curves, camber line, and reference point.

        Parameters
        ----------
        show_reference_point : bool, optional
            Whether to show the reference point marker. Default is True.
        show_camber_line : bool, optional
            Whether to show the camber line. Default is True.
        show_LE_circle : bool, optional
            Whether to show the leading edge circle. Default is True.
        show_closeups : bool, optional
            Whether to show close-up views of LE and TE. Default is True.

        Returns
        -------
        tuple
            Tuple from create_multi_view_plot containing figure and axes objects.
        """
        # Use multi-view plotting
        upper_curve, lower_curve = self.curves()

        # Define plot functions for multi-view plotting
        plot_functions = [
            lambda ax: upper_curve.plot(
                ax=ax, color="b", linewidth=2, label="Upper surface"
            ),
            lambda ax: lower_curve.plot(
                ax=ax, color="r", linewidth=2, label="Lower surface"
            ),
        ]

        # Add camber line if requested
        if show_camber_line:
            plot_functions.append(
                lambda ax: self.camber.line.plot(
                    ax=ax,
                    color="k",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                    label="Camber line",
                )
            )

        # Add reference point if requested
        if show_reference_point:
            ref_coords = self.get_reference_point_coords()
            ref_name = self.reference_point.display_name
            plot_functions.append(
                lambda ax: ax.plot(
                    ref_coords[0],
                    ref_coords[1],
                    "go",
                    markersize=8,
                    label=f"Reference point ({ref_name})",
                )
            )

        # Add leading edge circle if requested
        if show_LE_circle:
            sc, r = self.thickness.fit_LE_circle()
            centre = self.camber.line.interpolate([sc]).coords
            xc, yc = centre[0, 0], centre[0, 1]

            def add_circle(ax):
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

            plot_functions.append(add_circle)

        # Add axis formatting function
        def format_axes(ax):
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")
            ax.grid(True, alpha=0.3)

        plot_functions.append(format_axes)

        return create_multi_view_plot(
            plot_functions=plot_functions,
            camber_line=self.camber.line,
            title="Blade Section",
            show_closeups=show_closeups,
        )

    def plot_comparison(
        self,
        other,
        this_name: str = "Original",
        other_name: str = "Comparison",
        show_reference_points: bool = True,
        show_closeups: bool = True,
    ):
        """Plot comparison between this section and another Section or SectionPerimiter.

        Parameters
        ----------
        other : Section or SectionPerimiter
            Another Section or SectionPerimiter object to compare with.
        this_name : str, optional
            Name to use in labels for this section. Default is "Original".
        other_name : str, optional
            Name to use in labels for the other section. Default is "Comparison".
        show_reference_points : bool, optional
            Whether to show reference points for Section objects. Default is True.
        show_closeups : bool, optional
            Whether to show close-up views of LE and TE. Default is True.

        Returns
        -------
        tuple
            Tuple from create_multi_view_plot containing figure and axes objects.
        """
        from .section_perimiter import SectionPerimiter

        # Use multi-view plotting
        upper_curve, lower_curve = self.curves()

        # Define plot functions starting with this section
        plot_functions = [
            lambda ax: upper_curve.plot(
                ax=ax, color="b", label=f"{this_name} Upper surface"
            ),
            lambda ax: lower_curve.plot(
                ax=ax, color="r", label=f"{this_name} Lower surface"
            ),
            lambda ax: self.camber.line.plot(
                ax=ax,
                color="k",
                linestyle="--",
                alpha=0.7,
                label=f"{this_name} Camber line",
            ),
        ]

        # Add original reference point if requested
        if show_reference_points:
            ref_coords = self.get_reference_point_coords()
            ref_name = self.reference_point.display_name
            plot_functions.append(
                lambda ax: ax.plot(
                    ref_coords[0],
                    ref_coords[1],
                    "go",
                    markersize=8,
                    label=f"{this_name} Reference point ({ref_name})",
                )
            )

        # Add comparison object plotting functions
        if isinstance(other, Section):
            # Plot another Section
            other_upper_curve, other_lower_curve = other.curves()
            plot_functions.extend(
                [
                    lambda ax: other_upper_curve.plot(
                        ax=ax,
                        color="orange",
                        linestyle=":",
                        label=f"{other_name} Upper surface",
                    ),
                    lambda ax: other_lower_curve.plot(
                        ax=ax,
                        color="purple",
                        linestyle=":",
                        label=f"{other_name} Lower surface",
                    ),
                    lambda ax: other.camber.line.plot(
                        ax=ax,
                        color="gray",
                        linestyle="-.",
                        alpha=0.7,
                        label=f"{other_name} Camber line",
                    ),
                ]
            )

            # Plot comparison reference point
            if show_reference_points:
                other_ref_coords = other.get_reference_point_coords()
                other_ref_name = other.reference_point.display_name
                plot_functions.append(
                    lambda ax: ax.plot(
                        other_ref_coords[0],
                        other_ref_coords[1],
                        "mo",
                        markersize=8,
                        label=f"{other_name} Reference point ({other_ref_name})",
                    )
                )

        elif isinstance(other, SectionPerimiter):
            # Plot SectionPerimiter
            plot_functions.append(
                lambda ax: other.curve.plot(
                    ax=ax,
                    color="cyan",
                    linestyle=":",
                    label=f"{other_name} Perimeter",
                )
            )

        else:
            raise TypeError(
                f"Cannot compare with object of type {type(other)}. "
                "Expected Section or SectionPerimiter."
            )

        # Add axis formatting function
        def format_axes(ax):
            ax.grid(True, alpha=0.3)

        plot_functions.append(format_axes)

        return create_multi_view_plot(
            plot_functions=plot_functions,
            camber_line=self.camber.line,
            title="Section Comparison",
            show_closeups=show_closeups,
        )
