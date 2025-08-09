from dataclasses import dataclass, field
from enum import Enum
from .thickness import Thickness, ThicknessParams, create_thickness
from .camber import Camber, CamberParams, create_camber
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


@dataclass(frozen=True, init=False)
class Section:
    thickness: Thickness  # Thickness of the section along the camber line
    camber: Camber  # Camber line of this section on stream surface (m, c) coordinates
    stream_line: PlaneCurve | None  # Stream line curve in (x, r) meridional coordinates

    thickness_params: ThicknessParams | None
    camber_params: CamberParams | None

    reference_point: ReferencePoint

    def __init__(
        self,
        thickness: Thickness | ThicknessParams,
        camber: Camber | CamberParams,
        s: NDArray | None = None,
        stream_line: PlaneCurve | None = None,
        reference_point: ReferencePoint = ReferencePoint.Centroid,
    ):
        match camber, thickness:
            case Camber(), Thickness():
                camber_params = None
                thickness_params = None

            case CamberParams(), ThicknessParams():
                if s is None:
                    raise ValueError(
                        "s must be provided when using CamberParams and ThicknessParams"
                    )
                camber_params = camber
                thickness_params = thickness
                camber = create_camber(camber_params).eval(s)
                thickness = create_thickness(thickness_params).eval(s)

            case Camber(), ThicknessParams():
                if s is not None:
                    raise ValueError("s should not be provided when using Camber")
                camber_params = None
                thickness_params = thickness
                thickness = create_thickness(thickness_params).eval(camber.s)

            case CamberParams(), Thickness():
                if s is not None:
                    raise ValueError("s should not be provided when using Thickness")
                camber_params = camber
                thickness_params = None
                camber = create_camber(camber_params).eval(thickness.s)

        self._set("thickness", thickness)
        self._set("camber", camber)
        self._set("stream_line", stream_line)
        self._set("camber_params", camber_params)
        self._set("thickness_params", thickness_params)
        self._set("reference_point", reference_point)

    def with_thickness(
        self, thickness: Thickness | ThicknessParams, s: NDArray | None = None
    ) -> "Section":
        """Return a new Section with the specified thickness."""
        camber = self.camber_params or self.camber

        return Section(
            thickness=thickness,
            camber=camber,
            s=s,
            stream_line=self.stream_line,
            reference_point=self.reference_point,
        )

    def with_camber(
        self, camber: Camber | CamberParams, s: NDArray | None = None
    ) -> "Section":
        """Return a new Section with the specified camber."""
        thickness = self.thickness_params or self.thickness

        return Section(
            thickness=thickness,
            camber=camber,
            s=s,
            stream_line=self.stream_line,
            reference_point=self.reference_point,
        )

    def with_reference_point(self, reference_point: ReferencePoint) -> "Section":
        """Return a new Section with the specified reference point."""
        camber = self.camber_params or self.camber
        thickness = self.thickness_params or self.thickness
        s = self.camber.s
        return Section(
            thickness=thickness,
            camber=camber,
            s=s,
            stream_line=self.stream_line,
            reference_point=reference_point,
        )

    def with_stream_line(self, stream_line: PlaneCurve) -> "Section":
        """Return a new Section with the specified stream line."""
        camber = self.camber_params or self.camber
        thickness = self.thickness_params or self.thickness
        s = self.camber.s
        return Section(
            thickness=thickness,
            camber=camber,
            s=s,
            stream_line=stream_line,
            reference_point=self.reference_point,
        )

    def with_s(self, s: NDArray) -> "Section":
        """Return a new Section with the specified s values."""
        camber = self.camber_params or self.camber
        thickness = self.thickness_params or self.thickness

        if isinstance(camber, Camber):
            camber = camber.interpolate(s)
        if isinstance(thickness, Thickness):
            thickness = thickness.interpolate(s)

        return Section(
            thickness=thickness,
            camber=camber,
            s=s,
            stream_line=self.stream_line,
            reference_point=self.reference_point,
        )

    @property
    def is_camber_param(self) -> bool:
        """Check if the camber is parameterized."""
        return self.camber_params is not None

    @property
    def is_thickness_param(self) -> bool:
        """Check if the thickness is parameterized."""
        return self.thickness_params is not None

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

    def _create_plot_functions(
        self,
        name_prefix: str = "",
        show_reference_point: bool = True,
        show_camber_line: bool = True,
        show_LE_circle: bool = True,
        colors: dict | None = None,
    ) -> list:
        """Create plot functions for this section.

        Parameters
        ----------
        name_prefix : str, optional
            Prefix to add to labels. Default is "".
        show_reference_point : bool, optional
            Whether to include reference point plotting. Default is True.
        show_camber_line : bool, optional
            Whether to include camber line plotting. Default is True.
        show_LE_circle : bool, optional
            Whether to include LE circle plotting. Default is True.
        colors : dict, optional
            Color scheme to use. If None, uses default colors.

        Returns
        -------
        list
            List of plot functions for multi-view plotting.
        """
        if colors is None:
            colors = {
                "upper": "b",
                "lower": "r",
                "camber": "k",
                "reference": "go",
                "le_circle": "orange",
            }

        upper_curve, lower_curve = self.curves()
        plot_functions = []

        # Add upper and lower curves
        upper_label = f"{name_prefix} Upper surface" if name_prefix else "Upper surface"
        lower_label = f"{name_prefix} Lower surface" if name_prefix else "Lower surface"

        plot_functions.extend(
            [
                lambda ax: upper_curve.plot(
                    ax=ax, color=colors["upper"], label=upper_label
                ),
                lambda ax: lower_curve.plot(
                    ax=ax, color=colors["lower"], label=lower_label
                ),
            ]
        )

        # Add camber line if requested
        if show_camber_line:
            camber_label = (
                f"{name_prefix} Camber line" if name_prefix else "Camber line"
            )
            plot_functions.append(
                lambda ax: self.camber.line.plot(
                    ax=ax,
                    color=colors["camber"],
                    linestyle="--",
                    alpha=0.7,
                    label=camber_label,
                )
            )

        # Add reference point if requested
        if show_reference_point:
            ref_coords = self.get_reference_point_coords()
            ref_name = self.reference_point.display_name
            ref_label = (
                f"{name_prefix} Reference point ({ref_name})"
                if name_prefix
                else f"Reference point ({ref_name})"
            )
            plot_functions.append(
                lambda ax: ax.plot(
                    ref_coords[0],
                    ref_coords[1],
                    colors["reference"],
                    label=ref_label,
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
                    color=colors["le_circle"],
                    fill=False,
                    linestyle="--",
                    label="Leading edge circle",
                )
                ax.add_artist(circle)

            plot_functions.append(add_circle)

        return plot_functions

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
        # Get plot functions from shared method
        plot_functions = self._create_plot_functions(
            show_reference_point=show_reference_point,
            show_camber_line=show_camber_line,
            show_LE_circle=show_LE_circle,
        )

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
        show_reference_points: bool = False,
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

        # Get plot functions for this section (no LE circle in comparison plots)
        plot_functions = self._create_plot_functions(
            name_prefix=this_name,
            show_reference_point=show_reference_points,
            show_camber_line=True,
            show_LE_circle=False,
        )

        # Add comparison object plotting functions
        if isinstance(other, Section):
            # Define colors for comparison section
            other_colors = {
                "upper": "orange",
                "lower": "purple",
                "camber": "gray",
                "reference": "mo",
                "le_circle": "cyan",
            }

            # Get plot functions for other section
            other_plot_functions = other._create_plot_functions(
                name_prefix=other_name,
                show_reference_point=show_reference_points,
                show_camber_line=True,
                show_LE_circle=False,
                colors=other_colors,
            )

            # Modify line styles for comparison section
            for i, func in enumerate(other_plot_functions):
                if i < 3:  # upper, lower, camber lines
                    original_func = func

                    def modified_func(ax, orig=original_func):
                        # Call original function but modify the result to add linestyle
                        result = orig(ax)
                        # Get the last line added and modify its linestyle
                        if ax.lines:
                            if i < 2:  # upper and lower curves
                                ax.lines[-1].set_linestyle(":")
                            else:  # camber line
                                ax.lines[-1].set_linestyle("-.")
                        return result

                    other_plot_functions[i] = modified_func

            plot_functions.extend(other_plot_functions)

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

    def _set(self, name: str, value):
        """Set an attribute of the Section object."""
        if name in self.__dataclass_fields__:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Cannot set attribute '{name}' on Section")
