from dataclasses import dataclass
from typing import Callable, Generic, MutableMapping, Optional, TypeVar, Literal
from geometry.curves import PlaneCurve
from numpy.typing import NDArray
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt


@dataclass
class Section:
    camber_line: PlaneCurve
    thickness: NDArray
    stream_line: Optional[PlaneCurve]
    stream_distance: float = 0.0  # Distance along stream_line
    circumferential_position: float = 0.0  # y-coordinate of reference point
    reference_point: Literal["leading_edge", "trailing_edge", "centroid"] = (
        "leading_edge"
    )

    def __post_init__(self) -> None:
        assert self.normalised_arc_length[0] == 0
        assert self.normalised_arc_length[-1] == 1

        # Translate camber line so reference point is at origin
        self._translate_camber_line_to_origin()

    @property
    def chord(self) -> float:
        return self.camber_line.length()

    @property
    def normalised_arc_length(self) -> NDArray:
        return self.camber_line.param

    def upper_curve(self) -> PlaneCurve:
        cl = self.camber_line.coords
        thick = self._signed_thickness()
        upper_coords = cl + 0.5 * thick
        upper_param = self.camber_line.normalise().param
        return PlaneCurve(upper_coords, upper_param)

    def lower_curve(self) -> PlaneCurve:
        cl = self.camber_line.coords
        thick = self._signed_thickness()
        upper_coords = cl - 0.5 * thick
        upper_param = self.camber_line.normalise().param
        return PlaneCurve(upper_coords, upper_param)

    def upper_and_lower(self) -> PlaneCurve:
        cl = self.camber_line.coords
        thick = self._signed_thickness()

        upper_coords = cl + 0.5 * thick
        lower_coords = cl - 0.5 * thick
        coords = np.r_[lower_coords[1:][::-1], upper_coords]

        # Parameter 0 to -1 on lower and 0 to 1 on upper
        camber_param = self.camber_line.normalise().param
        param = np.r_[-camber_param[1:][::-1], camber_param]

        return PlaneCurve(coords, param)

    def _calculate_thickness_centroid(self) -> float:
        """Calculate the centroid of the thickness distribution.

        Returns:
            float: Normalized arc length position of the thickness centroid.
        """
        x = self.normalised_arc_length
        t = self.thickness
        s_centroid = np.trapezoid(x * t, x) / np.trapezoid(t, x)
        return s_centroid

    def _get_reference_point_coordinates(self) -> NDArray:
        """Get the coordinates of the reference point on the camber line."""
        camber_coords = self.camber_line.coords

        if self.reference_point == "leading_edge":
            return camber_coords[0]
        elif self.reference_point == "trailing_edge":
            return camber_coords[-1]
        elif self.reference_point == "centroid":
            s_centroid = self._calculate_thickness_centroid()
            return self.camber_line.interpolate([s_centroid]).coords[0]
        else:
            raise ValueError(f"Unknown reference point: {self.reference_point}")

    def _translate_camber_line_to_origin(self) -> None:
        """Translate the camber line so the reference point is at the origin."""
        reference_coords = self._get_reference_point_coordinates()

        # Create translation vector to move reference point to origin
        translation = -reference_coords

        # Translate camber line coordinates
        translated_coords = self.camber_line.coords + translation

        # Create new camber line with translated coordinates
        translated_camber_line = PlaneCurve(translated_coords, self.camber_line.param)
        self.camber_line = translated_camber_line

        # Store the translation that was applied (y-coordinate of reference point)
        self.circumferential_position = reference_coords[1]
        self.stream_distance = reference_coords[0]

    def _signed_thickness(self) -> NDArray:
        normal = self.camber_line.signed_normal().coords
        thickness = self.thickness[:, np.newaxis]
        return normal * thickness

    def remove_round_TE(
        self,
        mask_rear: NDArray[np.bool] | None = None,
        mask_high_curv: NDArray[np.bool] | None = None,
    ) -> "Section":
        s = self.normalised_arc_length
        # Mask for the rear section of the blade.
        if mask_rear is None:
            mask_rear = s >= 0.8

        # Mask for areas of high curvature on the blade
        if mask_high_curv is None:
            grad = np.gradient(self.thickness, s)
            curv = np.abs(np.gradient(grad, s))
            avg_curv = np.median(curv)
            mask_high_curv = curv > 4 * avg_curv
            assert mask_high_curv is not None

        # Fit straight line through linear part of thickness
        mask_linear = mask_rear & ~mask_high_curv  # linear part
        mask_round_TE = mask_rear & mask_high_curv  # round part
        poly = Polynomial.fit(s[mask_linear], self.thickness[mask_linear], deg=1)

        # Ensure new TE has continuous value.
        y1 = self.thickness[mask_linear][-1]
        y2 = poly(s[mask_linear][-1])
        poly -= y2 - y1

        # Extrapolate straight line to TE
        thickness = self.thickness.copy()
        thickness[mask_round_TE] = poly(s[mask_round_TE])

        return Section(
            self.camber_line,
            thickness,
            self.stream_line,
            self.stream_distance,
            self.circumferential_position,
            self.reference_point,
        )

    def add_round_TE(self) -> "Section":
        s = self.normalised_arc_length

        # Thickness relative to chord.
        t_over_c = self.thickness.copy() / self.chord

        # Find trailing edge angle.
        x1, x2 = s[[-2, -1]]
        y1, y2 = t_over_c[[-2, -1]]
        tan_angle_TE = (y1 - y2) / (x2 - x1)

        # Trailing edge thickness.
        t_TE = t_over_c[-1]

        # Analytical formula for round TE tangent to linear section.
        a = np.arctan(0.5 * tan_angle_TE)
        b = 2 * np.cos(a) - (1 - np.sin(a)) * tan_angle_TE
        rad_TE = t_TE / b

        # Create points on round TE and interpolate given `s` distribution.
        pts = np.linspace(a, np.pi / 2, 70)
        x_te = 1 - rad_TE + rad_TE * np.sin(pts)
        y_te = 2 * rad_TE * np.cos(pts)
        mask_round_TE = s > min(x_te)
        t_over_c[mask_round_TE] = np.interp(s[mask_round_TE], x_te, y_te)

        return Section(
            self.camber_line,
            t_over_c * self.chord,
            self.stream_line,
            self.stream_distance,
            self.circumferential_position,
            self.reference_point,
        )

    ###### Plotting ######

    def plot_section(self, ax=None, show_camber_line=True, *plot_args, **plot_kwargs):
        """Plot the full section shape including upper and lower surfaces."""
        if ax is None:
            _, ax = plt.subplots()

        # Get upper and lower curves
        upper = self.upper_curve()
        lower = self.lower_curve()

        # Plot upper and lower surfaces
        ax.plot(
            upper.coords[:, 0],
            upper.coords[:, 1],
            "b-",
            label="Upper surface",
            *plot_args,
            **plot_kwargs,
        )
        ax.plot(
            lower.coords[:, 0],
            lower.coords[:, 1],
            "r-",
            label="Lower surface",
            *plot_args,
            **plot_kwargs,
        )

        # Optionally plot camber line
        if show_camber_line:
            camber_coords = self.camber_line.coords
            ax.plot(
                camber_coords[:, 0],
                camber_coords[:, 1],
                "k--",
                alpha=0.7,
                label="Camber line",
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Section Shape")

        return ax

    def plot_thickness(self, ax=None, *plot_args, **plot_kwargs):
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.normalised_arc_length, self.thickness, *plot_args, **plot_kwargs)
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Thickness")

    def plot_thickness_to_chord(self, ax=None, *plot_args, **plot_kwargs):
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(
            self.normalised_arc_length,
            self.thickness / self.chord,
            *plot_args,
            **plot_kwargs,
        )

        # Add centroid visualization
        s_centroid = self._calculate_thickness_centroid()
        ax.axvline(
            s_centroid,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Thickness centroid",
        )

        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Thickness / chord")
        ax.legend()

    def plot_nondim_camber(self, ax=None, *plot_args, **plot_kwargs):
        if ax is None:
            _, ax = plt.subplots()

        angle = self.camber_line.turning_angle()
        nondim_camber = (angle - angle[0]) / (angle[-1] - angle[0])
        ax.plot(self.normalised_arc_length, nondim_camber, *plot_args, **plot_kwargs)
        ax.set_xlabel("Normalised arc length")
        ax.set_ylabel("Non-dimensional camber")

    def plot_summary(self, figsize=(12, 8)):
        """Plot a comprehensive summary of the section including shape, thickness/chord, camber, and meridional streamline if available."""
        # Determine layout based on whether stream line is available
        has_streamline = self.stream_line is not None

        if has_streamline:
            # 2x2 grid: section and meridional on top, thickness and camber on bottom
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                2, 2, figsize=figsize, layout="constrained"
            )

            # Top row: Section shape and meridional streamline
            self.plot_section(ax=ax1, show_camber_line=True)
            ax1.set_title("Section Shape")

            self.plot_meridional_streamline(ax=ax2)
            ax2.set_title("Meridional Stream Line")

            # Bottom row: Thickness and camber
            self.plot_thickness_to_chord(ax=ax3)
            ax3.set_title("Thickness/Chord Ratio")

            self.plot_nondim_camber(ax=ax4)
            ax4.set_title("Non-dimensional Camber")

            axes = ((ax1, ax2), (ax3, ax4))
        else:
            # 1x3 grid: original layout when no streamline
            fig, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=(15, 5), layout="constrained"
            )

            self.plot_section(ax=ax1, show_camber_line=True)
            ax1.set_title("Section Shape")

            self.plot_thickness_to_chord(ax=ax2)
            ax2.set_title("Thickness/Chord Ratio")

            self.plot_nondim_camber(ax=ax3)
            ax3.set_title("Non-dimensional Camber")

            axes = (ax1, ax2, ax3)

        # Add overall title with section info
        info_text = f"Reference Point: {self.reference_point.replace('_', ' ').title()}"
        if self.stream_distance != 0.0:
            info_text += f" | Stream Distance: {self.stream_distance:.3f}"
        if self.circumferential_position != 0.0:
            info_text += (
                f" | Circumferential Position: {self.circumferential_position:.3f}"
            )

        fig.suptitle(f"Section Summary\n{info_text}", fontsize=14)

        return fig, axes

    def plot_meridional_streamline(self, ax=None, *plot_args, **plot_kwargs):
        """Plot the stream line in the meridional (x, r) plane with key features highlighted.

        Highlights:
        - Leading edge
        - Trailing edge
        - Reference point
        - Total extent of the blade along the stream line
        """
        if self.stream_line is None:
            raise ValueError("No stream line available for this section")

        if ax is None:
            _, ax = plt.subplots()

        # Plot the stream line
        stream_coords = self.stream_line.coords
        ax.plot(
            stream_coords[:, 0],
            stream_coords[:, 1],
            "k-",
            linewidth=2,
            label="Stream line",
            *plot_args,
            **plot_kwargs,
        )

        # Map blade section key points to stream line using m-coordinate (distance along stream line)
        # Since camber line is translated to put reference point at origin, need to add back stream_distance

        # Calculate m-coordinates for key points
        le_m = self.camber_line.coords[0, 0] + self.stream_distance  # Leading edge
        te_m = self.camber_line.coords[-1, 0] + self.stream_distance  # Trailing edge
        ref_m = (
            self.stream_distance
        )  # Reference point (at origin in camber line coords)

        # Interpolate stream line at these m-coordinates to get (x, r) positions
        le_pos = self.stream_line.interpolate([le_m])
        te_pos = self.stream_line.interpolate([te_m])
        ref_pos = self.stream_line.interpolate([ref_m])

        # Highlight leading edge
        ax.plot(
            le_pos.coords[0, 0],
            le_pos.coords[0, 1],
            "go",
            markersize=8,
            label="Leading edge",
        )

        # Highlight trailing edge
        ax.plot(
            te_pos.coords[0, 0],
            te_pos.coords[0, 1],
            "ro",
            markersize=8,
            label="Trailing edge",
        )

        # Highlight reference point with x marker
        ax.plot(
            ref_pos.coords[0, 0],
            ref_pos.coords[0, 1],
            "bx",
            markersize=10,
            markeredgewidth=1,
            label=f'Reference point ({self.reference_point.replace("_", " ")})',
        )

        # Show total extent with vertical lines
        x_min, x_max = le_pos.coords[0, 0], te_pos.coords[0, 0]
        r_range = stream_coords[:, 1]
        r_min, r_max = np.min(r_range), np.max(r_range)

        # Vertical lines showing blade extent
        ax.axvline(x_min, color="gray", linestyle="--", alpha=0.5, label="Blade extent")
        ax.axvline(x_max, color="gray", linestyle="--", alpha=0.5)

        # # Add extent annotation
        # ax.annotate(
        #     f"Blade extent: {x_max - x_min:.3f}",
        #     xy=((x_min + x_max) / 2, np.mean(r_range)),
        #     xytext=(10, 10),
        #     textcoords="offset points",
        #     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        # )

        ax.set_xlabel("x (axial direction)")
        ax.set_ylabel("r (radial direction)")
        ax.set_title("Meridional Stream Line")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis("equal")

        return ax

    def set_reference_point(
        self, new_reference_point: Literal["leading_edge", "trailing_edge", "centroid"]
    ) -> None:
        """Update the reference point of this section.

        Args:
            new_reference_point: The new reference point to use
        """
        if new_reference_point not in ["leading_edge", "trailing_edge", "centroid"]:
            raise ValueError(
                f"Invalid reference point: {new_reference_point}. Must be one of: leading_edge, trailing_edge, centroid"
            )

        if new_reference_point == self.reference_point:
            return  # No change needed

        # First, translate back to original position
        camber_coords = self.camber_line.coords
        camber_coords += np.c_[self.stream_distance, self.circumferential_position]

        # Update reference point
        self.reference_point = new_reference_point

        # Re-translate to origin with new reference point
        self._translate_camber_line_to_origin()


@dataclass(frozen=True)
class FlatSection:
    curve: PlaneCurve
    stream_line: Optional[PlaneCurve]


@dataclass(frozen=True)
class FlatBlade:
    sections: list[FlatSection]
