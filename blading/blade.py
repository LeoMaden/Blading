from dataclasses import dataclass
from typing import List, Literal, cast
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .section import Section
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class Blade:
    sections: List[Section]

    def __post_init__(self) -> None:
        if not self.sections:
            raise ValueError("Blade must contain at least one section")

        # Enforce reference point consistency
        if not self._check_reference_point_consistency():
            raise ValueError("All sections must have the same reference point")

    @property
    def num_sections(self) -> int:
        return len(self.sections)

    @property
    def span_positions(self) -> NDArray:
        """Get the circumferential positions of all sections."""
        return np.array([section.circumferential_position for section in self.sections])

    @property
    def stream_positions(self) -> NDArray:
        """Get the stream distance positions of all sections."""
        return np.array([section.stream_distance for section in self.sections])

    def get_section(self, index: int) -> Section:
        """Get a section by index."""
        if not 0 <= index < len(self.sections):
            raise IndexError(
                f"Section index {index} out of range [0, {len(self.sections)-1}]"
            )
        return self.sections[index]

    def add_section(self, section: Section) -> None:
        """Add a new section to the blade."""
        # Check reference point consistency before adding
        if (
            self.sections
            and section.reference_point != self.sections[0].reference_point
        ):
            raise ValueError(
                f"Section reference point '{section.reference_point}' does not match blade reference point '{self.sections[0].reference_point}'"
            )

        self.sections.append(section)

    def remove_section(self, index: int) -> Section:
        """Remove and return a section by index."""
        if not 0 <= index < len(self.sections):
            raise IndexError(
                f"Section index {index} out of range [0, {len(self.sections)-1}]"
            )
        return self.sections.pop(index)

    def sort_sections_by_span(self) -> None:
        """Sort sections by their circumferential position."""
        self.sections.sort(key=lambda s: s.circumferential_position)

    def sort_sections_by_stream(self) -> None:
        """Sort sections by their stream distance."""
        self.sections.sort(key=lambda s: s.stream_distance)

    def set_all_reference_points(
        self, reference_point: Literal["leading_edge", "trailing_edge", "centroid"]
    ) -> None:
        """Set the reference point for all sections in the blade."""
        for section in self.sections:
            section.set_reference_point(reference_point)

    def _check_reference_point_consistency(self) -> bool:
        """Check if all sections have the same reference point."""
        if not self.sections:
            return True

        first_ref_point = self.sections[0].reference_point
        return all(
            section.reference_point == first_ref_point for section in self.sections
        )

    def get_common_reference_point(self) -> str:
        """Get the common reference point (guaranteed to be consistent)."""
        if not self.sections:
            raise ValueError("No sections in blade")

        return self.sections[0].reference_point

    def get_chord_distribution(self) -> NDArray:
        """Get the chord length for each section."""
        return np.array([section.chord for section in self.sections])

    def get_thickness_at_position(self, normalized_position: float) -> NDArray:
        """Get thickness at a normalized chord position for all sections."""
        if not 0 <= normalized_position <= 1:
            raise ValueError("Normalized position must be between 0 and 1")

        thicknesses = []
        for section in self.sections:
            # Interpolate thickness at the specified position
            thickness = np.interp(
                normalized_position, section.normalised_arc_length, section.thickness
            )
            thicknesses.append(thickness)

        return np.array(thicknesses)

    def plot_blade_3d(
        self,
        ax: Axes3D | None = None,
        show_camber: bool = True,
        use_stream_coords: bool = True,
        alpha: float = 0.7,
    ) -> Axes3D:
        """Plot the blade in 3D showing all sections.

        Args:
            ax: Matplotlib 3D axis to plot on
            show_camber: Whether to show camber lines
            use_stream_coords: If True, use stream line coordinates for proper 3D positioning.
                              If False, use simple span-wise positioning.
            alpha: Transparency for the plotted lines
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))

        if use_stream_coords:
            # Use proper stream line coordinates for 3D plotting
            sections_with_streamlines = [
                s for s in self.sections if s.stream_line is not None
            ]

            if not sections_with_streamlines:
                ax.text(
                    0.5,
                    0.5,
                    0.5,
                    "No sections with stream lines available for 3D plotting",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                return ax

            # Plot each section in 3D using stream coordinates
            for i, section in enumerate(sections_with_streamlines):
                # Only show labels for first section to avoid clutter
                show_labels = i == 0

                upper_label = "Upper surfaces" if show_labels else None
                lower_label = "Lower surfaces" if show_labels else None
                camber_label = "Camber lines" if show_labels else None

                section.plot_section_3d(
                    ax=ax,
                    show_camber_line=show_camber,
                    alpha=alpha,
                    upper_label=upper_label,
                    lower_label=lower_label,
                    camber_label=camber_label,
                )

            # Add surface filling between sections
            self._add_surface_fill(ax, sections_with_streamlines, alpha=alpha / 2)

            ax.set_xlabel("x (axial)")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(
                f"Blade 3D View - Stream Coordinates ({len(sections_with_streamlines)} sections)"
            )

            # Set equal aspect ratio for 3D plot
            ax.axis("equal")

            # Set default camera view to look in xy plane
            ax.view_init(elev=90, azim=-90)

        else:
            # Original simple span-wise plotting
            for section in self.sections:
                y_pos = section.circumferential_position

                # Plot upper and lower surfaces
                upper = section.upper_curve()
                lower = section.lower_curve()

                ax.plot(
                    upper.coords[:, 0],
                    [y_pos] * len(upper.coords),
                    upper.coords[:, 1],
                    "b-",
                    alpha=alpha,
                    linewidth=1,
                )
                ax.plot(
                    lower.coords[:, 0],
                    [y_pos] * len(lower.coords),
                    lower.coords[:, 1],
                    "r-",
                    alpha=alpha,
                    linewidth=1,
                )

                # Optionally plot camber line
                if show_camber:
                    camber = section.camber_line
                    ax.plot(
                        camber.coords[:, 0],
                        [y_pos] * len(camber.coords),
                        camber.coords[:, 1],
                        "k--",
                        alpha=0.5,
                        linewidth=0.5,
                    )

            ax.set_xlabel("x (chord direction)")
            ax.set_ylabel("y (span direction)")
            ax.set_zlabel("z (thickness direction)")
            ax.set_title(f"Blade 3D View ({self.num_sections} sections)")

            # Set equal aspect ratio for 3D plot
            ax.axis("equal")

            # Set default camera view to look in xy plane
            ax.view_init(elev=90, azim=-90)

        return ax

    def _add_surface_fill(
        self, ax: Axes3D, sections: List["Section"], alpha: float = 0.3
    ) -> None:
        """Add surface patches between blade sections for better visualization."""
        if len(sections) < 2:
            return

        for i in range(len(sections) - 1):
            section1 = sections[i]
            section2 = sections[i + 1]

            # Get 3D coordinates for both sections
            upper1_3d = section1._transform_to_3d(section1.upper_curve().coords)
            lower1_3d = section1._transform_to_3d(section1.lower_curve().coords)
            upper2_3d = section2._transform_to_3d(section2.upper_curve().coords)
            lower2_3d = section2._transform_to_3d(section2.lower_curve().coords)

            # Create surface patches between sections
            # Upper surface patches
            for j in range(len(upper1_3d) - 1):
                # Create quadrilateral patch
                vertices = [
                    upper1_3d[j],
                    upper1_3d[j + 1],
                    upper2_3d[j + 1],
                    upper2_3d[j],
                ]

                # Add upper surface patch
                ax.add_collection3d(
                    Poly3DCollection(
                        [vertices],
                        alpha=alpha,
                        facecolors="lightblue",
                        edgecolors="none",
                    )
                )

            # Lower surface patches
            for j in range(len(lower1_3d) - 1):
                # Create quadrilateral patch
                vertices = [
                    lower1_3d[j],
                    lower1_3d[j + 1],
                    lower2_3d[j + 1],
                    lower2_3d[j],
                ]

                # Add lower surface patch
                ax.add_collection3d(
                    Poly3DCollection(
                        [vertices],
                        alpha=alpha,
                        facecolors="lightcoral",
                        edgecolors="none",
                    )
                )

    def plot_span_wise_properties(self, figsize=(15, 10)):
        """Plot span-wise variation of blade properties."""
        fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")

        span_pos = self.span_positions

        # Chord distribution
        chords = self.get_chord_distribution()
        axes[0, 0].plot(span_pos, chords, "bo-", linewidth=2, markersize=6)
        axes[0, 0].set_xlabel("Circumferential Position")
        axes[0, 0].set_ylabel("Chord Length")
        axes[0, 0].set_title("Chord Distribution")
        axes[0, 0].grid(True, alpha=0.3)

        # Thickness distribution at different chord positions
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        colors = ["red", "orange", "green", "blue", "purple"]
        labels = [
            "Leading Edge",
            "25% Chord",
            "50% Chord",
            "75% Chord",
            "Trailing Edge",
        ]

        for pos, color, label in zip(positions, colors, labels):
            thickness = self.get_thickness_at_position(pos)
            axes[0, 1].plot(
                span_pos,
                thickness,
                "o-",
                color=color,
                linewidth=2,
                markersize=4,
                label=label,
            )

        axes[0, 1].set_xlabel("Circumferential Position")
        axes[0, 1].set_ylabel("Thickness")
        axes[0, 1].set_title("Thickness Distribution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Stream distance distribution
        stream_pos = self.stream_positions
        axes[1, 0].plot(span_pos, stream_pos, "go-", linewidth=2, markersize=6)
        axes[1, 0].set_xlabel("Circumferential Position")
        axes[1, 0].set_ylabel("Stream Distance")
        axes[1, 0].set_title("Stream Distance Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        # Reference point distribution
        ref_points = [section.reference_point for section in self.sections]
        unique_refs = list(set(ref_points))
        ref_counts = [ref_points.count(ref) for ref in unique_refs]

        axes[1, 1].pie(ref_counts, labels=unique_refs, autopct="%1.1f%%")
        axes[1, 1].set_title("Reference Point Distribution")

        fig.suptitle(
            f"Blade Span-wise Properties ({self.num_sections} sections)", fontsize=16
        )

        return fig, axes

    def plot_meridional_view(self, ax=None):
        """Plot the meridional view of the blade showing stream lines."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        sections_with_streamlines = [
            s for s in self.sections if s.stream_line is not None
        ]

        if not sections_with_streamlines:
            ax.text(
                0.5,
                0.5,
                "No sections with stream lines available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return ax

        # Plot stream lines without extent lines and with features for all sections
        for i, section in enumerate(sections_with_streamlines):
            # Only show stream line label for first section
            stream_label = "Stream lines" if i == 0 else None
            section.plot_meridional_streamline(
                ax=ax,
                alpha=0.7,
                label=stream_label,
                show_extent=False,  # Hide extent lines for blade view
                show_features=True,  # Show features for all sections
                show_feature_labels=(
                    i == 0
                ),  # Only show feature labels for first section
            )

        ax.set_title(f"Meridional View - {len(sections_with_streamlines)} Stream Lines")
        ax.legend()

        return ax

    def __len__(self) -> int:
        return len(self.sections)

    def __getitem__(self, index: int) -> Section:
        return self.sections[index]

    def __iter__(self):
        return iter(self.sections)
