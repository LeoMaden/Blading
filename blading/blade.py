import numpy as np
from dataclasses import dataclass
from .section import Section
import matplotlib.pyplot as plt


@dataclass
class Blade:
    sections: list[Section]

    def __post_init__(self):
        if not self.sections:
            raise ValueError("Blade must contain at least one section")

        # Enforce reference point consistency
        if not self._check_reference_point_consistency():
            raise ValueError("All sections must have the same reference point")

        if not all(s.stream_line for s in self.sections):
            raise ValueError("All sections must have a stream line defined")

        # Interpolate all sections to have the same number of points
        self._interpolate_sections_to_common_param()

    def plot_meridional(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        i_hub = 0
        i_cas = len(self.sections) - 1

        # Plot the stream line for each section
        for i, section in enumerate(self.sections):
            stream_line = section.stream_line
            assert stream_line is not None, "Expected section to have a stream line"
            if i == i_hub or i == i_cas:
                label = "Hub and Casing Lines" if i == i_hub else ""
                stream_line.plot(ax, "r-", label=label)
            else:
                stream_line.plot(ax, "k-", alpha=0.5)

        # Plot the leading and trailing edges
        N = len(self.sections)
        coords_LE = np.empty((N, 2))
        coords_TE = np.empty((N, 2))
        coords_ref = np.empty((N, 2))
        for i, section in enumerate(self.sections):
            m_LE = section.camber.line.x[0]
            m_TE = section.camber.line.x[-1]
            m_ref = section.get_reference_point_coords()[0]
            stream_line = section.stream_line
            assert stream_line is not None, "Expected section to have a stream line"
            coords_LE[i] = stream_line.interpolate([m_LE]).coords
            coords_TE[i] = stream_line.interpolate([m_TE]).coords
            coords_ref[i] = stream_line.interpolate([m_ref]).coords

        ax.plot(coords_LE[:, 0], coords_LE[:, 1], "b-", label="Leading Edge")
        ax.plot(coords_TE[:, 0], coords_TE[:, 1], "g-", label="Trailing Edge")

        ref_point_name = self.sections[0].reference_point.display_name
        ref_point_label = f"Reference Point ({ref_point_name})"
        ax.plot(
            coords_ref[:, 0], coords_ref[:, 1], "k-", alpha=0.5, label=ref_point_label
        )

        ax.set_title(f"Meridional View ({len(self.sections)} Sections)")

        ax.axis("equal")
        ax.legend()

    def _check_reference_point_consistency(self) -> bool:
        """Check if all sections have the same reference point."""
        first_ref_point = self.sections[0].reference_point
        return all(
            section.reference_point == first_ref_point for section in self.sections
        )

    def _interpolate_sections_to_common_param(self) -> None:
        """Interpolate all sections to have the same parameter distribution.

        Uses the parameter from the section with the highest number of points.
        """
        if len(self.sections) <= 1:
            return  # Nothing to interpolate

        # Find the section with the most points
        max_points = 0
        reference_param = None

        for section in self.sections:
            this_s = section.camber.s
            num_points = len(this_s)
            if num_points > max_points:
                max_points = num_points
                reference_param = this_s

        if reference_param is None:
            raise ValueError("Could not determine reference parameter")

        # Interpolate all sections to the reference parameter
        interpolated_sections = [sec.with_s(reference_param) for sec in self.sections]

        # Update the sections list (use object.__setattr__ for frozen dataclass)
        object.__setattr__(self, "sections", interpolated_sections)
