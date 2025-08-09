from typing import Any, Protocol
import numpy as np
from dataclasses import dataclass
from .section import Section, ReferencePoint
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from geometry.curves import PlaneCurve, SpaceCurve


class SectionFunc(Protocol):
    """Protocol for a function that takes a Section and returns a value."""

    def __call__(self, s: Section) -> Any:
        """Get the value for the given section."""
        ...


class MeridionalFunc(Protocol):
    """Protocol for a function that takes a Section and returns a meridional coordinate."""

    def __call__(self, s: Section) -> float:
        """Get the meridional coordinate for the given section."""
        ...


@dataclass
class Blade:
    sections: list[Section]

    def __post_init__(self):
        if not self.sections:
            raise ValueError("Blade must contain at least one section")

        # Enforce reference point consistency
        if not self._check_reference_point_consistency():
            raise ValueError("All sections must have the same reference point")

        # Enforce parameterisation consistency
        if not self._check_parameter_consistency():
            raise ValueError("All sections must have the same parameterisation")

        if not all(s.stream_line for s in self.sections):
            raise ValueError("All sections must have a stream line defined")

        # Interpolate all sections to have the same number of points
        self._interpolate_sections_to_common_param()

    @property
    def is_camber_param(self) -> bool:
        """Check if the blade has parameterised camber."""
        # Parameterisation consistency is ensured by __post_init__
        return self.sections[0].is_camber_param

    @property
    def is_thickness_param(self) -> bool:
        """Check if the blade has parameterised thickness."""
        # Parameterisation consistency is ensured by __post_init__
        return self.sections[0].is_thickness_param

    @property
    def reference_point(self) -> ReferencePoint:
        """Get the reference point used for all sections."""
        # Reference point consistency is ensured by __post_init__
        return self.sections[0].reference_point

    def get_spanwise_value(self, section_func: SectionFunc) -> NDArray:
        """Get a spanwise value for each section using the provided value function."""
        values = np.array([section_func(section) for section in self.sections])
        return values

    def meridional_spanwise(self, m_func: MeridionalFunc) -> PlaneCurve:
        """Get a meridional curve by interpolating the stream line for each section
        at the given meridional coordinate.
        """
        N = len(self.sections)
        coords = np.empty((N, 2))
        for i, section in enumerate(self.sections):
            m = m_func(section)
            stream_line = section.stream_line
            assert stream_line is not None, "Expected section to have a stream line"
            coords[i] = stream_line.interpolate([m]).coords
        return PlaneCurve.new_unit_speed(coords)

    def meridional_LE_curve(self) -> PlaneCurve:
        return self.meridional_spanwise(lambda s: s.camber.line.x[0])

    def meridional_TE_curve(self) -> PlaneCurve:
        return self.meridional_spanwise(lambda s: s.camber.line.x[-1])

    def meridional_ref_point_curve(self) -> PlaneCurve:
        return self.meridional_spanwise(lambda s: s.get_reference_point_coords()[0])

    def get_span(self) -> NDArray:
        """Get the relative spanwise location of each section."""
        ref_points = self.meridional_ref_point_curve()
        r_ref = ref_points.y  # y coordinate is r in meridional coordinates

        r_min = np.min(r_ref)
        r_max = np.max(r_ref)
        span = (r_ref - r_min) / (r_max - r_min)
        return span

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
        self.meridional_LE_curve().plot(ax, "b-", label="Leading Edge")
        self.meridional_TE_curve().plot(ax, "g-", label="Trailing Edge")

        ref_point_name = self.sections[0].reference_point.display_name
        ref_point_label = f"Reference Point ({ref_point_name})"
        self.meridional_ref_point_curve().plot(
            ax, "k-", alpha=0.5, label=ref_point_label
        )

        ax.set_title(f"Meridional View ({len(self.sections)} Sections)")

        ax.axis("equal")
        ax.legend()

    def plot_metal_angles(self, ax=None, show_turning: bool = True):
        if ax is None:
            _, ax = plt.subplots()

        span = self.get_span()

        funcs_labels = [
            (lambda s: s.camber_params.angles.LE, "Leading edge"),
            (lambda s: s.camber_params.angles.TE, "Trailing edge"),
        ]
        if show_turning:
            funcs_labels.append(
                (lambda s: s.camber_params.angles.delta, "Turning angle")
            )

        ax.set_title("Spanwise Metal Angles")
        for f, label in funcs_labels:
            angle = np.degrees(self.get_spanwise_value(f))
            ax.plot(angle, span, label=label)
        ax.plot([0], [0], "k-")  # Ensure origin is visible
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel(
            f"Spanwise location at {self.reference_point.display_name.lower()}"
        )
        ax.grid()
        ax.legend()
        return ax

    def _check_reference_point_consistency(self) -> bool:
        """Check if all sections have the same reference point."""
        first_ref_point = self.sections[0].reference_point
        return all(s.reference_point == first_ref_point for s in self.sections)

    def _check_parameter_consistency(self) -> bool:
        """Check if all sections have the same parameterisation.

        e.g. all have both parameterised camber and thickness
        """
        cp_type = type(self.sections[0].camber_params)
        camber_consistent = all(
            isinstance(s.camber_params, cp_type) for s in self.sections
        )
        tp_type = type(self.sections[0].thickness_params)
        thickness_consistent = all(
            isinstance(s.thickness_params, tp_type) for s in self.sections
        )
        return camber_consistent and thickness_consistent

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
