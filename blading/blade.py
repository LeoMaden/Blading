from pathlib import Path
from typing import Any, Protocol
import numpy as np
from dataclasses import dataclass
from .section import Section, ReferencePoint
from .section_perimiter import SectionPerimiter
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from geometry.curves import PlaneCurve, SpaceCurve
from matplotlib.axes import Axes


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
        # self._interpolate_sections_to_common_param()

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

    @property
    def span_length(self) -> float:
        return self.meridional_ref_point_curve().length()

    @property
    def avg_chord(self) -> float:
        """Get the average chord length of the blade."""
        return float(np.mean(self.get_spanwise_value(lambda s: s.camber.chord)))

    @property
    def aspect_ratio(self) -> float:
        return self.span_length / self.avg_chord

    @property
    def hub_tip_ratio(self) -> float:
        curve = self.meridional_ref_point_curve()
        r_hub = curve.start()[1]
        r_tip = curve.end()[1]
        return r_hub / r_tip

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

    def _get_section_idx(self, span: float) -> int:
        """Get the index of the section nearest to the given spanwise location."""
        span_values = self.get_span()
        idx = np.argmin(np.abs(span_values - span))
        return int(idx)

    def get_nearest_section(self, span: float) -> Section:
        """Get the section nearest to the given spanwise location."""
        idx = self._get_section_idx(span)
        return self.sections[idx]

    def plot_meridional(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        i_hub = 0
        i_cas = len(self.sections) - 1

        # Plot the stream line for each section
        hub_cas_color, stream_line_color = None, None

        for i, section in enumerate(self.sections):
            stream_line = section.stream_line
            assert stream_line is not None, "Expected section to have a stream line"

            if i == i_hub or i == i_cas:
                label = "Hub and Casing Lines" if i == i_hub else ""
                stream_line.plot(ax, label=label, color=hub_cas_color)
                if hub_cas_color is None:
                    hub_cas_color = ax.get_lines()[-1].get_color()
            else:
                label = "Stream Lines" if i == 1 else ""
                stream_line.plot(ax, alpha=0.2, label=label, color=stream_line_color)
                if stream_line_color is None:
                    stream_line_color = ax.get_lines()[-1].get_color()

        # Plot the leading and trailing edges
        self.meridional_LE_curve().plot(ax, label="Leading and Trailing Edges")
        c = ax.get_lines()[-1].get_color()
        self.meridional_TE_curve().plot(ax, color=c)

        ref_point_name = self.sections[0].reference_point.display_name
        ref_point_label = f"Reference Point ({ref_point_name})"
        self.meridional_ref_point_curve().plot(ax, alpha=0.5, label=ref_point_label)

        ax.set_title(f"Meridional View ({len(self.sections)} Sections)")

        ax.axis("equal")
        ax.legend()
        return ax

    def compare_section_at_span(
        self,
        span: float,
        compare: list[Section] | list[SectionPerimiter],
        title: str | None = None,
        this_name="This",
        other_name="Comparison",
    ):
        idx = self._get_section_idx(span)
        section = self.sections[idx]
        compare_section = compare[idx]
        val = section.plot_comparison(
            compare_section, this_name=this_name, other_name=other_name
        )
        val[1].set_title(title or f"Section at span {span:.2f}")
        return val

    def plot_angles(
        self,
        ax: Axes | None = None,
        show_turning: bool = True,
        show_stagger: bool = True,
        *plot_args,
        **plot_kwargs,
    ):
        if ax is None:
            _, ax = plt.subplots()

        funcs_labels = [
            (lambda s: np.degrees(s.camber.angle_LE), "Leading Edge"),
            (lambda s: np.degrees(s.camber.angle_TE), "Trailing Edge"),
        ]
        if show_turning:
            funcs_labels.append(
                (lambda s: np.degrees(s.camber.angle_delta), "Turning Angle")
            )
        if show_stagger:
            funcs_labels.append((lambda s: np.degrees(s.stagger), "Stagger Angle"))

        for func, label in funcs_labels:
            self.plot_spanwise(func, ax, *plot_args, label=label, **plot_kwargs)

        ax.set_xlabel("Angle (degrees)")
        ax.set_title("Spanwise Blade Angles")
        ax.legend()

        return ax

    def plot_chord(
        self,
        ax=None,
        include_axial_chord: bool = False,
        units: str | None = None,
        *plot_args,
        **plot_kwargs,
    ):
        if ax is None:
            _, ax = plt.subplots()

        funcs_labels = [(lambda s: s.camber.chord, "Chord")]

        if include_axial_chord:
            funcs_labels.append((lambda s: s.axial_chord, "Axial Chord"))

        for func, label in funcs_labels:
            self.plot_spanwise(func, ax, label=label, *plot_args, **plot_kwargs)

        ax.set_xlabel("Length" + (f" ({units})" if units else ""))
        ax.set_title("Spanwise Chord Distribution")
        ax.legend()

        return ax

    def plot_max_t(self, ax=None, units: str | None = None, *plot_args, **plot_kwargs):
        ax = self.plot_spanwise(
            lambda s: s.thickness.max_t, ax, *plot_args, **plot_kwargs
        )

        ax.set_xlabel("Thickness" + (f" ({units})" if units else ""))
        ax.set_title("Spanwise Maximum Thickness")

        return ax

    def plot_LE_radius(
        self, ax=None, units: str | None = None, *plot_args, **plot_kwargs
    ):
        ax = self.plot_spanwise(
            lambda s: s.thickness.fit_LE_circle()[1], ax, *plot_args, **plot_kwargs
        )

        ax.set_xlabel("Radius" + (f" ({units})" if units else ""))
        ax.set_title("Spanwise Leading Edge Radius")

        return ax

    def plot_s_max_t(self, ax=None, *plot_args, **plot_kwargs):
        ax = self.plot_spanwise(
            lambda s: s.thickness.s_max_t, ax, *plot_args, **plot_kwargs
        )

        ax.set_xlabel("Normalised arc length")
        ax.set_title("Spanwise Pos. of Max. Thickness")

        return ax

    def plot_spanwise(
        self,
        func: SectionFunc,
        ax: Axes | None = None,
        *plot_args,
        **plot_kwargs,
    ) -> Axes:

        if ax is None:
            _, ax = plt.subplots()

        span = self.get_span()
        var = np.array([func(s) for s in self.sections])

        ax.plot(var, span, *plot_args, **plot_kwargs)

        ax.set_ylabel(
            f"Spanwise position at {self.reference_point.display_name.lower()}"
        )
        ax.grid(True)

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

    # def _interpolate_sections_to_common_param(self) -> None:
    #     """Interpolate all sections to have the same parameter distribution.

    #     Uses the parameter from the section with the highest number of points.
    #     """
    #     if len(self.sections) <= 1:
    #         return  # Nothing to interpolate

    #     # Find the section with the most points
    #     max_points = 0
    #     reference_param = None

    #     for section in self.sections:
    #         this_s = section.camber.s
    #         num_points = len(this_s)
    #         if num_points > max_points:
    #             max_points = num_points
    #             reference_param = this_s

    #     if reference_param is None:
    #         raise ValueError("Could not determine reference parameter")

    #     # Interpolate all sections to the reference parameter
    #     interpolated_sections = [sec.with_s(reference_param) for sec in self.sections]

    #     # Update the sections list (use object.__setattr__ for frozen dataclass)
    #     object.__setattr__(self, "sections", interpolated_sections)

    def pickle(self, file: str | Path):
        with open(file, "wb") as f:
            import pickle

            pickle.dump(self, f)

    @staticmethod
    def unpickle(file: str | Path) -> "Blade":
        with open(file, "rb") as f:
            import pickle

            blade = pickle.load(f)
        if not isinstance(blade, Blade):
            raise TypeError("Unpickled object is not a Blade instance")
        return blade
