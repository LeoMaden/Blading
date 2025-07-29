"""Parameterised blade section module."""

from dataclasses import dataclass
from typing import Optional
from geometry.curves import PlaneCurve
from numpy.typing import NDArray
import numpy as np

from .section import Section
from .thickness import ThicknessResult, fit_thickness
from .camber import CamberResult, fit_camber


@dataclass
class ParameterisedSection:
    """A blade section with parameterised thickness and/or camber."""

    thickness_result: Optional[ThicknessResult] = None
    camber_result: Optional[CamberResult] = None
    base_camber_line: Optional[PlaneCurve] = None
    stream_line: Optional[PlaneCurve] = None

    def create_section(self, s: NDArray = None) -> Section:
        """Create a Section from the parameterised components."""
        if s is None and self.base_camber_line is not None:
            s = self.base_camber_line.param
        elif s is None:
            s = np.linspace(0, 1, 100)

        # Get camber line
        if self.camber_result is not None:
            camber_line = self.camber_result.create_camber_line(s)
        elif self.base_camber_line is not None:
            # Resample base camber line if needed
            camber_line = self.base_camber_line
        else:
            # Create a straight camber line as default
            coords = np.column_stack([s, np.zeros_like(s)])
            camber_line = PlaneCurve(coords, s)

        # Get thickness
        if self.thickness_result is not None:
            thickness = self.thickness_result.create_thickness_distribution(s)
        else:
            thickness = np.zeros_like(s)  # Flat section

        return Section(camber_line, thickness, self.stream_line)

    def plot_comparison(self, original_sec: Section):
        """Plot comparison with original section."""
        if self.thickness_result is not None:
            self.thickness_result.plot_fit_summary(original_sec)
        else:
            print("No thickness parameterisation available for plotting.")

    @classmethod
    def from_section(
        cls,
        sec: Section,
        fit_thickness_param: bool = True,
        fit_camber_param: bool = True,
        plot_intermediate: bool = False,
    ) -> "ParameterisedSection":
        """Create a parameterised section from an existing Section."""
        thickness_result = None
        camber_result = None

        if fit_thickness_param:
            thickness_result = fit_thickness(sec, plot_intermediate=plot_intermediate)

        if fit_camber_param:
            camber_result = fit_camber(sec, plot_intermediate=plot_intermediate)

        return cls(
            thickness_result=thickness_result,
            camber_result=camber_result,
            base_camber_line=sec.camber_line,
            stream_line=sec.stream_line,
        )
