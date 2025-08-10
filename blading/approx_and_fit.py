from dataclasses import dataclass
import logging
from .approx import ApproximateCamberResult, ApproximateCamberConfig, approximate_camber
from .thickness import fit_thickness, FitThicknessResult
from .camber import fit_camber, FitCamberResult
from .section_perimiter import SectionPerimiter
from .section import Section
from .blade import Blade


logger = logging.getLogger(__name__)


@dataclass
class SectionApproxFitResult:
    approx_result: ApproximateCamberResult
    fit_thickness_result: FitThicknessResult | None = None
    fit_camber_result: FitCamberResult | None = None
    section: Section | None = None
    error_message: str = ""
    success: bool = False

    def unwrap(self) -> Section:
        """Unwrap the result to get the fitted Section."""
        if self.section is None:
            raise ValueError(f"Section could not be fitted: {self.error_message}")
        return self.section


def approx_and_fit_section(
    section_perim: SectionPerimiter, approx_config: ApproximateCamberConfig
) -> SectionApproxFitResult:

    approx_res = approximate_camber(section_perim, approx_config)
    for l in approx_res.get_summary(True).splitlines():
        logger.info(l)
    if not approx_res.success:
        return SectionApproxFitResult(
            approx_result=approx_res,
            error_message=approx_res.error_message,
        )

    section = approx_res.unwrap()

    # Fit parameters
    fit_thickness_res = fit_thickness(section.thickness)
    fit_camber_res = fit_camber(section.camber)

    thickness_params = fit_thickness_res.result.params
    camber_params = fit_camber_res.result.params

    # Create parameterised section
    param_section = Section(
        thickness=thickness_params,
        camber=camber_params,
        s=section.camber.s,
        stream_line=section.stream_line,
        reference_point=section.reference_point,
    )
    return SectionApproxFitResult(
        approx_res,
        fit_thickness_res,
        fit_camber_res,
        param_section,
        success=True,
    )


@dataclass
class BladeApproxFitResult:
    results: list[SectionApproxFitResult]
    blade: Blade | None = None
    error_message: str = ""
    success: bool = False

    def unwrap(self) -> Blade:
        """Unwrap the result to get the fitted Blade."""
        if self.blade is None:
            raise ValueError(f"Blade could not be fitted: {self.error_message}")
        return self.blade


def approx_and_fit_blade(
    section_perims: list[SectionPerimiter],
    approx_config: ApproximateCamberConfig | None = None,
) -> BladeApproxFitResult:
    approx_config = approx_config or ApproximateCamberConfig()

    sections: list[Section] = []
    results: list[SectionApproxFitResult] = []
    num_sections = len(section_perims)

    for i, section_perim in enumerate(section_perims):

        logger.info(f"Processing section {i + 1}/{num_sections}")

        res = approx_and_fit_section(section_perim, approx_config)

        results.append(res)
        if not res.success:
            logger.error(f"Failed at section {i + 1}: {res.error_message}")
            return BladeApproxFitResult(
                results=results,
                error_message=res.error_message,
            )

        assert res.section is not None, "Expected section"
        sections.append(res.unwrap())

    blade = Blade(sections)
    return BladeApproxFitResult(
        results=results,
        blade=blade,
        success=True,
    )
