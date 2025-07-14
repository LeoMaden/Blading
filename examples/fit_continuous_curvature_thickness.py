from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from blading.section import Section
from blading.param import ContinuousCurvThickness
from hydra.blade_def import BladeDef
from cfd.hydra_blading import convert_blade_def
from blading.approx import approx_camber_line
from scipy.optimize import minimize, OptimizeResult
from blading.thickness import remove_round_TE
from scipy.interpolate import make_interp_spline


def main() -> None:
    file = Path(
        "/rds/user/lm859/hpc-work/projects/robust_transonic_rotor/data/baseline/I01R/n100/Blade_Definition"
    )
    bd = BladeDef.read(file)
    blade = convert_blade_def(bd)
    section = blade.sections[0]
    section = approx_camber_line(section).section
    no_TE_thickness = remove_round_TE(section.camber_line.param, section.thickness)
    section = Section(section.camber_line, no_TE_thickness, section.stream_line)

    # Maximum thickness of section - could be between points.
    thick_spl = make_interp_spline(section.camber_line.param, section.thickness, k=3)
    find_max_t = lambda s: -thick_spl(s)
    res = minimize(find_max_t, 0.5)
    assert res.success
    s_max_t = res.x[0]
    max_t = thick_spl(s_max_t)
    print(f"Maximum thickness = {max_t} at s = {s_max_t}")

    def calc_thickness(x):
        t_dist = ContinuousCurvThickness(
            radius_LE=x[0],
            thickness_TE=section.thickness[-1],
            wedge_angle=x[1],
            max_thickness=max_t,
            pos_max_t=s_max_t,
            curv_max_t=x[2],
            pos_knot=x[3],
        )
        f, _ = t_dist.make()
        return f(section.camber_line.param)

    err = lambda x: (np.max(np.abs(calc_thickness(x) - section.thickness)))

    x0 = [
        0.0001,  # radius_LE
        np.radians(0.5),  # wedge_angle
        -0.05,  # curv_max_t
        0.05,  # pos_knot
    ]
    bounds = [
        (0, None),
        (0, None),
        (None, 0),
        (0.02, 0.98),
    ]

    res: OptimizeResult = minimize(err, x0, bounds=bounds, tol=1e-15)
    assert res.success
    print(f"Error = {res.fun}")
    thickness_fit = calc_thickness(res.x)

    plt.figure()
    plt.plot(section.camber_line.param, section.thickness, "k-")
    plt.plot(section.camber_line.param, thickness_fit, "r--")

    plt.show()


if __name__ == "__main__":
    main()
