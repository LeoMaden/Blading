from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from blading.fitting import fit_section
from rolls_royce.blade_def import BladeDef
from blading.blade import (
    Section2D,
    SupersonicSplitCamberParam,
)
import blading
import blading.blade
from blading import misc
from blading import blade
from blading.fitting import NormalIntersectionsError

from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import cdist
from geometry.curves import cumulative_length
from scipy.interpolate import make_lsq_spline


def load_section():
    bd = BladeDef.read(
        Path(
            "/home/lm859/rds/hpc-work/projects/sensitivity_study/rows/original/I01R_SLS_n100_V604/Blade_Definition"
        )
    )
    i_sec = 0
    rtx = bd.sections[i_sec].rtx_blade[:-1]
    xy_section = np.c_[rtx[:, 2], rtx[:, 0] * rtx[:, 1]]
    return xy_section


def test_compare_original_and_fitted_section():
    xy_section = load_section()
    try:
        s = fit_section(xy_section)
    except NormalIntersectionsError as e:
        e.plot()

    plt.plot(xy_section[:, 0], xy_section[:, 1], "k-")
    plt.plot(s.xy_upper[:, 0], s.xy_upper[:, 1], "r.--")
    plt.plot(s.xy_lower[:, 0], s.xy_lower[:, 1], "r.--")
    plt.axis("equal")
    plt.show()


def test_camber_parameterisation():
    file = "/mnt/Tank/data/XWB_ESS_to_I01S_data/I01R_SLS_n100_V604/Blade_Definition"
    bd = BladeDef.read(Path(file))

    rtx = bd.sections[0].rtx_blade
    r, t, x = rtx[:, 0], rtx[:, 1], rtx[:, 2]
    xy_section = np.c_[x, r * t][:-1]

    res = fit_section(xy_section, n_fit=150)

    s = res.section.camber.s
    c_param = SupersonicSplitCamberParam.parameterise(res.section.camber)
    # t_param = ThicknessParam.parameterise(
    #     res.section.thickness.s, res.section.thickness.t
    # )

    # c_param.plot_spline(s)

    thickness_no_te = blading.remove_round_te(res.section.thickness)
    ss_no_te = thickness_no_te.calc_ss()

    camber = c_param.evaluate(s)
    # thickness = t_param.evaluate(s)
    section = Section2D.new(c_param.evaluate(s), res.section.thickness)

    plt.figure()
    plt.plot(s, res.section.camber.angle, "k-")
    # plt.plot(s, angle, "b--")
    plt.plot(s, camber.angle, "r--")
    print(f"{c_param.ss_pos=}, {c_param.ss_turning=}")

    plt.figure()
    plt.plot(s, res.section.thickness.t, "k-")
    plt.plot(s, thickness_no_te.t, "r--")
    # plt.plot(s, thickness.t, "r--")

    plt.figure()
    plt.plot(thickness_no_te.s, ss_no_te, "k-")

    plt.figure()
    plt.plot(res.section.xy_upper[:, 0], res.section.xy_upper[:, 1], "k-")
    plt.plot(res.section.xy_lower[:, 0], res.section.xy_lower[:, 1], "k-")
    plt.plot(section.xy_upper[:, 0], section.xy_upper[:, 1], "r--")
    plt.plot(section.xy_lower[:, 0], section.xy_lower[:, 1], "r--")
    plt.axis("equal")

    plt.show()


def test_BP33ThicknessParam_spline():
    # Parameters for NACA 0008-34 from Rogalsky (2004)
    tp = blading.blade.BP33ThicknessParam(
        le_radius=0.00157,
        pos_max_t=0.398,
        max_t=0.0402 * 2,
        curv_max_t=0.264,
        te_thickness=0,
        te_wedge=9.2 * 2,
    )
    # tp = blading.blade.BP33ThicknessParam(
    #     le_radius=0.00157,
    #     pos_max_t=0.398,
    #     max_t=0.0402,
    #     curv_max_t=0.4,
    #     te_thickness=0.001,
    #     te_wedge=5,
    # )
    s = np.linspace(0, 1, 100)
    tp.plot_spline(s)


def test_BP33ThicknessParam_parameterise():
    bd = BladeDef.read(
        Path(
            "/home/lm859/rds/hpc-work/projects/sensitivity_study/rows/original/I01R_SLS_n100_V604/Blade_Definition"
        )
    )
    i_sec = 0
    rtx = bd.sections[i_sec].rtx_blade[:-1]
    xy_section = np.c_[rtx[:, 2], rtx[:, 0] * rtx[:, 1]]
    res = fit_section(xy_section)
    thickness = blade.remove_round_te(res.section.thickness)
    t_param = blading.blade.BP33ThicknessParam.parameterise(thickness)
    print(t_param)
    t_param.plot_spline(thickness.s)

    thickness_fit = t_param.evaluate(thickness.s)

    plt.plot(thickness.s, thickness.t, "k--")
    plt.plot(thickness_fit.s, thickness_fit.t, "r-")
    plt.show()


def test_ShapeSpaceThicknessParam_parameterise():
    bd = BladeDef.read(
        Path(
            "/home/lm859/rds/hpc-work/projects/sensitivity_study/rows/original/I01R_SLS_n100_V604/Blade_Definition"
        )
    )
    i_sec = 20
    rtx = bd.sections[i_sec].rtx_blade[:-1]
    xy_section = np.c_[rtx[:, 2], rtx[:, 0] * rtx[:, 1]]
    res = fit_section(xy_section)
    thickness = blade.remove_round_te(res.section.thickness)
    # thickness = res.section.thickness
    # plt.plot(thickness.s, thickness.t)
    # plt.show()
    # return
    t_param = blading.blade.ShapeSpaceThicknessParam.parameterise(thickness)

    thickness_fit = t_param.evaluate(thickness.s)

    plt.plot(thickness.s, thickness.t, "k--")
    plt.plot(thickness_fit.s, thickness_fit.t, "r-")
    plt.show()


if __name__ == "__main__":
    # test_voroni()
    test_compare_original_and_fitted_section()

    # test_camber_parameterisation()

    # test_BP33ThicknessParam_spline()
    # test_BP33ThicknessParam_parameterise()

    # test_ShapeSpaceThicknessParam_parameterise()
