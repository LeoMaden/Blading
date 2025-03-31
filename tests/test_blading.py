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


if __name__ == "__main__":
    test_camber_parameterisation()
