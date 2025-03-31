from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rolls_royce.blade_def import BladeDef
from blading.fitting import (
    fit_section,
    NormalIntersectionsError,
    ImproveCamberConvergenceError,
)


def test():
    file = "/mnt/Tank/geometry/rolls-royce/TEN10/TEN_I01R"
    bd = BladeDef.read(Path(file))
    rtx = bd.sections[0].rtx
    r, t, x = rtx[:, 0], rtx[:, 1], rtx[:, 2]
    xy_section = np.c_[x, r * t][:-1]

    try:
        res = fit_section(xy_section, n_fit=150, n_fine=250)
    except NormalIntersectionsError as e:
        print(e)
        e.plot()
        raise e
    except ImproveCamberConvergenceError as e:
        print(e)
        e.plot()
        raise e

    print(f"{res.n_arr=}")

    plt.figure()
    plt.plot(xy_section[:, 0], xy_section[:, 1], "k-")
    plt.plot(res.section.camber.xy[:, 0], res.section.camber.xy[:, 1], ".-")

    plt.plot(res.section.xy_upper[:, 0], res.section.xy_upper[:, 1], "r.--")
    plt.plot(res.section.xy_lower[:, 0], res.section.xy_lower[:, 1], "y.--")
    plt.axis("equal")

    plt.figure()
    plt.plot(res.section.thickness.s, res.section.thickness.t, ".-")

    plt.figure()
    plt.plot(res.section.camber.s, res.section.camber.angle, ".-")

    plt.figure()
    plt.semilogy(res.conv, "x-")

    plt.show()


test()
