import matplotlib.pyplot as plt
from typing import Mapping
from blading.blade import Section, PSection
import numpy as np
from geometry.curves import PlaneCurve, plot_plane_curve


def create_section(params: Mapping[str, float]) -> Section:
    angle = np.radians(np.linspace(params["bia"], params["boa"], 100))

    s = np.linspace(0, params["chord"], len(angle))
    s_over_c = s / params["chord"]
    t_over_c = 2 * params["max t/c"] * np.sqrt(0.25 - (s_over_c - 0.5) ** 2)
    t = t_over_c * params["chord"]

    camber = PlaneCurve.from_angle(angle, s)
    offset = np.array([[params["xle"], params["yle"]]])
    camber = camber + offset
    return Section(camber, t, None)


def main() -> None:
    params = {
        "bia": 70,
        "boa": 120,
        "chord": 1,
        "max t/c": 0.08,
        "xle": 0.1,
        "yle": 0.2,
    }

    psec = PSection(params, create_section)
    sec2 = psec.update({"boa": -55, "max t/c": 0.02, "xle": 0, "yle": 0}).section()
    sec = psec.section()

    up_and_lo = sec.upper_and_lower()
    i_lower = up_and_lo.param <= 0
    i_upper = up_and_lo.param >= 0

    fig, ax = plt.subplots()
    plot_plane_curve(sec.camber_line, ax)
    plot_plane_curve(up_and_lo[i_lower], ax)
    plot_plane_curve(up_and_lo[i_upper], ax)
    plot_plane_curve(sec2.upper_and_lower(), ax)
    plt.axis("equal")

    plt.show()


if __name__ == "__main__":
    main()
