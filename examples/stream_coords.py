from pathlib import Path
from hydra.blade_def import BladeDef
from geometry.curves import PlaneCurve, plot_plane_curve
import matplotlib.pyplot as plt
from blading.annulus import Annulus


def main() -> None:
    file = Path("rows/I01R/n100/Blade_Definition")
    blade_def = BladeDef.read(file)
    hub = PlaneCurve.new_unit_speed(blade_def.sections[0].xr_streamtube)
    cas = PlaneCurve.new_unit_speed(blade_def.sections[-1].xr_streamtube)

    annulus = Annulus(hub, cas)

    for sec in blade_def.sections:
        ss_physical = PlaneCurve.new_unit_speed(sec.xr_streamtube)
        ss_stream = PlaneCurve.new_unit_speed(annulus.stream_coords(ss_physical.coords))

        plt.figure(0)
        plot_plane_curve(ss_physical)
        plt.axis("equal")
        plt.figure(1)
        plot_plane_curve(ss_stream)
        plt.axis("equal")

    plt.show()


if __name__ == "__main__":
    main()
