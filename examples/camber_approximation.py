from pathlib import Path
from hydra.blade_def import BladeDef
from cfd.hydra_blading import convert_blade_def
from blading.approx import approx_camber_line
import matplotlib.pyplot as plt
from geometry.curves import plot_plane_curve


def main() -> None:
    file = Path("data/baseline/I01R/n100/Blade_Definition")
    blade_def = BladeDef.read(file)

    blade = convert_blade_def(blade_def)
    section = blade.sections[0]
    res = approx_camber_line(section, 1e-8)
    new_sec = res.section
    deltas = [i.delta for i in res.camber_iterations]

    # Plot original and approximated geometries.
    fig, ax = plt.subplots()
    plot_plane_curve(section.curve, ax, "k.-")
    plot_plane_curve(new_sec.upper_and_lower(), ax, "r.-")
    plot_plane_curve(new_sec.camber_line, ax, "b.-")
    plt.axis("equal")

    # Show convergence of the camber approximation.
    fig, ax = plt.subplots()
    plt.semilogy(deltas)

    plt.show()


if __name__ == "__main__":
    main()
