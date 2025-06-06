from blading import blade
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    s = np.linspace(0, 1, 100)
    angle = np.linspace(50, -50, 100)
    rel_thick = np.linspace(0.1, 1, 100)

    c = blade.Camber(s, angle, blade.Cartesian.empty())
    t = blade.Thickness(s, rel_thick, maxt=0.1)
    b = blade.DistrSection(c, t)

    plt.plot(b.upper_coords.x, b.upper_coords.y)
    plt.plot(b.lower_coords.x, b.lower_coords.y)
    plt.plot(b.camber.coords.x, b.camber.coords.y)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
