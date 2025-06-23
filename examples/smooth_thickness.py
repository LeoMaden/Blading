import matplotlib.pyplot as plt
import numpy as np
from blading.param import ContinuousCurvThickness


def cluster_left(a, b, N):
    theta = np.linspace(0, np.pi / 2, N)
    cos = np.cos(theta)
    return b - (b - a) * cos


def main() -> None:
    t__dist = ContinuousCurvThickness(
        0.0001,
        0.005,
        np.radians(0.5),
        0.02,
        0.3,
        -0.05,
        0.05,
    )
    eval, spl = t__dist.make()

    x = np.r_[
        cluster_left(0, 0.1, 50),
        np.linspace(0.1, 1, 75)[1:],
    ]

    plt.figure()
    plt.plot(x, eval(x))

    plt.figure()
    plt.plot(x, spl(x))

    plt.show()


if __name__ == "__main__":
    main()
