from blading.param import *
import numpy as np
from geometry.curves import plot_plane_curve
import matplotlib.pyplot as plt


def cluster_left(a, b, N):
    theta = np.linspace(0, np.pi / 2, N)
    cos = np.cos(theta)
    return b - (b - a) * cos


def cluster_right(a, b, N):
    theta = np.linspace(np.pi / 2, 0, N)
    cos = np.cos(theta)
    return a + (b - a) * cos


def main() -> None:
    transonic_camber = TransonicCamber(
        ss_turning_ratio=0.3,
        ss_chord_ratio=0.5,
        alpha=0.8,
        beta=0.3,
        ss_scale=(1, 1),
        sb_scale=(1, 1),
    )
    camber_distr = transonic_camber.make()

    s_norm = np.r_[
        cluster_left(0, 0.1, 25),
        np.linspace(0.1, 0.9, 50)[1:-1],
        cluster_right(0.9, 1.0, 25),
    ]
    s = s_norm * 3.3
    offset = [1.1, 0.4]
    camber = create_camber(camber_distr, 80, 70, s, offset)

    plot_plane_curve(camber, None, "k-")
    plt.axis("equal")

    plt.figure()
    plt.plot(s_norm, camber_distr(s_norm))
    plt.plot(transonic_camber.t_star, transonic_camber.coefs, "x")
    plt.plot(camber_distr.t, np.zeros_like(camber_distr.t), "o")

    plt.show()


if __name__ == "__main__":
    main()
