from scipy.interpolate import BSpline, make_interp_spline
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    knots = [0, 0, 0, 0, 0.05, 0.3, 1, 1, 1, 1]
    m = len(knots) - 1
    p = 3
    n = m - p - 1

    basis: list = [0] * (n + 1)
    for i in range(n + 1):  # number of basis functions
        coef = np.astype(np.arange(n + 1) == i, np.floating)
        basis[i] = BSpline(knots, coef, p)

    # Create system of equations
    A = np.zeros((n + 1, n + 1))
    A[0, :] = [b(0.4) for b in basis]
    A[1, :] = [b.derivative()(0.4) for b in basis]
    A[2, :] = [b.derivative().derivative()(0.4) for b in basis]
    A[3, :] = [b(0) for b in basis]
    A[4, :] = [b(1) for b in basis]
    A[5, :] = [b(0.2) for b in basis]
    b = np.r_[2, 0, -5, 1, 1, 2]
    c = np.linalg.solve(A, b)

    x = np.linspace(0, 1, 100)
    for i, b in enumerate(basis):
        plt.plot(x, b(x), label=i)

    spl = BSpline(knots, c, p)
    plt.plot(x, spl(x), "--", label="Spline")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
