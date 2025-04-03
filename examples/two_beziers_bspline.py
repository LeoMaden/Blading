import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt


control_points = np.array(
    [[0, 0], [0, 0.1], [0.2, 0.3], [0.4, 0.3], [0.6, 0.3], [0.8, 0.1], [1, 0.05]]
)

n = len(control_points)
k = 3

knots = [*np.zeros(k), *np.linspace(0, 1, n - k + 1), *np.ones(k)]

if len(knots) != n + k + 1:
    raise ValueError(f"{len(knots)=} must be {n+k+1=}")

spline = BSpline(t=knots, c=control_points, k=k)

t_vals = np.linspace(0, 1, 100)
curve = spline(t_vals)

plt.plot(curve[:, 0], curve[:, 1], "b.-")
plt.plot(control_points[:, 0], control_points[:, 1], "ro--")
plt.show()
