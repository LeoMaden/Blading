import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Define control points
control_points = np.array([[0, 0], [1, 2], [3, 3], [4, 2], [5, 0]])
x, y = control_points[:, 0], control_points[:, 1]

# Degree of the B-spline (Cubic)
k = 3

# Number of control points
n = len(control_points)

# Create a clamped knot vector (first and last knots repeated k+1 times)
t = np.concatenate(
    (
        np.zeros(k),  # Repeat 0 for clamping at the start
        np.linspace(0, 1, n - k + 1),  # Internal knots uniformly spaced
        np.ones(k),  # Repeat 1 for clamping at the end
    )
)

# Create B-spline object
spl = BSpline(t, control_points, k)

# Evaluate B-spline curve
t_vals = np.linspace(0, 1, 100)
curve = spl(t_vals)

# Plot the B-spline curve and control points
plt.plot(curve[:, 0], curve[:, 1], "b-", label="B-Spline Curve")
plt.plot(x, y, "ro--", label="Control Polygon")
plt.legend()
plt.show()
