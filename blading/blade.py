from abc import ABC
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Protocol, Self, TypeVar
from scipy.interpolate import make_interp_spline
from scipy.integrate import cumulative_trapezoid
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import vtk
from scipy.optimize import minimize
from numpy.polynomial import Polynomial
from circle_fit import taubinSVD
from scipy.optimize import least_squares
from blading.misc import calc_te_t_param

import geometry
import geometry.surfaces

import shapespace


# ======================================================================================


@dataclass
class Camber:
    s: NDArray
    chord: float
    xy: NDArray
    angle: NDArray
    angle_le: float
    angle_te: float
    non_dim: NDArray

    @staticmethod
    def from_xy(xy):
        s = geometry.cumulative_length(xy)
        chord = s[-1]
        s /= chord  # normalise
        angle = np.degrees(np.arctan(np.gradient(xy[:, 1], xy[:, 0])))
        angle_le = angle[0]
        angle_te = angle[-1]
        non_dim = (angle - angle_le) / (angle_te - angle_le)

        return Camber(
            s=s,
            chord=chord,
            xy=xy,
            angle=angle,
            angle_le=angle_le,
            angle_te=angle_te,
            non_dim=non_dim,
        )

    @staticmethod
    def from_angle(angle: NDArray, s: NDArray, chord: float, xy_offset: NDArray):
        x = np.r_[0, cumulative_trapezoid(np.cos(np.radians(angle)), s)]
        y = np.r_[0, cumulative_trapezoid(np.sin(np.radians(angle)), s)]

        xy = np.c_[x, y]
        xy /= geometry.total_length(xy)
        xy = xy * chord + xy_offset

        angle_le = angle[0]
        angle_te = angle[-1]
        non_dim = (angle - angle_le) / (angle_te - angle_le)

        return Camber(
            s=s,
            chord=chord,
            xy=xy,
            angle=angle,
            angle_le=angle_le,
            angle_te=angle_te,
            non_dim=non_dim,
        )


@dataclass
class Thickness:
    s: NDArray
    t: NDArray

    def calc_ss(self):
        t_te = self.t[-1]
        valid = np.arange(1, len(self.s) - 1)
        ss = np.nan * np.ones_like(self.s)
        ss[valid] = shapespace.value(self.s[valid], self.t[valid], t_te)
        return ss


@dataclass
class Section2D:
    camber: Camber
    thickness: Thickness

    xy_upper: NDArray
    xy_lower: NDArray

    @staticmethod
    def new(camber: Camber, thickness: Thickness):
        c = camber.xy
        t = thickness.t
        chord = camber.chord

        n = geometry.calc_normals(c)
        xy_upper = c + 0.5 * n * chord * t[:, np.newaxis]
        xy_lower = c - 0.5 * n * chord * t[:, np.newaxis]

        return Section2D(
            camber=camber, thickness=thickness, xy_upper=xy_upper, xy_lower=xy_lower
        )

    def to_3d(self, xr_streamsurf: NDArray):
        def to_xrrt(xy, xr):
            x = xy[:, 0]
            y = xy[:, 1]
            r = np.interp(x, xr[:, 0], xr[:, 1])
            return np.c_[x, r, y]

        xrrt_camber = to_xrrt(self.camber.xy, xr_streamsurf)
        xrrt_upper = to_xrrt(self.xy_upper, xr_streamsurf)
        xrrt_lower = to_xrrt(self.xy_lower, xr_streamsurf)

        return Section3D(
            camber=self.camber,
            thickness=self.thickness,
            xr_streamsurf=xr_streamsurf,
            xrrt_camber=xrrt_camber,
            xrrt_lower=xrrt_lower,
            xrrt_upper=xrrt_upper,
        )


@dataclass
class Section3D:
    camber: Camber
    thickness: Thickness
    xr_streamsurf: NDArray

    xrrt_camber: NDArray
    xrrt_upper: NDArray
    xrrt_lower: NDArray

    @staticmethod
    def new(camber: Camber, thickness: Thickness, xr_streamsurf: NDArray):
        return Section2D.new(camber, thickness).to_3d(xr_streamsurf)


@dataclass
class Blade:
    sections: list[Section3D]

    xrrt_camber: NDArray
    xrrt_upper: NDArray
    xrrt_lower: NDArray

    @staticmethod
    def new(sections: list[Section3D]):
        xrrt_camber = np.stack([s.xrrt_camber for s in sections], axis=-1)
        xrrt_upper = np.stack([s.xrrt_upper for s in sections], axis=-1)
        xrrt_lower = np.stack([s.xrrt_lower for s in sections], axis=-1)

        xrrt_camber = np.swapaxes(xrrt_camber, 2, 1)
        xrrt_upper = np.swapaxes(xrrt_upper, 2, 1)
        xrrt_lower = np.swapaxes(xrrt_lower, 2, 1)

        return Blade(
            sections=sections,
            xrrt_camber=xrrt_camber,
            xrrt_upper=xrrt_upper,
            xrrt_lower=xrrt_lower,
        )

    def to_vtk(self, filename: Path):
        path = Path(filename).with_suffix(".vtm")

        multiblock = vtk.vtkMultiBlockDataSet()

        def convert_to_vtk_grid(surf):
            points = vtk.vtkPoints()
            ni, nj = surf.shape[:2]
            for j in range(nj):
                for i in range(ni):
                    x, y, z = surf[i, j]
                    points.InsertNextPoint(x, y, z)

            grid = vtk.vtkStructuredGrid()
            ni, nj, _ = surf.shape
            grid.SetDimensions(ni, nj, 1)
            grid.SetPoints(points)
            return grid

        a = convert_to_vtk_grid(self.xrrt_camber)
        multiblock.SetBlock(0, a)
        multiblock.SetBlock(1, convert_to_vtk_grid(self.xrrt_lower))
        multiblock.SetBlock(2, convert_to_vtk_grid(self.xrrt_upper))

        writer = vtk.vtkXMLMultiBlockDataWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(multiblock)
        writer.Write()


# ======================================================================================


@dataclass
class ThicknessParam:
    rad_le: float
    max_t: float
    x_max_t: float
    rad_max_t: float
    angle_te: float
    t_te: float
    x_join: float
    x_stretch: float
    x_split: float

    def __post_init__(self):
        self._construct()

    def evaluate(self, s: NDArray) -> Thickness:
        # plt.figure()

        # Stretch coordinates
        xs = np.copy(s)
        i = s < self.x_split
        xs[i] = s[i] - self.ps(s[i])

        # Before join
        i = xs < self.x_join
        ss1 = self.p1(xs[i])

        # After join
        i = xs >= self.x_join
        ss2 = self.p2(xs[i])

        ss = np.r_[ss1, ss2]
        t = shapespace.inverse(s, ss, self.t_te)

        # Round trailing edge
        a = np.arctan(0.5 * np.tan(np.radians(self.angle_te)))
        b = 2 * np.cos(a) - (1 - np.sin(a)) * np.tan(np.radians(self.angle_te))
        rad_te = self.t_te / b
        pts = np.linspace(a, np.pi / 2, 40)
        x_te = 1 - rad_te + rad_te * np.sin(pts)
        y_te = 2 * rad_te * np.cos(pts)
        i = xs > min(x_te)
        t[i] = np.interp(xs[i], x_te, y_te)

        self.t = t
        self.ss = ss

        return Thickness(s, t)

    def __call__(self, x):
        return self.evaluate(x)

    def _construct(self):
        ss_le = np.sqrt(2 * self.rad_le)
        ss_mt = shapespace.value(self.x_max_t, self.max_t, self.t_te)
        ss1_mt = shapespace.deriv1(self.x_max_t, self.max_t, self.t_te, 0)
        ss2_mt = shapespace.deriv2(
            self.x_max_t, self.max_t, self.t_te, 0, -1 / self.rad_max_t
        )
        ss_te = np.tan(np.radians(self.angle_te)) + self.t_te

        # Rear cubic
        b = [ss_mt, ss1_mt, ss2_mt, ss_te]
        A = np.zeros((4, 4))
        x1 = self.x_max_t
        x2 = 1
        A[0, :] = [x1**3, x1**2, x1, 1]
        A[1, :] = [3 * x1**2, 2 * x1, 1, 0]
        A[2, :] = [6 * x1, 2, 0, 0]
        A[3, :] = [x2**3, x2**2, x2, 1]
        c2 = np.linalg.solve(A, b)

        # Values at join
        p2 = Polynomial(c2[::-1])
        ssj = p2(self.x_join)
        ss1_j = p2.deriv(1)(self.x_join)
        ss2_j = p2.deriv(2)(self.x_join)

        # Front cubic
        b = [ss_le, ssj, ss1_j, ss2_j]
        A = np.zeros((4, 4))
        x1 = -self.x_stretch
        x2 = self.x_join
        A[0, :] = [x1**3, x1**2, x1, 1]
        A[1, :] = [x2**3, x2**2, x2, 1]
        A[2, :] = [3 * x2**2, 2 * x2, 1, 0]
        A[3, :] = [6 * x2, 2, 0, 0]
        c1 = np.linalg.solve(A, b)
        p1 = Polynomial(c1[::-1])

        # Stretch near leading edge
        b = [self.x_stretch, 0, 0, 0]
        A = np.zeros((4, 4))
        x1 = 0
        x2 = self.x_split
        A[0, :] = [x1**3, x1**2, x1, 1]
        A[1, :] = [x2**3, x2**2, x2, 1]
        A[2, :] = [3 * x2**2, 2 * x2, 1, 0]
        A[3, :] = [6 * x2, 2, 0, 0]
        cs = np.linalg.solve(A, b)
        ps = Polynomial(cs[::-1])

        self.p1 = p1
        self.p2 = p2
        self.ps = ps

    @staticmethod
    def parameterise(x, t):
        # Maximum thickness
        i_max_t = np.argmax(t)
        max_t = t[i_max_t]
        x_max_t = x[i_max_t]

        # # Trailing edge
        dtdx = np.gradient(t, x)
        # thresh = np.interp(0.8, x, dtdx)  # dt/dx at 80% chord
        # i_te = np.min(np.where((np.abs(dtdx) > 2 * np.abs(thresh)) & (x > 0.8))[0])
        # # angle_te = np.degrees(np.arctan(-dtdx[i_te - 1]))

        # # Extend trailing edge wedge
        # i = (x <= x[i_te]) & (x >= x[i_te] - 0.1)
        # te_poly = Polynomial.fit(x[i], t[i], 1)
        # t_te = te_poly(1)
        # angle_te = np.degrees(np.arctan(-te_poly.deriv(1)(1)))

        t_te, angle_te = calc_te_t_param(x, t)

        # State space
        ss = shapespace.value(x, t, t_te)

        # Curvature at max thickness
        d2tdx2 = np.gradient(dtdx, x)
        rad_max_t = -1 / d2tdx2[i_max_t]

        # Leading edge radius
        i_le = 5
        x_le, t_le = x[:i_le], t[:i_le]
        upper = np.c_[x_le, t_le / 2]
        lower = np.c_[x_le, -t_le / 2][::-1]
        coords = np.r_[lower, upper[1:]]
        xc, yc, rad_le, _ = taubinSVD(coords)

        # Optimize join and stretch parameters
        t_param = ThicknessParam(
            rad_le=rad_le,
            max_t=max_t,
            x_max_t=x_max_t,
            rad_max_t=rad_max_t,
            angle_te=angle_te,
            t_te=t_te,
            x_join=0.98 * x_max_t,
            x_stretch=0.35,
            x_split=0.15,
        )

        w = np.interp(x, [0, 0.1, 0.2, 1], [0.1, 0.1, 1, 1])
        # w = np.ones_like(x)

        def err(params):
            a, b, c, d = params
            t_param.x_join = a
            t_param.x_stretch = b
            t_param.x_split = c
            t_param.rad_max_t = d
            t_param._construct()
            thickness = t_param.evaluate(x)
            t_fit = thickness.t

            return np.trapz(w * (t - t_fit) ** 2, x)

        x0 = [0.4, 0.3, 0.1, rad_max_t]
        constraints = [
            {"type": "ineq", "fun": lambda p: p[0] - p[2]},
            {"type": "ineq", "fun": lambda p: p[3]},
            {"type": "ineq", "fun": lambda p: x_max_t - p[0]},
            {"type": "ineq", "fun": lambda p: p[2] + 1e-5},
        ]
        res = minimize(
            err, x0, constraints=constraints, tol=1e-16, options={"maxiter": 10_000}
        )
        if not res.success:
            raise Exception(f"Error fitting params: {res.message}\n{res.x}")
        else:
            print(f"{res.x=}")

        a, b, c, d = res.x
        t_param.x_join = a
        t_param.x_stretch = b
        t_param.x_split = c
        t_param.rad_max_t = d
        return t_param


# ======================================================================================


class CamberParam(Protocol):
    def evaluate(self, s: NDArray) -> Camber: ...

    @staticmethod
    def parameterise(camber: Camber) -> "CamberParam": ...


@dataclass
class MCACamberParam:
    chord: float
    xy_offset: NDArray

    angle_le: float
    angle_te: float

    ss_turning: float
    ss_pos: float

    def evaluate(self, s: NDArray) -> Camber:
        spline = make_interp_spline([0, self.ss_pos, 1], [0, self.ss_turning, 1], k=1)
        non_dim = spline(s)
        angle = non_dim * (self.angle_te - self.angle_le) + self.angle_le
        return Camber.from_angle(angle, s, self.chord, self.xy_offset)

    @staticmethod
    def parameterise(camber: Camber) -> "MCACamberParam":
        # Extract parameters from given camber line
        chord = camber.chord
        xy_offset = camber.xy[0]

        # Fit other parameters using least squares
        def create(x):
            return MCACamberParam(
                chord=chord,
                xy_offset=xy_offset,
                angle_le=x[2],
                angle_te=x[3],
                ss_turning=x[0],
                ss_pos=x[1],
            )

        def fun(x):
            c_param = create(x)
            camber_fit = c_param.evaluate(camber.s)
            return camber.angle - camber_fit.angle

        x0 = [0.5, 0.5, camber.angle_le, camber.angle_te]
        bounds = ([0, 0, -90, -90], [1, 1, 90, 90])
        res = least_squares(fun, x0, bounds=bounds, loss="soft_l1")
        if not res.success:
            raise Exception("Could not parameterise")

        return create(res.x)


@dataclass
class SupersonicSplitCamberParam:
    chord: float
    xy_offset: NDArray

    angle_le: float
    angle_te: float

    ss_pos: float
    ss_turning: float

    ss1: float
    ss2: float
    sb1: float
    sb2: float

    def _create_spline(self):
        x_ss1 = 0.3 * self.ss_pos
        x_ss2 = 0.7 * self.ss_pos
        x_sb1 = self.ss_pos + 0.3 * (1 - self.ss_pos)
        x_sb2 = self.ss_pos + 0.7 * (1 - self.ss_pos)

        y_ss1 = self.ss1 * self.ss_turning
        y_ss2 = self.ss2 * self.ss_turning
        y_sb1 = self.ss_turning + self.sb1 * (1 - self.ss_turning)
        y_sb2 = self.ss_turning + self.sb2 * (1 - self.ss_turning)

        spline = make_interp_spline(
            [0, x_ss1, x_ss2, self.ss_pos, x_sb1, x_sb2, 1],
            [0, y_ss1, y_ss2, self.ss_turning, y_sb1, y_sb2, 1],
            k=3,
            bc_type="natural",
        )
        return spline

    def evaluate(self, s: NDArray) -> Camber:
        spline = self._create_spline()
        non_dim = spline(s)
        angle = non_dim * (self.angle_te - self.angle_le) + self.angle_le
        return Camber.from_angle(angle, s, self.chord, self.xy_offset)

    @staticmethod
    def parameterise(camber: Camber) -> "SupersonicSplitCamberParam":
        # Extract parameters from given camber line
        chord = camber.chord
        xy_offset = camber.xy[0]

        # Fit other parameters using least squares
        def create(x):
            return SupersonicSplitCamberParam(
                chord=chord,
                xy_offset=xy_offset,
                angle_le=x[2],
                angle_te=x[3],
                ss_pos=x[0],
                ss_turning=x[1],
                ss1=x[4],
                ss2=x[5],
                sb1=x[6],
                sb2=x[7],
            )

        def fun(x):
            c_param = create(x)
            camber_fit = c_param.evaluate(camber.s)
            return camber.angle - camber_fit.angle

        x0 = np.array([0.5, 0.5, camber.angle_le, camber.angle_te, 0.3, 0.7, 0.3, 0.7])
        bounds = (
            [0, 0, -90, -90, *[0] * 4],
            [1, 1, 90, 90, *[1] * 4],
        )
        res = least_squares(fun, x0, bounds=bounds, loss="arctan")
        if not res.success:
            raise Exception("Could not parameterise")

        return create(res.x)

    def plot_spline(self, s: NDArray):
        spline = self._create_spline()
        non_dim = spline(s)
        x_knots = spline.t[3:-3]
        y_knots = spline(x_knots)
        plt.figure()
        plt.plot(s, non_dim, "k-")
        plt.plot(x_knots[[0, 3, 6]], y_knots[[0, 3, 6]], "rx")
        plt.plot(x_knots[[1, 2, 4, 5]], y_knots[[1, 2, 4, 5]], "bx")
        plt.show()


# ======================================================================================


# # Meridional
# @dataclass
# class MeridionalCurve:
#     xr: NDArray

#     @staticmethod
#     def new(xr: NDArray):
#         return MeridionalCurve(xr)

#     @staticmethod
#     def from_params(
#         x: ArrayLike,
#         r: ArrayLike,
#         # tangent_angle: Pair,
#         # tangent_scale: Pair,
#         discretisation: int,
#     ):
#         # Tangent vectors
#         def tangent_vec(i: int):
#             angle_rad = np.radians(tangent_angle[i])
#             tangent = np.r_[np.cos(angle_rad), np.sin(angle_rad)]
#             tangent *= tangent_scale[i]
#             return tangent

#         # Spline
#         x = np.asarray(x)
#         abscissa = np.linspace(0, 1, len(x))
#         points = np.c_[x, r]
#         deriv_l = [(1, tangent_vec(0))]
#         deriv_r = [(1, tangent_vec(1))]
#         spline = make_interp_spline(abscissa, points, bc_type=(deriv_l, deriv_r))

#         num_segments = len(x) - 1
#         s = np.linspace(0, 1, num_segments * discretisation)
#         xr = spline(s)
#         return MeridionalCurve(xr)


# ======================================================================================


@dataclass
class Section2DParam:
    c_param: CamberParam
    t_param: ThicknessParam

    camber: Camber
    thickness: Thickness

    xy_upper: NDArray
    xy_lower: NDArray

    @staticmethod
    def new(c_param: CamberParam, t_param: ThicknessParam, s: NDArray):
        camber = c_param.evaluate(s)
        thickness = t_param.evaluate(s)

        c = camber.xy
        t = thickness.t
        chord = camber.chord

        n = geometry.calc_normals(c)
        xy_upper = c + 0.5 * n * chord * t[:, np.newaxis]
        xy_lower = c - 0.5 * n * chord * t[:, np.newaxis]

        return Section2DParam(
            c_param=c_param,
            t_param=t_param,
            camber=camber,
            thickness=thickness,
            xy_upper=xy_upper,
            xy_lower=xy_lower,
        )

    def to_3d(self, xr_streamsurf: NDArray):
        def to_xrrt(xy, xr):
            x = xy[:, 0]
            y = xy[:, 1]
            r = np.interp(x, xr[:, 0], xr[:, 1])
            return np.c_[x, r, y]

        xrrt_camber = to_xrrt(self.camber.xy, xr_streamsurf)
        xrrt_upper = to_xrrt(self.xy_upper, xr_streamsurf)
        xrrt_lower = to_xrrt(self.xy_lower, xr_streamsurf)

        return Section3DParam(
            c_param=self.c_param,
            t_param=self.t_param,
            camber=self.camber,
            thickness=self.thickness,
            xr_streamsurf=xr_streamsurf,
            xrrt_camber=xrrt_camber,
            xrrt_lower=xrrt_lower,
            xrrt_upper=xrrt_upper,
        )


@dataclass
class Section3DParam:
    c_param: CamberParam
    t_param: ThicknessParam

    camber: Camber
    thickness: Thickness
    xr_streamsurf: NDArray

    xrrt_camber: NDArray
    xrrt_upper: NDArray
    xrrt_lower: NDArray

    @staticmethod
    def new(
        c_param: CamberParam,
        t_param: ThicknessParam,
        s: NDArray,
        xr_streamsurf: NDArray,
    ):
        return Section2DParam.new(c_param, t_param, s).to_3d(xr_streamsurf)


@dataclass
class BladeParam:
    sections: list[Section3DParam]

    xrrt_camber: NDArray
    xrrt_upper: NDArray
    xrrt_lower: NDArray

    @staticmethod
    def new(sections: list[Section3DParam]):
        xrrt_camber = np.stack([s.xrrt_camber for s in sections], axis=-1)
        xrrt_upper = np.stack([s.xrrt_upper for s in sections], axis=-1)
        xrrt_lower = np.stack([s.xrrt_lower for s in sections], axis=-1)

        xrrt_camber = np.swapaxes(xrrt_camber, 2, 1)
        xrrt_upper = np.swapaxes(xrrt_upper, 2, 1)
        xrrt_lower = np.swapaxes(xrrt_lower, 2, 1)

        return BladeParam(
            sections=sections,
            xrrt_camber=xrrt_camber,
            xrrt_upper=xrrt_upper,
            xrrt_lower=xrrt_lower,
        )
