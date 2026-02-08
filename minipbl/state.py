"""Model state container for prognostic and diagnostic fields."""

import copy

import numpy as np

from .grid import Grid


class State:
    """Holds prognostic fields (theta, u, v, w) and diagnostic fields.

    1D (dim=1): theta(nz), heat_flux(nz+1), K_h(nz+1), bl_height scalar.
                u, v, w, p are None.
    2D (dim=2): theta(nx,nz), u(nx,nz), w(nx,nz+1), p(nx,nz),
                heat_flux(nx,nz+1), K_h(nx,nz+1), K_m(nx,nz+1),
                bl_height(nx,).  v is None.
    3D (dim=3): theta(nx,ny,nz), u(nx,ny,nz), v(nx,ny,nz), w(nx,ny,nz+1),
                p(nx,ny,nz), heat_flux(nx,ny,nz+1), K_h(nx,ny,nz+1),
                K_m(nx,ny,nz+1), bl_height(nx,ny).
    """

    def __init__(self, grid: Grid):
        self.grid = grid
        nx, ny, nz = grid.nx, grid.ny, grid.nz

        if grid.dim >= 3:
            self.theta = np.zeros((nx, ny, nz))
            self.u = np.zeros((nx, ny, nz))
            self.v = np.zeros((nx, ny, nz))
            self.w = np.zeros((nx, ny, nz + 1))
            self.p = np.zeros((nx, ny, nz))
            self.heat_flux = np.zeros((nx, ny, nz + 1))
            self.K_h = np.zeros((nx, ny, nz + 1))
            self.K_m = np.zeros((nx, ny, nz + 1))
            self.bl_height = np.zeros((nx, ny))
        elif grid.dim >= 2:
            self.theta = np.zeros((nx, nz))
            self.u = np.zeros((nx, nz))
            self.v = None
            self.w = np.zeros((nx, nz + 1))
            self.p = np.zeros((nx, nz))
            self.heat_flux = np.zeros((nx, nz + 1))
            self.K_h = np.zeros((nx, nz + 1))
            self.K_m = np.zeros((nx, nz + 1))
            self.bl_height = np.zeros(nx)
        else:
            self.theta = np.zeros(nz)
            self.heat_flux = np.zeros(nz + 1)
            self.K_h = np.zeros(nz + 1)
            self.bl_height = 0.0
            self.u = None
            self.v = None
            self.w = None
            self.p = None
            self.K_m = None

    def copy(self) -> "State":
        """Deep copy for RK3 sub-stages."""
        return copy.deepcopy(self)

    def initialize_theta(self, theta0: float, mixed_layer_height: float,
                         lapse_rate: float):
        """Set initial theta profile: mixed layer + stable stratification above."""
        if self.grid.dim >= 3:
            for k in range(self.grid.nz):
                z = self.grid.z_center[k]
                if z <= mixed_layer_height:
                    self.theta[:, :, k] = theta0
                else:
                    self.theta[:, :, k] = theta0 + lapse_rate * (z - mixed_layer_height)
        elif self.grid.dim >= 2:
            for k in range(self.grid.nz):
                z = self.grid.z_center[k]
                if z <= mixed_layer_height:
                    self.theta[:, k] = theta0
                else:
                    self.theta[:, k] = theta0 + lapse_rate * (z - mixed_layer_height)
        else:
            for k in range(self.grid.nz):
                z = self.grid.z_center[k]
                if z <= mixed_layer_height:
                    self.theta[k] = theta0
                else:
                    self.theta[k] = theta0 + lapse_rate * (z - mixed_layer_height)

    def initialize_u(self, u_geo: float, z0: float = 0.1,
                     bl_height: float = None):
        """Initialize u with a log wind profile below BL height.

        u(z) = u_geo * ln(z/z0) / ln(h/z0)   for z < h
        u(z) = u_geo                           for z >= h
        """
        if self.u is None:
            return

        if bl_height is None:
            bl_height = self.grid.Lz * 0.5

        log_denom = np.log(bl_height / z0)

        if self.grid.dim >= 3:
            for k in range(self.grid.nz):
                z = self.grid.z_center[k]
                if z <= z0:
                    self.u[:, :, k] = 0.0
                elif z < bl_height:
                    self.u[:, :, k] = u_geo * np.log(z / z0) / log_denom
                else:
                    self.u[:, :, k] = u_geo
        else:
            for k in range(self.grid.nz):
                z = self.grid.z_center[k]
                if z <= z0:
                    self.u[:, k] = 0.0
                elif z < bl_height:
                    self.u[:, k] = u_geo * np.log(z / z0) / log_denom
                else:
                    self.u[:, k] = u_geo

    def initialize_v(self, v_geo: float):
        """Initialize v uniformly (3D only)."""
        if self.v is None:
            return
        self.v[:] = v_geo
