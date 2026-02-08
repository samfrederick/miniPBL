"""Staggered grid for the 1D vertical column (extensible to 2D/3D)."""

import numpy as np


class Grid:
    """Arakawa C-grid: scalars at cell centers, fluxes at cell faces.

    For nz cells over domain [0, Lz]:
      - z_face has nz+1 points: 0, dz, 2*dz, ..., Lz
      - z_center has nz points at midpoints: dz/2, 3*dz/2, ...

    For 2D (dim >= 2) with nx cells over periodic domain [0, Lx):
      - x_face has nx+1 points: 0, dx, ..., Lx  (face nx wraps to face 0)
      - x_center has nx points at midpoints

    For 3D (dim >= 3) with ny cells over periodic domain [0, Ly):
      - y_face has ny+1 points: 0, dy, ..., Ly  (face ny wraps to face 0)
      - y_center has ny points at midpoints
    """

    def __init__(self, nz: int, Lz: float, dim: int = 1,
                 nx: int = 1, Lx: float = 6000.0,
                 ny: int = 1, Ly: float = 6000.0):
        self.nz = nz
        self.Lz = Lz
        self.dim = dim
        self.dz = Lz / nz

        self.z_face = np.linspace(0.0, Lz, nz + 1)
        self.z_center = 0.5 * (self.z_face[:-1] + self.z_face[1:])

        # Horizontal grid — x
        self.nx = nx
        self.Lx = Lx
        self.dx = Lx / nx if nx > 1 else Lx
        self.x_face = np.linspace(0.0, Lx, nx + 1)
        self.x_center = 0.5 * (self.x_face[:-1] + self.x_face[1:])

        # Horizontal grid — y
        self.ny = ny
        self.Ly = Ly
        self.dy = Ly / ny if ny > 1 else Ly
        self.y_face = np.linspace(0.0, Ly, ny + 1)
        self.y_center = 0.5 * (self.y_face[:-1] + self.y_face[1:])
