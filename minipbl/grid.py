"""Staggered grid for the 1D vertical column (extensible to 2D/3D)."""

import numpy as np


class Grid:
    """1D staggered grid: scalars at cell centers, fluxes at cell faces.

    For nz cells over domain [0, Lz]:
      - z_face has nz+1 points: 0, dz, 2*dz, ..., Lz
      - z_center has nz points at midpoints: dz/2, 3*dz/2, ...
    """

    def __init__(self, nz: int, Lz: float, dim: int = 1):
        self.nz = nz
        self.Lz = Lz
        self.dim = dim
        self.dz = Lz / nz

        self.z_face = np.linspace(0.0, Lz, nz + 1)
        self.z_center = 0.5 * (self.z_face[:-1] + self.z_face[1:])
