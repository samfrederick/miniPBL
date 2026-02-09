"""Staggered grid for the 1D vertical column (extensible to 2D/3D)."""

import numpy as np


class Grid:
    """Arakawa C-grid: scalars at cell centers, fluxes at cell faces.

    For nz cells over domain [0, Lz]:
      - z_face has nz+1 points: 0, dz, 2*dz, ..., Lz
      - z_center has nz points at midpoints: dz/2, 3*dz/2, ...
      - dz_center[k] = z_face[k+1] - z_face[k]  (cell thickness, nz values)
      - dz_face[k] = z_center[k] - z_center[k-1]  (distance between centers, nz+1 values)

    When stretch_factor > 1.0, the first nz_uniform cells have uniform spacing
    and cells above grow geometrically with ratio stretch_factor, normalized so
    the total depth equals Lz.

    For 2D (dim >= 2) with nx cells over periodic domain [0, Lx):
      - x_face has nx+1 points: 0, dx, ..., Lx  (face nx wraps to face 0)
      - x_center has nx points at midpoints

    For 3D (dim >= 3) with ny cells over periodic domain [0, Ly):
      - y_face has ny+1 points: 0, dy, ..., Ly  (face ny wraps to face 0)
      - y_center has ny points at midpoints
    """

    def __init__(self, nz: int, Lz: float, dim: int = 1,
                 nx: int = 1, Lx: float = 6000.0,
                 ny: int = 1, Ly: float = 6000.0,
                 stretch_factor: float = 1.0, nz_uniform: int = 0):
        self.nz = nz
        self.Lz = Lz
        self.dim = dim

        # Build vertical grid
        self.z_face = self._build_z_faces(nz, Lz, stretch_factor, nz_uniform)
        self.z_center = 0.5 * (self.z_face[:-1] + self.z_face[1:])

        # Cell thickness at centers: dz_center[k] = z_face[k+1] - z_face[k], shape (nz,)
        self.dz_center = np.diff(self.z_face)

        # Distance between cell centers at faces, shape (nz+1,)
        # Interior: dz_face[k] = z_center[k] - z_center[k-1] for k=1..nz-1
        # Boundaries: half-cell widths
        self.dz_face = np.zeros(nz + 1)
        self.dz_face[1:nz] = self.z_center[1:nz] - self.z_center[:nz-1]
        self.dz_face[0] = self.z_center[0] - self.z_face[0]      # half-cell at bottom
        self.dz_face[nz] = self.z_face[nz] - self.z_center[nz-1]  # half-cell at top

        # Scalar dz for backward compatibility (uniform grid value)
        self.dz = Lz / nz

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

    @staticmethod
    def _build_z_faces(nz, Lz, stretch_factor, nz_uniform):
        """Build z_face array with optional geometric stretching.

        For stretch_factor == 1.0: uniform spacing dz = Lz / nz.
        For stretch_factor > 1.0: first nz_uniform cells are uniform,
        then geometric growth dz[k] = dz[k-1] * stretch_factor, all
        normalized so that the total depth sums to Lz.
        """
        if stretch_factor <= 1.0 or nz_uniform >= nz:
            return np.linspace(0.0, Lz, nz + 1)

        # Number of stretched cells
        n_stretch = nz - nz_uniform

        # Build un-normalized cell thicknesses
        dz_raw = np.ones(nz)
        # Geometric growth above uniform region
        for k in range(nz_uniform, nz):
            dz_raw[k] = dz_raw[nz_uniform - 1 if nz_uniform > 0 else 0] * \
                         stretch_factor ** (k - nz_uniform + (1 if nz_uniform > 0 else 0))

        # For nz_uniform == 0: all cells geometric starting from base dz=1
        if nz_uniform == 0:
            for k in range(nz):
                dz_raw[k] = stretch_factor ** k

        # Normalize so sum = Lz
        dz_raw *= Lz / np.sum(dz_raw)

        z_face = np.zeros(nz + 1)
        for k in range(nz):
            z_face[k + 1] = z_face[k] + dz_raw[k]
        z_face[-1] = Lz  # ensure exact top boundary

        return z_face
