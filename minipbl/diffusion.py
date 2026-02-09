"""Vertical diffusion operator using second-order centered differences.

Supports variable vertical spacing via grid.dz_center and grid.dz_face arrays.
"""

import numpy as np

from .grid import Grid


# ---------------------------------------------------------------------------
# 1D (variable dz)
# ---------------------------------------------------------------------------

def compute_diffusion_tendency(theta: np.ndarray, K_h: np.ndarray,
                               grid: Grid,
                               surface_heat_flux: float) -> np.ndarray:
    """Compute d(theta)/dt from vertical turbulent diffusion.

    Uses the flux form: dtheta/dt = -d(w'theta')/dz
    where w'theta' = -K_h * d(theta)/dz  (positive upward)

    w'theta' is defined on cell faces (nz+1 points).
    theta is at cell centers (nz points).
    Returns tendency at cell centers (nz points).
    """
    nz = grid.nz
    dz_c = grid.dz_center  # (nz,)
    dz_f = grid.dz_face    # (nz+1,)

    # Compute kinematic heat flux w'theta' = -K_h * dtheta/dz on interior faces
    wtheta = np.zeros(nz + 1)
    for i in range(1, nz):
        wtheta[i] = -K_h[i] * (theta[i] - theta[i - 1]) / dz_f[i]

    # Boundary conditions
    wtheta[0] = surface_heat_flux    # prescribed surface kinematic heat flux
    wtheta[-1] = 0.0                 # zero flux at top (insulating lid)

    # Tendency: dtheta/dt = -d(w'theta')/dz
    tendency = np.zeros(nz)
    for k in range(nz):
        tendency[k] = -(wtheta[k + 1] - wtheta[k]) / dz_c[k]

    return tendency


# ---------------------------------------------------------------------------
# 2D diffusion tendencies
# ---------------------------------------------------------------------------

def _horizontal_laplacian_center(f, dx):
    """Compute d2f/dx2 for a cell-center field (nx, nz), periodic in x."""
    return (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / (dx * dx)


def _horizontal_laplacian_zface(f, dx):
    """Compute d2f/dx2 for a z-face field (nx, nz+1), periodic in x."""
    return (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / (dx * dx)


def compute_diffusion_tendency_theta(theta, K_h, grid, surface_heat_flux,
                                     K_horiz=0.0):
    """Vertical + horizontal turbulent diffusion of theta for 2D fields.

    theta: (nx, nz), K_h: (nx, nz+1)
    Returns (nx, nz).
    """
    nx, nz = grid.nx, grid.nz
    dz_c = grid.dz_center  # (nz,)
    dz_f = grid.dz_face    # (nz+1,)

    # Vertical: heat flux on z-faces
    wtheta = np.zeros((nx, nz + 1))
    wtheta[:, 1:nz] = -K_h[:, 1:nz] * (theta[:, 1:nz] - theta[:, :nz-1]) / dz_f[1:nz]
    wtheta[:, 0] = surface_heat_flux
    wtheta[:, -1] = 0.0

    tendency = -(wtheta[:, 1:] - wtheta[:, :-1]) / dz_c[np.newaxis, :]

    # Horizontal diffusion (periodic)
    if K_horiz > 0:
        tendency += K_horiz * _horizontal_laplacian_center(theta, grid.dx)

    return tendency


def compute_diffusion_tendency_u(u, K_m, grid, K_horiz=0.0):
    """Vertical + horizontal turbulent diffusion of u for 2D fields.

    u: (nx, nz), K_m: (nx, nz+1)
    Returns (nx, nz).

    BCs: du/dz = 0 at top, no-slip approximation at surface
    (surface stress = -K_m * u / dz_face[0]).
    """
    nx, nz = grid.nx, grid.nz
    dz_c = grid.dz_center  # (nz,)
    dz_f = grid.dz_face    # (nz+1,)

    # Vertical: momentum flux on z-faces
    tau = np.zeros((nx, nz + 1))
    tau[:, 1:nz] = -K_m[:, 1:nz] * (u[:, 1:nz] - u[:, :nz-1]) / dz_f[1:nz]
    # Surface: no-slip => u=0 at z=0, so du/dz ~ u[0]/dz_face[0]
    tau[:, 0] = -K_m[:, 0] * u[:, 0] / dz_f[0]
    tau[:, -1] = 0.0

    tendency = -(tau[:, 1:] - tau[:, :-1]) / dz_c[np.newaxis, :]

    # Horizontal diffusion (periodic)
    if K_horiz > 0:
        tendency += K_horiz * _horizontal_laplacian_center(u, grid.dx)

    return tendency


def compute_diffusion_tendency_w(w, K_m, grid, K_horiz=0.0):
    """Vertical + horizontal turbulent diffusion of w for 2D fields.

    w: (nx, nz+1), K_m: (nx, nz+1)
    Returns (nx, nz+1).  Boundary values stay zero (rigid lid).
    """
    nx, nz = grid.nx, grid.nz
    dz_c = grid.dz_center  # (nz,)
    dz_f = grid.dz_face    # (nz+1,)

    tendency = np.zeros((nx, nz + 1))

    # Vertical: K_m at cell centers (interpolate from faces)
    K_m_center = 0.5 * (K_m[:, :-1] + K_m[:, 1:])  # (nx, nz)
    dwdz = (w[:, 1:] - w[:, :-1]) / dz_c[np.newaxis, :]  # (nx, nz)
    flux = -K_m_center * dwdz
    tendency[:, 1:nz] = -(flux[:, 1:nz] - flux[:, :nz-1]) / dz_f[1:nz]

    # Horizontal diffusion (periodic), interior faces only
    if K_horiz > 0:
        tendency[:, 1:nz] += K_horiz * _horizontal_laplacian_zface(w, grid.dx)[:, 1:nz]

    return tendency


# ---------------------------------------------------------------------------
# 3D diffusion tendencies
# ---------------------------------------------------------------------------

def _horizontal_laplacian_center_3d(f, dx, dy):
    """Laplacian d2f/dx2 + d2f/dy2 for a cell-center field (nx, ny, nz), periodic in x and y."""
    return ((np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / (dx * dx)
            + (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / (dy * dy))


def _horizontal_laplacian_zface_3d(f, dx, dy):
    """Laplacian d2f/dx2 + d2f/dy2 for a z-face field (nx, ny, nz+1), periodic in x and y."""
    return ((np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / (dx * dx)
            + (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / (dy * dy))


def compute_diffusion_tendency_theta_3d(theta, K_h, grid, surface_heat_flux,
                                        K_horiz=0.0):
    """Vertical + horizontal diffusion of theta for 3D fields.

    theta: (nx, ny, nz), K_h: (nx, ny, nz+1)
    Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dz_c = grid.dz_center  # (nz,)
    dz_f = grid.dz_face    # (nz+1,)

    wtheta = np.zeros((nx, ny, nz + 1))
    wtheta[:, :, 1:nz] = -K_h[:, :, 1:nz] * (theta[:, :, 1:nz] - theta[:, :, :nz-1]) / dz_f[1:nz]
    wtheta[:, :, 0] = surface_heat_flux
    wtheta[:, :, -1] = 0.0

    tendency = -(wtheta[:, :, 1:] - wtheta[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]

    if K_horiz > 0:
        tendency += K_horiz * _horizontal_laplacian_center_3d(theta, grid.dx, grid.dy)

    return tendency


def compute_diffusion_tendency_u_3d(u, K_m, grid, K_horiz=0.0):
    """Vertical + horizontal diffusion of u for 3D fields.

    u: (nx, ny, nz), K_m: (nx, ny, nz+1)
    Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dz_c = grid.dz_center
    dz_f = grid.dz_face

    tau = np.zeros((nx, ny, nz + 1))
    tau[:, :, 1:nz] = -K_m[:, :, 1:nz] * (u[:, :, 1:nz] - u[:, :, :nz-1]) / dz_f[1:nz]
    tau[:, :, 0] = -K_m[:, :, 0] * u[:, :, 0] / dz_f[0]
    tau[:, :, -1] = 0.0

    tendency = -(tau[:, :, 1:] - tau[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]

    if K_horiz > 0:
        tendency += K_horiz * _horizontal_laplacian_center_3d(u, grid.dx, grid.dy)

    return tendency


def compute_diffusion_tendency_v_3d(v, K_m, grid, K_horiz=0.0):
    """Vertical + horizontal diffusion of v for 3D fields.

    v: (nx, ny, nz), K_m: (nx, ny, nz+1)
    Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dz_c = grid.dz_center
    dz_f = grid.dz_face

    tau = np.zeros((nx, ny, nz + 1))
    tau[:, :, 1:nz] = -K_m[:, :, 1:nz] * (v[:, :, 1:nz] - v[:, :, :nz-1]) / dz_f[1:nz]
    tau[:, :, 0] = -K_m[:, :, 0] * v[:, :, 0] / dz_f[0]
    tau[:, :, -1] = 0.0

    tendency = -(tau[:, :, 1:] - tau[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]

    if K_horiz > 0:
        tendency += K_horiz * _horizontal_laplacian_center_3d(v, grid.dx, grid.dy)

    return tendency


def compute_diffusion_tendency_w_3d(w, K_m, grid, K_horiz=0.0):
    """Vertical + horizontal diffusion of w for 3D fields.

    w: (nx, ny, nz+1), K_m: (nx, ny, nz+1)
    Returns (nx, ny, nz+1).  Boundary values stay zero.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dz_c = grid.dz_center
    dz_f = grid.dz_face

    tendency = np.zeros((nx, ny, nz + 1))

    K_m_center = 0.5 * (K_m[:, :, :-1] + K_m[:, :, 1:])
    dwdz = (w[:, :, 1:] - w[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]
    flux = -K_m_center * dwdz
    tendency[:, :, 1:nz] = -(flux[:, :, 1:nz] - flux[:, :, :nz-1]) / dz_f[1:nz]

    if K_horiz > 0:
        tendency[:, :, 1:nz] += K_horiz * _horizontal_laplacian_zface_3d(
            w, grid.dx, grid.dy)[:, :, 1:nz]

    return tendency
