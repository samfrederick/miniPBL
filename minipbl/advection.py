"""Advection operators: 1D stub and 2D/3D flux-form centered advection on C-grid.

Supports variable vertical spacing via grid.dz_center and grid.dz_face arrays.
"""

import numpy as np

from .grid import Grid


# ---------------------------------------------------------------------------
# 1D stub (backward-compatible)
# ---------------------------------------------------------------------------

def compute_advection_tendency(theta: np.ndarray, w: np.ndarray,
                               grid: Grid) -> np.ndarray:
    """Compute advection tendency -w * d(theta)/dz.

    Currently a stub that returns zero tendency (no mean vertical velocity
    in a pure convective boundary layer column).
    """
    return np.zeros(grid.nz)


# ---------------------------------------------------------------------------
# 2D flux-form centered advection on Arakawa C-grid
# ---------------------------------------------------------------------------

def compute_advection_tendency_theta(theta, u, w, grid):
    """Flux-form advection tendency for theta (cell-center scalar).

    d(theta)/dt = -d(u*theta)/dx - d(w*theta)/dz

    Returns (nx, nz).
    """
    nx, nz = grid.nx, grid.nz
    dx = grid.dx
    dz_c = grid.dz_center  # (nz,)

    # --- x-flux at left x-face of cell i: u[i] * theta_interp ---
    theta_at_xface = 0.5 * (theta + np.roll(theta, 1, axis=0))
    flux_x = u * theta_at_xface
    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- z-flux at z-faces: w * theta_interp ---
    flux_z = np.zeros((nx, nz + 1))
    flux_z[:, 1:nz] = w[:, 1:nz] * 0.5 * (theta[:, :nz-1] + theta[:, 1:nz])

    dflux_z = (flux_z[:, 1:] - flux_z[:, :-1]) / dz_c[np.newaxis, :]

    return -(dflux_x + dflux_z)


def compute_advection_tendency_u(u, w, grid):
    """Flux-form advection tendency for u (at left x-face).

    d(u)/dt = -d(uu)/dx - d(wu)/dz

    Returns (nx, nz).
    """
    nx, nz = grid.nx, grid.nz
    dx = grid.dx
    dz_c = grid.dz_center

    # --- x-flux of u-momentum at cell centers ---
    u_right = np.roll(u, -1, axis=0)
    u_center = 0.5 * (u + u_right)
    flux_x_center = u_center * u_center
    dflux_x = (flux_x_center - np.roll(flux_x_center, 1, axis=0)) / dx

    # --- z-flux of u-momentum at (x-face, z-face) ---
    w_at_xface = 0.5 * (w + np.roll(w, 1, axis=0))
    flux_z = np.zeros((nx, nz + 1))
    flux_z[:, 1:nz] = w_at_xface[:, 1:nz] * 0.5 * (u[:, :nz-1] + u[:, 1:nz])

    dflux_z = (flux_z[:, 1:] - flux_z[:, :-1]) / dz_c[np.newaxis, :]

    return -(dflux_x + dflux_z)


def compute_advection_tendency_w(u, w, grid):
    """Flux-form advection tendency for w (at z-faces).

    d(w)/dt = -d(uw)/dx - d(ww)/dz

    Returns (nx, nz+1).  Boundary values (k=0, k=nz) stay zero.
    """
    nx, nz = grid.nx, grid.nz
    dx = grid.dx
    dz_c = grid.dz_center  # (nz,)
    dz_f = grid.dz_face    # (nz+1,)

    tendency = np.zeros((nx, nz + 1))

    # --- x-flux of w-momentum at (x-face, z-face) ---
    u_at_zface = np.zeros((nx, nz + 1))
    u_at_zface[:, 1:nz] = 0.5 * (u[:, :nz-1] + u[:, 1:nz])
    w_at_xface = 0.5 * (w + np.roll(w, 1, axis=0))
    flux_x = u_at_zface * w_at_xface

    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- z-flux of w-momentum at cell centers (z-center) ---
    w_at_zcenter = 0.5 * (w[:, :-1] + w[:, 1:])  # (nx, nz)
    flux_z_center = w_at_zcenter * w_at_zcenter

    tendency[:, 1:nz] = -(dflux_x[:, 1:nz]
                           + (flux_z_center[:, 1:nz] - flux_z_center[:, :nz-1]) / dz_f[1:nz])

    return tendency


# ---------------------------------------------------------------------------
# 3D flux-form centered advection on Arakawa C-grid
# ---------------------------------------------------------------------------

def compute_advection_tendency_theta_3d(theta, u, v, w, grid):
    """Flux-form advection tendency for theta in 3D.

    Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    dz_c = grid.dz_center

    # --- x-flux ---
    theta_at_xface = 0.5 * (theta + np.roll(theta, 1, axis=0))
    flux_x = u * theta_at_xface
    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- y-flux ---
    theta_at_yface = 0.5 * (theta + np.roll(theta, 1, axis=1))
    flux_y = v * theta_at_yface
    dflux_y = (np.roll(flux_y, -1, axis=1) - flux_y) / dy

    # --- z-flux ---
    flux_z = np.zeros((nx, ny, nz + 1))
    flux_z[:, :, 1:nz] = w[:, :, 1:nz] * 0.5 * (theta[:, :, :nz-1] + theta[:, :, 1:nz])

    dflux_z = (flux_z[:, :, 1:] - flux_z[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]

    return -(dflux_x + dflux_y + dflux_z)


def compute_advection_tendency_u_3d(u, v, w, grid):
    """Flux-form advection tendency for u in 3D.

    Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    dz_c = grid.dz_center

    # --- x-flux: uu at cell centers ---
    u_right = np.roll(u, -1, axis=0)
    u_center = 0.5 * (u + u_right)
    flux_x_center = u_center * u_center
    dflux_x = (flux_x_center - np.roll(flux_x_center, 1, axis=0)) / dx

    # --- y-flux: v at (x-face, y-face) * u at y-face ---
    v_at_xface = 0.5 * (v + np.roll(v, 1, axis=0))
    u_at_yface = 0.5 * (u + np.roll(u, 1, axis=1))
    flux_y = v_at_xface * u_at_yface
    dflux_y = (np.roll(flux_y, -1, axis=1) - flux_y) / dy

    # --- z-flux: w at (x-face, z-face) * u at z-face ---
    w_at_xface = 0.5 * (w + np.roll(w, 1, axis=0))
    flux_z = np.zeros((nx, ny, nz + 1))
    flux_z[:, :, 1:nz] = w_at_xface[:, :, 1:nz] * 0.5 * (u[:, :, :nz-1] + u[:, :, 1:nz])
    dflux_z = (flux_z[:, :, 1:] - flux_z[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]

    return -(dflux_x + dflux_y + dflux_z)


def compute_advection_tendency_v_3d(u, v, w, grid):
    """Flux-form advection tendency for v in 3D.

    Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    dz_c = grid.dz_center

    # --- x-flux: u at (x-face, y-face) * v at x-face ---
    u_at_yface = 0.5 * (u + np.roll(u, 1, axis=1))
    v_at_xface = 0.5 * (v + np.roll(v, 1, axis=0))
    flux_x = u_at_yface * v_at_xface
    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- y-flux: vv at cell centers ---
    v_front = np.roll(v, -1, axis=1)
    v_center = 0.5 * (v + v_front)
    flux_y_center = v_center * v_center
    dflux_y = (flux_y_center - np.roll(flux_y_center, 1, axis=1)) / dy

    # --- z-flux: w at (y-face, z-face) * v at z-face ---
    w_at_yface = 0.5 * (w + np.roll(w, 1, axis=1))
    flux_z = np.zeros((nx, ny, nz + 1))
    flux_z[:, :, 1:nz] = w_at_yface[:, :, 1:nz] * 0.5 * (v[:, :, :nz-1] + v[:, :, 1:nz])
    dflux_z = (flux_z[:, :, 1:] - flux_z[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]

    return -(dflux_x + dflux_y + dflux_z)


def compute_advection_tendency_w_3d(u, v, w, grid):
    """Flux-form advection tendency for w in 3D.

    Returns (nx, ny, nz+1).  Boundary values (k=0, k=nz) stay zero.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    dz_c = grid.dz_center
    dz_f = grid.dz_face

    tendency = np.zeros((nx, ny, nz + 1))

    # --- x-flux ---
    u_at_zface = np.zeros((nx, ny, nz + 1))
    u_at_zface[:, :, 1:nz] = 0.5 * (u[:, :, :nz-1] + u[:, :, 1:nz])
    w_at_xface = 0.5 * (w + np.roll(w, 1, axis=0))
    flux_x = u_at_zface * w_at_xface
    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- y-flux ---
    v_at_zface = np.zeros((nx, ny, nz + 1))
    v_at_zface[:, :, 1:nz] = 0.5 * (v[:, :, :nz-1] + v[:, :, 1:nz])
    w_at_yface = 0.5 * (w + np.roll(w, 1, axis=1))
    flux_y = v_at_zface * w_at_yface
    dflux_y = (np.roll(flux_y, -1, axis=1) - flux_y) / dy

    # --- z-flux ---
    w_at_zcenter = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    flux_z_center = w_at_zcenter * w_at_zcenter

    tendency[:, :, 1:nz] = -(dflux_x[:, :, 1:nz]
                              + dflux_y[:, :, 1:nz]
                              + (flux_z_center[:, :, 1:nz] - flux_z_center[:, :, :nz-1]) / dz_f[1:nz])

    return tendency
