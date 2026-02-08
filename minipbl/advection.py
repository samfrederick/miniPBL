"""Advection operators: 1D stub and 2D flux-form centered advection on C-grid."""

import numpy as np

from .grid import Grid


# ---------------------------------------------------------------------------
# 1D stub (backward-compatible)
# ---------------------------------------------------------------------------

def compute_advection_tendency(theta: np.ndarray, w: np.ndarray,
                               grid: Grid) -> np.ndarray:
    """Compute advection tendency -w * d(theta)/dz.

    Currently a stub that returns zero tendency (no mean vertical velocity
    in a pure convective boundary layer column). Interface is defined for
    future 2D/3D extension.

    Parameters
    ----------
    theta : array of shape (nz,) — scalar at cell centers
    w : array of shape (nz+1,) — vertical velocity at cell faces
    grid : Grid instance
    """
    return np.zeros(grid.nz)


# ---------------------------------------------------------------------------
# 2D flux-form centered advection on Arakawa C-grid
# ---------------------------------------------------------------------------
# Convention:
#   - Cell-center quantities: theta(nx, nz) at (x_center[i], z_center[k])
#   - u(nx, nz) at left x-face of cell i:  x_face[i], z_center[k]
#   - w(nx, nz+1) at bottom z-face of cell k:  x_center[i], z_face[k]
#
# So for cell (i, k):
#   left  face velocity = u[i, k]
#   right face velocity = u[i+1, k]  (periodic: u[(i+1)%nx, k])
#   bottom face velocity = w[i, k]
#   top    face velocity = w[i, k+1]
#
# Divergence at cell center (i, k):
#   (u[i+1,k] - u[i,k]) / dx + (w[i,k+1] - w[i,k]) / dz
#
# This matches pressure.py's project_velocity convention.
# ---------------------------------------------------------------------------


def compute_advection_tendency_theta(theta, u, w, grid):
    """Flux-form advection tendency for theta (cell-center scalar).

    d(theta)/dt = -d(u*theta)/dx - d(w*theta)/dz

    Returns (nx, nz).
    """
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz

    # --- x-flux at left x-face of cell i: u[i] * theta_interp ---
    # theta at left x-face[i] = average of cell i-1 and cell i
    theta_at_xface = 0.5 * (theta + np.roll(theta, 1, axis=0))  # (nx, nz)
    flux_x = u * theta_at_xface  # flux at left face of cell i

    # Divergence at cell center i: (flux_right - flux_left) / dx
    # flux_right of cell i = flux at left face of cell i+1 = flux_x[(i+1)%nx]
    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- z-flux at z-faces: w * theta_interp ---
    flux_z = np.zeros((nx, nz + 1))
    flux_z[:, 1:nz] = w[:, 1:nz] * 0.5 * (theta[:, :nz-1] + theta[:, 1:nz])
    # flux_z at boundaries = 0 (rigid lid: w=0)

    # Divergence at cell center k: (flux_z[k+1] - flux_z[k]) / dz
    dflux_z = (flux_z[:, 1:] - flux_z[:, :-1]) / dz

    return -(dflux_x + dflux_z)


def compute_advection_tendency_u(u, w, grid):
    """Flux-form advection tendency for u (at left x-face).

    d(u)/dt = -d(uu)/dx - d(wu)/dz

    Returns (nx, nz).
    """
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz

    # --- x-flux of u-momentum at cell centers ---
    # u at cell center i = 0.5*(u[i] + u[i+1])
    u_right = np.roll(u, -1, axis=0)
    u_center = 0.5 * (u + u_right)  # u interpolated to cell center i
    flux_x_center = u_center * u_center  # momentum flux at cell center i

    # Divergence at x-face i (between cell i-1 and cell i):
    # (flux at cell center i) - (flux at cell center i-1) / dx
    dflux_x = (flux_x_center - np.roll(flux_x_center, 1, axis=0)) / dx

    # --- z-flux of u-momentum at (x-face, z-face) ---
    # w at x-face i = average of w in cell i-1 and cell i
    w_at_xface = 0.5 * (w + np.roll(w, 1, axis=0))  # (nx, nz+1)
    # u at z-face: average vertical neighbours
    flux_z = np.zeros((nx, nz + 1))
    flux_z[:, 1:nz] = w_at_xface[:, 1:nz] * 0.5 * (u[:, :nz-1] + u[:, 1:nz])
    # top and bottom: w=0 so flux=0

    dflux_z = (flux_z[:, 1:] - flux_z[:, :-1]) / dz

    return -(dflux_x + dflux_z)


def compute_advection_tendency_w(u, w, grid):
    """Flux-form advection tendency for w (at z-faces).

    d(w)/dt = -d(uw)/dx - d(ww)/dz

    Returns (nx, nz+1).  Boundary values (k=0, k=nz) stay zero.
    """
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz

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
                           + (flux_z_center[:, 1:nz] - flux_z_center[:, :nz-1]) / dz)

    return tendency


# ---------------------------------------------------------------------------
# 3D flux-form centered advection on Arakawa C-grid
# ---------------------------------------------------------------------------
# Convention:
#   - Cell-center quantities: theta(nx, ny, nz) at (x_center[i], y_center[j], z_center[k])
#   - u(nx, ny, nz) at left x-face of cell i
#   - v(nx, ny, nz) at front y-face of cell j
#   - w(nx, ny, nz+1) at bottom z-face of cell k
# ---------------------------------------------------------------------------


def compute_advection_tendency_theta_3d(theta, u, v, w, grid):
    """Flux-form advection tendency for theta in 3D.

    Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

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

    dflux_z = (flux_z[:, :, 1:] - flux_z[:, :, :-1]) / dz

    return -(dflux_x + dflux_y + dflux_z)


def compute_advection_tendency_u_3d(u, v, w, grid):
    """Flux-form advection tendency for u in 3D.

    Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    # --- x-flux: uu at cell centers ---
    u_right = np.roll(u, -1, axis=0)
    u_center = 0.5 * (u + u_right)
    flux_x_center = u_center * u_center
    dflux_x = (flux_x_center - np.roll(flux_x_center, 1, axis=0)) / dx

    # --- y-flux: v at (x-face, y-face) * u at y-face ---
    # v at x-face: average of v[i-1,j] and v[i,j]
    v_at_xface = 0.5 * (v + np.roll(v, 1, axis=0))
    # u at y-face: average of u[i,j-1] and u[i,j]
    u_at_yface = 0.5 * (u + np.roll(u, 1, axis=1))
    flux_y = v_at_xface * u_at_yface  # at (x-face, y-face)
    dflux_y = (np.roll(flux_y, -1, axis=1) - flux_y) / dy

    # --- z-flux: w at (x-face, z-face) * u at z-face ---
    w_at_xface = 0.5 * (w + np.roll(w, 1, axis=0))
    flux_z = np.zeros((nx, ny, nz + 1))
    flux_z[:, :, 1:nz] = w_at_xface[:, :, 1:nz] * 0.5 * (u[:, :, :nz-1] + u[:, :, 1:nz])
    dflux_z = (flux_z[:, :, 1:] - flux_z[:, :, :-1]) / dz

    return -(dflux_x + dflux_y + dflux_z)


def compute_advection_tendency_v_3d(u, v, w, grid):
    """Flux-form advection tendency for v in 3D.

    Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    # --- x-flux: u at (x-face, y-face) * v at x-face ---
    u_at_yface = 0.5 * (u + np.roll(u, 1, axis=1))
    v_at_xface = 0.5 * (v + np.roll(v, 1, axis=0))
    flux_x = u_at_yface * v_at_xface  # at (x-face, y-face)
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
    dflux_z = (flux_z[:, :, 1:] - flux_z[:, :, :-1]) / dz

    return -(dflux_x + dflux_y + dflux_z)


def compute_advection_tendency_w_3d(u, v, w, grid):
    """Flux-form advection tendency for w in 3D.

    Returns (nx, ny, nz+1).  Boundary values (k=0, k=nz) stay zero.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

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
                              + (flux_z_center[:, :, 1:nz] - flux_z_center[:, :, :nz-1]) / dz)

    return tendency
