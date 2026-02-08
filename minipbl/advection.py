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
    # u at z-face k: average u from levels k-1 and k
    u_at_zface = np.zeros((nx, nz + 1))
    u_at_zface[:, 1:nz] = 0.5 * (u[:, :nz-1] + u[:, 1:nz])
    # w at x-face: average from cell i-1 and cell i (same as for u-momentum)
    # But here we need w at (x-face[i], z-face[k])
    # w at x_face[i] = average of w[i-1] and w[i]
    # Actually, u_at_zface is at (x-face[i], z-face[k])
    # and we need w there too — w at x-face = 0.5*(w[i-1] + w[i])
    # flux = u_at_zface * w_at_xface at (x-face[i], z-face[k])
    w_at_xface = 0.5 * (w + np.roll(w, 1, axis=0))
    flux_x = u_at_zface * w_at_xface  # at (x-face, z-face)

    # Divergence at (x-center[i], z-face[k]):
    # (flux at x-face[i+1]) - (flux at x-face[i]) / dx
    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- z-flux of w-momentum at cell centers (z-center) ---
    w_at_zcenter = 0.5 * (w[:, :-1] + w[:, 1:])  # (nx, nz)
    flux_z_center = w_at_zcenter * w_at_zcenter  # at cell centers

    # Divergence at z-face k (between cell k-1 and cell k):
    # (flux at cell center k) - (flux at cell center k-1) / dz
    # For interior z-faces k=1..nz-1:
    tendency[:, 1:nz] = -(dflux_x[:, 1:nz]
                           + (flux_z_center[:, 1:nz] - flux_z_center[:, :nz-1]) / dz)

    return tendency
