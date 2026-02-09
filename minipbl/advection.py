"""Advection operators: 1D stub, 2D/3D flux-form centered, and 5th-order upwind.

Supports variable vertical spacing via grid.dz_center and grid.dz_face arrays.
The 5th-order upwind scheme (Wicker & Skamarock 2002) is applied to horizontal
(periodic) fluxes only; vertical fluxes remain 2nd-order centered.
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


# ---------------------------------------------------------------------------
# 5th-order upwind-biased advection (Wicker & Skamarock 2002)
# ---------------------------------------------------------------------------

def _upwind5_interp_periodic(phi, vel_at_face, axis):
    """5th-order upwind interpolation of phi to cell faces along a periodic axis.

    phi: cell-centered field
    vel_at_face: velocity at the left face of each cell (same shape as phi)
    axis: 0 (x) or 1 (y)

    Returns phi interpolated to the left face of each cell, shape = phi.shape.

    The face between cell (i-1) and cell (i) uses velocity vel_at_face[i].
    For vel >= 0 (flow from i-1 toward i), the stencil is biased left:
        (2*phi[i-3] - 13*phi[i-2] + 47*phi[i-1] + 27*phi[i] - 3*phi[i+1]) / 60
    For vel < 0 (flow from i toward i-1), the stencil is biased right:
        (-3*phi[i-2] + 27*phi[i-1] + 47*phi[i] - 13*phi[i+1] + 2*phi[i+2]) / 60
    """
    # Pre-compute shifted arrays (periodic via np.roll)
    pm3 = np.roll(phi, 3, axis=axis)   # phi[i-3]
    pm2 = np.roll(phi, 2, axis=axis)   # phi[i-2]
    pm1 = np.roll(phi, 1, axis=axis)   # phi[i-1]
    pp0 = phi                           # phi[i]
    pp1 = np.roll(phi, -1, axis=axis)  # phi[i+1]
    pp2 = np.roll(phi, -2, axis=axis)  # phi[i+2]

    phi_pos = (2.0*pm3 - 13.0*pm2 + 47.0*pm1 + 27.0*pp0 - 3.0*pp1) / 60.0
    phi_neg = (-3.0*pm2 + 27.0*pm1 + 47.0*pp0 - 13.0*pp1 + 2.0*pp2) / 60.0

    return np.where(vel_at_face >= 0, phi_pos, phi_neg)


# ---------------------------------------------------------------------------
# 2D upwind5 advection
# ---------------------------------------------------------------------------

def compute_advection_tendency_theta_upwind5(theta, u, w, grid):
    """Flux-form advection tendency for theta using 5th-order upwind in x.

    Vertical fluxes remain 2nd-order centered.  Returns (nx, nz).
    """
    nx, nz = grid.nx, grid.nz
    dx = grid.dx
    dz_c = grid.dz_center

    # --- x-flux: u * theta_upwind5 at left x-face of cell i ---
    theta_at_xface = _upwind5_interp_periodic(theta, u, axis=0)
    flux_x = u * theta_at_xface
    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- z-flux (2nd-order centered, same as original) ---
    flux_z = np.zeros((nx, nz + 1))
    flux_z[:, 1:nz] = w[:, 1:nz] * 0.5 * (theta[:, :nz-1] + theta[:, 1:nz])
    dflux_z = (flux_z[:, 1:] - flux_z[:, :-1]) / dz_c[np.newaxis, :]

    return -(dflux_x + dflux_z)


def compute_advection_tendency_u_upwind5(u, w, grid):
    """Flux-form advection tendency for u using 5th-order upwind in x.

    For the uu flux, u lives at x-faces.  We interpolate u to cell centers
    using upwind5, then compute the flux divergence.
    Vertical fluxes remain 2nd-order centered.  Returns (nx, nz).
    """
    nx, nz = grid.nx, grid.nz
    dx = grid.dx
    dz_c = grid.dz_center

    # --- x-flux of u-momentum at cell centers ---
    # u is at x-faces; interpolate u to cell centers (center is right of face i)
    # u_center sits at center of cell i, between face i and face i+1
    # Velocity carrying the flux is u_center itself (centered between faces).
    u_center_2nd = 0.5 * (u + np.roll(u, -1, axis=0))

    # 5th-order interpolation of u to cell centers using u_center as advecting vel.
    # roll(-1) shifts u[i+1] -> position i, so np.roll(u, -1, axis=0) is "right" face.
    # For upwind5 at center between face i and face i+1, the "left" field is u[i].
    # We need phi at the right face of cell i = left face of cell i+1.
    # Use _upwind5_interp_periodic with roll conventions adjusted:
    # The face we want is between face(i) and face(i+1). The "cell-center"
    # interpretation: u at x-faces treated as cell values, interpolated to
    # midpoints (which are the cell centers). The velocity at that midpoint
    # is u_center_2nd.
    # np.roll(u, -k) with axis=0 shifts by -k, matching _upwind5_interp_periodic
    # with the face at position i being between u[i-1] and u[i] — but we want
    # face at center(i) between u[i] and u[i+1].
    # Shifting frame: let phi = roll(u, -1), vel = roll(u_center_2nd, -1)
    # then _upwind5 gives phi at left face = u at center. But simpler:
    # compute explicitly.
    um2 = np.roll(u, 2, axis=0)
    um1 = np.roll(u, 1, axis=0)
    u0 = u
    up1 = np.roll(u, -1, axis=0)
    up2 = np.roll(u, -2, axis=0)
    up3 = np.roll(u, -3, axis=0)

    u_at_center_pos = (2.0*um2 - 13.0*um1 + 47.0*u0 + 27.0*up1 - 3.0*up2) / 60.0
    u_at_center_neg = (-3.0*um1 + 27.0*u0 + 47.0*up1 - 13.0*up2 + 2.0*up3) / 60.0
    u_at_center = np.where(u_center_2nd >= 0, u_at_center_pos, u_at_center_neg)

    flux_x_center = u_center_2nd * u_at_center
    dflux_x = (flux_x_center - np.roll(flux_x_center, 1, axis=0)) / dx

    # --- z-flux (2nd-order centered, same as original) ---
    w_at_xface = 0.5 * (w + np.roll(w, 1, axis=0))
    flux_z = np.zeros((nx, nz + 1))
    flux_z[:, 1:nz] = w_at_xface[:, 1:nz] * 0.5 * (u[:, :nz-1] + u[:, 1:nz])
    dflux_z = (flux_z[:, 1:] - flux_z[:, :-1]) / dz_c[np.newaxis, :]

    return -(dflux_x + dflux_z)


def compute_advection_tendency_w_upwind5(u, w, grid):
    """Flux-form advection tendency for w using 5th-order upwind in x.

    Vertical fluxes remain 2nd-order centered.
    Returns (nx, nz+1).  Boundary values (k=0, k=nz) stay zero.
    """
    nx, nz = grid.nx, grid.nz
    dx = grid.dx
    dz_c = grid.dz_center
    dz_f = grid.dz_face

    tendency = np.zeros((nx, nz + 1))

    # --- x-flux of w-momentum at (x-face, z-face) ---
    # Advecting velocity at x-faces (interpolated from u to z-faces)
    u_at_zface = np.zeros((nx, nz + 1))
    u_at_zface[:, 1:nz] = 0.5 * (u[:, :nz-1] + u[:, 1:nz])

    # w lives at z-faces (center-like in x); interpolate to x-faces using upwind5
    w_at_xface = _upwind5_interp_periodic(w, u_at_zface, axis=0)
    flux_x = u_at_zface * w_at_xface
    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- z-flux (2nd-order centered, same as original) ---
    w_at_zcenter = 0.5 * (w[:, :-1] + w[:, 1:])
    flux_z_center = w_at_zcenter * w_at_zcenter

    tendency[:, 1:nz] = -(dflux_x[:, 1:nz]
                           + (flux_z_center[:, 1:nz] - flux_z_center[:, :nz-1]) / dz_f[1:nz])

    return tendency


# ---------------------------------------------------------------------------
# 3D upwind5 advection
# ---------------------------------------------------------------------------

def compute_advection_tendency_theta_3d_upwind5(theta, u, v, w, grid):
    """Flux-form advection tendency for theta in 3D using 5th-order upwind in x,y.

    Vertical fluxes remain 2nd-order centered.  Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    dz_c = grid.dz_center

    # --- x-flux ---
    theta_at_xface = _upwind5_interp_periodic(theta, u, axis=0)
    flux_x = u * theta_at_xface
    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- y-flux ---
    theta_at_yface = _upwind5_interp_periodic(theta, v, axis=1)
    flux_y = v * theta_at_yface
    dflux_y = (np.roll(flux_y, -1, axis=1) - flux_y) / dy

    # --- z-flux (2nd-order centered) ---
    flux_z = np.zeros((nx, ny, nz + 1))
    flux_z[:, :, 1:nz] = w[:, :, 1:nz] * 0.5 * (theta[:, :, :nz-1] + theta[:, :, 1:nz])
    dflux_z = (flux_z[:, :, 1:] - flux_z[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]

    return -(dflux_x + dflux_y + dflux_z)


def compute_advection_tendency_u_3d_upwind5(u, v, w, grid):
    """Flux-form advection tendency for u in 3D using 5th-order upwind in x,y.

    Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    dz_c = grid.dz_center

    # --- x-flux: uu at cell centers (explicit stencil, same as 2D) ---
    u_center_2nd = 0.5 * (u + np.roll(u, -1, axis=0))

    um2 = np.roll(u, 2, axis=0)
    um1 = np.roll(u, 1, axis=0)
    u0 = u
    up1 = np.roll(u, -1, axis=0)
    up2 = np.roll(u, -2, axis=0)
    up3 = np.roll(u, -3, axis=0)

    u_at_center_pos = (2.0*um2 - 13.0*um1 + 47.0*u0 + 27.0*up1 - 3.0*up2) / 60.0
    u_at_center_neg = (-3.0*um1 + 27.0*u0 + 47.0*up1 - 13.0*up2 + 2.0*up3) / 60.0
    u_at_center = np.where(u_center_2nd >= 0, u_at_center_pos, u_at_center_neg)

    flux_x_center = u_center_2nd * u_at_center
    dflux_x = (flux_x_center - np.roll(flux_x_center, 1, axis=0)) / dx

    # --- y-flux: v at (x-face, y-face) * u interpolated to y-face ---
    v_at_xface = 0.5 * (v + np.roll(v, 1, axis=0))
    u_at_yface = _upwind5_interp_periodic(u, v_at_xface, axis=1)
    flux_y = v_at_xface * u_at_yface
    dflux_y = (np.roll(flux_y, -1, axis=1) - flux_y) / dy

    # --- z-flux (2nd-order centered) ---
    w_at_xface = 0.5 * (w + np.roll(w, 1, axis=0))
    flux_z = np.zeros((nx, ny, nz + 1))
    flux_z[:, :, 1:nz] = w_at_xface[:, :, 1:nz] * 0.5 * (u[:, :, :nz-1] + u[:, :, 1:nz])
    dflux_z = (flux_z[:, :, 1:] - flux_z[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]

    return -(dflux_x + dflux_y + dflux_z)


def compute_advection_tendency_v_3d_upwind5(u, v, w, grid):
    """Flux-form advection tendency for v in 3D using 5th-order upwind in x,y.

    Returns (nx, ny, nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    dz_c = grid.dz_center

    # --- x-flux: u at (x-face, y-face) * v interpolated to x-face ---
    u_at_yface = 0.5 * (u + np.roll(u, 1, axis=1))
    v_at_xface = _upwind5_interp_periodic(v, u_at_yface, axis=0)
    flux_x = u_at_yface * v_at_xface
    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- y-flux: vv at cell centers (explicit stencil) ---
    v_center_2nd = 0.5 * (v + np.roll(v, -1, axis=1))

    vm2 = np.roll(v, 2, axis=1)
    vm1 = np.roll(v, 1, axis=1)
    v0 = v
    vp1 = np.roll(v, -1, axis=1)
    vp2 = np.roll(v, -2, axis=1)
    vp3 = np.roll(v, -3, axis=1)

    v_at_center_pos = (2.0*vm2 - 13.0*vm1 + 47.0*v0 + 27.0*vp1 - 3.0*vp2) / 60.0
    v_at_center_neg = (-3.0*vm1 + 27.0*v0 + 47.0*vp1 - 13.0*vp2 + 2.0*vp3) / 60.0
    v_at_center = np.where(v_center_2nd >= 0, v_at_center_pos, v_at_center_neg)

    flux_y_center = v_center_2nd * v_at_center
    dflux_y = (flux_y_center - np.roll(flux_y_center, 1, axis=1)) / dy

    # --- z-flux (2nd-order centered) ---
    w_at_yface = 0.5 * (w + np.roll(w, 1, axis=1))
    flux_z = np.zeros((nx, ny, nz + 1))
    flux_z[:, :, 1:nz] = w_at_yface[:, :, 1:nz] * 0.5 * (v[:, :, :nz-1] + v[:, :, 1:nz])
    dflux_z = (flux_z[:, :, 1:] - flux_z[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]

    return -(dflux_x + dflux_y + dflux_z)


def compute_advection_tendency_w_3d_upwind5(u, v, w, grid):
    """Flux-form advection tendency for w in 3D using 5th-order upwind in x,y.

    Vertical fluxes remain 2nd-order centered.
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
    w_at_xface = _upwind5_interp_periodic(w, u_at_zface, axis=0)
    flux_x = u_at_zface * w_at_xface
    dflux_x = (np.roll(flux_x, -1, axis=0) - flux_x) / dx

    # --- y-flux ---
    v_at_zface = np.zeros((nx, ny, nz + 1))
    v_at_zface[:, :, 1:nz] = 0.5 * (v[:, :, :nz-1] + v[:, :, 1:nz])
    w_at_yface = _upwind5_interp_periodic(w, v_at_zface, axis=1)
    flux_y = v_at_zface * w_at_yface
    dflux_y = (np.roll(flux_y, -1, axis=1) - flux_y) / dy

    # --- z-flux (2nd-order centered) ---
    w_at_zcenter = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    flux_z_center = w_at_zcenter * w_at_zcenter

    tendency[:, :, 1:nz] = -(dflux_x[:, :, 1:nz]
                              + dflux_y[:, :, 1:nz]
                              + (flux_z_center[:, :, 1:nz] - flux_z_center[:, :, :nz-1]) / dz_f[1:nz])

    return tendency
