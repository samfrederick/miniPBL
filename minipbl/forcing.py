"""External forcing: sponge / Rayleigh damping and large-scale subsidence."""

import numpy as np

from .grid import Grid
from .config import PhysicsConfig


# ---------------------------------------------------------------------------
# Sponge / Rayleigh damping layer
# ---------------------------------------------------------------------------

def compute_sponge_tendency(field, field_ref, grid, phys_cfg, z_axis=-1):
    """Compute Rayleigh damping tendency in the upper domain.

    d(phi)/dt += -alpha(z) * (phi - phi_ref)

    alpha(z) = alpha_max * sin^2(pi/2 * (z - z_sponge) / (Lz - z_sponge))
    for z > z_sponge, zero below.

    Parameters
    ----------
    field : ndarray — prognostic field at cell centers (..., nz)
    field_ref : ndarray or None — reference state (same shape as field).
                If None, uses zero (appropriate for w, tke).
    grid : Grid
    phys_cfg : PhysicsConfig
    z_axis : int — axis index for the z dimension

    Returns tendency array with same shape as field.
    """
    alpha_max = phys_cfg.sponge_alpha_max
    sponge_frac = phys_cfg.sponge_fraction

    if alpha_max <= 0 or sponge_frac <= 0:
        return np.zeros_like(field)

    z_sponge = grid.Lz * (1.0 - sponge_frac)
    nz = grid.nz

    # Build damping coefficient profile at cell centers
    alpha = np.zeros(nz)
    for k in range(nz):
        z = grid.z_center[k]
        if z > z_sponge:
            frac = (z - z_sponge) / (grid.Lz - z_sponge)
            alpha[k] = alpha_max * np.sin(0.5 * np.pi * frac) ** 2

    # Reshape alpha for broadcasting
    shape = [1] * field.ndim
    shape[z_axis] = nz
    alpha = alpha.reshape(shape)

    if field_ref is None:
        field_ref = 0.0

    return -alpha * (field - field_ref)


def compute_sponge_tendency_w(w, grid, phys_cfg):
    """Sponge tendency for w (on z-faces). Reference is zero.

    w: (..., nz+1)
    Returns tendency with same shape.
    """
    alpha_max = phys_cfg.sponge_alpha_max
    sponge_frac = phys_cfg.sponge_fraction

    if alpha_max <= 0 or sponge_frac <= 0:
        return np.zeros_like(w)

    z_sponge = grid.Lz * (1.0 - sponge_frac)

    # Build damping coefficient at z-faces
    alpha = np.zeros(grid.nz + 1)
    for k in range(grid.nz + 1):
        z = grid.z_face[k]
        if z > z_sponge:
            frac = (z - z_sponge) / (grid.Lz - z_sponge)
            alpha[k] = alpha_max * np.sin(0.5 * np.pi * frac) ** 2

    # Reshape for broadcasting
    shape = [1] * w.ndim
    shape[-1] = grid.nz + 1
    alpha = alpha.reshape(shape)

    return -alpha * w


# ---------------------------------------------------------------------------
# Large-scale subsidence
# ---------------------------------------------------------------------------

def compute_subsidence_tendency(theta, grid, phys_cfg, z_axis=-1):
    """Compute subsidence tendency for theta.

    d(theta)/dt += -w_s(z) * d(theta)/dz

    where w_s(z) = -D * z  (D = subsidence_divergence, positive = subsidence)

    Uses centered differences for dtheta/dz interpolated to cell centers.

    Parameters
    ----------
    theta : ndarray — potential temperature at cell centers (..., nz)
    grid : Grid
    phys_cfg : PhysicsConfig
    z_axis : int — axis index for the z dimension

    Returns tendency array with same shape as theta.
    """
    D = phys_cfg.subsidence_divergence
    if D == 0:
        return np.zeros_like(theta)

    nz = grid.nz
    dz_f = grid.dz_face

    # Subsidence velocity at cell centers: w_s = -D * z
    w_sub = -D * grid.z_center  # (nz,) — negative = downward

    # dtheta/dz at cell centers using centered differences
    # Interior: average of face gradients above and below
    # Boundaries: one-sided
    dtheta_dz = np.zeros_like(theta)

    if theta.ndim == 1:
        # 1D
        for k in range(1, nz - 1):
            grad_below = (theta[k] - theta[k - 1]) / dz_f[k]
            grad_above = (theta[k + 1] - theta[k]) / dz_f[k + 1]
            dtheta_dz[k] = 0.5 * (grad_below + grad_above)
        dtheta_dz[0] = (theta[1] - theta[0]) / dz_f[1]
        dtheta_dz[nz - 1] = (theta[nz - 1] - theta[nz - 2]) / dz_f[nz - 1]
    elif theta.ndim == 2:
        # 2D: (nx, nz)
        for k in range(1, nz - 1):
            grad_below = (theta[:, k] - theta[:, k - 1]) / dz_f[k]
            grad_above = (theta[:, k + 1] - theta[:, k]) / dz_f[k + 1]
            dtheta_dz[:, k] = 0.5 * (grad_below + grad_above)
        dtheta_dz[:, 0] = (theta[:, 1] - theta[:, 0]) / dz_f[1]
        dtheta_dz[:, nz - 1] = (theta[:, nz - 1] - theta[:, nz - 2]) / dz_f[nz - 1]
    elif theta.ndim == 3:
        # 3D: (nx, ny, nz)
        for k in range(1, nz - 1):
            grad_below = (theta[:, :, k] - theta[:, :, k - 1]) / dz_f[k]
            grad_above = (theta[:, :, k + 1] - theta[:, :, k]) / dz_f[k + 1]
            dtheta_dz[:, :, k] = 0.5 * (grad_below + grad_above)
        dtheta_dz[:, :, 0] = (theta[:, :, 1] - theta[:, :, 0]) / dz_f[1]
        dtheta_dz[:, :, nz - 1] = (theta[:, :, nz - 1] - theta[:, :, nz - 2]) / dz_f[nz - 1]

    # Reshape w_sub for broadcasting
    shape = [1] * theta.ndim
    shape[z_axis] = nz
    w_sub = w_sub.reshape(shape)

    return -w_sub * dtheta_dz
