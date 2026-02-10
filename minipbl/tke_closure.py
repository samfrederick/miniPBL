"""Deardorff TKE subgrid-scale closure.

Prognostic TKE equation: de/dt = Shear + Buoyancy - Dissipation + Diffusion + Advection

Eddy viscosity and diffusivity are computed from TKE and a mixing length.
"""

import numpy as np
from numba import njit

from .grid import Grid
from .config import TurbulenceConfig, PhysicsConfig


@njit
def _compute_mixing_length_jit(tke_col, dz_face, dx, dy, dim, nz,
                                dtheta_dz, c_l, tke_min, g, theta_ref):
    """Compute mixing length for a single column (JIT-compiled).

    Returns l: (nz+1,) at faces.
    """
    l = np.zeros(nz + 1)

    for k in range(1, nz):
        # TKE at face k: average of adjacent centers
        e_face = 0.5 * (tke_col[k - 1] + tke_col[k])
        e_face = max(e_face, tke_min)

        # Grid scale: geometric mean of dx and local dz
        dz_local = dz_face[k]
        if dim >= 3:
            Delta = (dx * dy * dz_local) ** (1.0 / 3.0)
        elif dim >= 2:
            Delta = (dx * dz_local) ** 0.5
        else:
            Delta = dz_local

        # Brunt-Vaisala frequency squared at face k
        N2 = (g / theta_ref) * dtheta_dz[k]

        if N2 > 0:
            N = np.sqrt(N2)
            l[k] = min(c_l * np.sqrt(e_face) / N, Delta)
        else:
            l[k] = Delta

    return l


def _compute_mixing_length(tke_col, grid, dtheta_dz, turb_cfg, phys_cfg):
    """Compute mixing length for a single column.

    Python wrapper preserving the original API for external callers.
    """
    return _compute_mixing_length_jit(
        tke_col, grid.dz_face, grid.dx, grid.dy, grid.dim, grid.nz,
        dtheta_dz, turb_cfg.tke_c_l, turb_cfg.tke_min,
        phys_cfg.g, phys_cfg.reference_theta,
    )


@njit
def _compute_tke_closure_column_jit(tke_col, theta_col, u_col, v_col,
                                     nz, dz_center, dz_face, dx, dy, dim,
                                     c_m, c_eps_base, tke_min, c_l,
                                     g, theta_ref, background_K,
                                     has_u, has_v):
    """Compute K_m, K_h, and TKE tendency for a single column (JIT-compiled).

    Returns (K_m, K_h, tke_tend).
    """
    # Gradients at faces
    dtheta_dz = np.zeros(nz + 1)
    du_dz = np.zeros(nz + 1)
    dv_dz = np.zeros(nz + 1)

    for k in range(1, nz):
        dtheta_dz[k] = (theta_col[k] - theta_col[k - 1]) / dz_face[k]
        if has_u:
            du_dz[k] = (u_col[k] - u_col[k - 1]) / dz_face[k]
        if has_v:
            dv_dz[k] = (v_col[k] - v_col[k - 1]) / dz_face[k]

    # Mixing length at faces
    l = _compute_mixing_length_jit(tke_col, dz_face, dx, dy, dim, nz,
                                    dtheta_dz, c_l, tke_min, g, theta_ref)

    # Grid scale Delta at faces
    Delta = np.zeros(nz + 1)
    for k in range(1, nz):
        dz_local = dz_face[k]
        if dim >= 3:
            Delta[k] = (dx * dy * dz_local) ** (1.0 / 3.0)
        elif dim >= 2:
            Delta[k] = (dx * dz_local) ** 0.5
        else:
            Delta[k] = dz_local

    # Eddy viscosity and diffusivity at faces
    K_m = np.full(nz + 1, background_K)
    K_h = np.full(nz + 1, background_K)

    for k in range(1, nz):
        e_face = max(0.5 * (tke_col[k - 1] + tke_col[k]), tke_min)
        K_m[k] = max(c_m * l[k] * np.sqrt(e_face), background_K)
        if Delta[k] > 0:
            l_over_Delta = l[k] / Delta[k]
        else:
            l_over_Delta = 0.0
        K_h[k] = max((1.0 + 2.0 * l_over_Delta) * K_m[k], background_K)

    # TKE tendency at cell centers
    tke_tend = np.zeros(nz)

    for k in range(nz):
        # Shear production
        shear_bot = K_m[k] * (du_dz[k] ** 2 + dv_dz[k] ** 2)
        shear_top = K_m[k + 1] * (du_dz[k + 1] ** 2 + dv_dz[k + 1] ** 2)
        shear = 0.5 * (shear_bot + shear_top)

        # Buoyancy production
        buoy_bot = -(g / theta_ref) * K_h[k] * dtheta_dz[k]
        buoy_top = -(g / theta_ref) * K_h[k + 1] * dtheta_dz[k + 1]
        buoyancy = 0.5 * (buoy_bot + buoy_top)

        # Dissipation
        e_k = max(tke_col[k], tke_min)
        l_center = 0.5 * (l[k] + l[k + 1])
        l_center = max(l_center, 1e-10)
        Delta_center = 0.5 * (Delta[k] + Delta[k + 1])
        Delta_center = max(Delta_center, 1e-10)
        c_eps = c_eps_base + 0.51 * l_center / Delta_center
        dissipation = c_eps * e_k ** 1.5 / l_center

        tke_tend[k] = shear + buoyancy - dissipation

    # TKE diffusion: d/dz(2*K_m * de/dz)
    tke_flux = np.zeros(nz + 1)
    for k in range(1, nz):
        tke_flux[k] = -2.0 * K_m[k] * (tke_col[k] - tke_col[k - 1]) / dz_face[k]

    for k in range(nz):
        tke_tend[k] += -(tke_flux[k + 1] - tke_flux[k]) / dz_center[k]

    return K_m, K_h, tke_tend


def compute_tke_closure_column(tke_col, theta_col, u_col, v_col, grid,
                                turb_cfg, phys_cfg):
    """Compute K_m, K_h, and TKE tendency for a single column.

    Python wrapper preserving the original API for external callers.
    """
    has_u = u_col is not None
    has_v = v_col is not None
    dummy = np.zeros(grid.nz)
    return _compute_tke_closure_column_jit(
        tke_col, theta_col,
        u_col if has_u else dummy,
        v_col if has_v else dummy,
        grid.nz, grid.dz_center, grid.dz_face, grid.dx, grid.dy, grid.dim,
        turb_cfg.tke_c_m, turb_cfg.tke_c_eps_base, turb_cfg.tke_min,
        turb_cfg.tke_c_l,
        phys_cfg.g, phys_cfg.reference_theta, turb_cfg.background_K,
        has_u, has_v,
    )


def compute_tke_closure_2d(state, grid, turb_cfg, phys_cfg):
    """Compute K_m, K_h, and TKE tendency for 2D fields.

    Returns (K_h, K_m, tke_tend) with shapes (nx, nz+1), (nx, nz+1), (nx, nz).
    """
    nx, nz = grid.nx, grid.nz
    K_h = np.zeros((nx, nz + 1))
    K_m = np.zeros((nx, nz + 1))
    tke_tend = np.zeros((nx, nz))

    # Extract scalars/arrays once
    dz_center = grid.dz_center
    dz_face = grid.dz_face
    dx = grid.dx
    dy = grid.dy
    dim = grid.dim
    c_m = turb_cfg.tke_c_m
    c_eps_base = turb_cfg.tke_c_eps_base
    tke_min = turb_cfg.tke_min
    c_l = turb_cfg.tke_c_l
    g = phys_cfg.g
    theta_ref = phys_cfg.reference_theta
    background_K = turb_cfg.background_K

    has_u = state.u is not None
    dummy = np.zeros(nz)

    for i in range(nx):
        u_col = state.u[i, :] if has_u else dummy
        K_m[i, :], K_h[i, :], tke_tend[i, :] = _compute_tke_closure_column_jit(
            state.tke[i, :], state.theta[i, :], u_col, dummy,
            nz, dz_center, dz_face, dx, dy, dim,
            c_m, c_eps_base, tke_min, c_l,
            g, theta_ref, background_K,
            has_u, False,
        )

    return K_h, K_m, tke_tend


def compute_tke_closure_3d(state, grid, turb_cfg, phys_cfg):
    """Compute K_m, K_h, and TKE tendency for 3D fields.

    Returns (K_h, K_m, tke_tend) with shapes (nx,ny,nz+1), (nx,ny,nz+1), (nx,ny,nz).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    K_h = np.zeros((nx, ny, nz + 1))
    K_m = np.zeros((nx, ny, nz + 1))
    tke_tend = np.zeros((nx, ny, nz))

    # Extract scalars/arrays once
    dz_center = grid.dz_center
    dz_face = grid.dz_face
    dx = grid.dx
    dy = grid.dy
    dim = grid.dim
    c_m = turb_cfg.tke_c_m
    c_eps_base = turb_cfg.tke_c_eps_base
    tke_min = turb_cfg.tke_min
    c_l = turb_cfg.tke_c_l
    g = phys_cfg.g
    theta_ref = phys_cfg.reference_theta
    background_K = turb_cfg.background_K

    has_u = state.u is not None
    has_v = state.v is not None
    dummy = np.zeros(nz)

    for i in range(nx):
        for j in range(ny):
            u_col = state.u[i, j, :] if has_u else dummy
            v_col = state.v[i, j, :] if has_v else dummy
            K_m[i, j, :], K_h[i, j, :], tke_tend[i, j, :] = _compute_tke_closure_column_jit(
                state.tke[i, j, :], state.theta[i, j, :], u_col, v_col,
                nz, dz_center, dz_face, dx, dy, dim,
                c_m, c_eps_base, tke_min, c_l,
                g, theta_ref, background_K,
                has_u, has_v,
            )

    return K_h, K_m, tke_tend
