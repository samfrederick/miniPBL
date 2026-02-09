"""Deardorff TKE subgrid-scale closure.

Prognostic TKE equation: de/dt = Shear + Buoyancy - Dissipation + Diffusion + Advection

Eddy viscosity and diffusivity are computed from TKE and a mixing length.
"""

import numpy as np

from .grid import Grid
from .config import TurbulenceConfig, PhysicsConfig


def _compute_mixing_length(tke_col, grid, dtheta_dz, turb_cfg, phys_cfg):
    """Compute mixing length for a single column.

    l = min(c_l * sqrt(e) / N, Delta)  where N = sqrt(max(0, (g/theta_ref)*dtheta/dz))
    If unstable (N^2 <= 0), l = Delta.
    Delta = sqrt(dx * dz_c[k]) as the geometric mean grid spacing.

    tke_col: (nz,), dtheta_dz: (nz+1,) at faces.
    Returns l: (nz+1,) at faces.
    """
    nz = grid.nz
    c_l = turb_cfg.tke_c_l
    g = phys_cfg.g
    theta_ref = phys_cfg.reference_theta

    l = np.zeros(nz + 1)

    for k in range(1, nz):
        # TKE at face k: average of adjacent centers
        e_face = 0.5 * (tke_col[k - 1] + tke_col[k])
        e_face = max(e_face, turb_cfg.tke_min)

        # Grid scale: geometric mean of dx and local dz
        dz_local = grid.dz_face[k]
        if grid.dim >= 3:
            Delta = (grid.dx * grid.dy * dz_local) ** (1.0 / 3.0)
        elif grid.dim >= 2:
            Delta = (grid.dx * dz_local) ** 0.5
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


def compute_tke_closure_column(tke_col, theta_col, u_col, v_col, grid,
                                turb_cfg, phys_cfg):
    """Compute K_m, K_h, and TKE tendency for a single column.

    tke_col: (nz,) at cell centers
    theta_col: (nz,) at cell centers
    u_col: (nz,) at cell centers (or None for 1D)
    v_col: (nz,) at cell centers (or None for 1D/2D)
    grid: Grid
    turb_cfg: TurbulenceConfig
    phys_cfg: PhysicsConfig

    Returns:
        K_m: (nz+1,) eddy viscosity at faces
        K_h: (nz+1,) eddy diffusivity at faces
        tke_tend: (nz,) TKE tendency at centers
    """
    nz = grid.nz
    dz_c = grid.dz_center
    dz_f = grid.dz_face
    c_m = turb_cfg.tke_c_m
    c_eps_base = turb_cfg.tke_c_eps_base
    tke_min = turb_cfg.tke_min
    g = phys_cfg.g
    theta_ref = phys_cfg.reference_theta
    background_K = turb_cfg.background_K

    # Gradients at faces
    dtheta_dz = np.zeros(nz + 1)
    du_dz = np.zeros(nz + 1)
    dv_dz = np.zeros(nz + 1)

    for k in range(1, nz):
        dtheta_dz[k] = (theta_col[k] - theta_col[k - 1]) / dz_f[k]
        if u_col is not None:
            du_dz[k] = (u_col[k] - u_col[k - 1]) / dz_f[k]
        if v_col is not None:
            dv_dz[k] = (v_col[k] - v_col[k - 1]) / dz_f[k]

    # Mixing length at faces
    l = _compute_mixing_length(tke_col, grid, dtheta_dz, turb_cfg, phys_cfg)

    # Grid scale Delta at faces (for c_eps and K_h ratio)
    Delta = np.zeros(nz + 1)
    for k in range(1, nz):
        dz_local = dz_f[k]
        if grid.dim >= 3:
            Delta[k] = (grid.dx * grid.dy * dz_local) ** (1.0 / 3.0)
        elif grid.dim >= 2:
            Delta[k] = (grid.dx * dz_local) ** 0.5
        else:
            Delta[k] = dz_local

    # Eddy viscosity and diffusivity at faces
    K_m = np.full(nz + 1, background_K)
    K_h = np.full(nz + 1, background_K)

    for k in range(1, nz):
        e_face = max(0.5 * (tke_col[k - 1] + tke_col[k]), tke_min)
        K_m[k] = max(c_m * l[k] * np.sqrt(e_face), background_K)
        l_over_Delta = l[k] / Delta[k] if Delta[k] > 0 else 0.0
        K_h[k] = max((1.0 + 2.0 * l_over_Delta) * K_m[k], background_K)

    # TKE tendency at cell centers
    tke_tend = np.zeros(nz)

    for k in range(nz):
        # Shear and buoyancy production: use face averages
        # Shear: S = K_m * (|du/dz|^2 + |dv/dz|^2)
        # Average the face contributions to the cell center
        shear_bot = K_m[k] * (du_dz[k] ** 2 + dv_dz[k] ** 2)
        shear_top = K_m[k + 1] * (du_dz[k + 1] ** 2 + dv_dz[k + 1] ** 2)
        shear = 0.5 * (shear_bot + shear_top)

        # Buoyancy: B = -(g/theta_ref) * K_h * dtheta/dz
        buoy_bot = -(g / theta_ref) * K_h[k] * dtheta_dz[k]
        buoy_top = -(g / theta_ref) * K_h[k + 1] * dtheta_dz[k + 1]
        buoyancy = 0.5 * (buoy_bot + buoy_top)

        # Dissipation: eps = c_eps * e^(3/2) / l
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
        tke_flux[k] = -2.0 * K_m[k] * (tke_col[k] - tke_col[k - 1]) / dz_f[k]
    # Zero flux at boundaries
    tke_flux[0] = 0.0
    tke_flux[nz] = 0.0

    for k in range(nz):
        tke_tend[k] += -(tke_flux[k + 1] - tke_flux[k]) / dz_c[k]

    return K_m, K_h, tke_tend


def compute_tke_closure_2d(state, grid, turb_cfg, phys_cfg):
    """Compute K_m, K_h, and TKE tendency for 2D fields.

    Returns (K_h, K_m, tke_tend) with shapes (nx, nz+1), (nx, nz+1), (nx, nz).
    """
    nx, nz = grid.nx, grid.nz
    K_h = np.zeros((nx, nz + 1))
    K_m = np.zeros((nx, nz + 1))
    tke_tend = np.zeros((nx, nz))

    for i in range(nx):
        u_col = state.u[i, :] if state.u is not None else None
        v_col = None  # v not prognosed in 2D
        K_m[i, :], K_h[i, :], tke_tend[i, :] = compute_tke_closure_column(
            state.tke[i, :], state.theta[i, :], u_col, v_col,
            grid, turb_cfg, phys_cfg
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

    for i in range(nx):
        for j in range(ny):
            u_col = state.u[i, j, :] if state.u is not None else None
            v_col = state.v[i, j, :] if state.v is not None else None
            K_m[i, j, :], K_h[i, j, :], tke_tend[i, j, :] = compute_tke_closure_column(
                state.tke[i, j, :], state.theta[i, j, :], u_col, v_col,
                grid, turb_cfg, phys_cfg
            )

    return K_h, K_m, tke_tend
