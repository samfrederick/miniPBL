"""Turbulence closure: K-profile parameterization for eddy diffusivity."""

import numpy as np

from .config import TurbulenceConfig, PhysicsConfig
from .grid import Grid
from .state import State

G = 9.81  # m/s^2


def diagnose_bl_height(theta: np.ndarray, grid: Grid,
                       theta_excess: float) -> float:
    """Find boundary layer height as the lowest level where theta exceeds
    the near-surface value by more than theta_excess."""
    theta_sfc = theta[0]
    for k in range(1, grid.nz):
        if theta[k] - theta_sfc > theta_excess:
            # Linearly interpolate between k-1 and k
            dtheta_below = theta[k - 1] - theta_sfc
            dtheta_above = theta[k] - theta_sfc
            frac = (theta_excess - dtheta_below) / (dtheta_above - dtheta_below)
            return grid.z_center[k - 1] + frac * grid.dz
    # If no inversion found, return domain height
    return grid.Lz


def compute_k_profile(state: State, grid: Grid,
                      turb_cfg: TurbulenceConfig,
                      phys_cfg: PhysicsConfig) -> np.ndarray:
    """Compute eddy diffusivity K_h on cell faces using K-profile closure.

    K_h(z) = kappa * w_star * z * (1 - z/h)^2   for z < h
    K_h(z) = background_K                         for z >= h

    where w_star = (g/theta_0 * sfc_flux * h)^(1/3)
    """
    kappa = turb_cfg.k_profile_kappa
    background = turb_cfg.background_K
    theta_excess = turb_cfg.theta_excess_threshold
    sfc_flux = phys_cfg.surface_heat_flux
    theta_ref = phys_cfg.reference_theta

    h = diagnose_bl_height(state.theta, grid, theta_excess)
    state.bl_height = h

    # Convective velocity scale
    if sfc_flux > 0 and h > 0:
        w_star = (G / theta_ref * sfc_flux * h) ** (1.0 / 3.0)
    else:
        w_star = 0.0

    K_h = np.full(grid.nz + 1, background)

    for i in range(grid.nz + 1):
        z = grid.z_face[i]
        if 0 < z < h:
            K_h[i] = kappa * w_star * z * (1.0 - z / h) ** 2
            K_h[i] = max(K_h[i], background)

    return K_h
