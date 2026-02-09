"""Monin-Obukhov Similarity Theory (MOST) surface layer parameterization.

Iteratively solves for u_star and theta_star given wind speed and temperature
at the first grid level, using Businger-Dyer stability functions.
"""

import numpy as np

KAPPA = 0.4  # von Karman constant


def _psi_m_unstable(zeta):
    """Businger-Dyer momentum stability function for unstable conditions (zeta < 0)."""
    x = (1.0 - 16.0 * zeta) ** 0.25
    return 2.0 * np.log(0.5 * (1.0 + x)) + np.log(0.5 * (1.0 + x * x)) - 2.0 * np.arctan(x) + 0.5 * np.pi


def _psi_h_unstable(zeta):
    """Businger-Dyer heat stability function for unstable conditions (zeta < 0)."""
    y = (1.0 - 16.0 * zeta) ** 0.5
    return 2.0 * np.log(0.5 * (1.0 + y))


def _psi_m_stable(zeta):
    """Momentum stability function for stable conditions (zeta >= 0)."""
    return -5.0 * zeta


def _psi_h_stable(zeta):
    """Heat stability function for stable conditions (zeta >= 0)."""
    return -5.0 * zeta


def compute_surface_fluxes(u1, v1, theta1, theta_s, z1, z0, g, theta_ref,
                           max_iter=20, tol=1e-4):
    """Compute surface fluxes using MOST for a single column.

    Parameters
    ----------
    u1, v1 : float — wind components at first grid level
    theta1 : float — potential temperature at first grid level
    theta_s : float — surface temperature
    z1 : float — height of first grid level (m)
    z0 : float — surface roughness length (m)
    g : float — gravitational acceleration
    theta_ref : float — reference potential temperature
    max_iter : int — maximum iterations
    tol : float — convergence tolerance for u_star

    Returns
    -------
    u_star : float — friction velocity (m/s)
    theta_star : float — temperature scale (K)
    sfc_heat_flux : float — w'theta' at surface (K m/s)
    tau_x, tau_y : float — surface momentum flux components (m^2/s^2)
    """
    M = max(np.sqrt(u1 ** 2 + v1 ** 2), 0.01)  # wind speed, prevent zero
    log_z1_z0 = np.log(z1 / z0)

    # Initial guess: neutral
    u_star = KAPPA * M / log_z1_z0
    theta_star = 0.0

    for _ in range(max_iter):
        u_star_old = u_star

        # Obukhov length
        if abs(theta_star) > 1e-10 and u_star > 1e-10:
            L = u_star ** 2 * theta_ref / (KAPPA * g * theta_star)
        else:
            L = 1e10  # neutral limit

        zeta = z1 / L

        # Clamp zeta to prevent extreme values
        zeta = max(min(zeta, 10.0), -10.0)

        # Stability functions
        if zeta < 0:
            psi_m = _psi_m_unstable(zeta)
            psi_h = _psi_h_unstable(zeta)
        else:
            psi_m = _psi_m_stable(zeta)
            psi_h = _psi_h_stable(zeta)

        # Update u_star and theta_star
        denom_m = log_z1_z0 - psi_m
        denom_h = log_z1_z0 - psi_h

        # Prevent zero or negative denominators
        denom_m = max(denom_m, 0.5)
        denom_h = max(denom_h, 0.5)

        u_star = KAPPA * M / denom_m
        theta_star = KAPPA * (theta1 - theta_s) / denom_h

        if abs(u_star - u_star_old) < tol * max(u_star_old, 1e-6):
            break

    # Surface fluxes
    sfc_heat_flux = -u_star * theta_star
    tau_x = -u_star ** 2 * u1 / M
    tau_y = -u_star ** 2 * v1 / M

    return u_star, theta_star, sfc_heat_flux, tau_x, tau_y


def apply_most_surface_fluxes_2d(state, grid, phys_cfg):
    """Compute MOST surface fluxes for 2D fields.

    Returns (sfc_heat_flux, tau_x) arrays of shape (nx,).
    """
    nx = grid.nx
    z1 = grid.z_center[0]
    z0 = phys_cfg.z0
    g = phys_cfg.g
    theta_ref = phys_cfg.reference_theta
    theta_s = phys_cfg.theta_surface

    sfc_heat_flux = np.zeros(nx)
    tau_x = np.zeros(nx)

    for i in range(nx):
        u1 = state.u[i, 0]
        v1 = 0.0  # no v in 2D
        theta1 = state.theta[i, 0]

        _, _, sfc_heat_flux[i], tau_x[i], _ = compute_surface_fluxes(
            u1, v1, theta1, theta_s, z1, z0, g, theta_ref
        )

    return sfc_heat_flux, tau_x


def apply_most_surface_fluxes_3d(state, grid, phys_cfg):
    """Compute MOST surface fluxes for 3D fields.

    Returns (sfc_heat_flux, tau_x, tau_y) arrays of shape (nx, ny).
    """
    nx, ny = grid.nx, grid.ny
    z1 = grid.z_center[0]
    z0 = phys_cfg.z0
    g = phys_cfg.g
    theta_ref = phys_cfg.reference_theta
    theta_s = phys_cfg.theta_surface

    sfc_heat_flux = np.zeros((nx, ny))
    tau_x = np.zeros((nx, ny))
    tau_y = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            u1 = state.u[i, j, 0]
            v1 = state.v[i, j, 0]
            theta1 = state.theta[i, j, 0]

            _, _, sfc_heat_flux[i, j], tau_x[i, j], tau_y[i, j] = \
                compute_surface_fluxes(
                    u1, v1, theta1, theta_s, z1, z0, g, theta_ref
                )

    return sfc_heat_flux, tau_x, tau_y
