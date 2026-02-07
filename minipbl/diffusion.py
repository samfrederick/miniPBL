"""Vertical diffusion operator using second-order centered differences."""

import numpy as np

from .grid import Grid


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
    dz = grid.dz

    # Compute kinematic heat flux w'theta' = -K_h * dtheta/dz on interior faces
    # Positive upward (flux is up-gradient correction: if theta decreases with z,
    # dtheta/dz < 0, so w'theta' = -K * (negative) > 0 = upward flux)
    wtheta = np.zeros(nz + 1)
    for i in range(1, nz):
        wtheta[i] = -K_h[i] * (theta[i] - theta[i - 1]) / dz

    # Boundary conditions
    wtheta[0] = surface_heat_flux    # prescribed surface kinematic heat flux
    wtheta[-1] = 0.0                 # zero flux at top (insulating lid)

    # Tendency: dtheta/dt = -d(w'theta')/dz
    tendency = np.zeros(nz)
    for k in range(nz):
        tendency[k] = -(wtheta[k + 1] - wtheta[k]) / dz

    return tendency
