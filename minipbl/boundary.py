"""Boundary condition handlers for 1D and 2D."""

import numpy as np

from .grid import Grid


# ---------------------------------------------------------------------------
# 1D boundary conditions (unchanged)
# ---------------------------------------------------------------------------

def apply_surface_flux(flux_array: np.ndarray, surface_heat_flux: float):
    """Set the surface (bottom) boundary heat flux."""
    flux_array[0] = surface_heat_flux


def apply_top_zero_flux(flux_array: np.ndarray):
    """Set zero-flux (insulating lid) at the top boundary."""
    flux_array[-1] = 0.0


def apply_top_fixed_gradient(theta: np.ndarray, grid: Grid, lapse_rate: float):
    """Enforce a fixed lapse rate in the topmost cell (Neumann-like).

    This prevents the top boundary from contaminating the interior solution.
    """
    theta[-1] = theta[-2] + lapse_rate * grid.dz


# ---------------------------------------------------------------------------
# 2D boundary conditions
# ---------------------------------------------------------------------------

def apply_rigid_lid_w(w):
    """Enforce w = 0 at bottom (z=0) and top (z=Lz) boundaries.

    w: (nx, nz+1)
    """
    w[:, 0] = 0.0
    w[:, -1] = 0.0


def apply_top_fixed_gradient_2d(theta, grid, lapse_rate):
    """Enforce fixed lapse rate in topmost cell for 2D fields.

    theta: (nx, nz)
    """
    theta[:, -1] = theta[:, -2] + lapse_rate * grid.dz
