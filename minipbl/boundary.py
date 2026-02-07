"""Boundary condition handlers for the 1D column."""

import numpy as np

from .grid import Grid


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
