"""Advection operator (placeholder for 1D CBL — no mean vertical velocity)."""

import numpy as np

from .grid import Grid


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
