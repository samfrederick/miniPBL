"""Model state container for prognostic and diagnostic fields."""

import copy

import numpy as np

from .grid import Grid


class State:
    """Holds prognostic fields (theta) and diagnostic fields (heat_flux, K_h)."""

    def __init__(self, grid: Grid):
        self.grid = grid
        # Prognostic: potential temperature at cell centers
        self.theta = np.zeros(grid.nz)
        # Diagnostics on cell faces (nz+1)
        self.heat_flux = np.zeros(grid.nz + 1)
        self.K_h = np.zeros(grid.nz + 1)
        # Boundary layer height diagnostic
        self.bl_height = 0.0

    def copy(self) -> "State":
        """Deep copy for RK3 sub-stages."""
        return copy.deepcopy(self)

    def initialize_theta(self, theta0: float, mixed_layer_height: float,
                         lapse_rate: float):
        """Set initial theta profile: mixed layer + stable stratification above."""
        for k in range(self.grid.nz):
            z = self.grid.z_center[k]
            if z <= mixed_layer_height:
                self.theta[k] = theta0
            else:
                self.theta[k] = theta0 + lapse_rate * (z - mixed_layer_height)
