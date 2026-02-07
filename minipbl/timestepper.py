"""Low-storage 3rd-order Runge-Kutta time integration (Williamson 1980)."""

import numpy as np

from .state import State
from .grid import Grid

# Standard 3-stage RK3 coefficients (Wicker & Skamarock 2002 variant)
# Stage 1: q* = q^n + (dt/3) * R(q^n)
# Stage 2: q** = q^n + (dt/2) * R(q*)
# Stage 3: q^(n+1) = q^n + dt * R(q**)
RK3_ALPHA = [1.0 / 3.0, 1.0 / 2.0, 1.0]


def rk3_step(state: State, grid: Grid, dt: float,
             compute_tendency) -> State:
    """Advance state by one timestep using 3-stage RK3.

    Parameters
    ----------
    state : current State
    grid : Grid instance
    dt : timestep
    compute_tendency : callable(State, Grid) -> np.ndarray
        Returns dtheta/dt given current state and grid.
    """
    theta_n = state.theta.copy()

    for alpha in RK3_ALPHA:
        tendency = compute_tendency(state, grid)
        state.theta = theta_n + alpha * dt * tendency

    return state
