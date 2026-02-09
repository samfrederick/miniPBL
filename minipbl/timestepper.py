"""Low-storage 3rd-order Runge-Kutta time integration (Williamson 1980)."""

import numpy as np

from .state import State
from .grid import Grid
from .boundary import apply_rigid_lid_w, apply_rigid_lid_w_3d
from .pressure import project_velocity, project_velocity_3d

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


def rk3_step_2d(state, grid, dt, compute_tendencies, poisson_solver):
    """Advance 2D state by one timestep using RK3 with pressure projection.

    Parameters
    ----------
    state : State with theta(nx,nz), u(nx,nz), w(nx,nz+1), optionally tke(nx,nz)
    grid : Grid
    dt : timestep
    compute_tendencies : callable(State, Grid) -> dict
        Returns {'theta': ..., 'u': ..., 'w': ...} and optionally 'tke'
    poisson_solver : PoissonSolver instance
    """
    theta_n = state.theta.copy()
    u_n = state.u.copy()
    w_n = state.w.copy()
    tke_n = state.tke.copy() if state.tke is not None else None

    for alpha in RK3_ALPHA:
        tend = compute_tendencies(state, grid)

        # Advance provisionally
        state.theta = theta_n + alpha * dt * tend['theta']
        u_star = u_n + alpha * dt * tend['u']
        w_star = w_n + alpha * dt * tend['w']

        # Advance TKE if present
        if tke_n is not None and 'tke' in tend:
            state.tke = tke_n + alpha * dt * tend['tke']

        # Apply rigid-lid BCs before projection
        apply_rigid_lid_w(w_star)

        # Pressure projection to enforce incompressibility
        state.u, state.w, state.p = project_velocity(
            u_star, w_star, grid, poisson_solver, alpha * dt
        )

        # Enforce rigid lid again after projection
        apply_rigid_lid_w(state.w)

    return state


def rk3_step_3d(state, grid, dt, compute_tendencies, poisson_solver):
    """Advance 3D state by one timestep using RK3 with pressure projection.

    Parameters
    ----------
    state : State with theta(nx,ny,nz), u, v, w, optionally tke(nx,ny,nz)
    grid : Grid
    dt : timestep
    compute_tendencies : callable(State, Grid) -> dict
        Returns {'theta': ..., 'u': ..., 'v': ..., 'w': ...} and optionally 'tke'
    poisson_solver : PoissonSolver3D instance
    """
    theta_n = state.theta.copy()
    u_n = state.u.copy()
    v_n = state.v.copy()
    w_n = state.w.copy()
    tke_n = state.tke.copy() if state.tke is not None else None

    for alpha in RK3_ALPHA:
        tend = compute_tendencies(state, grid)

        state.theta = theta_n + alpha * dt * tend['theta']
        u_star = u_n + alpha * dt * tend['u']
        v_star = v_n + alpha * dt * tend['v']
        w_star = w_n + alpha * dt * tend['w']

        # Advance TKE if present
        if tke_n is not None and 'tke' in tend:
            state.tke = tke_n + alpha * dt * tend['tke']

        apply_rigid_lid_w_3d(w_star)

        state.u, state.v, state.w, state.p = project_velocity_3d(
            u_star, v_star, w_star, grid, poisson_solver, alpha * dt
        )

        apply_rigid_lid_w_3d(state.w)

    return state
