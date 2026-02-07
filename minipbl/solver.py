"""Main solver: orchestrates one timestep and the simulation loop."""

import numpy as np

from .config import SimConfig
from .grid import Grid
from .state import State
from .turbulence import compute_k_profile
from .diffusion import compute_diffusion_tendency
from .timestepper import rk3_step
from .output import NetCDFWriter


class Solver:
    """Owns the grid, state, and orchestrates the time integration."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.grid = Grid(cfg.grid.nz, cfg.grid.Lz, cfg.grid.dim)
        self.state = State(self.grid)

        # Initialize theta profile
        self.state.initialize_theta(
            cfg.physics.theta_initial,
            cfg.physics.mixed_layer_height,
            cfg.physics.lapse_rate,
        )

        self.time = 0.0
        self.step_count = 0

    def compute_tendency(self, state: State, grid: Grid) -> np.ndarray:
        """Compute total tendency for theta: turbulence closure + diffusion."""
        # Update eddy diffusivity
        K_h = compute_k_profile(state, grid, self.cfg.turbulence, self.cfg.physics)
        state.K_h[:] = K_h

        # Diffusion tendency
        tendency = compute_diffusion_tendency(
            state.theta, K_h, grid, self.cfg.physics.surface_heat_flux
        )

        # Store heat flux diagnostic: w'theta' = -K_h * dtheta/dz (positive upward)
        dz = grid.dz
        for i in range(1, grid.nz):
            state.heat_flux[i] = -K_h[i] * (state.theta[i] - state.theta[i - 1]) / dz
        state.heat_flux[0] = self.cfg.physics.surface_heat_flux
        state.heat_flux[-1] = 0.0

        return tendency

    def step(self):
        """Advance one timestep with RK3."""
        self.state = rk3_step(
            self.state, self.grid, self.cfg.time.dt, self.compute_tendency
        )
        self.time += self.cfg.time.dt
        self.step_count += 1

    def run(self):
        """Run the full simulation."""
        dt = self.cfg.time.dt
        t_end = self.cfg.time.t_end
        output_interval = self.cfg.time.output_interval
        next_output_time = 0.0

        writer = NetCDFWriter(self.cfg, self.grid)

        n_steps = int(t_end / dt)
        print(f"Starting simulation: {n_steps} steps, dt={dt} s, t_end={t_end} s")

        # Write initial state
        self.compute_tendency(self.state, self.grid)  # populate diagnostics
        writer.write(self.state, self.time)
        next_output_time += output_interval

        for n in range(1, n_steps + 1):
            self.step()

            if self.time >= next_output_time - 0.5 * dt:
                writer.write(self.state, self.time)
                next_output_time += output_interval
                bl_h = self.state.bl_height
                theta_sfc = self.state.theta[0]
                print(f"  t={self.time:8.1f} s  "
                      f"BL height={bl_h:7.1f} m  "
                      f"theta_sfc={theta_sfc:.2f} K")

        writer.close()
        print(f"Simulation complete. Output in {self.cfg.output.output_dir}/")
        return writer.filepath
