"""Main solver: orchestrates one timestep and the simulation loop."""

import os
import numpy as np

from .config import SimConfig
from .grid import Grid
from .state import State
from .turbulence import compute_k_profile, compute_k_profile_2d, compute_k_profile_3d
from .diffusion import (compute_diffusion_tendency,
                        compute_diffusion_tendency_theta,
                        compute_diffusion_tendency_u,
                        compute_diffusion_tendency_w,
                        compute_diffusion_tendency_theta_3d,
                        compute_diffusion_tendency_u_3d,
                        compute_diffusion_tendency_v_3d,
                        compute_diffusion_tendency_w_3d)
from .advection import (compute_advection_tendency_theta,
                        compute_advection_tendency_u,
                        compute_advection_tendency_w,
                        compute_advection_tendency_theta_3d,
                        compute_advection_tendency_u_3d,
                        compute_advection_tendency_v_3d,
                        compute_advection_tendency_w_3d)
from .boundary import (apply_rigid_lid_w, apply_top_fixed_gradient_2d,
                       apply_top_fixed_gradient_3d)
from .pressure import PoissonSolver, PoissonSolver3D
from .timestepper import rk3_step, rk3_step_2d, rk3_step_3d
from .output import NetCDFWriter


class Solver:
    """Owns the grid, state, and orchestrates the time integration."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.grid = Grid(cfg.grid.nz, cfg.grid.Lz, cfg.grid.dim,
                         cfg.grid.nx, cfg.grid.Lx,
                         cfg.grid.ny, cfg.grid.Ly)
        self.state = State(self.grid)

        # Initialize theta profile
        self.state.initialize_theta(
            cfg.physics.theta_initial,
            cfg.physics.mixed_layer_height,
            cfg.physics.lapse_rate,
        )

        # 3D initialization
        if self.grid.dim >= 3:
            self.state.initialize_u(
                cfg.physics.geostrophic_u,
                z0=cfg.physics.z0,
                bl_height=cfg.physics.mixed_layer_height,
            )
            self.state.initialize_v(cfg.physics.geostrophic_v)
            self.poisson_solver = PoissonSolver3D(self.grid)
            rng = np.random.default_rng(seed=42)
            n_perturb = min(5, self.grid.nz)
            self.state.theta[:, :, :n_perturb] += rng.standard_normal(
                (self.grid.nx, self.grid.ny, n_perturb)) * 0.01
        elif self.grid.dim >= 2:
            self.state.initialize_u(
                cfg.physics.geostrophic_u,
                z0=cfg.physics.z0,
                bl_height=cfg.physics.mixed_layer_height,
            )
            self.poisson_solver = PoissonSolver(self.grid)
            rng = np.random.default_rng(seed=42)
            n_perturb = min(5, self.grid.nz)
            self.state.theta[:, :n_perturb] += rng.standard_normal(
                (self.grid.nx, n_perturb)) * 0.01
        else:
            self.poisson_solver = None

        self.time = 0.0
        self.step_count = 0

    # ------------------------------------------------------------------
    # 1D tendency (unchanged)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 2D tendencies
    # ------------------------------------------------------------------

    def compute_tendencies_2d(self, state, grid):
        """Compute tendencies for all prognostic variables in 2D.

        Returns dict {'theta': ..., 'u': ..., 'w': ...}.
        """
        phys = self.cfg.physics
        turb = self.cfg.turbulence

        # Turbulence closure (column-by-column)
        K_h, K_m = compute_k_profile_2d(state, grid, turb, phys)
        state.K_h[:] = K_h
        state.K_m[:] = K_m

        K_horiz = turb.K_horizontal

        # --- Theta tendency ---
        adv_theta = compute_advection_tendency_theta(
            state.theta, state.u, state.w, grid)
        diff_theta = compute_diffusion_tendency_theta(
            state.theta, K_h, grid, phys.surface_heat_flux, K_horiz=K_horiz)
        tend_theta = adv_theta + diff_theta

        # --- u tendency ---
        adv_u = compute_advection_tendency_u(state.u, state.w, grid)
        diff_u = compute_diffusion_tendency_u(state.u, K_m, grid, K_horiz=K_horiz)
        # Coriolis: in 2D x-z with v not prognosed, the u-equation gets
        # du/dt = ... + f*v_geo from the Coriolis+PGF balance.
        coriolis_u = phys.coriolis_f * phys.geostrophic_v
        tend_u = adv_u + diff_u + coriolis_u

        # --- w tendency ---
        adv_w = compute_advection_tendency_w(state.u, state.w, grid)
        diff_w = compute_diffusion_tendency_w(state.w, K_m, grid, K_horiz=K_horiz)
        # Buoyancy: dw/dt += g * (theta' / theta_ref)
        # theta' = theta - theta_mean(z) for each level
        theta_mean = np.mean(state.theta, axis=0, keepdims=True)  # (1, nz)
        buoyancy = np.zeros_like(state.w)
        # Buoyancy at z-faces: interpolate theta perturbation
        theta_prime = state.theta - theta_mean  # (nx, nz)
        buoyancy[:, 1:grid.nz] = phys.g / phys.reference_theta * 0.5 * (
            theta_prime[:, :grid.nz-1] + theta_prime[:, 1:grid.nz])
        tend_w = adv_w + diff_w + buoyancy

        # Store heat flux diagnostic
        dz = grid.dz
        state.heat_flux[:, 1:grid.nz] = -K_h[:, 1:grid.nz] * (
            state.theta[:, 1:grid.nz] - state.theta[:, :grid.nz-1]) / dz
        state.heat_flux[:, 0] = phys.surface_heat_flux
        state.heat_flux[:, -1] = 0.0

        return {'theta': tend_theta, 'u': tend_u, 'w': tend_w}

    # ------------------------------------------------------------------
    # 3D tendencies
    # ------------------------------------------------------------------

    def compute_tendencies_3d(self, state, grid):
        """Compute tendencies for all prognostic variables in 3D.

        Returns dict {'theta': ..., 'u': ..., 'v': ..., 'w': ...}.
        """
        phys = self.cfg.physics
        turb = self.cfg.turbulence

        # Turbulence closure (column-by-column)
        K_h, K_m = compute_k_profile_3d(state, grid, turb, phys)
        state.K_h[:] = K_h
        state.K_m[:] = K_m

        K_horiz = turb.K_horizontal

        # --- Theta tendency ---
        adv_theta = compute_advection_tendency_theta_3d(
            state.theta, state.u, state.v, state.w, grid)
        diff_theta = compute_diffusion_tendency_theta_3d(
            state.theta, K_h, grid, phys.surface_heat_flux, K_horiz=K_horiz)
        tend_theta = adv_theta + diff_theta

        # --- u tendency ---
        adv_u = compute_advection_tendency_u_3d(state.u, state.v, state.w, grid)
        diff_u = compute_diffusion_tendency_u_3d(state.u, K_m, grid, K_horiz=K_horiz)
        # Coriolis: du/dt += f * (v_bar - v_geo)
        # v at x-faces: average v[i-1,j] and v[i,j]
        v_at_xface = 0.5 * (state.v + np.roll(state.v, 1, axis=0))
        coriolis_u = phys.coriolis_f * (v_at_xface - phys.geostrophic_v)
        tend_u = adv_u + diff_u + coriolis_u

        # --- v tendency ---
        adv_v = compute_advection_tendency_v_3d(state.u, state.v, state.w, grid)
        diff_v = compute_diffusion_tendency_v_3d(state.v, K_m, grid, K_horiz=K_horiz)
        # Coriolis: dv/dt += -f * (u_bar - u_geo)
        # u at y-faces: average u[i,j-1] and u[i,j]
        u_at_yface = 0.5 * (state.u + np.roll(state.u, 1, axis=1))
        coriolis_v = -phys.coriolis_f * (u_at_yface - phys.geostrophic_u)
        tend_v = adv_v + diff_v + coriolis_v

        # --- w tendency ---
        adv_w = compute_advection_tendency_w_3d(state.u, state.v, state.w, grid)
        diff_w = compute_diffusion_tendency_w_3d(state.w, K_m, grid, K_horiz=K_horiz)
        # Buoyancy
        theta_mean = np.mean(state.theta, axis=(0, 1), keepdims=True)  # (1, 1, nz)
        buoyancy = np.zeros_like(state.w)
        theta_prime = state.theta - theta_mean
        buoyancy[:, :, 1:grid.nz] = phys.g / phys.reference_theta * 0.5 * (
            theta_prime[:, :, :grid.nz-1] + theta_prime[:, :, 1:grid.nz])
        tend_w = adv_w + diff_w + buoyancy

        # Store heat flux diagnostic
        dz = grid.dz
        state.heat_flux[:, :, 1:grid.nz] = -K_h[:, :, 1:grid.nz] * (
            state.theta[:, :, 1:grid.nz] - state.theta[:, :, :grid.nz-1]) / dz
        state.heat_flux[:, :, 0] = phys.surface_heat_flux
        state.heat_flux[:, :, -1] = 0.0

        return {'theta': tend_theta, 'u': tend_u, 'v': tend_v, 'w': tend_w}

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self):
        """Advance one timestep."""
        if self.grid.dim >= 3:
            self.state = rk3_step_3d(
                self.state, self.grid, self.cfg.time.dt,
                self.compute_tendencies_3d, self.poisson_solver
            )
            apply_top_fixed_gradient_3d(
                self.state.theta, self.grid, self.cfg.physics.lapse_rate)
        elif self.grid.dim >= 2:
            self.state = rk3_step_2d(
                self.state, self.grid, self.cfg.time.dt,
                self.compute_tendencies_2d, self.poisson_solver
            )
            # Enforce top theta BC
            apply_top_fixed_gradient_2d(
                self.state.theta, self.grid, self.cfg.physics.lapse_rate)
        else:
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
        print(f"Starting simulation: {n_steps} steps, dt={dt} s, t_end={t_end} s"
              f"  dim={self.grid.dim}")

        # Progress log file for monitoring long simulations
        log_path = os.path.join(self.cfg.output.output_dir, "progress.log")
        log_interval = max(1, n_steps // 200)  # ~200 log entries total

        def _log_progress(n, msg):
            with open(log_path, "a") as lf:
                lf.write(f"step {n}/{n_steps}  {msg}\n")

        # Clear old log
        with open(log_path, "w") as lf:
            lf.write(f"# miniPBL progress: {n_steps} steps, dt={dt}, dim={self.grid.dim}\n")

        # Write initial state
        if self.grid.dim >= 3:
            self.compute_tendencies_3d(self.state, self.grid)
        elif self.grid.dim >= 2:
            self.compute_tendencies_2d(self.state, self.grid)
        else:
            self.compute_tendency(self.state, self.grid)
        writer.write(self.state, self.time)
        next_output_time += output_interval

        for n in range(1, n_steps + 1):
            self.step()

            # Write progress log periodically
            if n % log_interval == 0:
                if self.grid.dim >= 2:
                    _log_progress(n, f"t={self.time:.1f}s  "
                                  f"max|w|={np.max(np.abs(self.state.w)):.4f}")
                else:
                    _log_progress(n, f"t={self.time:.1f}s  "
                                  f"BLh={self.state.bl_height:.1f}m")

            if self.time >= next_output_time - 0.5 * dt:
                writer.write(self.state, self.time)
                next_output_time += output_interval

                if self.grid.dim >= 3:
                    bl_h = np.mean(self.state.bl_height)
                    theta_sfc = np.mean(self.state.theta[:, :, 0])
                    max_w = np.max(np.abs(self.state.w))
                    print(f"  t={self.time:8.1f} s  "
                          f"BL height={bl_h:7.1f} m  "
                          f"theta_sfc={theta_sfc:.2f} K  "
                          f"max|w|={max_w:.3f} m/s")
                elif self.grid.dim >= 2:
                    bl_h = np.mean(self.state.bl_height)
                    theta_sfc = np.mean(self.state.theta[:, 0])
                    max_w = np.max(np.abs(self.state.w))
                    print(f"  t={self.time:8.1f} s  "
                          f"BL height={bl_h:7.1f} m  "
                          f"theta_sfc={theta_sfc:.2f} K  "
                          f"max|w|={max_w:.3f} m/s")
                else:
                    bl_h = self.state.bl_height
                    theta_sfc = self.state.theta[0]
                    print(f"  t={self.time:8.1f} s  "
                          f"BL height={bl_h:7.1f} m  "
                          f"theta_sfc={theta_sfc:.2f} K")

        writer.close()
        _log_progress(n_steps, "COMPLETE")
        print(f"Simulation complete. Output in {self.cfg.output.output_dir}/")
        return writer.filepath
