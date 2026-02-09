"""Main solver: orchestrates one timestep and the simulation loop."""

import os
import numpy as np

from .config import SimConfig
from .grid import Grid
from .state import State
from .turbulence import compute_k_profile, compute_k_profile_2d, compute_k_profile_3d
from .tke_closure import compute_tke_closure_2d, compute_tke_closure_3d
from .surface_layer import apply_most_surface_fluxes_2d, apply_most_surface_fluxes_3d
from .forcing import (compute_sponge_tendency, compute_sponge_tendency_w,
                      compute_subsidence_tendency)
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
                         cfg.grid.ny, cfg.grid.Ly,
                         cfg.grid.stretch_factor, cfg.grid.nz_uniform)
        self.state = State(self.grid)

        # Initialize theta profile
        self.state.initialize_theta(
            cfg.physics.theta_initial,
            cfg.physics.mixed_layer_height,
            cfg.physics.lapse_rate,
        )

        # Initialize TKE if using deardorff-tke scheme
        if cfg.turbulence.scheme == "deardorff-tke":
            self.state.initialize_tke(
                tke_min=cfg.turbulence.tke_min,
                surface_heat_flux=cfg.physics.surface_heat_flux,
                mixed_layer_height=cfg.physics.mixed_layer_height,
                g=cfg.physics.g,
                reference_theta=cfg.physics.reference_theta,
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
    # 1D tendency
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
        for i in range(1, grid.nz):
            state.heat_flux[i] = -K_h[i] * (state.theta[i] - state.theta[i - 1]) / grid.dz_face[i]
        state.heat_flux[0] = self.cfg.physics.surface_heat_flux
        state.heat_flux[-1] = 0.0

        return tendency

    # ------------------------------------------------------------------
    # 2D tendencies
    # ------------------------------------------------------------------

    def compute_tendencies_2d(self, state, grid):
        """Compute tendencies for all prognostic variables in 2D.

        Returns dict {'theta': ..., 'u': ..., 'w': ...} and optionally 'tke'.
        """
        phys = self.cfg.physics
        turb = self.cfg.turbulence

        # --- Turbulence closure ---
        tke_tend = None
        if turb.scheme == "deardorff-tke" and state.tke is not None:
            K_h, K_m, tke_tend = compute_tke_closure_2d(state, grid, turb, phys)
        else:
            K_h, K_m = compute_k_profile_2d(state, grid, turb, phys)
        state.K_h[:] = K_h
        state.K_m[:] = K_m

        K_horiz = turb.K_horizontal

        # --- Surface fluxes ---
        if phys.surface_flux_scheme == "most":
            sfc_heat_flux_arr, tau_x_arr = apply_most_surface_fluxes_2d(
                state, grid, phys)
        else:
            sfc_heat_flux_arr = None  # use scalar prescribed value
            tau_x_arr = None

        # --- Theta tendency ---
        sfc_hf = sfc_heat_flux_arr if sfc_heat_flux_arr is not None else phys.surface_heat_flux
        adv_theta = compute_advection_tendency_theta(
            state.theta, state.u, state.w, grid)
        diff_theta = compute_diffusion_tendency_theta(
            state.theta, K_h, grid, sfc_hf, K_horiz=K_horiz)
        tend_theta = adv_theta + diff_theta

        # Subsidence
        tend_theta += compute_subsidence_tendency(state.theta, grid, phys)

        # Sponge on theta: relax toward horizontal mean
        theta_ref = np.mean(state.theta, axis=0, keepdims=True)
        tend_theta += compute_sponge_tendency(state.theta, theta_ref, grid, phys)

        # --- u tendency ---
        adv_u = compute_advection_tendency_u(state.u, state.w, grid)

        # If MOST, override surface stress in diffusion by applying tau directly
        if tau_x_arr is not None:
            # Build custom diffusion: use MOST stress at surface
            diff_u = _diffusion_u_with_most_2d(state.u, K_m, grid, tau_x_arr, K_horiz)
        else:
            diff_u = compute_diffusion_tendency_u(state.u, K_m, grid, K_horiz=K_horiz)

        coriolis_u = phys.coriolis_f * phys.geostrophic_v
        tend_u = adv_u + diff_u + coriolis_u

        # Sponge on u: relax toward horizontal mean
        u_ref = np.mean(state.u, axis=0, keepdims=True)
        tend_u += compute_sponge_tendency(state.u, u_ref, grid, phys)

        # --- w tendency ---
        adv_w = compute_advection_tendency_w(state.u, state.w, grid)
        diff_w = compute_diffusion_tendency_w(state.w, K_m, grid, K_horiz=K_horiz)
        theta_mean = np.mean(state.theta, axis=0, keepdims=True)
        buoyancy = np.zeros_like(state.w)
        theta_prime = state.theta - theta_mean
        buoyancy[:, 1:grid.nz] = phys.g / phys.reference_theta * 0.5 * (
            theta_prime[:, :grid.nz-1] + theta_prime[:, 1:grid.nz])
        tend_w = adv_w + diff_w + buoyancy

        # Sponge on w: relax toward zero
        tend_w += compute_sponge_tendency_w(state.w, grid, phys)

        # Store heat flux diagnostic
        dz_f = grid.dz_face
        state.heat_flux[:, 1:grid.nz] = -K_h[:, 1:grid.nz] * (
            state.theta[:, 1:grid.nz] - state.theta[:, :grid.nz-1]) / dz_f[1:grid.nz]
        if sfc_heat_flux_arr is not None:
            state.heat_flux[:, 0] = sfc_heat_flux_arr
        else:
            state.heat_flux[:, 0] = phys.surface_heat_flux
        state.heat_flux[:, -1] = 0.0

        result = {'theta': tend_theta, 'u': tend_u, 'w': tend_w}

        # TKE tendency
        if tke_tend is not None:
            # Sponge on TKE: relax toward zero
            tke_tend += compute_sponge_tendency(state.tke, None, grid, phys)
            result['tke'] = tke_tend

        return result

    # ------------------------------------------------------------------
    # 3D tendencies
    # ------------------------------------------------------------------

    def compute_tendencies_3d(self, state, grid):
        """Compute tendencies for all prognostic variables in 3D.

        Returns dict {'theta': ..., 'u': ..., 'v': ..., 'w': ...} and optionally 'tke'.
        """
        phys = self.cfg.physics
        turb = self.cfg.turbulence

        # --- Turbulence closure ---
        tke_tend = None
        if turb.scheme == "deardorff-tke" and state.tke is not None:
            K_h, K_m, tke_tend = compute_tke_closure_3d(state, grid, turb, phys)
        else:
            K_h, K_m = compute_k_profile_3d(state, grid, turb, phys)
        state.K_h[:] = K_h
        state.K_m[:] = K_m

        K_horiz = turb.K_horizontal

        # --- Surface fluxes ---
        if phys.surface_flux_scheme == "most":
            sfc_heat_flux_arr, tau_x_arr, tau_y_arr = apply_most_surface_fluxes_3d(
                state, grid, phys)
        else:
            sfc_heat_flux_arr = None
            tau_x_arr = None
            tau_y_arr = None

        # --- Theta tendency ---
        sfc_hf = sfc_heat_flux_arr if sfc_heat_flux_arr is not None else phys.surface_heat_flux
        adv_theta = compute_advection_tendency_theta_3d(
            state.theta, state.u, state.v, state.w, grid)
        diff_theta = compute_diffusion_tendency_theta_3d(
            state.theta, K_h, grid, sfc_hf, K_horiz=K_horiz)
        tend_theta = adv_theta + diff_theta

        # Subsidence
        tend_theta += compute_subsidence_tendency(state.theta, grid, phys)

        # Sponge on theta: relax toward horizontal mean
        theta_ref = np.mean(state.theta, axis=(0, 1), keepdims=True)
        tend_theta += compute_sponge_tendency(state.theta, theta_ref, grid, phys)

        # --- u tendency ---
        adv_u = compute_advection_tendency_u_3d(state.u, state.v, state.w, grid)

        if tau_x_arr is not None:
            diff_u = _diffusion_u_with_most_3d(state.u, K_m, grid, tau_x_arr, K_horiz)
        else:
            diff_u = compute_diffusion_tendency_u_3d(state.u, K_m, grid, K_horiz=K_horiz)

        v_at_xface = 0.5 * (state.v + np.roll(state.v, 1, axis=0))
        coriolis_u = phys.coriolis_f * (v_at_xface - phys.geostrophic_v)
        tend_u = adv_u + diff_u + coriolis_u

        # Sponge on u
        u_ref = np.mean(state.u, axis=(0, 1), keepdims=True)
        tend_u += compute_sponge_tendency(state.u, u_ref, grid, phys)

        # --- v tendency ---
        adv_v = compute_advection_tendency_v_3d(state.u, state.v, state.w, grid)

        if tau_y_arr is not None:
            diff_v = _diffusion_v_with_most_3d(state.v, K_m, grid, tau_y_arr, K_horiz)
        else:
            diff_v = compute_diffusion_tendency_v_3d(state.v, K_m, grid, K_horiz=K_horiz)

        u_at_yface = 0.5 * (state.u + np.roll(state.u, 1, axis=1))
        coriolis_v = -phys.coriolis_f * (u_at_yface - phys.geostrophic_u)
        tend_v = adv_v + diff_v + coriolis_v

        # Sponge on v
        v_ref = np.mean(state.v, axis=(0, 1), keepdims=True)
        tend_v += compute_sponge_tendency(state.v, v_ref, grid, phys)

        # --- w tendency ---
        adv_w = compute_advection_tendency_w_3d(state.u, state.v, state.w, grid)
        diff_w = compute_diffusion_tendency_w_3d(state.w, K_m, grid, K_horiz=K_horiz)
        theta_mean = np.mean(state.theta, axis=(0, 1), keepdims=True)
        buoyancy = np.zeros_like(state.w)
        theta_prime = state.theta - theta_mean
        buoyancy[:, :, 1:grid.nz] = phys.g / phys.reference_theta * 0.5 * (
            theta_prime[:, :, :grid.nz-1] + theta_prime[:, :, 1:grid.nz])
        tend_w = adv_w + diff_w + buoyancy

        # Sponge on w
        tend_w += compute_sponge_tendency_w(state.w, grid, phys)

        # Store heat flux diagnostic
        dz_f = grid.dz_face
        state.heat_flux[:, :, 1:grid.nz] = -K_h[:, :, 1:grid.nz] * (
            state.theta[:, :, 1:grid.nz] - state.theta[:, :, :grid.nz-1]) / dz_f[1:grid.nz]
        if sfc_heat_flux_arr is not None:
            state.heat_flux[:, :, 0] = sfc_heat_flux_arr
        else:
            state.heat_flux[:, :, 0] = phys.surface_heat_flux
        state.heat_flux[:, :, -1] = 0.0

        result = {'theta': tend_theta, 'u': tend_u, 'v': tend_v, 'w': tend_w}

        # TKE tendency
        if tke_tend is not None:
            tke_tend += compute_sponge_tendency(state.tke, None, grid, phys)
            result['tke'] = tke_tend

        return result

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
            apply_top_fixed_gradient_2d(
                self.state.theta, self.grid, self.cfg.physics.lapse_rate)
        else:
            self.state = rk3_step(
                self.state, self.grid, self.cfg.time.dt, self.compute_tendency
            )

        # Clamp TKE to minimum
        if self.state.tke is not None:
            np.clip(self.state.tke, self.cfg.turbulence.tke_min, None,
                    out=self.state.tke)

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
        log_interval = max(1, n_steps // 200)

        def _log_progress(n, msg):
            with open(log_path, "a") as lf:
                lf.write(f"step {n}/{n_steps}  {msg}\n")

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


# ---------------------------------------------------------------------------
# Helpers: diffusion with MOST surface stress override
# ---------------------------------------------------------------------------

def _diffusion_u_with_most_2d(u, K_m, grid, tau_x, K_horiz):
    """Vertical diffusion of u with MOST-derived surface stress for 2D.

    tau_x: (nx,) — surface momentum flux from MOST (m^2/s^2).
    """
    from .diffusion import _horizontal_laplacian_center

    nx, nz = grid.nx, grid.nz
    dz_c = grid.dz_center
    dz_f = grid.dz_face

    tau = np.zeros((nx, nz + 1))
    tau[:, 1:nz] = -K_m[:, 1:nz] * (u[:, 1:nz] - u[:, :nz-1]) / dz_f[1:nz]
    # MOST surface stress replaces no-slip approximation
    tau[:, 0] = tau_x  # tau_x = -u_star^2 * u/M (already signed)
    tau[:, -1] = 0.0

    tendency = -(tau[:, 1:] - tau[:, :-1]) / dz_c[np.newaxis, :]

    if K_horiz > 0:
        tendency += K_horiz * _horizontal_laplacian_center(u, grid.dx)

    return tendency


def _diffusion_u_with_most_3d(u, K_m, grid, tau_x, K_horiz):
    """Vertical diffusion of u with MOST-derived surface stress for 3D.

    tau_x: (nx, ny) — surface momentum flux from MOST.
    """
    from .diffusion import _horizontal_laplacian_center_3d

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dz_c = grid.dz_center
    dz_f = grid.dz_face

    tau = np.zeros((nx, ny, nz + 1))
    tau[:, :, 1:nz] = -K_m[:, :, 1:nz] * (u[:, :, 1:nz] - u[:, :, :nz-1]) / dz_f[1:nz]
    tau[:, :, 0] = tau_x
    tau[:, :, -1] = 0.0

    tendency = -(tau[:, :, 1:] - tau[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]

    if K_horiz > 0:
        tendency += K_horiz * _horizontal_laplacian_center_3d(u, grid.dx, grid.dy)

    return tendency


def _diffusion_v_with_most_3d(v, K_m, grid, tau_y, K_horiz):
    """Vertical diffusion of v with MOST-derived surface stress for 3D.

    tau_y: (nx, ny) — surface momentum flux from MOST.
    """
    from .diffusion import _horizontal_laplacian_center_3d

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dz_c = grid.dz_center
    dz_f = grid.dz_face

    tau = np.zeros((nx, ny, nz + 1))
    tau[:, :, 1:nz] = -K_m[:, :, 1:nz] * (v[:, :, 1:nz] - v[:, :, :nz-1]) / dz_f[1:nz]
    tau[:, :, 0] = tau_y
    tau[:, :, -1] = 0.0

    tendency = -(tau[:, :, 1:] - tau[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]

    if K_horiz > 0:
        tendency += K_horiz * _horizontal_laplacian_center_3d(v, grid.dx, grid.dy)

    return tendency
