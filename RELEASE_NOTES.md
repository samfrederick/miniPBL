# miniPBL v4.1.0

5th-order upwind advection scheme, MOST convective velocity fix, and LES cold-start improvements.

## New Features

### 5th-Order Upwind Advection (Wicker & Skamarock 2002)
- Upwind-biased advection for horizontal (periodic) fluxes using a 6-point stencil
- Provides implicit numerical dissipation that selectively damps grid-scale noise while preserving resolved convective structures
- Vertical fluxes remain 2nd-order centered (z is non-periodic and may be stretched)
- Eliminates the need for explicit horizontal diffusion (`K_horizontal`) — automatically set to zero when `advection_scheme = "upwind5"`
- Available for all prognostic variables (theta, u, v, w) in both 2D and 3D
- Activated by setting `advection_scheme: "upwind5"` in the turbulence config (default: `"centered"`)
- Requires `nx >= 6` (and `ny >= 6` for 3D) for the 6-point stencil

### Beljaars (1994) Convective Velocity Scale in MOST
- Effective wind speed includes convective velocity: `M_eff = sqrt(M² + (1.2·w*)²)`
- Convective velocity scale: `w* = (g/theta_ref · w'theta'_sfc · h)^(1/3)`
- Prevents MOST flux collapse in free-convective (low/zero mean wind) conditions where the wind speed floor alone gave near-zero friction velocity and heat flux
- Boundary layer height diagnosed from the theta profile, with fallback to `mixed_layer_height`

## Bug Fixes
- **TKE cold-start**: Initialize TKE to `tke_min` uniformly instead of convective scaling profile. The previous initialization (w*²-based profile reaching ~4 m²/s² at surface) caused excessive initial SGS dissipation that killed resolved convection during spin-up. Standard LES practice (PALM, DALES, WRF-LES) is to start with minimal SGS TKE and let turbulence develop organically from theta perturbations and surface fluxes.
- **Theta perturbation amplitude**: Increased from 0.01 K to 0.1 K, consistent with standard LES initialization practice.
- **theta_surface**: Increased from 302 K to 307 K in `cbl_2d.yaml` for consistency with the target surface heat flux (~0.24 K m/s) under MOST.

## Configuration Changes

### TurbulenceConfig
- `advection_scheme: str = "centered"` — `"centered"` (2nd-order + K_horizontal) or `"upwind5"` (5th-order upwind, no explicit diffusion)

### PhysicsConfig
- `theta_surface` default updated to 307.0 K in `cbl_2d.yaml`

## Files Added
- `config/cbl_2d_minimal.yaml` — Minimal 2D config for isolated physics testing (prescribed flux, no MOST, no subsidence, no sponge, no wind, no Coriolis)

## Files Modified
- `minipbl/advection.py` — 5th-order upwind interpolation helper and 7 new advection functions (3 for 2D, 4 for 3D)
- `minipbl/config.py` — `advection_scheme` field and validation
- `minipbl/solver.py` — Advection scheme dispatch, grid size validation, 0.1 K perturbation amplitude
- `minipbl/state.py` — TKE initialization changed to uniform `tke_min`
- `minipbl/surface_layer.py` — Beljaars convective velocity in MOST, `bl_height` parameter
- `config/cbl_2d.yaml` — `advection_scheme: "upwind5"`, `K_horizontal: 0.0`, `theta_surface: 307.0`
- `config/cbl_3d.yaml` — `advection_scheme: "upwind5"`

## Dependencies
- numpy
- scipy
- matplotlib
- pyyaml

---

# miniPBL v4.0.0

Addition of advanced physics parameterizations to miniPBL: vertical grid stretching, Deardorff prognostic TKE subgrid closure, Monin-Obukhov similarity theory surface layer, Rayleigh sponge damping, and large-scale subsidence. All operators updated to support variable vertical spacing. Existing 1D, 2D, and 3D modes are fully backward-compatible.

## New Features

### Vertical Grid Stretching
- Optional geometric stretching of the vertical grid with configurable `stretch_factor` (ratio between successive cell thicknesses) and `nz_uniform` (number of uniform cells near the surface before stretching begins)
- New grid arrays `dz_center[k]` (cell thickness) and `dz_face[k]` (distance between cell centers) replace the scalar `dz` throughout all operators
- When `stretch_factor = 1.0` (default), the grid is identical to the previous uniform spacing

### Variable-dz Support in All Operators
- Diffusion: flux `F[k] = -K[k] * (f[k] - f[k-1]) / dz_face[k]`; tendency `(F[k+1] - F[k]) / dz_center[k]`
- Advection: flux divergence uses local `dz_center[k]` and `dz_face[k]`
- Pressure Poisson solver: tridiagonal coefficients `a[k] = 1/(dz_c[k]*dz_f[k])`, `c[k] = 1/(dz_c[k]*dz_f[k+1])` for variable spacing
- Divergence and pressure gradient correction use position-dependent spacing
- Boundary conditions (top gradient, surface stress) use local cell thickness

### Deardorff TKE Subgrid-Scale Closure
- Prognostic TKE equation: `de/dt = Shear + Buoyancy - Dissipation + Diffusion`
- Shear production: `S = K_m * (|du/dz|^2 + |dv/dz|^2)`
- Buoyancy production/destruction: `B = -(g/theta_ref) * K_h * dtheta/dz`
- Dissipation: `eps = c_eps * e^(3/2) / l` with `c_eps = 0.19 + 0.51*l/Delta`
- Stability-dependent mixing length: `l = min(c_l * sqrt(e) / N, Delta)` (Delta for unstable conditions)
- Eddy viscosity: `K_m = c_m * l * sqrt(e)`; eddy diffusivity: `K_h = (1 + 2*l/Delta) * K_m`
- TKE advanced by RK3 alongside theta, u, v, w; clamped to configurable minimum
- Column-by-column computation for 2D and 3D fields
- Activated by setting `scheme: "deardorff-tke"` in the turbulence config

### Monin-Obukhov Similarity Theory (MOST) Surface Layer
- Iterative solver for friction velocity (u_star) and temperature scale (theta_star) given wind speed and temperature at the first grid level
- Businger-Dyer stability functions for momentum (psi_m) and heat (psi_h) under both stable and unstable conditions
- Surface heat flux: `w'theta'_sfc = -u_star * theta_star`
- Surface momentum flux: `tau = -u_star^2 * (u,v) / |V|`
- Replaces both the prescribed surface heat flux and the no-slip surface stress approximation when activated
- Activated by setting `surface_flux_scheme: "most"` in the physics config

### Rayleigh Sponge Damping Layer
- Rayleigh damping in the upper domain: `d(phi)/dt += -alpha(z) * (phi - phi_ref)`
- Damping profile: `alpha(z) = alpha_max * sin^2(pi/2 * (z - z_sponge) / (Lz - z_sponge))`
- Applied to theta (relaxed toward horizontal mean), u and v (horizontal mean), w and TKE (zero)
- Configurable `sponge_fraction` (default 0.25) and `sponge_alpha_max` (default 0, i.e. off)

### Large-Scale Subsidence
- Prescribed subsidence profile: `w_s(z) = -D * z` where D is the divergence rate
- Theta tendency: `d(theta)/dt += -w_s * d(theta)/dz`
- Prevents unbounded boundary layer growth in long simulations
- Activated by setting `subsidence_divergence > 0` in the physics config

## Configuration Changes

### GridConfig
- `stretch_factor: float = 1.0` — geometric stretch ratio (1.0 = uniform)
- `nz_uniform: int = 0` — uniform cells near surface before stretching

### PhysicsConfig
- `surface_flux_scheme: str = "prescribed"` — `"prescribed"` or `"most"`
- `theta_surface: float = 302.0` — surface temperature for MOST (K)
- `subsidence_divergence: float = 0.0` — large-scale divergence rate (1/s)
- `sponge_fraction: float = 0.25` — fraction of domain for sponge layer
- `sponge_alpha_max: float = 0.01` — maximum Rayleigh damping rate (1/s)

### TurbulenceConfig
- `scheme` now supports `"k-profile"` (existing) and `"deardorff-tke"` (new)
- `tke_c_m: float = 0.1` — eddy viscosity coefficient
- `tke_c_eps_base: float = 0.19` — base dissipation coefficient
- `tke_c_l: float = 0.76` — mixing length coefficient
- `tke_min: float = 1e-4` — minimum TKE floor (m^2/s^2)

## Backward Compatibility

All new features default to off. Existing configurations produce identical results:
- `stretch_factor = 1.0`: uniform grid (identical to v3.0.0)
- `scheme = "k-profile"`: diagnostic K-profile closure (unchanged)
- `surface_flux_scheme = "prescribed"`: fixed surface heat flux (unchanged)
- `sponge_alpha_max = 0.0`: no sponge damping
- `subsidence_divergence = 0.0`: no subsidence

## Files Added
- `minipbl/tke_closure.py` — Deardorff prognostic TKE closure
- `minipbl/surface_layer.py` — Monin-Obukhov similarity theory surface fluxes
- `minipbl/forcing.py` — Rayleigh sponge damping and large-scale subsidence

## Files Modified
- `minipbl/config.py` — New parameters in GridConfig, PhysicsConfig, TurbulenceConfig
- `minipbl/grid.py` — Vertical stretching, `dz_center`/`dz_face` arrays
- `minipbl/state.py` — TKE field and `initialize_tke()` method
- `minipbl/diffusion.py` — All operators use variable `dz_center`/`dz_face`
- `minipbl/advection.py` — All operators use variable `dz_center`/`dz_face`
- `minipbl/pressure.py` — Variable-dz tridiagonal coefficients and projection
- `minipbl/boundary.py` — Top gradient BCs use `dz_center[-1]`
- `minipbl/turbulence.py` — BL height interpolation uses `dz_face[k]`
- `minipbl/solver.py` — Wired TKE closure, MOST, sponge, subsidence into tendency computation
- `minipbl/timestepper.py` — RK3 advances TKE alongside other prognostic variables
- `config/cbl_2d.yaml` — New physics parameters with comments
- `config/cbl_3d.yaml` — New physics parameters with comments

## Dependencies
- numpy
- scipy
- matplotlib
- pyyaml

---

# miniPBL v3.0.0

Extension of miniPBL from 2D x-z to full 3D x-y-z Boussinesq solver with a prognostic v velocity, Coriolis coupling on both u and v, and 3D pressure projection. The 1D and 2D modes are fully backward-compatible.

## New Features

### 3D x-y-z Boussinesq Solver
- Prognostic v velocity on Arakawa C-grid y-faces, alongside existing theta, u, and w
- All arrays stored internally as `(nx, ny, nz)` with NetCDF output transposed to `(time, z, y, x)` for VisIT
- Automatic 3D mode activation when both `nx > 1` and `ny > 1` in the configuration

### 3D Pressure Solver
- `PoissonSolver3D`: 2D FFT in x and y (both periodic) + tridiagonal solve in z per wavenumber pair
- Combined eigenvalues `lambda_xy[m,n] = lambda_x[m] + lambda_y[n]` for each `(kx, ky)` mode
- kx=0, ky=0 mode: pressure pinned to zero (null-space treatment)
- `project_velocity_3d`: 3D divergence, pressure correction for u, v, and w

### 3D Advection
- Flux-form centered advection for theta, u, v, and w with y-flux terms
- Periodic y boundary conditions via `np.roll` on axis=1

### 3D Diffusion
- Horizontal Laplacian in x and y for cell-center and z-face fields
- Vertical + horizontal diffusion for theta, u, v, and w
- No-slip surface stress for both u and v

### Coriolis Forcing
- Full Coriolis coupling: `du/dt += f*(v - v_geo)`, `dv/dt += -f*(u - u_geo)`
- v interpolated to x-faces for u tendency; u interpolated to y-faces for v tendency

### 3D Turbulence
- Column-by-column K-profile closure over `(nx, ny)` columns
- Boundary layer height diagnosed per column: `bl_height(nx, ny)`

### 3D Boundary Conditions
- Rigid-lid: `w[:,:,0] = 0`, `w[:,:,-1] = 0`
- Fixed lapse rate enforcement at domain top for 3D theta fields

### 3D Output and Diagnostics
- NetCDF dimensions: `(time, z_center, y_center, x_center)` and `(time, z_face, y_center, x_center)`
- Variables: theta, u, v, w, p, heat_flux, K_h, bl_height
- Diagnostic plots:
  - x-z cross-sections at mid-y (theta, u, v, w)
  - x-y horizontal cross-sections at mid-z and near-surface (theta, w)
  - xy-averaged theta profiles
  - xy-averaged BL height time series

### Configuration
- New grid parameters: `ny`, `Ly`
- New config file: `config/cbl_3d.yaml` (64x64x64, dt=0.5s, 1 hour simulation)

## Backward Compatibility

- `ny=1` (or omitting `ny`): dimensionality stays 1D or 2D; all existing code paths unchanged
- `nx > 1, ny > 1`: activates 3D code paths
- `geostrophic_v` remains as a large-scale forcing parameter (used for Coriolis balance in 2D and 3D)

## Files Added
- `config/cbl_3d.yaml` — 3D configuration file

## Files Modified
- `minipbl/config.py` — Added `ny`, `Ly` to `GridConfig`; 3D auto-detection
- `minipbl/grid.py` — Added y-direction arrays (`ny`, `Ly`, `dy`, `y_face`, `y_center`)
- `minipbl/state.py` — 3D field arrays `(nx, ny, nz)`, prognostic v, `initialize_v()`
- `minipbl/advection.py` — 3D flux-form advection for theta, u, v, w
- `minipbl/diffusion.py` — 3D horizontal Laplacian; diffusion tendencies for theta, u, v, w
- `minipbl/pressure.py` — `PoissonSolver3D` and `project_velocity_3d`
- `minipbl/boundary.py` — 3D rigid-lid and top-gradient boundary conditions
- `minipbl/turbulence.py` — `compute_k_profile_3d` (double loop over nx, ny columns)
- `minipbl/timestepper.py` — `rk3_step_3d` with v projection
- `minipbl/solver.py` — `compute_tendencies_3d` (Coriolis on u and v, buoyancy), 3D constructor/step/run
- `minipbl/output.py` — `_close_3d` with `(time, z, y, x)` ordering, v variable
- `minipbl/plotting.py` — `_plot_results_3d` with x-z, x-y, and profile diagnostics

## Dependencies
- numpy
- scipy
- matplotlib
- pyyaml

---

# miniPBL v2.0.0

Extension of miniPBL from a 1D vertical column solver to a 2D x-z Boussinesq solver with resolved convection, momentum equations, and pressure projection. The 1D mode is fully backward-compatible with v1.0.0.

## New Features

### 2D x-z Boussinesq Solver
- Prognostic equations for horizontal velocity (u), vertical velocity (w), and potential temperature (theta) on an Arakawa C-grid
- Periodic boundary conditions in x, rigid-lid (w=0) at top and bottom
- Automatic 2D mode activation when `nx > 1` in the configuration

### Pressure Solver
- FFT-based Poisson solver for enforcing incompressibility (divergence-free velocity)
- FFT in x (periodic) with tridiagonal solve in z per wavenumber
- Neumann boundary conditions in z; mean pressure pinned to zero for the kx=0 mode
- Fractional-step pressure projection applied at each RK3 sub-stage

### Advection
- Second-order flux-form centered advection for theta, u, and w
- Proper C-grid staggering with periodic x boundary conditions via `np.roll`

### Momentum Physics
- Buoyancy forcing: vertical velocity driven by horizontal potential temperature perturbations (theta - theta_mean)
- Coriolis forcing with prescribed geostrophic wind (u_geo, v_geo)
- No-slip surface stress parameterization for u
- Vertical turbulent momentum diffusion using K_m = K_m_ratio * K_h

### Horizontal Diffusion
- Configurable horizontal diffusivity (`K_horizontal`) applied to theta, u, and w
- Prevents grid-scale noise accumulation from the non-dissipative centered advection scheme

### Initialization
- Logarithmic wind profile initialization: u(z) = u_geo * ln(z/z0) / ln(h/z0) below the boundary layer height, with configurable surface roughness length (z0)
- Small random theta perturbation seeded near the surface to trigger convective instability

### Turbulence (2D)
- Column-by-column K-profile closure reusing the existing 1D scheme
- Separate K_h (heat) and K_m (momentum) eddy diffusivities with configurable ratio

### Output and Diagnostics
- 2D NetCDF output with x_center dimension and u, w, p variables
- x-z cross-section plots (pcolormesh) of theta, u, w at selected times
- x-averaged theta profile plots for comparison with 1D results
- Progress log file (`output/progress.log`) for monitoring long simulations

### Configuration
- New parameters: `nx`, `Lx`, `g`, `coriolis_f`, `geostrophic_u`, `geostrophic_v`, `z0`, `K_m_ratio`, `K_horizontal`
- New 2D config file: `config/cbl_2d.yaml`

## Backward Compatibility

The 1D solver path is completely unchanged. Running with `nx=1` (or omitting `nx` from the config) produces identical results to v1.0.0.

## Files Added
- `minipbl/pressure.py` — Poisson solver and velocity projection
- `config/cbl_2d.yaml` — 2D configuration file

## Files Modified
- `minipbl/config.py` — New grid, physics, and turbulence parameters
- `minipbl/grid.py` — Horizontal grid arrays (x_face, x_center, dx)
- `minipbl/state.py` — 2D field arrays (u, w, p, K_m) and log wind initialization
- `minipbl/advection.py` — Flux-form centered advection operators for theta, u, w
- `minipbl/diffusion.py` — Horizontal diffusion; vectorized 2D vertical diffusion
- `minipbl/boundary.py` — Rigid-lid and 2D top-gradient boundary conditions
- `minipbl/turbulence.py` — Column-by-column K-profile for 2D with K_m output
- `minipbl/timestepper.py` — RK3 with fractional-step pressure projection
- `minipbl/solver.py` — 2D tendency assembly (advection + diffusion + buoyancy + Coriolis), progress logging
- `minipbl/output.py` — 2D NetCDF writer with x dimension and velocity fields
- `minipbl/plotting.py` — 2D cross-section and x-averaged profile plots

## Dependencies
- numpy
- scipy
- matplotlib
- pyyaml

---

# miniPBL v1.0.0

Initial release of miniPBL, a lightweight Python solver for planetary boundary layer simulation.

## Features

### Physics
- 1D vertical column model for the convective boundary layer (CBL)
- Prognostic potential temperature (theta) equation driven by vertical turbulent heat flux divergence
- K-profile turbulence closure with convective velocity scaling
- Automatic boundary layer height diagnosis with linear interpolation
- Prescribed surface kinematic heat flux (lower boundary)
- Zero-flux insulating lid (upper boundary)
- Optional fixed lapse rate enforcement at the domain top

### Numerics
- Staggered vertical grid (scalars at cell centers, fluxes at cell faces)
- Second-order centered finite differences for diffusion
- Third-order Runge-Kutta (RK3) time integration (Wicker & Skamarock 2002)

### Configuration
- YAML-based configuration for grid, physics, turbulence, time stepping, and output settings
- Input validation with descriptive error messages

### Output
- NetCDF output of theta, heat flux, eddy diffusivity, and boundary layer height at configurable intervals
- Automatic diagnostic plots: theta profiles, BL height time series, and heat flux profiles

## Dependencies
- numpy
- scipy
- matplotlib
- pyyaml
