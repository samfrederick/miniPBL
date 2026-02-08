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
