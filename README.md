# miniPBL

A lightweight Python solver for simulating the planetary boundary layer (PBL). miniPBL is designed for learning, prototyping turbulence parameterizations, and experimenting with boundary layer dynamics before implementing in larger models.

## Current Capabilities

### Physics
- **1D vertical column model** simulating the convective boundary layer (CBL)
- **2D x-z Boussinesq solver** with resolved convection, momentum equations, and pressure projection
- **3D x-y-z Boussinesq solver** with prognostic u, v, w velocities and full pressure projection
- Prognostic equations for potential temperature (theta), horizontal velocities (u, v), vertical velocity (w), and optionally turbulent kinetic energy (TKE)
- **Two turbulence closures**:
  - **K-profile** — diagnostic closure with convective velocity scaling (w*) and automatic boundary layer height diagnosis
  - **Deardorff TKE** — prognostic subgrid TKE with shear/buoyancy production, dissipation, and stability-dependent mixing length; eddy viscosity K_m = c_m * l * sqrt(e) and diffusivity K_h = (1 + 2l/Delta) * K_m; inner column loops compiled to machine code via Numba `@njit` for ~200x speedup
- **Monin-Obukhov Similarity Theory (MOST)** surface layer — iterative solver for u_star and theta_star with Businger-Dyer stability functions; replaces prescribed surface fluxes with interactive fluxes based on near-surface wind and temperature; includes Beljaars (1994) convective velocity scale to prevent flux collapse in free-convective conditions
- **Rayleigh sponge damping** in the upper domain (configurable fraction and strength) to absorb gravity waves and prevent spurious reflections
- **Large-scale subsidence** — prescribed w_s(z) = -D*z profile with configurable divergence rate to limit boundary layer growth
- Buoyancy forcing on vertical velocity from potential temperature perturbations
- Coriolis forcing on u and v with prescribed geostrophic wind (Ekman balance)
- Logarithmic wind profile initialization with configurable surface roughness length
- **Third-order Runge-Kutta (RK3)** time integration (Wicker & Skamarock 2002)
- Fractional-step pressure projection at each RK3 sub-stage for incompressibility

### Numerics
- **Arakawa C-grid** with scalars at cell centers, u at x-faces, v at y-faces, w at z-faces
- **Vertical grid stretching** — optional geometric stretching with configurable stretch factor and uniform near-surface region; all operators support variable dz via `dz_center[k]` and `dz_face[k]` arrays
- Second-order centered finite differences for diffusion (vertical and horizontal)
- **Two advection schemes**:
  - **Centered** — 2nd-order flux-form centered advection (requires explicit horizontal diffusion K_horizontal to suppress grid-scale noise)
  - **5th-order upwind** — Wicker & Skamarock (2002) upwind-biased advection for horizontal fluxes with implicit numerical dissipation; vertical fluxes remain 2nd-order centered. Eliminates the need for explicit horizontal diffusion.
- **Pressure Poisson solver**: FFT in x (periodic) + tridiagonal in z (2D), or 2D FFT in x-y + tridiagonal in z (3D), with Neumann BCs and variable-dz tridiagonal coefficients
- Periodic boundary conditions in x and y, rigid-lid (w=0) at top and bottom
- Surface stress from MOST or no-slip parameterization for momentum (u and v)
- Configurable grid resolution, domain size, timestep, and simulation duration

### Configuration
- YAML-based configuration system (see `config/cbl_1d.yaml`, `config/cbl_2d.yaml`, and `config/cbl_3d.yaml`)
- Configurable parameters include grid dimensions, vertical stretching, surface flux scheme, turbulence closure, advection scheme, sponge layer, subsidence, geostrophic wind, Coriolis parameter, and output options
- Automatic dimensionality detection: `nx > 1` enables 2D; `nx > 1` and `ny > 1` enables 3D

### Output
- NetCDF output via SciPy with snapshots of theta, u, v, w, pressure, heat flux, eddy diffusivity, and boundary layer height
- Solver stores arrays as `(nx, ny, nz)` internally; NetCDF transposes to `(time, z, y, x)` for VisIT compatibility
- Automatic generation of diagnostic plots:
  - **1D**: theta profiles, BL height time series, heat flux profiles
  - **2D**: x-z cross-sections of theta/u/w at selected times, x-averaged theta profiles, BL height time series
  - **3D**: x-z cross-sections at mid-y (theta/u/v/w), x-y horizontal slices at mid-z and near-surface, xy-averaged theta profiles, BL height time series
- Progress log file (`output/progress.log`) for monitoring long simulations

## Getting Started

### Installation
```bash
pip install -r requirements.txt   # includes numba for JIT-compiled TKE closure
```

### Running
```bash
# Run 1D convective boundary layer
python run.py config/cbl_1d.yaml

# Run 2D resolved convection
python run.py config/cbl_2d.yaml

# Run 3D resolved convection
python run.py config/cbl_3d.yaml
```

Output files are written to the `output/` directory.

## Project Structure
```
miniPBL/
├── run.py                  # Entry point
├── config/
│   ├── cbl_1d.yaml         # 1D column configuration
│   ├── cbl_2d.yaml         # 2D x-z configuration
│   ├── cbl_2d_minimal.yaml # 2D minimal config (prescribed flux, no MOST/sponge/subsidence)
│   └── cbl_3d.yaml         # 3D x-y-z configuration
├── minipbl/
│   ├── __init__.py
│   ├── config.py            # YAML config loading and validation
│   ├── grid.py              # Arakawa C-grid with optional vertical stretching
│   ├── state.py             # Prognostic state (theta, u, v, w, p, tke)
│   ├── solver.py            # Main simulation loop and tendency assembly
│   ├── turbulence.py        # K-profile turbulence closure (column-by-column)
│   ├── tke_closure.py       # Deardorff prognostic TKE closure (Numba JIT-compiled)
│   ├── surface_layer.py     # Monin-Obukhov similarity theory surface fluxes
│   ├── forcing.py           # Sponge/Rayleigh damping and large-scale subsidence
│   ├── diffusion.py         # Vertical + horizontal diffusion operators
│   ├── advection.py         # Flux-form advection: centered and 5th-order upwind (2D/3D)
│   ├── pressure.py          # Poisson solver and velocity projection (2D/3D)
│   ├── boundary.py          # Boundary condition application
│   ├── timestepper.py       # RK3 time integration with pressure projection
│   ├── output.py            # NetCDF writer (1D, 2D, and 3D)
│   └── plotting.py          # Diagnostic plot generation (1D, 2D, and 3D)
└── requirements.txt
```

## Roadmap

- **Moisture and tracers** — additional prognostic variables beyond potential temperature
- **Spatially and temporally varying surface fluxes** — support heterogeneous and time-dependent lower boundary forcing
- **Radiative tendencies** — prescribed or interactive radiation schemes
- **Higher-order advection** — WENO schemes for improved accuracy
