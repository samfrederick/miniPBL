# miniPBL

A lightweight Python solver for simulating the planetary boundary layer (PBL). miniPBL is designed for learning, prototyping turbulence parameterizations, and experimenting with boundary layer dynamics before implementing in larger models.

## Current Capabilities

### Physics
- **1D vertical column model** simulating the convective boundary layer (CBL)
- **2D x-z Boussinesq solver** with resolved convection, momentum equations, and pressure projection
- **3D x-y-z Boussinesq solver** with prognostic u, v, w velocities and full pressure projection
- Prognostic equations for potential temperature (theta), horizontal velocities (u, v), and vertical velocity (w)
- **K-profile turbulence closure** with convective velocity scaling (w*) and automatic boundary layer height diagnosis, applied column-by-column in 2D and 3D
- Eddy diffusivity for both heat (K_h) and momentum (K_m) with configurable ratio
- Horizontal diffusion in x and y for numerical stability of resolved motions
- Buoyancy forcing on vertical velocity from potential temperature perturbations
- Coriolis forcing on u and v with prescribed geostrophic wind (Ekman balance)
- Logarithmic wind profile initialization with configurable surface roughness length
- **Third-order Runge-Kutta (RK3)** time integration (Wicker & Skamarock 2002)
- Fractional-step pressure projection at each RK3 sub-stage for incompressibility

### Numerics
- **Arakawa C-grid** with scalars at cell centers, u at x-faces, v at y-faces, w at z-faces
- Second-order centered finite differences for diffusion (vertical and horizontal)
- Flux-form centered advection for all prognostic variables
- **Pressure Poisson solver**: FFT in x (periodic) + tridiagonal in z (2D), or 2D FFT in x-y + tridiagonal in z (3D), with Neumann BCs
- Periodic boundary conditions in x and y, rigid-lid (w=0) at top and bottom
- No-slip surface stress parameterization for momentum (u and v)
- Configurable grid resolution, domain size, timestep, and simulation duration

### Configuration
- YAML-based configuration system (see `config/cbl_1d.yaml`, `config/cbl_2d.yaml`, and `config/cbl_3d.yaml`)
- Configurable parameters include grid dimensions, surface heat flux, initial thermodynamic profile, turbulence scheme settings, geostrophic wind, Coriolis parameter, and output options
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
pip install -r requirements.txt
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
│   └── cbl_3d.yaml         # 3D x-y-z configuration
├── minipbl/
│   ├── __init__.py
│   ├── config.py            # YAML config loading and validation
│   ├── grid.py              # Arakawa C-grid construction (1D/2D/3D)
│   ├── state.py             # Prognostic state (theta, u, v, w, p)
│   ├── solver.py            # Main simulation loop and tendency assembly
│   ├── turbulence.py        # Turbulence closure (K-profile, column-by-column)
│   ├── diffusion.py         # Vertical + horizontal diffusion operators
│   ├── advection.py         # Flux-form centered advection (2D/3D)
│   ├── pressure.py          # Poisson solver and velocity projection (2D/3D)
│   ├── boundary.py          # Boundary condition application
│   ├── timestepper.py       # RK3 time integration with pressure projection
│   ├── output.py            # NetCDF writer (1D, 2D, and 3D)
│   └── plotting.py          # Diagnostic plot generation (1D, 2D, and 3D)
└── requirements.txt
```

## Roadmap

- **Additional turbulence closures** — TKE-based and other schemes beyond the current K-profile parameterization
- **Monin-Obukhov similarity theory (MOST)** — surface layer parameterization for consistent surface fluxes
- **Moisture and tracers** — additional prognostic variables beyond potential temperature
- **Spatially and temporally varying surface fluxes** — support heterogeneous and time-dependent lower boundary forcing
- **Large-scale forcings** — prescribed subsidence and radiative tendencies
- **Higher-order advection** — upwind-biased or WENO schemes for improved accuracy
