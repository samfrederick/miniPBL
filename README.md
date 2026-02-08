# miniPBL

A lightweight Python solver for simulating the planetary boundary layer (PBL). miniPBL is designed for learning, prototyping turbulence parameterizations, and experimenting with boundary layer dynamics before implementing in larger models.

## Current Capabilities

### Physics
- **1D vertical column model** simulating the convective boundary layer (CBL)
- **2D x-z Boussinesq solver** with resolved convection, momentum equations, and pressure projection
- Prognostic equations for potential temperature (theta), horizontal velocity (u), and vertical velocity (w)
- **K-profile turbulence closure** with convective velocity scaling (w*) and automatic boundary layer height diagnosis, applied column-by-column in 2D
- Eddy diffusivity for both heat (K_h) and momentum (K_m) with configurable ratio
- Horizontal diffusion for numerical stability of resolved motions
- Buoyancy forcing on vertical velocity from potential temperature perturbations
- Coriolis forcing with prescribed geostrophic wind
- Logarithmic wind profile initialization with configurable surface roughness length
- **Third-order Runge-Kutta (RK3)** time integration (Wicker & Skamarock 2002)
- Fractional-step pressure projection at each RK3 sub-stage for incompressibility

### Numerics
- **Arakawa C-grid** with scalars at cell centers, u at x-faces, w at z-faces
- Second-order centered finite differences for diffusion (vertical and horizontal)
- Flux-form centered advection for all prognostic variables
- **Pressure Poisson solver**: FFT in x (periodic) + tridiagonal solve in z with Neumann BCs
- Periodic boundary conditions in x, rigid-lid (w=0) at top and bottom
- No-slip surface stress parameterization for momentum
- Configurable grid resolution, domain size, timestep, and simulation duration

### Configuration
- YAML-based configuration system (see `config/cbl_1d.yaml` and `config/cbl_2d.yaml`)
- Configurable parameters include grid dimensions, surface heat flux, initial thermodynamic profile, turbulence scheme settings, geostrophic wind, Coriolis parameter, and output options
- Automatic dimensionality detection: setting `nx > 1` enables the 2D solver

### Output
- NetCDF output via SciPy with snapshots of theta, u, w, pressure, heat flux, eddy diffusivity, and boundary layer height
- Automatic generation of diagnostic plots:
  - **1D**: theta profiles, BL height time series, heat flux profiles
  - **2D**: x-z cross-sections of theta/u/w at selected times, x-averaged theta profiles, BL height time series
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
```

Output files are written to the `output/` directory.

## Project Structure
```
miniPBL/
├── run.py                  # Entry point
├── config/
│   ├── cbl_1d.yaml         # 1D column configuration
│   └── cbl_2d.yaml         # 2D x-z configuration
├── minipbl/
│   ├── __init__.py
│   ├── config.py            # YAML config loading and validation
│   ├── grid.py              # Arakawa C-grid construction (1D/2D)
│   ├── state.py             # Prognostic state (theta, u, w, p)
│   ├── solver.py            # Main simulation loop and tendency assembly
│   ├── turbulence.py        # Turbulence closure (K-profile, column-by-column)
│   ├── diffusion.py         # Vertical + horizontal diffusion operators
│   ├── advection.py         # Flux-form centered advection (2D)
│   ├── pressure.py          # Poisson solver and velocity projection
│   ├── boundary.py          # Boundary condition application
│   ├── timestepper.py       # RK3 time integration with pressure projection
│   ├── output.py            # NetCDF writer (1D and 2D)
│   └── plotting.py          # Diagnostic plot generation (1D and 2D)
└── requirements.txt
```

## Roadmap

- **Additional turbulence closures** — TKE-based and other schemes beyond the current K-profile parameterization
- **3D extension** — extend to full 3D grids with y-direction and prognostic v-equation
- **Moisture and tracers** — additional prognostic variables beyond potential temperature
- **Spatially and temporally varying surface fluxes** — support heterogeneous and time-dependent lower boundary forcing
- **Large-scale forcings** — prescribed subsidence and radiative tendencies
- **Higher-order advection** — upwind-biased or WENO schemes for improved accuracy
