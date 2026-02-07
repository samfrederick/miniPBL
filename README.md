# miniPBL

A lightweight Python solver for simulating the planetary boundary layer (PBL). miniPBL is designed for learning, prototyping turbulence parameterizations, and experimenting with boundary layer dynamics before implementing in larger models.

## Current Capabilities

### Physics
- **1D vertical column model** simulating the convective boundary layer (CBL)
- Solves the conservation equation for potential temperature driven by turbulent heat flux
- **K-profile turbulence closure** with convective velocity scaling (w*) and automatic boundary layer height diagnosis
- **Third-order Runge-Kutta (RK3)** time integration (Wicker & Skamarock 2002)
- Prescribed surface kinematic heat flux as the lower boundary condition
- Zero-flux upper boundary condition

### Numerics
- Staggered vertical grid with scalars at cell centers and fluxes at cell faces
- Second-order centered finite differences for vertical diffusion
- Configurable grid resolution, domain height, timestep, and simulation duration

### Configuration
- YAML-based configuration system (see `config/cbl_1d.yaml`)
- Configurable parameters include grid dimensions, surface heat flux, initial thermodynamic profile, turbulence scheme settings, and output options

### Output
- NetCDF output via SciPy with snapshots of potential temperature, heat flux, eddy diffusivity, and boundary layer height
- Automatic generation of diagnostic plots (theta profiles, BL height time series, heat flux profiles)

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Running
```bash
# Run with default config
python run.py

# Specify a config file
python run.py config/cbl_1d.yaml
```

Output files are written to the `output/` directory.

## Project Structure
```
miniPBL/
├── run.py                  # Entry point
├── config/
│   └── cbl_1d.yaml         # Default configuration
├── minipbl/
│   ├── __init__.py
│   ├── config.py            # YAML config loading and validation
│   ├── grid.py              # Staggered grid construction
│   ├── state.py             # Prognostic state initialization
│   ├── solver.py            # Main simulation loop
│   ├── turbulence.py        # Turbulence closure (K-profile)
│   ├── diffusion.py         # Vertical diffusion operator
│   ├── advection.py         # Advection operator (stub)
│   ├── boundary.py          # Boundary condition application
│   ├── timestepper.py       # RK3 time integration
│   ├── output.py            # NetCDF writer
│   └── plotting.py          # Diagnostic plot generation
└── requirements.txt
```

## Roadmap

- **Additional turbulence closures** — TKE-based and other schemes beyond the current K-profile parameterization
- **Large-eddy simulation (LES)** — extend to 2D and 3D grids with explicit resolution of turbulent eddies
- **Moisture and tracers** — additional prognostic variables beyond potential temperature
- **Advection** — implement mean-flow and resolved advection operators (interface exists in `advection.py`)
- **Spatially and temporally varying surface fluxes** — support heterogeneous and time-dependent lower boundary forcing
- **Large-scale forcings** — prescribed geostrophic wind, subsidence, and radiative tendencies
