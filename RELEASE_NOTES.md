# miniPBL v1.0.0

Initial release of miniPBL, a lightweight Python solver for planetary boundary layer simulation.

## Features

### Physics
- 1D vertical column model for the convective boundary layer (CBL)
- Prognostic potential temperature (θ) equation driven by vertical turbulent heat flux divergence
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
- NetCDF output of θ, heat flux, eddy diffusivity, and boundary layer height at configurable intervals
- Automatic diagnostic plots: θ profiles, BL height time series, and heat flux profiles

## Dependencies
- numpy
- scipy
- matplotlib
- pyyaml
