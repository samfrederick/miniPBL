"""Configuration dataclasses parsed from YAML."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class GridConfig:
    nz: int = 100
    Lz: float = 3000.0
    dim: int = 1


@dataclass
class PhysicsConfig:
    surface_heat_flux: float = 0.24
    theta_initial: float = 300.0
    brunt_vaisala_freq: float = 0.01
    reference_theta: float = 300.0
    mixed_layer_height: float = 1000.0
    lapse_rate: float = 0.003  # K/m above mixed layer


@dataclass
class TurbulenceConfig:
    scheme: str = "k-profile"
    k_profile_kappa: float = 0.4
    theta_excess_threshold: float = 0.5  # K
    background_K: float = 0.1  # m^2/s


@dataclass
class TimeConfig:
    dt: float = 1.0
    t_end: float = 14400.0
    output_interval: float = 300.0


@dataclass
class OutputConfig:
    output_dir: str = "output"
    fields_to_save: List[str] = field(
        default_factory=lambda: ["theta", "heat_flux", "K_h"]
    )


@dataclass
class SimConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str) -> SimConfig:
    """Load and validate a YAML configuration file into SimConfig."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return SimConfig()

    cfg = SimConfig()

    if "grid" in raw:
        cfg.grid = GridConfig(**raw["grid"])
    if "physics" in raw:
        cfg.physics = PhysicsConfig(**raw["physics"])
    if "turbulence" in raw:
        cfg.turbulence = TurbulenceConfig(**raw["turbulence"])
    if "time" in raw:
        cfg.time = TimeConfig(**raw["time"])
    if "output" in raw:
        cfg.output = OutputConfig(**raw["output"])

    # Validate
    assert cfg.grid.nz > 0, "nz must be positive"
    assert cfg.grid.Lz > 0, "Lz must be positive"
    assert cfg.grid.dim in (1, 2, 3), "dim must be 1, 2, or 3"
    assert cfg.time.dt > 0, "dt must be positive"
    assert cfg.time.t_end > cfg.time.dt, "t_end must exceed dt"

    return cfg
