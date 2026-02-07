"""NetCDF output writer using scipy.io.netcdf."""

import os
from typing import List

import numpy as np
from scipy.io import netcdf_file

from .config import SimConfig
from .grid import Grid
from .state import State


class NetCDFWriter:
    """Accumulates snapshots in memory and writes a NetCDF file at close."""

    def __init__(self, cfg: SimConfig, grid: Grid):
        os.makedirs(cfg.output.output_dir, exist_ok=True)
        self.filepath = os.path.join(cfg.output.output_dir, "cbl_output.nc")
        self.grid = grid

        self._times: List[float] = []
        self._theta: List[np.ndarray] = []
        self._heat_flux: List[np.ndarray] = []
        self._K_h: List[np.ndarray] = []
        self._bl_height: List[float] = []

    def write(self, state: State, time: float):
        """Append a snapshot to the in-memory buffer."""
        self._times.append(time)
        self._theta.append(state.theta.copy())
        self._heat_flux.append(state.heat_flux.copy())
        self._K_h.append(state.K_h.copy())
        self._bl_height.append(state.bl_height)

    def close(self):
        """Write all accumulated data to a NetCDF file."""
        nt = len(self._times)
        nc = netcdf_file(self.filepath, "w")

        nc.createDimension("time", nt)
        nc.createDimension("z_center", self.grid.nz)
        nc.createDimension("z_face", self.grid.nz + 1)

        # Coordinate variables
        t_var = nc.createVariable("time", "f8", ("time",))
        t_var[:] = np.array(self._times)
        t_var.units = "s"

        z_c = nc.createVariable("z_center", "f8", ("z_center",))
        z_c[:] = self.grid.z_center
        z_c.units = "m"

        z_f = nc.createVariable("z_face", "f8", ("z_face",))
        z_f[:] = self.grid.z_face
        z_f.units = "m"

        # Data variables
        theta_var = nc.createVariable("theta", "f8", ("time", "z_center"))
        theta_var[:] = np.array(self._theta)

        hf_var = nc.createVariable("heat_flux", "f8", ("time", "z_face"))
        hf_var[:] = np.array(self._heat_flux)

        kh_var = nc.createVariable("K_h", "f8", ("time", "z_face"))
        kh_var[:] = np.array(self._K_h)

        bl_var = nc.createVariable("bl_height", "f8", ("time",))
        bl_var[:] = np.array(self._bl_height)

        nc.close()
