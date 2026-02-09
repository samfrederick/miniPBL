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
        self.dim = grid.dim

        self._times: List[float] = []
        self._theta: List[np.ndarray] = []
        self._heat_flux: List[np.ndarray] = []
        self._K_h: List[np.ndarray] = []
        self._bl_height: List = []

        self._tke: List[np.ndarray] = []

        # 2D/3D-specific buffers
        if self.dim >= 2:
            self._u: List[np.ndarray] = []
            self._w: List[np.ndarray] = []
            self._p: List[np.ndarray] = []
        if self.dim >= 3:
            self._v: List[np.ndarray] = []

    def write(self, state: State, time: float):
        """Append a snapshot to the in-memory buffer."""
        self._times.append(time)
        self._theta.append(state.theta.copy())
        self._heat_flux.append(state.heat_flux.copy())
        self._K_h.append(state.K_h.copy())
        if hasattr(state, 'tke') and state.tke is not None:
            self._tke.append(state.tke.copy())

        if self.dim >= 3:
            self._bl_height.append(state.bl_height.copy())
            self._u.append(state.u.copy())
            self._v.append(state.v.copy())
            self._w.append(state.w.copy())
            self._p.append(state.p.copy())
        elif self.dim >= 2:
            self._bl_height.append(state.bl_height.copy())
            self._u.append(state.u.copy())
            self._w.append(state.w.copy())
            self._p.append(state.p.copy())
        else:
            self._bl_height.append(state.bl_height)

    def close(self):
        """Write all accumulated data to a NetCDF file."""
        if self.dim >= 3:
            self._close_3d()
        elif self.dim >= 2:
            self._close_2d()
        else:
            self._close_1d()

    def _close_1d(self):
        """Write 1D output (unchanged from v1.0.0)."""
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

    def _close_2d(self):
        """Write 2D output with x dimension and velocity fields."""

        nt = len(self._times)
        nx = self.grid.nx
        nz = self.grid.nz
        nc = netcdf_file(self.filepath, "w")

        # --------------------
        # Dimensions
        # --------------------
        nc.createDimension("time", nt)
        nc.createDimension("x_center", nx)
        nc.createDimension("z_center", nz)
        nc.createDimension("z_face", nz + 1)

        # --------------------
        # Coordinate variables
        # --------------------
        t_var = nc.createVariable("time", "f8", ("time",))
        t_var[:] = np.array(self._times)
        t_var.units = "s"
        t_var.axis = "T"

        x_c = nc.createVariable("x_center", "f8", ("x_center",))
        x_c[:] = self.grid.x_center
        x_c.units = "m"
        x_c.axis = "X"
        x_c.standard_name = "projection_x_coordinate"

        z_c = nc.createVariable("z_center", "f8", ("z_center",))
        z_c[:] = self.grid.z_center
        z_c.units = "m"
        z_c.axis = "Z"
        z_c.standard_name = "height"
        z_c.positive = "up"

        z_f = nc.createVariable("z_face", "f8", ("z_face",))
        z_f[:] = self.grid.z_face
        z_f.units = "m"
        z_f.axis = "Z"
        z_f.standard_name = "height"
        z_f.positive = "up"

        # --------------------
        # Prognostic / diagnostic variables
        # NetCDF dimension order: (time, z, x)
        # Solver storage order: (time, x, z)
        # --------------------

        # Theta
        theta_var = nc.createVariable(
            "theta", "f8",
            ("time", "z_center", "x_center")
        )
        theta_var[:] = np.asarray(self._theta).transpose(0, 2, 1)
        theta_var.coordinates = "time z_center x_center"

        # u velocity
        u_var = nc.createVariable(
            "u", "f8",
            ("time", "z_center", "x_center")
        )
        u_var[:] = np.asarray(self._u).transpose(0, 2, 1)
        u_var.coordinates = "time z_center x_center"

        # w velocity (defined on faces)
        w_var = nc.createVariable(
            "w", "f8",
            ("time", "z_face", "x_center")
        )
        w_var[:] = np.asarray(self._w).transpose(0, 2, 1)
        w_var.coordinates = "time z_face x_center"

        # pressure
        p_var = nc.createVariable(
            "p", "f8",
            ("time", "z_center", "x_center")
        )
        p_var[:] = np.asarray(self._p).transpose(0, 2, 1)
        p_var.coordinates = "time z_center x_center"

        # heat flux (faces)
        hf_var = nc.createVariable(
            "heat_flux", "f8",
            ("time", "z_face", "x_center")
        )
        hf_var[:] = np.asarray(self._heat_flux).transpose(0, 2, 1)
        hf_var.coordinates = "time z_face x_center"

        # K_h (faces)
        kh_var = nc.createVariable(
            "K_h", "f8",
            ("time", "z_face", "x_center")
        )
        kh_var[:] = np.asarray(self._K_h).transpose(0, 2, 1)
        kh_var.coordinates = "time z_face x_center"

        # TKE (centers)
        if self._tke:
            tke_var = nc.createVariable(
                "tke", "f8",
                ("time", "z_center", "x_center")
            )
            tke_var[:] = np.asarray(self._tke).transpose(0, 2, 1)
            tke_var.coordinates = "time z_center x_center"

        # Boundary-layer height
        bl_var = nc.createVariable(
            "bl_height", "f8",
            ("time", "x_center")
        )
        bl_var[:] = np.asarray(self._bl_height)
        bl_var.coordinates = "time x_center"

        nc.close()

    def _close_3d(self):
        """Write 3D output with x, y dimensions and velocity fields."""

        nt = len(self._times)
        nx = self.grid.nx
        ny = self.grid.ny
        nz = self.grid.nz
        nc = netcdf_file(self.filepath, "w")

        # --------------------
        # Dimensions
        # --------------------
        nc.createDimension("time", nt)
        nc.createDimension("x_center", nx)
        nc.createDimension("y_center", ny)
        nc.createDimension("z_center", nz)
        nc.createDimension("z_face", nz + 1)

        # --------------------
        # Coordinate variables
        # --------------------
        t_var = nc.createVariable("time", "f8", ("time",))
        t_var[:] = np.array(self._times)
        t_var.units = "s"
        t_var.axis = "T"

        x_c = nc.createVariable("x_center", "f8", ("x_center",))
        x_c[:] = self.grid.x_center
        x_c.units = "m"
        x_c.axis = "X"
        x_c.standard_name = "projection_x_coordinate"

        y_c = nc.createVariable("y_center", "f8", ("y_center",))
        y_c[:] = self.grid.y_center
        y_c.units = "m"
        y_c.axis = "Y"
        y_c.standard_name = "projection_y_coordinate"

        z_c = nc.createVariable("z_center", "f8", ("z_center",))
        z_c[:] = self.grid.z_center
        z_c.units = "m"
        z_c.axis = "Z"
        z_c.standard_name = "height"
        z_c.positive = "up"

        z_f = nc.createVariable("z_face", "f8", ("z_face",))
        z_f[:] = self.grid.z_face
        z_f.units = "m"
        z_f.axis = "Z"
        z_f.standard_name = "height"
        z_f.positive = "up"

        # --------------------
        # Prognostic / diagnostic variables
        # NetCDF dimension order: (time, z, y, x)
        # Solver storage order: (time, x, y, z)
        # Transpose: (nt, nx, ny, nz) -> (nt, nz, ny, nx) via .transpose(0, 3, 2, 1)
        # --------------------

        # Theta
        theta_var = nc.createVariable(
            "theta", "f8",
            ("time", "z_center", "y_center", "x_center")
        )
        theta_var[:] = np.asarray(self._theta).transpose(0, 3, 2, 1)
        theta_var.coordinates = "time z_center y_center x_center"

        # u velocity
        u_var = nc.createVariable(
            "u", "f8",
            ("time", "z_center", "y_center", "x_center")
        )
        u_var[:] = np.asarray(self._u).transpose(0, 3, 2, 1)
        u_var.coordinates = "time z_center y_center x_center"

        # v velocity
        v_var = nc.createVariable(
            "v", "f8",
            ("time", "z_center", "y_center", "x_center")
        )
        v_var[:] = np.asarray(self._v).transpose(0, 3, 2, 1)
        v_var.coordinates = "time z_center y_center x_center"

        # w velocity (defined on faces)
        w_var = nc.createVariable(
            "w", "f8",
            ("time", "z_face", "y_center", "x_center")
        )
        w_var[:] = np.asarray(self._w).transpose(0, 3, 2, 1)
        w_var.coordinates = "time z_face y_center x_center"

        # pressure
        p_var = nc.createVariable(
            "p", "f8",
            ("time", "z_center", "y_center", "x_center")
        )
        p_var[:] = np.asarray(self._p).transpose(0, 3, 2, 1)
        p_var.coordinates = "time z_center y_center x_center"

        # heat flux (faces)
        hf_var = nc.createVariable(
            "heat_flux", "f8",
            ("time", "z_face", "y_center", "x_center")
        )
        hf_var[:] = np.asarray(self._heat_flux).transpose(0, 3, 2, 1)
        hf_var.coordinates = "time z_face y_center x_center"

        # K_h (faces)
        kh_var = nc.createVariable(
            "K_h", "f8",
            ("time", "z_face", "y_center", "x_center")
        )
        kh_var[:] = np.asarray(self._K_h).transpose(0, 3, 2, 1)
        kh_var.coordinates = "time z_face y_center x_center"

        # TKE (centers)
        if self._tke:
            tke_var = nc.createVariable(
                "tke", "f8",
                ("time", "z_center", "y_center", "x_center")
            )
            tke_var[:] = np.asarray(self._tke).transpose(0, 3, 2, 1)
            tke_var.coordinates = "time z_center y_center x_center"

        # Boundary-layer height: (nt, nx, ny) -> (nt, ny, nx)
        bl_var = nc.createVariable(
            "bl_height", "f8",
            ("time", "y_center", "x_center")
        )
        bl_var[:] = np.asarray(self._bl_height).transpose(0, 2, 1)
        bl_var.coordinates = "time y_center x_center"

        nc.close()
