"""Post-run visualization of CBL simulation output."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import netcdf_file


def plot_results(filepath: str, output_dir: str = "output"):
    """Generate standard diagnostic plots from NetCDF output.

    Dispatches to 1D, 2D, or 3D plotting based on dimension presence.
    """
    nc = netcdf_file(filepath, "r", mmap=False)
    has_y = "y_center" in nc.dimensions
    has_x = "x_center" in nc.dimensions
    nc.close()
    if has_y:
        _plot_results_3d(filepath, output_dir)
    elif has_x:
        _plot_results_2d(filepath, output_dir)
    else:
        _plot_results_1d(filepath, output_dir)


# ---------------------------------------------------------------------------
# 1D plotting (unchanged from v1.0.0)
# ---------------------------------------------------------------------------

def _plot_results_1d(filepath, output_dir):
    nc = netcdf_file(filepath, "r", mmap=False)

    time = nc.variables["time"].data.copy()
    z_center = nc.variables["z_center"].data.copy()
    z_face = nc.variables["z_face"].data.copy()
    theta = nc.variables["theta"].data.copy()
    heat_flux = nc.variables["heat_flux"].data.copy()
    bl_height = nc.variables["bl_height"].data.copy()
    nc.close()

    nt = len(time)

    # Select indices at roughly 0, 1, 2, 3, 4 hours
    indices = [0]
    for target_t in [3600, 7200, 10800, 14400]:
        idx = int(np.argmin(np.abs(time - target_t)))
        if idx not in indices:
            indices.append(idx)
    if nt - 1 not in indices:
        indices.append(nt - 1)

    # --- Plot 1: Theta profile evolution ---
    fig, ax = plt.subplots(figsize=(6, 8))
    for i in sorted(indices):
        t_hr = time[i] / 3600.0
        ax.plot(theta[i, :], z_center, label=f"t = {t_hr:.1f} h")
    ax.set_xlabel("Potential Temperature (K)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Theta Profile Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/theta_profiles.png", dpi=150)
    print(f"  Saved {output_dir}/theta_profiles.png")

    # --- Plot 2: Boundary layer height vs time ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time / 3600.0, bl_height, "k-", linewidth=1.5)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("BL Height (m)")
    ax.set_title("Boundary Layer Height Evolution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/bl_height.png", dpi=150)
    print(f"  Saved {output_dir}/bl_height.png")

    # --- Plot 3: Heat flux profiles at selected times ---
    fig, ax = plt.subplots(figsize=(6, 8))
    for i in sorted(indices):
        t_hr = time[i] / 3600.0
        ax.plot(heat_flux[i, :], z_face, label=f"t = {t_hr:.1f} h")
    ax.set_xlabel("Heat Flux (K m/s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Heat Flux Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/heat_flux_profiles.png", dpi=150)
    print(f"  Saved {output_dir}/heat_flux_profiles.png")

    plt.close("all")


# ---------------------------------------------------------------------------
# 2D plotting
# ---------------------------------------------------------------------------

def _plot_results_2d(filepath, output_dir):
    nc = netcdf_file(filepath, "r", mmap=False)

    time = nc.variables["time"].data.copy()
    x_center = nc.variables["x_center"].data.copy()
    z_center = nc.variables["z_center"].data.copy()
    z_face = nc.variables["z_face"].data.copy()
    theta = nc.variables["theta"].data.copy()     # (nt, nz, nx)
    u = nc.variables["u"].data.copy()             # (nt, nz, nx)
    w = nc.variables["w"].data.copy()             # (nt, nz+1, nx)
    bl_height = nc.variables["bl_height"].data.copy()  # (nt, nx)
    nc.close()

    nt = len(time)
    x_km = x_center / 1000.0

    # Select time indices
    indices = [0]
    for target_t in [3600, 7200, 10800, 14400]:
        idx = int(np.argmin(np.abs(time - target_t)))
        if idx not in indices:
            indices.append(idx)
    if nt - 1 not in indices:
        indices.append(nt - 1)

    # --- x-z cross-sections at selected times ---
    for i in sorted(indices):
        t_hr = time[i] / 3600.0

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Theta
        ax = axes[0]
        pc = ax.pcolormesh(
            x_km,
            z_center,
            theta[i, :, :],     
            shading="auto",
            cmap="RdYlBu_r"
        )
        fig.colorbar(pc, ax=ax, label="K")
        ax.set_title(f"Theta  t={t_hr:.1f} h")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (m)")

        # u
        ax = axes[1]
        pc = ax.pcolormesh(
            x_km,
            z_center,
            u[i, :, :],         
            shading="auto",
            cmap="RdBu_r"
        )
        fig.colorbar(pc, ax=ax, label="m/s")
        ax.set_title(f"u  t={t_hr:.1f} h")
        ax.set_xlabel("x (km)")

        # w
        ax = axes[2]
        # Interpolate w to cell centers for plotting
        w_center = 0.5 * (w[i, :-1, :] + w[i, 1:, :])
        wmax = max(np.max(np.abs(w_center)), 1e-6)

        pc = ax.pcolormesh(
            x_km,
            z_center,
            w_center,           
            shading="auto",
            cmap="RdBu_r",
            vmin=-wmax,
            vmax=wmax
        )
        fig.colorbar(pc, ax=ax, label="m/s")
        ax.set_title(f"w  t={t_hr:.1f} h")
        ax.set_xlabel("x (km)")

        fig.tight_layout()
        fname = f"{output_dir}/xz_t{time[i]:07.0f}s.png"
        fig.savefig(fname, dpi=150)
        print(f"  Saved {fname}")

    # --- x-averaged theta profiles (comparison with 1D) ---
    fig, ax = plt.subplots(figsize=(6, 8))
    for i in sorted(indices):
        t_hr = time[i] / 3600.0
        theta_mean = np.mean(theta[i, :, :], axis=1)
        ax.plot(theta_mean, z_center, label=f"t = {t_hr:.1f} h")
    ax.set_xlabel("Potential Temperature (K)")
    ax.set_ylabel("Height (m)")
    ax.set_title("x-Averaged Theta Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/theta_profiles_xavg.png", dpi=150)
    print(f"  Saved {output_dir}/theta_profiles_xavg.png")

    # --- BL height vs time (x-averaged) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    bl_mean = np.mean(bl_height, axis=1)
    ax.plot(time / 3600.0, bl_mean, "k-", linewidth=1.5)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("BL Height (m)")
    ax.set_title("x-Averaged Boundary Layer Height")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/bl_height.png", dpi=150)
    print(f"  Saved {output_dir}/bl_height.png")

    plt.close("all")


# ---------------------------------------------------------------------------
# 3D plotting
# ---------------------------------------------------------------------------

def _plot_results_3d(filepath, output_dir):
    nc = netcdf_file(filepath, "r", mmap=False)

    time = nc.variables["time"].data.copy()
    x_center = nc.variables["x_center"].data.copy()
    y_center = nc.variables["y_center"].data.copy()
    z_center = nc.variables["z_center"].data.copy()
    z_face = nc.variables["z_face"].data.copy()
    theta = nc.variables["theta"].data.copy()       # (nt, nz, ny, nx)
    u = nc.variables["u"].data.copy()               # (nt, nz, ny, nx)
    v = nc.variables["v"].data.copy()               # (nt, nz, ny, nx)
    w = nc.variables["w"].data.copy()               # (nt, nz+1, ny, nx)
    bl_height = nc.variables["bl_height"].data.copy()  # (nt, ny, nx)
    nc.close()

    nt = len(time)
    nz = len(z_center)
    ny = len(y_center)
    nx = len(x_center)
    x_km = x_center / 1000.0
    y_km = y_center / 1000.0
    mid_y = ny // 2
    mid_z = nz // 2
    near_sfc_z = min(2, nz - 1)

    # Select time indices
    indices = [0]
    for target_t in [3600, 7200, 10800, 14400]:
        idx = int(np.argmin(np.abs(time - target_t)))
        if idx not in indices:
            indices.append(idx)
    if nt - 1 not in indices:
        indices.append(nt - 1)

    # --- x-z cross-sections at mid-y ---
    for i in sorted(indices):
        t_hr = time[i] / 3600.0

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Theta
        ax = axes[0]
        pc = ax.pcolormesh(x_km, z_center, theta[i, :, mid_y, :],
                           shading="auto", cmap="RdYlBu_r")
        fig.colorbar(pc, ax=ax, label="K")
        ax.set_title(f"Theta  t={t_hr:.1f} h")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (m)")

        # u
        ax = axes[1]
        pc = ax.pcolormesh(x_km, z_center, u[i, :, mid_y, :],
                           shading="auto", cmap="RdBu_r")
        fig.colorbar(pc, ax=ax, label="m/s")
        ax.set_title(f"u  t={t_hr:.1f} h")
        ax.set_xlabel("x (km)")

        # v
        ax = axes[2]
        pc = ax.pcolormesh(x_km, z_center, v[i, :, mid_y, :],
                           shading="auto", cmap="RdBu_r")
        fig.colorbar(pc, ax=ax, label="m/s")
        ax.set_title(f"v  t={t_hr:.1f} h")
        ax.set_xlabel("x (km)")

        # w
        ax = axes[3]
        w_center = 0.5 * (w[i, :-1, mid_y, :] + w[i, 1:, mid_y, :])
        wmax = max(np.max(np.abs(w_center)), 1e-6)
        pc = ax.pcolormesh(x_km, z_center, w_center,
                           shading="auto", cmap="RdBu_r",
                           vmin=-wmax, vmax=wmax)
        fig.colorbar(pc, ax=ax, label="m/s")
        ax.set_title(f"w  t={t_hr:.1f} h")
        ax.set_xlabel("x (km)")

        fig.tight_layout()
        fname = f"{output_dir}/xz_midy_t{time[i]:07.0f}s.png"
        fig.savefig(fname, dpi=150)
        print(f"  Saved {fname}")

    # --- x-y horizontal cross-sections at mid-z and near-surface ---
    for i in sorted(indices):
        t_hr = time[i] / 3600.0

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Theta near surface
        ax = axes[0, 0]
        pc = ax.pcolormesh(x_km, y_km, theta[i, near_sfc_z, :, :],
                           shading="auto", cmap="RdYlBu_r")
        fig.colorbar(pc, ax=ax, label="K")
        ax.set_title(f"Theta z={z_center[near_sfc_z]:.0f}m  t={t_hr:.1f}h")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

        # w at mid-z
        ax = axes[0, 1]
        w_center_xy = 0.5 * (w[i, mid_z, :, :] + w[i, mid_z + 1, :, :])
        wmax = max(np.max(np.abs(w_center_xy)), 1e-6)
        pc = ax.pcolormesh(x_km, y_km, w_center_xy,
                           shading="auto", cmap="RdBu_r",
                           vmin=-wmax, vmax=wmax)
        fig.colorbar(pc, ax=ax, label="m/s")
        ax.set_title(f"w z={z_center[mid_z]:.0f}m  t={t_hr:.1f}h")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

        # Theta at mid-z
        ax = axes[1, 0]
        pc = ax.pcolormesh(x_km, y_km, theta[i, mid_z, :, :],
                           shading="auto", cmap="RdYlBu_r")
        fig.colorbar(pc, ax=ax, label="K")
        ax.set_title(f"Theta z={z_center[mid_z]:.0f}m  t={t_hr:.1f}h")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

        # w near surface
        ax = axes[1, 1]
        w_sfc = 0.5 * (w[i, near_sfc_z, :, :] + w[i, near_sfc_z + 1, :, :])
        wmax_sfc = max(np.max(np.abs(w_sfc)), 1e-6)
        pc = ax.pcolormesh(x_km, y_km, w_sfc,
                           shading="auto", cmap="RdBu_r",
                           vmin=-wmax_sfc, vmax=wmax_sfc)
        fig.colorbar(pc, ax=ax, label="m/s")
        ax.set_title(f"w z={z_center[near_sfc_z]:.0f}m  t={t_hr:.1f}h")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

        fig.tight_layout()
        fname = f"{output_dir}/xy_t{time[i]:07.0f}s.png"
        fig.savefig(fname, dpi=150)
        print(f"  Saved {fname}")

    # --- xy-averaged theta profiles ---
    fig, ax = plt.subplots(figsize=(6, 8))
    for i in sorted(indices):
        t_hr = time[i] / 3600.0
        theta_mean = np.mean(theta[i, :, :, :], axis=(1, 2))
        ax.plot(theta_mean, z_center, label=f"t = {t_hr:.1f} h")
    ax.set_xlabel("Potential Temperature (K)")
    ax.set_ylabel("Height (m)")
    ax.set_title("xy-Averaged Theta Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/theta_profiles_xyavg.png", dpi=150)
    print(f"  Saved {output_dir}/theta_profiles_xyavg.png")

    # --- BL height vs time (xy-averaged) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    bl_mean = np.mean(bl_height, axis=(1, 2))
    ax.plot(time / 3600.0, bl_mean, "k-", linewidth=1.5)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("BL Height (m)")
    ax.set_title("xy-Averaged Boundary Layer Height")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/bl_height.png", dpi=150)
    print(f"  Saved {output_dir}/bl_height.png")

    plt.close("all")
