"""Post-run visualization of CBL simulation output."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import netcdf_file


def plot_results(filepath: str, output_dir: str = "output"):
    """Generate standard diagnostic plots from NetCDF output."""
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
