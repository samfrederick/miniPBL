"""Pressure Poisson solver for enforcing incompressibility.

FFT in x (periodic) + tridiagonal solve in z per wavenumber.
Neumann BCs in z (dphi/dz = 0 at rigid boundaries).

Supports variable vertical spacing via grid.dz_center and grid.dz_face arrays.
"""

import numpy as np
from scipy.linalg import solve_banded


class PoissonSolver:
    """Precomputed Poisson solver for 2D (x-z) with periodic x, rigid-lid z."""

    def __init__(self, grid):
        nx, nz = grid.nx, grid.nz
        dx = grid.dx

        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz_center = grid.dz_center  # (nz,)
        self.dz_face = grid.dz_face      # (nz+1,)

        # Eigenvalues of the discrete second-derivative in x (periodic, FFT)
        k = np.arange(nx)
        self.lambda_x = 2.0 / (dx * dx) * (np.cos(2.0 * np.pi * k / nx) - 1.0)

    def solve(self, rhs):
        """Solve the Poisson equation  nabla^2 phi = rhs.

        rhs: (nx, nz) — divergence field at cell centers.
        Returns phi: (nx, nz).
        """
        nx, nz = self.nx, self.nz
        dz_c = self.dz_center
        dz_f = self.dz_face

        # Forward FFT in x
        rhs_hat = np.fft.fft(rhs, axis=0)

        phi_hat = np.zeros_like(rhs_hat)

        for m in range(nx):
            lam = self.lambda_x[m]

            a = np.zeros(nz)
            b = np.zeros(nz)
            c = np.zeros(nz)
            r = rhs_hat[m, :]

            # Interior rows: variable-dz tridiagonal coefficients
            # d/dz(dphi/dz) at center k:
            #   (1/dz_c[k]) * [ (phi[k+1]-phi[k])/dz_f[k+1] - (phi[k]-phi[k-1])/dz_f[k] ]
            for k in range(1, nz - 1):
                a[k] = 1.0 / (dz_c[k] * dz_f[k])
                c[k] = 1.0 / (dz_c[k] * dz_f[k + 1])
                b[k] = -(a[k] + c[k]) + lam

            # Bottom Neumann: ghost phi[-1] = phi[0]
            # d2phi/dz2 ~ (phi[1]-phi[0])/(dz_c[0]*dz_f[1])
            c[0] = 1.0 / (dz_c[0] * dz_f[1])
            b[0] = -c[0] + lam

            # Top Neumann: ghost phi[nz] = phi[nz-1]
            a[nz - 1] = 1.0 / (dz_c[nz - 1] * dz_f[nz - 1])
            b[nz - 1] = -a[nz - 1] + lam

            # Handle kx=0 mode: singular system — pin phi[0] = 0
            if m == 0:
                b[0] = 1.0
                c[0] = 0.0
                r = r.copy()
                r[0] = 0.0

            ab = np.zeros((3, nz), dtype=complex)
            ab[0, 1:] = c[:nz - 1]
            ab[1, :] = b
            ab[2, :nz - 1] = a[1:]

            phi_hat[m, :] = solve_banded((1, 1), ab, r)

        phi = np.fft.ifft(phi_hat, axis=0).real

        return phi


class PoissonSolver3D:
    """Precomputed Poisson solver for 3D (x-y-z) with periodic x,y and rigid-lid z."""

    def __init__(self, grid):
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        dx, dy = grid.dx, grid.dy

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz_center = grid.dz_center
        self.dz_face = grid.dz_face

        # Eigenvalues of discrete second-derivative in x and y
        kx = np.arange(nx)
        ky = np.arange(ny)
        lambda_x = 2.0 / (dx * dx) * (np.cos(2.0 * np.pi * kx / nx) - 1.0)
        lambda_y = 2.0 / (dy * dy) * (np.cos(2.0 * np.pi * ky / ny) - 1.0)
        self.lambda_xy = lambda_x[:, np.newaxis] + lambda_y[np.newaxis, :]

    def solve(self, rhs):
        """Solve nabla^2 phi = rhs.

        rhs: (nx, ny, nz) — divergence field at cell centers.
        Returns phi: (nx, ny, nz).
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        dz_c = self.dz_center
        dz_f = self.dz_face

        rhs_hat = np.fft.fft2(rhs, axes=(0, 1))

        phi_hat = np.zeros_like(rhs_hat)

        for m in range(nx):
            for n in range(ny):
                lam = self.lambda_xy[m, n]

                a = np.zeros(nz)
                b = np.zeros(nz)
                c = np.zeros(nz)
                r = rhs_hat[m, n, :]

                # Interior rows
                for k in range(1, nz - 1):
                    a[k] = 1.0 / (dz_c[k] * dz_f[k])
                    c[k] = 1.0 / (dz_c[k] * dz_f[k + 1])
                    b[k] = -(a[k] + c[k]) + lam

                # Bottom Neumann
                c[0] = 1.0 / (dz_c[0] * dz_f[1])
                b[0] = -c[0] + lam

                # Top Neumann
                a[nz - 1] = 1.0 / (dz_c[nz - 1] * dz_f[nz - 1])
                b[nz - 1] = -a[nz - 1] + lam

                # Handle kx=0, ky=0 mode: pin phi[0]=0
                if m == 0 and n == 0:
                    b[0] = 1.0
                    c[0] = 0.0
                    r = r.copy()
                    r[0] = 0.0

                ab = np.zeros((3, nz), dtype=complex)
                ab[0, 1:] = c[:nz - 1]
                ab[1, :] = b
                ab[2, :nz - 1] = a[1:]

                phi_hat[m, n, :] = solve_banded((1, 1), ab, r)

        phi = np.fft.ifft2(phi_hat, axes=(0, 1)).real

        return phi


def project_velocity_3d(u_star, v_star, w_star, grid, poisson_solver, dt):
    """Project provisional velocity to divergence-free field in 3D.

    Returns (u, v, w, p).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    dz_c = grid.dz_center
    dz_f = grid.dz_face

    # 3D divergence at cell centers
    div = ((np.roll(u_star, -1, axis=0) - u_star) / dx
           + (np.roll(v_star, -1, axis=1) - v_star) / dy
           + (w_star[:, :, 1:] - w_star[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :])

    phi = poisson_solver.solve(div / dt)

    # Correct velocities
    dphi_dx = (phi - np.roll(phi, 1, axis=0)) / dx
    u = u_star - dt * dphi_dx

    dphi_dy = (phi - np.roll(phi, 1, axis=1)) / dy
    v = v_star - dt * dphi_dy

    dphi_dz = np.zeros((nx, ny, nz + 1))
    dphi_dz[:, :, 1:nz] = (phi[:, :, 1:nz] - phi[:, :, :nz-1]) / dz_f[1:nz]
    w = w_star - dt * dphi_dz

    return u, v, w, phi * dt


def project_velocity(u_star, w_star, grid, poisson_solver, dt):
    """Project provisional velocity to divergence-free field.

    u_star: (nx, nz) — provisional u at x-faces
    w_star: (nx, nz+1) — provisional w at z-faces
    Returns (u, w, p).
    """
    nx, nz = grid.nx, grid.nz
    dx = grid.dx
    dz_c = grid.dz_center
    dz_f = grid.dz_face

    # Compute divergence of (u_star, w_star) at cell centers
    div = (np.roll(u_star, -1, axis=0) - u_star) / dx \
        + (w_star[:, 1:] - w_star[:, :-1]) / dz_c[np.newaxis, :]

    phi = poisson_solver.solve(div / dt)

    # Correct velocities
    dphi_dx = (phi - np.roll(phi, 1, axis=0)) / dx
    u = u_star - dt * dphi_dx

    dphi_dz = np.zeros((nx, nz + 1))
    dphi_dz[:, 1:nz] = (phi[:, 1:nz] - phi[:, :nz-1]) / dz_f[1:nz]
    w = w_star - dt * dphi_dz

    return u, w, phi * dt
