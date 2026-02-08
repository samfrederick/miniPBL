"""Pressure Poisson solver for enforcing incompressibility.

FFT in x (periodic) + tridiagonal solve in z per wavenumber.
Neumann BCs in z (dphi/dz = 0 at rigid boundaries).
"""

import numpy as np
from scipy.linalg import solve_banded


class PoissonSolver:
    """Precomputed Poisson solver for 2D (x-z) with periodic x, rigid-lid z."""

    def __init__(self, grid):
        nx, nz = grid.nx, grid.nz
        dx, dz = grid.dx, grid.dz

        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz

        # Eigenvalues of the discrete second-derivative in x (periodic, FFT)
        # lambda_k = (2/dx^2) * (cos(2*pi*k/nx) - 1)
        k = np.arange(nx)
        self.lambda_x = 2.0 / (dx * dx) * (np.cos(2.0 * np.pi * k / nx) - 1.0)

    def solve(self, rhs):
        """Solve the Poisson equation  nabla^2 phi = rhs.

        rhs: (nx, nz) — divergence field at cell centers.
        Returns phi: (nx, nz).
        """
        nx, nz = self.nx, self.nz
        dz = self.dz

        # Forward FFT in x
        rhs_hat = np.fft.fft(rhs, axis=0)  # (nx, nz) complex

        phi_hat = np.zeros_like(rhs_hat)

        # For each wavenumber, solve tridiagonal system in z
        # d^2 phi / dz^2 + lambda_x * phi = rhs_hat
        # With Neumann BCs: dphi/dz = 0 at z=0 and z=Lz
        #
        # Interior: (phi[k-1] - 2*phi[k] + phi[k+1])/dz^2 + lam*phi[k] = rhs[k]
        # k=0 (Neumann bottom): (phi[1] - phi[0])/dz^2 + lam*phi[0] = rhs[0]
        #   equivalent to: (-1/dz^2 + lam)*phi[0] + (1/dz^2)*phi[1] = rhs[0]
        # k=nz-1 (Neumann top): (phi[nz-2] - phi[nz-1])/dz^2 + lam*phi[nz-1] = rhs[nz-1]

        dz2_inv = 1.0 / (dz * dz)

        for m in range(nx):
            lam = self.lambda_x[m]

            # Build tridiagonal: a (sub), b (main), c (super)
            a = np.zeros(nz)  # sub-diagonal (index 0 unused)
            b = np.zeros(nz)  # main diagonal
            c = np.zeros(nz)  # super-diagonal (index nz-1 unused)
            r = rhs_hat[m, :]  # right-hand side

            # Interior rows
            for k in range(1, nz - 1):
                a[k] = dz2_inv
                b[k] = -2.0 * dz2_inv + lam
                c[k] = dz2_inv

            # Bottom Neumann: ghost phi[-1] = phi[0] => phi[-1]-2*phi[0]+phi[1] = phi[0]-2*phi[0]+phi[1]
            b[0] = -dz2_inv + lam
            c[0] = dz2_inv

            # Top Neumann: ghost phi[nz] = phi[nz-1]
            a[nz - 1] = dz2_inv
            b[nz - 1] = -dz2_inv + lam

            # Handle kx=0 mode: singular system (constant null space)
            # Pin phi[0] = 0 to remove the singularity
            if m == 0:
                b[0] = 1.0
                c[0] = 0.0
                r = r.copy()
                r[0] = 0.0

            # Pack into banded form for solve_banded: ab[0]=super, ab[1]=main, ab[2]=sub
            ab = np.zeros((3, nz), dtype=complex)
            ab[0, 1:] = c[:nz - 1]   # super-diagonal
            ab[1, :] = b             # main diagonal
            ab[2, :nz - 1] = a[1:]   # sub-diagonal

            phi_hat[m, :] = solve_banded((1, 1), ab, r)

        # Inverse FFT in x
        phi = np.fft.ifft(phi_hat, axis=0).real

        return phi


class PoissonSolver3D:
    """Precomputed Poisson solver for 3D (x-y-z) with periodic x,y and rigid-lid z."""

    def __init__(self, grid):
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        dx, dy, dz = grid.dx, grid.dy, grid.dz

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz

        # Eigenvalues of discrete second-derivative in x and y
        kx = np.arange(nx)
        ky = np.arange(ny)
        lambda_x = 2.0 / (dx * dx) * (np.cos(2.0 * np.pi * kx / nx) - 1.0)
        lambda_y = 2.0 / (dy * dy) * (np.cos(2.0 * np.pi * ky / ny) - 1.0)
        # Combined eigenvalue for each (kx, ky) pair: (nx, ny)
        self.lambda_xy = lambda_x[:, np.newaxis] + lambda_y[np.newaxis, :]

    def solve(self, rhs):
        """Solve nabla^2 phi = rhs.

        rhs: (nx, ny, nz) — divergence field at cell centers.
        Returns phi: (nx, ny, nz).
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        dz = self.dz

        # Forward 2D FFT in x and y
        rhs_hat = np.fft.fft2(rhs, axes=(0, 1))  # (nx, ny, nz) complex

        phi_hat = np.zeros_like(rhs_hat)

        dz2_inv = 1.0 / (dz * dz)

        for m in range(nx):
            for n in range(ny):
                lam = self.lambda_xy[m, n]

                a = np.zeros(nz)
                b = np.zeros(nz)
                c = np.zeros(nz)
                r = rhs_hat[m, n, :]

                # Interior rows
                for k in range(1, nz - 1):
                    a[k] = dz2_inv
                    b[k] = -2.0 * dz2_inv + lam
                    c[k] = dz2_inv

                # Bottom Neumann
                b[0] = -dz2_inv + lam
                c[0] = dz2_inv

                # Top Neumann
                a[nz - 1] = dz2_inv
                b[nz - 1] = -dz2_inv + lam

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

        # Inverse 2D FFT
        phi = np.fft.ifft2(phi_hat, axes=(0, 1)).real

        return phi


def project_velocity_3d(u_star, v_star, w_star, grid, poisson_solver, dt):
    """Project provisional velocity to divergence-free field in 3D.

    Returns (u, v, w, p).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    # 3D divergence at cell centers
    div = ((np.roll(u_star, -1, axis=0) - u_star) / dx
           + (np.roll(v_star, -1, axis=1) - v_star) / dy
           + (w_star[:, :, 1:] - w_star[:, :, :-1]) / dz)

    phi = poisson_solver.solve(div / dt)

    # Correct velocities
    dphi_dx = (phi - np.roll(phi, 1, axis=0)) / dx
    u = u_star - dt * dphi_dx

    dphi_dy = (phi - np.roll(phi, 1, axis=1)) / dy
    v = v_star - dt * dphi_dy

    dphi_dz = np.zeros((nx, ny, nz + 1))
    dphi_dz[:, :, 1:nz] = (phi[:, :, 1:nz] - phi[:, :, :nz-1]) / dz
    w = w_star - dt * dphi_dz

    return u, v, w, phi * dt


def project_velocity(u_star, w_star, grid, poisson_solver, dt):
    """Project provisional velocity to divergence-free field.

    u_star: (nx, nz) — provisional u at x-faces
    w_star: (nx, nz+1) — provisional w at z-faces
    Returns (u, w, p).
    """
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz

    # Compute divergence of (u_star, w_star) at cell centers
    # du/dx: u at right face minus u at left face
    div = (np.roll(u_star, -1, axis=0) - u_star) / dx \
        + (w_star[:, 1:] - w_star[:, :-1]) / dz

    # Solve Poisson equation: nabla^2 phi = div / dt
    phi = poisson_solver.solve(div / dt)

    # Correct velocities: u = u_star - dt * dphi/dx, w = w_star - dt * dphi/dz
    # dphi/dx at x-faces (between cell centers i-1 and i, periodic)
    dphi_dx = (phi - np.roll(phi, 1, axis=0)) / dx  # (nx, nz)
    u = u_star - dt * dphi_dx

    # dphi/dz at z-faces (between cell centers k-1 and k)
    dphi_dz = np.zeros((nx, nz + 1))
    dphi_dz[:, 1:nz] = (phi[:, 1:nz] - phi[:, :nz-1]) / dz
    # dphi/dz = 0 at boundaries (Neumann)
    w = w_star - dt * dphi_dz

    return u, w, phi * dt  # return pressure = dt * phi for diagnostics
