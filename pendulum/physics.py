# pendulum/physics.py

"""Multi-link pendulum on a cart – Lagrangian dynamics.

Generalised coordinates:
    q = [x, θ₁, θ₂, …, θₙ]
where x is the horizontal cart position and θᵢ is the angle of link i
measured from the *upward* vertical (θ = 0 ⟹ balanced upright).

Sign convention: positive θ is clockwise from upright.

The engine builds the mass matrix **M(q)** and the force vector **f(q, q̇, u)**
so that the equations of motion are  M q̈ = f.  This formulation naturally
extends to any number of links.

Zero friction everywhere.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .config import PendulumConfig


def _build_mass_matrix(
    cfg: PendulumConfig,
    theta: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return the (1+N)×(1+N) mass matrix M(q).

    Parameters
    ----------
    cfg : PendulumConfig
    theta : array of shape (N,) – link angles
    """
    n = cfg.num_links
    M_cart = cfg.cart_mass
    dim = 1 + n
    M = np.zeros((dim, dim))

    lengths = np.asarray(cfg.link_lengths)
    masses = np.asarray(cfg.link_masses)
    # Half-lengths (pivot to centre-of-mass, assuming uniform rods)
    half = lengths / 2.0

    # Total mass felt by cart row / column
    total_mass = M_cart + masses.sum()
    M[0, 0] = total_mass

    for i in range(n):
        # Coupling between cart (index 0) and link i (index i+1)
        c = masses[i] * half[i] * np.cos(theta[i])
        # Add contributions from links further out that also depend on link i
        for j in range(i + 1, n):
            c += masses[j] * lengths[i] * np.cos(theta[i])
        M[0, i + 1] = c
        M[i + 1, 0] = c

    for i in range(n):
        for j in range(n):
            val = 0.0
            if i == j:
                # Diagonal: moment of inertia of link i about its pivot
                val = masses[i] * half[i] ** 2
                # Contributions of outer links through this pivot
                for k in range(i + 1, n):
                    val += masses[k] * lengths[i] ** 2
            else:
                # Off-diagonal coupling between links i and j
                ii, jj = min(i, j), max(i, j)
                # The inner link uses full length, outer uses half-length
                l_inner = lengths[ii]
                l_outer = half[jj]
                m_outer = masses[jj]
                val = m_outer * l_inner * l_outer * np.cos(theta[ii] - theta[jj])
                # Contributions of links further out
                for k in range(jj + 1, n):
                    val += masses[k] * lengths[ii] * lengths[jj] * np.cos(
                        theta[ii] - theta[jj]
                    )
            M[i + 1, j + 1] = val

    return M


def _gravity_vector(
    cfg: PendulumConfig,
    theta: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute  −∂V/∂q  (gravity generalised forces).

    V = Σᵢ mᵢ g yᵢ  where yᵢ is the height of each link's centre of mass.
    Returns the *negative* gradient so that  M q̈ = … + grav_vec + …
    """
    n = cfg.num_links
    g = cfg.gravity
    dim = 1 + n
    grav = np.zeros(dim)
    # ∂V/∂x = 0  →  grav[0] = 0
    lengths = np.asarray(cfg.link_lengths)
    masses = np.asarray(cfg.link_masses)
    half = lengths / 2.0

    for k in range(n):
        # −∂V/∂θ_k = g sin(θ_k) [m_k l_k/2 + l_k Σ_{j>k} m_j]
        val = masses[k] * g * half[k] * np.sin(theta[k])
        for j in range(k + 1, n):
            val += masses[j] * g * lengths[k] * np.sin(theta[k])
        grav[k + 1] = val

    return grav


def _coriolis_vector(
    cfg: PendulumConfig,
    theta: NDArray[np.floating],
    qdot: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the Coriolis / centripetal vector  C(q, q̇) q̇.

    Uses Christoffel symbols of the first kind computed via central finite
    differences of the mass matrix.  This is robust for any number of links.
    """
    n = cfg.num_links
    dim = 1 + n
    eps = 1e-7

    # ∂M/∂q_k  (M is independent of x, so index 0 is zero)
    dMdq = [np.zeros((dim, dim))]  # ∂M/∂x = 0
    for k in range(n):
        theta_p = theta.copy()
        theta_p[k] += eps
        theta_m = theta.copy()
        theta_m[k] -= eps
        dMdq.append(
            (_build_mass_matrix(cfg, theta_p) - _build_mass_matrix(cfg, theta_m))
            / (2.0 * eps)
        )

    # c_i = Σ_{j,k} Γ_{ijk} q̇_j q̇_k
    # Γ_{ijk} = ½ (∂M_{ij}/∂q_k + ∂M_{ik}/∂q_j − ∂M_{jk}/∂q_i)
    c = np.zeros(dim)
    for i in range(dim):
        ci = 0.0
        for j in range(dim):
            for k in range(dim):
                gamma = 0.5 * (
                    dMdq[k][i, j] + dMdq[j][i, k] - dMdq[i][j, k]
                )
                ci += gamma * qdot[j] * qdot[k]
        c[i] = ci
    return c


def _build_force_vector(
    cfg: PendulumConfig,
    theta: NDArray[np.floating],
    qdot: NDArray[np.floating],
    u: float,
) -> NDArray[np.floating]:
    """Return the right-hand-side force vector  f(q, q̇, u).

    Equation of motion:  M(q) q̈ = f  where
        f = τ  +  (−∂V/∂q)  −  C(q, q̇) q̇

    Parameters
    ----------
    cfg : PendulumConfig
    theta : array of shape (N,) – link angles
    qdot  : array of shape (1+N,) – generalised velocities [ẋ, θ̇₁, …]
    u : float – horizontal force applied to cart
    """
    dim = 1 + cfg.num_links

    tau = np.zeros(dim)
    tau[0] = u

    grav = _gravity_vector(cfg, theta)
    coriolis = _coriolis_vector(cfg, theta, qdot)

    return tau + grav - coriolis


def derivatives(
    cfg: PendulumConfig,
    state: NDArray[np.floating],
    u: float,
) -> NDArray[np.floating]:
    """Compute state derivatives  ṡ = [q̇, q̈].

    Parameters
    ----------
    cfg : PendulumConfig
    state : array [x, θ₁, …, θₙ, ẋ, θ̇₁, …, θ̇ₙ]  length 2*(1+N)
    u : float – force on cart

    Returns
    -------
    ds : array of same shape as *state*
    """
    n = cfg.num_links
    dim = 1 + n
    q = state[:dim]
    qd = state[dim:]

    theta = q[1:]

    M = _build_mass_matrix(cfg, theta)
    f = _build_force_vector(cfg, theta, qd, u)

    # Solve  M q̈ = f  for q̈
    qdd = np.linalg.solve(M, f)

    return np.concatenate([qd, qdd])


def step(
    cfg: PendulumConfig,
    state: NDArray[np.floating],
    u: float,
    dt: float = 0.02,
) -> NDArray[np.floating]:
    """Advance the simulation by *dt* seconds using RK4 integration.

    Parameters
    ----------
    cfg : PendulumConfig
    state : current state vector
    u : force applied to cart
    dt : time step

    Returns
    -------
    new_state : state after dt seconds
    """
    sol = solve_ivp(
        fun=lambda _t, s: derivatives(cfg, s, u),
        t_span=(0.0, dt),
        y0=state,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )
    new_state = sol.y[:, -1].copy()
    # Normalise angles to [-π, π]
    n = cfg.num_links
    new_state[1: 1 + n] = (new_state[1: 1 + n] + np.pi) % (2 * np.pi) - np.pi
    return new_state
