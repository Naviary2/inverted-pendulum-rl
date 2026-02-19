"""Tests for the multi-link pendulum physics engine."""

import numpy as np
import pytest

from pendulum.config import PendulumConfig
from pendulum.physics import _build_mass_matrix, derivatives, step


class TestMassMatrix:
    """Mass matrix construction."""

    def test_single_link_shape(self):
        cfg = PendulumConfig(num_links=1, link_lengths=[1.0], link_masses=[0.1])
        M = _build_mass_matrix(cfg, np.array([0.0]))
        assert M.shape == (2, 2)

    def test_double_link_shape(self):
        cfg = PendulumConfig(
            num_links=2, link_lengths=[1.0, 0.5], link_masses=[0.1, 0.05]
        )
        M = _build_mass_matrix(cfg, np.array([0.0, 0.0]))
        assert M.shape == (3, 3)

    def test_mass_matrix_symmetric(self):
        cfg = PendulumConfig(
            num_links=2, link_lengths=[1.0, 0.5], link_masses=[0.1, 0.05]
        )
        theta = np.array([0.3, -0.2])
        M = _build_mass_matrix(cfg, theta)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_mass_matrix_positive_definite(self):
        cfg = PendulumConfig(num_links=1, link_lengths=[1.0], link_masses=[0.1])
        M = _build_mass_matrix(cfg, np.array([0.5]))
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0)


class TestDerivatives:
    """State derivative computation."""

    def test_stationary_upright_no_force(self):
        """A pendulum balanced upright with no force should stay still."""
        cfg = PendulumConfig()
        state = np.zeros(4)  # [x, θ, ẋ, θ̇] = 0
        ds = derivatives(cfg, state, u=0.0)
        # Velocities are zero, accelerations should be zero (equilibrium)
        np.testing.assert_allclose(ds, 0.0, atol=1e-12)

    def test_gravity_acts_on_tilted(self):
        """A slightly tilted pendulum should accelerate due to gravity."""
        cfg = PendulumConfig()
        state = np.array([0.0, 0.1, 0.0, 0.0])  # θ = 0.1 rad
        ds = derivatives(cfg, state, u=0.0)
        # θ̈ should be non-zero (gravity pulling it down)
        assert ds[3] != 0.0  # angular acceleration


class TestStep:
    """Full integration step."""

    def test_step_returns_correct_shape(self):
        cfg = PendulumConfig()
        state = np.zeros(4)
        new_state = step(cfg, state, u=0.0, dt=0.02)
        assert new_state.shape == (4,)

    def test_double_link_step(self):
        cfg = PendulumConfig(
            num_links=2, link_lengths=[1.0, 0.5], link_masses=[0.1, 0.05]
        )
        state = np.zeros(6)  # [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
        new_state = step(cfg, state, u=0.0, dt=0.02)
        assert new_state.shape == (6,)

    def test_energy_conservation_no_force(self):
        """With no external force the total energy should be conserved."""
        cfg = PendulumConfig(
            cart_mass=1.0, num_links=1, link_lengths=[1.0], link_masses=[0.1]
        )
        # Start tilted
        state = np.array([0.0, 0.3, 0.0, 0.0])

        def total_energy(s):
            x, theta, x_dot, theta_dot = s
            m = cfg.link_masses[0]
            M = cfg.cart_mass
            l = cfg.link_lengths[0] / 2  # to CoM
            g = cfg.gravity
            KE_cart = 0.5 * M * x_dot**2
            # Velocity of pendulum CoM
            vx = x_dot + l * theta_dot * np.cos(theta)
            vy = -l * theta_dot * np.sin(theta)
            KE_pend = 0.5 * m * (vx**2 + vy**2)
            PE = m * g * l * np.cos(theta)
            return KE_cart + KE_pend + PE

        E0 = total_energy(state)
        for _ in range(500):
            state = step(cfg, state, u=0.0, dt=0.01)
        E1 = total_energy(state)
        np.testing.assert_allclose(E1, E0, rtol=1e-4)

    def test_force_moves_cart(self):
        """Applying a constant force should accelerate the cart."""
        cfg = PendulumConfig()
        state = np.zeros(4)
        state[1] = 0.0  # upright
        for _ in range(50):
            state = step(cfg, state, u=5.0, dt=0.02)
        # Cart should have moved to the right
        assert state[0] > 0.0
        assert state[2] > 0.0  # positive velocity


class TestMultiLink:
    """Verify the engine generalises to more links."""

    def test_triple_pendulum(self):
        cfg = PendulumConfig(
            num_links=3,
            link_lengths=[1.0, 0.8, 0.6],
            link_masses=[0.1, 0.08, 0.05],
        )
        state = np.zeros(8)
        state[1] = 0.05
        new_state = step(cfg, state, u=0.0, dt=0.02)
        assert new_state.shape == (8,)

    def test_quadruple_pendulum(self):
        cfg = PendulumConfig(
            num_links=4,
            link_lengths=[1.0, 0.8, 0.6, 0.4],
            link_masses=[0.1, 0.08, 0.05, 0.03],
        )
        state = np.zeros(10)
        new_state = step(cfg, state, u=1.0, dt=0.02)
        assert new_state.shape == (10,)
