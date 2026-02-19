"""Gymnasium environment for balancing a single inverted pendulum on a cart.

Observation (4-dim continuous):
    0  x    – cart position
    1  ẋ    – cart velocity
    2  θ₁   – angle of link 1 from vertical (0 = upright)
    3  θ̇₁   – angular velocity of link 1

Action (1-dim continuous):
    Cart acceleration (clipped to ±force_limit / cart_mass equivalent).

The action is interpreted as a desired cart acceleration; internally it is
converted to a force  F = (M_cart + Σmᵢ) · a_desired  so that the
cart-only response roughly matches the requested acceleration.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import PendulumConfig
from .physics import step as physics_step


class CartPendulumEnv(gym.Env):
    """Single inverted pendulum on a cart (extensible to N links)."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        pendulum_config: PendulumConfig | None = None,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = pendulum_config or PendulumConfig()
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.dt = 0.02  # 50 Hz physics

        n = self.cfg.num_links

        # --- action: cart acceleration (scalar, continuous) ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # --- observation ---
        # For single-link: [x, x_dot, theta, theta_dot]
        high = np.array(
            [
                self.cfg.track_length / 2,  # x
                np.finfo(np.float32).max,    # x_dot
            ]
            + [np.pi, np.finfo(np.float32).max] * n,  # theta_i, theta_dot_i
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # internal state: [x, θ₁, …, ẋ, θ̇₁, …]
        self._state: np.ndarray | None = None
        self._step_count = 0

    # -----------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Map internal state to observation vector."""
        n = self.cfg.num_links
        s = self._state
        # state layout: [x, θ₁…θₙ, ẋ, θ̇₁…θ̇ₙ]
        x = s[0]
        x_dot = s[1 + n]
        obs = [x, x_dot]
        for i in range(n):
            obs.append(s[1 + i])        # θᵢ
            obs.append(s[1 + n + 1 + i])  # θ̇ᵢ
        return np.array(obs, dtype=np.float32)

    # -----------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        n = self.cfg.num_links
        # Start near upright with small random perturbation
        state = np.zeros(2 * (1 + n))
        state[1: 1 + n] = self.np_random.uniform(-0.05, 0.05, size=n)  # θ
        self._state = state
        self._step_count = 0
        return self._get_obs(), {}

    # -----------------------------------------------------------------
    def step(self, action):
        assert self._state is not None, "Call reset() first"
        self._step_count += 1

        # Convert normalised action [-1,1] → force
        acc = float(np.clip(action[0], -1.0, 1.0))
        force = acc * self.cfg.force_limit

        # Physics step
        self._state = physics_step(self.cfg, self._state, force, self.dt)

        obs = self._get_obs()

        # --- reward ---
        n = self.cfg.num_links
        x = self._state[0]
        theta = self._state[1: 1 + n]

        # Reward: stay upright + stay centred
        angle_penalty = float(np.sum(theta ** 2))
        position_penalty = (x / (self.cfg.track_length / 2)) ** 2
        reward = 1.0 - 0.5 * angle_penalty - 0.5 * position_penalty

        # --- termination ---
        terminated = False
        # Fall over: any link angle > 90°
        if np.any(np.abs(theta) > np.pi / 2):
            terminated = True
            reward = 0.0
        # Cart off track
        if np.abs(x) > self.cfg.track_length / 2:
            terminated = True
            reward = 0.0

        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}
