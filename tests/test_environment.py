"""Tests for the Gymnasium environment."""

import numpy as np
import pytest

from pendulum.config import PendulumConfig
from pendulum.environment import CartPendulumEnv


class TestEnvInterface:
    """Standard Gymnasium API checks."""

    def test_reset_returns_obs_and_info(self):
        env = CartPendulumEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (4,)
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self):
        env = CartPendulumEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5  # obs, reward, terminated, truncated, info

    def test_obs_within_bounds(self):
        env = CartPendulumEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_episode_terminates_on_fall(self):
        """Force the pendulum past 90° and check termination."""
        env = CartPendulumEnv()
        env.reset(seed=0)
        # Manually set state to large angle
        env._state[1] = np.pi / 2 + 0.01
        _, _, terminated, _, _ = env.step(np.array([0.0]))
        assert terminated

    def test_episode_terminates_off_track(self):
        env = CartPendulumEnv()
        env.reset(seed=0)
        env._state[0] = env.cfg.track_length / 2 + 0.1
        _, _, terminated, _, _ = env.step(np.array([0.0]))
        assert terminated

    def test_truncation_at_max_steps(self):
        env = CartPendulumEnv(max_episode_steps=5)
        env.reset(seed=42)
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(np.array([0.0]))
            if terminated:
                break
        # If it didn't terminate early, it should truncate
        if not terminated:
            assert truncated


class TestReward:
    """Reward shaping sanity checks."""

    def test_max_reward_at_upright_centre(self):
        env = CartPendulumEnv()
        env.reset(seed=42)
        env._state = np.zeros(4)  # perfect upright, centred
        _, reward, _, _, _ = env.step(np.array([0.0]))
        # Should be close to 1.0 (max reward)
        assert reward > 0.9

    def test_reward_decreases_with_angle(self):
        env = CartPendulumEnv()
        env.reset(seed=42)
        env._state = np.array([0.0, 0.0, 0.0, 0.0])
        _, r1, _, _, _ = env.step(np.array([0.0]))

        env.reset(seed=42)
        env._state = np.array([0.0, 0.3, 0.0, 0.0])
        _, r2, _, _, _ = env.step(np.array([0.0]))
        assert r1 > r2
