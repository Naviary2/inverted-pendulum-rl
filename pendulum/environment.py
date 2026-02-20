# pendulum/environment.py

"""
Gymnasium environment for balancing a single inverted pendulum on a cart,
powered by the MuJoCo physics engine.

This file acts as a wrapper around Gymnasium's 'InvertedPendulum-v4'
to maintain compatibility with the existing training and visualization code.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import os
import tempfile
from gymnasium import spaces

from .config import PendulumConfig


class CartPendulumEnv(gym.Env):
    """
    A wrapper for the MuJoCo 'InvertedPendulum-v4' environment.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        pendulum_config: PendulumConfig | None = None,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
    ):
        super().__init__()
        # Although pendulum_config is unused, we keep it for API consistency
        self.cfg = pendulum_config or PendulumConfig()
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        xml_path = os.path.join(os.path.dirname(__file__), "..", "assets", "inverted_pendulum.xml")

        # Generate a modified XML with config values as the single source of truth
        modified_xml_path = self._create_config_xml(os.path.abspath(xml_path))
        self._tmp_xml_path = modified_xml_path

        # Create the underlying MuJoCo environment
        self._mujoco_env = gym.make(
            "InvertedPendulum-v5",
            render_mode=render_mode,
            xml_file=modified_xml_path,
            # This tells Gym: "Run X physics steps for every 1 call to .step()"
            frame_skip=self.cfg.physics_substeps,
            # This forces the inner env to respect our custom limit set in visualize.py (instead of its maximum 1000 steps)
            max_episode_steps=max_episode_steps,
            disable_env_checker=True,
        )

        # Calculate the high-resolution physics timestep
        # Example: 60 FPS * 4 substeps = 240Hz physics -> 0.00416s timestep
        phys_freq = self.cfg.fps * self.cfg.physics_substeps
        phys_timestep = 1.0 / phys_freq
        # Overwrite the XML's timestep to match our desired FPS exactly.
        # This ensures real-time simulation: 60 FPS -> 0.0166s timestep.
        self._mujoco_env.unwrapped.model.opt.timestep = phys_timestep

        # --- Define observation and action spaces to match the original setup ---

        # The PPO agent expects a normalised action of [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # The observation space definition from the original environment
        n = self.cfg.num_links
        high = np.array(
            [
                self.cfg.track_length / 2,  # x
                np.finfo(np.float32).max,    # x_dot
            ]
            + [np.pi, np.finfo(np.float32).max] * n,  # theta_i, theta_dot_i
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # The _state variable is crucial for the visualizer.
        # MuJoCo's observation is [x, θ, ẋ, θ̇], which we store directly here.
        self._state: np.ndarray | None = None
        self._step_count = 0

    def _create_config_xml(self, base_xml_path: str) -> str:
        """
        Read the base MuJoCo XML template and substitute placeholders with
        values from PendulumConfig so that config.py is the single source of truth.
        Returns the path to a temporary XML file with the substituted values.
        """
        with open(base_xml_path, 'r') as f:
            xml_template = f.read()

        xml_content = xml_template.format(
            gravity=self.cfg.gravity,
            half_track=self.cfg.track_length / 2.0,
            pole_length=self.cfg.link_lengths[0],
            node_radius=self.cfg.node_radius,
            force_circle_radius=self.cfg.force_circle_radius,
        )

        # Write to a temporary file
        fd, tmp_path = tempfile.mkstemp(suffix='.xml')
        os.close(fd)
        with open(tmp_path, 'w') as f:
            f.write(xml_content)
        return tmp_path

    def _get_obs(self) -> np.ndarray:
        """
        Map the internal MuJoCo state to the observation format expected by the agent.
        - MuJoCo state/obs: [x, θ, ẋ, θ̇]
        - Original agent obs: [x, ẋ, θ, θ̇]
        This function performs that re-ordering.
        """
        assert self._state is not None
        x, theta, x_dot, theta_dot = self._state
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the underlying MuJoCo environment
        self._mujoco_env.reset(seed=seed, options=options)
        
        # Get pointers to the physics state
        qpos = self._mujoco_env.unwrapped.data.qpos
        qvel = self._mujoco_env.unwrapped.data.qvel
        
        # qpos[0] is the cart position (center)
        qpos[0] = 0.0
        
        # qpos[1] is the angle. 0 is up, pi is down.
        qpos[1] = np.pi 
        
        # Zero out all velocities so it starts perfectly still
        qvel[:] = 0.0
        
        # Apply this state to the simulation
        self._mujoco_env.unwrapped.set_state(qpos, qvel)
        
        # Get the observation
        mujoco_obs = self._mujoco_env.unwrapped._get_obs()

        # Store the state in the format required by the visualizer: [x, θ, ẋ, θ̇]
        self._state = mujoco_obs
        self._step_count = 0
        
        return self._get_obs(), {}

    def step(self, action):
        assert self._state is not None, "Call reset() first"
        self._step_count += 1

        # 1. Scale the agent's action from [-1, 1] to the MuJoCo env's expected range [-3, 3]
        scaled_action = np.clip(action, -1.0, 1.0) * self.cfg.force_magnitude

        # 2. Step the underlying MuJoCo environment
        mujoco_obs, _reward, terminated, truncated, info = self._mujoco_env.step(scaled_action)

        # --- Custom Reward Shaping ---
        # Linear reward based on the angle of the pendulum. 0 when hanging down, 1 when perfectly upright.
        theta = mujoco_obs[1]
        theta_normalized = ((theta + np.pi) % (2 * np.pi)) - np.pi
        reward = 1.0 - (abs(theta_normalized) / np.pi)

        # 3. Update the internal state for the visualizer
        self._state = mujoco_obs
        
        # 4. Use MuJoCo's termination/truncation logic, but also respect max_episode_steps
        if self._step_count >= self.max_episode_steps > 0:
            truncated = True

        # Override the termination signal from MuJoCo to allow free swinging.
        # Otherwise whenever the pendulum falls past 15 degrees, it gets reset.
        terminated = False

        # 5. Return the re-ordered observation for the agent
        return self._get_obs(), reward, terminated, truncated, info
    
    def close(self):
        """Close the underlying MuJoCo environment and clean up temp files."""
        self._mujoco_env.close()
        if hasattr(self, '_tmp_xml_path') and os.path.exists(self._tmp_xml_path):
            os.unlink(self._tmp_xml_path)