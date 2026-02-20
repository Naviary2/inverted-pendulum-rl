# pendulum/config.py

"""Configuration for pendulum simulation and training."""

import multiprocessing
from dataclasses import dataclass, field
from typing import List


@dataclass
class PendulumConfig:
    """Physical parameters for the multi-link pendulum on a cart."""

    # Cart
    force_magnitude: float = 200          # Maximum force the cart can apply on any one frame, in Newtons.
    track_length: float = 2.4 * 2    # m (total track, cart can move ±half)

    # fps: int = 60                    # Hz (Controls both render speed and physics timestep)
    fps: int = 240                    # Hz (Controls both render speed and physics timestep)
    physics_substeps: int = 1        # How many physics step per 1 render step. Higher = more accurate physics but slower training.

    # Links (lists allow N-link pendulums)
    num_links: int = 1
    link_lengths: List[float] = field(default_factory=lambda: [1.0])   # Pendulum lengths in metres. Length of link i is link_lengths[i].

    # Physics
    gravity: float = 9.81            # m/s² ~8 Looks normal
    force_limit: float = 8.0        # N  (max force applied to cart)

    # Collision geometry
    node_radius: float = 0.13      # m  (tip node sphere radius)
    force_circle_radius: float = 0.16  # m  (force circle sphere radius)

    def __post_init__(self):
        assert len(self.link_lengths) == self.num_links


@dataclass
class TrainingConfig:
    """Hyper-parameters for PPO training."""

    total_timesteps: int = 300_000
    n_envs: int = field(default_factory=lambda: multiprocessing.cpu_count())
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    max_episode_steps: int = PendulumConfig.fps * int(1000 / 60)  # 1000 steps at 60 FPS
    model_save_path: str = "models/ppo_pendulum"


@dataclass
class VisualizationConfig:
    """Display settings for the Qt viewer."""

    width: int = 1920
    height: int = 1080
    fps: int = 60
    scale: float = 200.0 # pixels per metre

    track_h: float = 0.08 # m  (track height)
    track_thick: int = 5 # px (constant line thickness, ~50% thicker than original)
    track_rad: float = 0.032 # m  (corner roundness)

    # Cart body (white rectangle behind pivot node)
    cart_body_width: float = 0.57
    # Strut rectangles on the left and right sides of the cart body
    cart_strut_width: float = 0.07
    cart_strut_height: float = 0.3

    node_fill_color: tuple = (50, 160, 30)
    node_outline_width: float = 0.022 # m

    pendulum_width: float = 0.045 # m
    bg_color: tuple = (35, 35, 35) # grey
    fg_color: tuple = (255, 255, 255) # white

    # Force circle (toggled with "F" key)
    force_circle_thickness: float = 0.016  # m
    force_circle_color: tuple = (255, 0, 0)  # red

    # Cart lock (toggled with "G" key)
    cart_locked_wheel_tint: tuple = (0, 0, 0, 150)  # semi-transparent black overlay
