# pendulum/config.py

"""Configuration for pendulum simulation and training."""

import multiprocessing
from dataclasses import dataclass, field
from typing import List


@dataclass
class PendulumConfig:
    """Physical parameters for the multi-link pendulum on a cart."""

    # Cart
    force_magnitude: float = 20          # Maximum force the cart can apply on any one frame
    cart_mass: float = 1.0           # kg
    track_length: float = 2.4 * 2    # m (total track, cart can move ±half) - MUST MATCH XML!!!!!!

    fps: int = 60                    # Hz (Controls both render speed and physics timestep)
    physics_substeps: int = 1        # How many physics step per 1 render step. Higher = more accurate physics but slower training.

    # Links (lists allow N-link pendulums)
    num_links: int = 1
    link_lengths: List[float] = field(default_factory=lambda: [1.0])   # m - MUST MATCH XML!!!!!!

    # Physics
    gravity: float = 9.81            # m/s²
    force_limit: float = 10.0        # N  (max force applied to cart)

    def __post_init__(self):
        assert len(self.link_lengths) == self.num_links


@dataclass
class TrainingConfig:
    """Hyper-parameters for PPO training."""

    total_timesteps: int = 60 * 100
    n_envs: int = field(default_factory=lambda: multiprocessing.cpu_count())
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    max_episode_steps: int = 1000
    model_save_path: str = "models/ppo_pendulum"


@dataclass
class VisualizationConfig:
    """Display settings for the pygame viewer."""

    width: int = 1920
    height: int = 1080
    fps: int = 60
    scale: float = 250.0             # pixels per metre

    track_h = 30        # Height requested
    track_thick = 3     # Constant thickness (line width)
    track_rad = 8       # Corner roundness (half height = full pill shape)

    cart_width: int = 80             # px
    cart_height: int = 24            # px
    cart_rad: int = 6              # px
    cart_node_radius: int = 7            # px

    node_radius: int = 24            # px
    node_fill_color: tuple = (50, 160, 30)
    node_outline_width: int = 4

    pendulum_width: int = 8          # px
    bg_color: tuple = (35, 35, 35) # grey
    fg_color: tuple = (255, 255, 255) # white
    cart_fill_alpha: int = 128        # 0-255 transparency for cart fill
