# pendulum/config.py

"""Configuration for pendulum simulation and training."""

import multiprocessing
from dataclasses import dataclass, field
from typing import List

# Warmup duration: physics and interaction are held for this many seconds
# after the window opens (or after pressing R to reset).
WARMUP_DURATION_SECS: float = 1.0


@dataclass
class PendulumConfig:
    """Physical parameters for the multi-link pendulum on a cart."""

    # Cart
    force_magnitude: float = 200          # Maximum force the cart can apply on any one frame, in Newtons.
    track_length: float = 2.5 * 2    # m (total track, cart can move ±half)

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
    force_circle_radius: float = 0.24  # m  (force circle sphere radius)

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
    model_load_path: str = ""  # Path to an existing model to continue training (empty = train from scratch)
    tensorboard_log: str = "logs/tensorboard"  # Directory for TensorBoard logs (empty = disabled)


@dataclass
class VisualizationConfig:
    """Display settings for the Qt viewer."""

    bg_color: tuple = (60, 60, 60) # grey

    width: int = 1920
    height: int = 1080
    scale: float = 200.0 # pixels per metre

    track_h: float = 0.09 # m  (track height)
    track_thick: float = 0.02 # m (track outline thickness)
    track_rad: float = 0.05 # m  (corner roundness)

    # Cart body (white rectangle behind pivot node)
    cart_body_width: float = 0.65
    # Strut rectangles on the left and right sides of the cart body
    cart_strut_width: float = 0.08
    cart_strut_height: float = 0.36
    cart_strut_center_y: float = 0.013
    # Strut tilt: each strut is rotated this many degrees toward the cart centre
    # (tops lean inward, bottoms lean outward)
    cart_strut_angle: float = -20.0

    node_outline_width: float = 0.022 # m

    pendulum_width: float = 0.045 # m
    fg_color: tuple = (255, 255, 255) # white

    # Force circle (toggled with "F" key)
    force_circle_thickness: float = 0.02  # m
    force_circle_color: tuple = (255, 255, 255)  # white

    # Tick ruler (graduated markings below the track)
    tick_range: int = 3               # whole-meter radius of the ruler (draws ±tick_range)
    tick_gap: float = 0.35            # m  gap from track bottom edge to tick centre line
    tick_zero_height: float = 0.18    # m  height of the x=0 tick
    tick_int_height: float = 0.13     # m  height of integer ticks
    tick_half_height: float = 0.09    # m  height of half-integer ticks
    tick_tenth_height: float = 0.055  # m  height of tenth-step ticks
    tick_zero_alpha: int = 255        # opacity of the x=0 tick
    tick_int_alpha: int = 210         # opacity (0-255) of integer ticks
    tick_half_alpha: int = 130        # opacity of half-integer ticks
    tick_tenth_alpha: int = 65        # opacity of tenth-step ticks
    tick_zero_width: float = 0.019    # m  stroke width of the x=0 tick
    tick_int_width: float = 0.014     # m  stroke width of integer ticks
    tick_half_width: float = 0.010    # m  stroke width of half-integer ticks
    tick_tenth_width: float = 0.007   # m  stroke width of tenth-step ticks
    tick_label_font_size: float = 20.0 # pt  font size for integer tick labels
    tick_label_font_family: str = "Courier New" # preferred font family for tick labels (monospace)
    tick_label_gap: float = 0.03     # m  gap from bottom of tallest tick to label top
    tick_label_height: float = 0.1   # m  reserved height for label text
