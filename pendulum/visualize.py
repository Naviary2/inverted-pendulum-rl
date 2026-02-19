# pendulum/visualize.py

"""Pygame visualisation of the trained pendulum model.

Renders:
  • Black background
  • White horizontal track
  • White-outlined cart with semi-transparent white fill
  • White pendulum rod(s)
  • White nodes at each joint / tip

Usage:
    python -m pendulum.visualize                        # random actions
    python -m pendulum.visualize --model models/ppo_pendulum
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pygame
import pygame.gfxdraw

from .config import PendulumConfig, TrainingConfig, VisualizationConfig
from .environment import CartPendulumEnv


def _load_model(path: str):
    """Load a trained PPO model (returns None when *path* is empty)."""
    if not path:
        return None
    from stable_baselines3 import PPO

    return PPO.load(path)


def run(
    model_path: str = "",
    pendulum_cfg: PendulumConfig | None = None,
    vis_cfg: VisualizationConfig | None = None,
):
    """Open a pygame window and run the pendulum simulation."""
    p_cfg = pendulum_cfg or PendulumConfig()
    v = vis_cfg or VisualizationConfig()

    t_cfg = TrainingConfig() # Load training config to access total_timesteps
    # Overwrite the maximum episode steps to match total timesteps for a full-length run
    env = CartPendulumEnv(pendulum_config=p_cfg, max_episode_steps=t_cfg.total_timesteps)
    model = _load_model(model_path)

    pygame.init()
    screen = pygame.display.set_mode((v.width, v.height))
    pygame.display.set_caption("Inverted Pendulum")
    clock = pygame.time.Clock()

    obs, _ = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, _ = env.reset()

        # --- action ---
        # Apply random force
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Randomly choose max-left or max-right force.
            action = np.random.choice([-1, 1], size=(1,))
        # Apply no force
        # action = np.array([0.0])

        obs, _reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

        # --- draw ---
        screen.fill(v.bg_color)

        cx = v.width // 2   # centre x of screen
        cy = v.height // 2  # vertical centre (track line)

        # Track
        track_len_px = int(p_cfg.track_length * v.scale) + v.cart_width + v.cart_node_radius
        # Create a rectangle centered at (cx, cy)
        track_rect = pygame.Rect(
            cx - track_len_px // 2,  # Left
            cy - v.track_h // 2,       # Top
            track_len_px,            # Width
            v.track_h                  # Height
        )
        # Draw hollow rounded rectangle
        pygame.draw.rect(
            screen,
            v.fg_color,
            track_rect,
            v.track_thick,             # Thickness (makes it hollow)
            border_radius=v.track_rad  # Roundness
        )

        # Cart position
        state = env._state
        cart_x_px = cx + int(state[0] * v.scale)

        # Define the hard limit based on the track length
        half_length = p_cfg.track_length / 2.0
        # Clamp the cart's x position so it visually never leaves the track
        # (Even if MuJoCo calculates a slight penetration of the wall)
        cart_x = float(np.clip(state[0], -half_length, half_length))
        # Use this clamped value to calculate pixels
        cart_x_px = cx + int(cart_x * v.scale)

        # Cart rectangle (outlined + translucent fill)
        cart_rect = pygame.Rect(
            cart_x_px - v.cart_width // 2,
            cy - v.cart_height // 2,
            v.cart_width,
            v.cart_height,
        )

        # Draw Fill
        pygame.draw.rect(
            screen,
            v.node_fill_color,
            cart_rect,
            border_radius=v.cart_rad  # Roundness
        )
        # Draw hollow rounded rectangle (Outline)
        pygame.draw.rect(
            screen,
            v.fg_color,
            cart_rect,
            v.track_thick,             # Thickness (makes it hollow)
            border_radius=v.cart_rad  # Roundness
        )

        # --- pendulum links & nodes ---
        n = p_cfg.num_links
        lengths = p_cfg.link_lengths

        # First node sits at the cart pivot (top-centre of cart)
        pivot_x, pivot_y = cart_x_px, cy

        # Draw first node
        pygame.gfxdraw.aacircle(screen, pivot_x, pivot_y, v.cart_node_radius, v.fg_color) ## AA outline
        pygame.gfxdraw.filled_circle(screen, pivot_x, pivot_y, v.cart_node_radius, v.fg_color) # Filled circle

        for i in range(n):
            theta_i = state[1 + i]
            end_x = pivot_x + int(lengths[i] * v.scale * np.sin(theta_i))
            end_y = pivot_y - int(lengths[i] * v.scale * np.cos(theta_i))

            # Rod
            pygame.draw.line(
                screen,
                v.fg_color,
                (pivot_x, pivot_y),
                (end_x, end_y),
                v.pendulum_width,
            )

            # --- Draw Tip Node ---
            # Using gfxdraw for AA
            
            # 1. Draw Outline (Outer Circle)
            pygame.gfxdraw.aacircle(screen, end_x, end_y, v.node_radius, v.fg_color) ## AA outline
            pygame.gfxdraw.filled_circle(screen, end_x, end_y, v.node_radius, v.fg_color) # Filled circle
            
            # 2. Draw Fill (Inner Circle)
            inner_radius = v.node_radius - v.node_outline_width
            pygame.gfxdraw.aacircle(screen, end_x, end_y, inner_radius, v.node_fill_color) # AA outline
            pygame.gfxdraw.filled_circle(screen, end_x, end_y, inner_radius, v.node_fill_color) # Filled circle

            # Next pivot is this tip
            pivot_x, pivot_y = end_x, end_y

        pygame.display.flip()
        clock.tick(p_cfg.fps) # Sync render loop to physics config

    pygame.quit()


# ---- CLI entry point -------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="Visualise pendulum")
    parser.add_argument("--model", type=str, default="",
                        help="Path to trained model (omit for random actions)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(model_path=args.model)