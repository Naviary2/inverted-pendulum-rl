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

from .config import PendulumConfig, VisualizationConfig
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

    env = CartPendulumEnv(pendulum_config=p_cfg, max_episode_steps=0)
    model = _load_model(model_path)

    pygame.init()
    screen = pygame.display.set_mode((v.width, v.height))
    pygame.display.set_caption("Inverted Pendulum")
    clock = pygame.time.Clock()

    # Semi-transparent cart surface (needs per-pixel alpha)
    cart_surf = pygame.Surface((v.cart_width, v.cart_height), pygame.SRCALPHA)

    obs, _ = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, _ = env.reset()

        # --- action ---
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, _reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

        # --- draw ---
        screen.fill(v.bg_color)

        cx = v.width // 2   # centre x of screen
        cy = v.height // 2  # vertical centre (track line)

        # Track
        track_half_px = int(p_cfg.track_length / 2 * v.scale)
        pygame.draw.line(
            screen,
            v.fg_color,
            (cx - track_half_px, cy),
            (cx + track_half_px, cy),
            1,
        )

        # Cart position
        state = env._state
        cart_x_px = cx + int(state[0] * v.scale)

        # Cart rectangle (outlined + translucent fill)
        cart_rect = pygame.Rect(
            cart_x_px - v.cart_width // 2,
            cy - v.cart_height // 2,
            v.cart_width,
            v.cart_height,
        )
        cart_surf.fill((0, 0, 0, 0))  # clear
        pygame.draw.rect(
            cart_surf,
            (*v.fg_color, v.cart_fill_alpha),
            cart_surf.get_rect(),
        )
        pygame.draw.rect(cart_surf, (*v.fg_color, 255), cart_surf.get_rect(), 2)
        screen.blit(cart_surf, cart_rect.topleft)

        # --- pendulum links & nodes ---
        n = p_cfg.num_links
        lengths = p_cfg.link_lengths

        # First node sits at the cart pivot (top-centre of cart)
        pivot_x, pivot_y = cart_x_px, cy

        # Draw first node
        pygame.draw.circle(screen, v.fg_color, (pivot_x, pivot_y), v.node_radius)

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

            # Tip node
            pygame.draw.circle(screen, v.fg_color, (end_x, end_y), v.node_radius)

            # Next pivot is this tip
            pivot_x, pivot_y = end_x, end_y

        pygame.display.flip()
        clock.tick(v.fps)

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
