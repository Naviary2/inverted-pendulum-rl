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
import asyncio
import math
import time

import numpy as np
import pygame
import pygame.gfxdraw

from .config import PendulumConfig, TrainingConfig, VisualizationConfig
from .environment import CartPendulumEnv
from .interaction import CartDragController, ForceCircleController


def _load_model(path: str):
    """Load a trained PPO model (returns None when *path* is empty)."""
    if not path:
        return None
    from stable_baselines3 import PPO

    return PPO.load(path)


def _draw_thick_aaline(surface, p1, p2, width, color):
    """Draws a thick anti-aliased line (simulated via polygon)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    if length == 0:
        return

    # Normalized direction vector components
    ux = dx / length
    uy = dy / length

    # Perpendicular vector ((-uy, ux) corresponds to rotation by 90 degrees)
    # Scaled by half width to get offset from center line
    half_width = width / 2
    px = -uy * half_width
    py = ux * half_width

    # Calculate 4 corners of the rectangle
    corners = [
        (int(x1 + px), int(y1 + py)),
        (int(x1 - px), int(y1 - py)),
        (int(x2 - px), int(y2 - py)),
        (int(x2 + px), int(y2 + py)),
    ]

    # Draw filled polygon (the body)
    pygame.gfxdraw.filled_polygon(surface, corners, color)
    # Draw anti-aliased outline (the smooth edges)
    pygame.gfxdraw.aapolygon(surface, corners, color)


async def _async_run(
    env: CartPendulumEnv,
    model,
    p_cfg: PendulumConfig,
    v: VisualizationConfig,
    screen: pygame.Surface
):
    """Async main loop to handle physics and rendering with smoother VSync."""
    loop = asyncio.get_running_loop()
    
    obs, _ = env.reset()
    running = True

    # Timing variables for VSync framerate limiting
    limit_frame_duration = 1.0 / p_cfg.fps
    next_frame_target = 0.0

    # Initialize the drag controller
    drag_controller = CartDragController(env, p_cfg, v)
    
    # Initialize the force circle controller
    force_circle_controller = ForceCircleController(env, v)

    while running:
        # --- Framerate Limiter ---
        # Logic from: https://glyph.twistedmatrix.com/2022/02/a-better-pygame-mainloop.html
        if limit_frame_duration:
            this_frame = time.time()
            delay = next_frame_target - this_frame
            if delay > 0:
                await asyncio.sleep(delay)
            next_frame_target = time.time() + limit_frame_duration

        # Calculate screen geometry for rendering & input
        cx = v.width // 2
        cy = v.height // 2
        mouse_pos = pygame.mouse.get_pos()
        
        # Fetch all events once per frame
        events = pygame.event.get()

        # Handle General Events
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, _ = env.reset()
                
        # --- Manual Interaction ---
        drag_action = drag_controller.update(events, mouse_pos, cx, cy)
        
        # --- Force Circle Interaction ---
        force_circle_controller.update(events, mouse_pos, cx, cy)

        # --- action ---
        if drag_action is not None:
            # Overridden by manual dragging
            action = drag_action
        else:
            if model is not None:
                # Normal AI control
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Randomly choose max-left or max-right force.
                action = np.random.choice([-1, 1], size=(1,))
                # action = np.random.choice([-0.1, 0.1], size=(1,))
                # Apply no force
                action = np.array([0.0])

        obs, _reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            if not drag_controller.is_dragging:
                obs, _ = env.reset()

        # --- draw ---
        screen.fill(v.bg_color)

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

        # Cart position (Recalculate for rendering after step)
        state = env._state

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

            # Using custom helper for AA thick line,
            # Since gfxdraw.line doesn't support width and pygame.draw.line isn't anti-aliased.
            _draw_thick_aaline(
                screen,
                (pivot_x, pivot_y),
                (end_x, end_y),
                v.pendulum_width,
                v.fg_color
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

        # --- Draw Force Circle ---
        force_circle_controller.draw(screen, mouse_pos)

        # Run flip in an executor to avoid blocking the event loop during VSync wait
        await loop.run_in_executor(None, pygame.display.flip)


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
    
    # Use SCALED and vsync=1 as recommended by:
    # https://glyph.twistedmatrix.com/2022/02/a-better-pygame-mainloop.html
    screen = pygame.display.set_mode(
        (v.width, v.height), 
        flags=pygame.SCALED, 
        vsync=1
    )
    
    pygame.display.set_caption("Inverted Pendulum")
    
    # Use asyncio.run to execute the async main loop
    try:
        asyncio.run(_async_run(env, model, p_cfg, v, screen))
    except KeyboardInterrupt:
        pass
    finally:
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