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
import time

import numpy as np
import pygame

from .config import PendulumConfig, TrainingConfig, VisualizationConfig
from .environment import CartPendulumEnv
from .interaction import CartDragController, ForceCircleController
from .renderer import SceneRenderer


def _load_model(path: str):
    """Load a trained PPO model (returns None when *path* is empty)."""
    if not path:
        return None
    from stable_baselines3 import PPO

    return PPO.load(path)


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

    # Initialize the scene renderer
    renderer = SceneRenderer(p_cfg, v)

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
        renderer.draw(screen, env._state, cx, cy, mouse_pos, force_circle_controller)

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