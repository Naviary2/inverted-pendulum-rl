# pendulum/visualize.py

"""PySide6 visualisation of the trained pendulum model.

Renders:
  • Dark background
  • White horizontal track
  • White-outlined cart with semi-transparent fill
  • White pendulum rod(s) with round caps
  • White nodes at each joint / tip

Usage:
    python -m pendulum.visualize                        # random actions
    python -m pendulum.visualize --model models/ppo_pendulum
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QMainWindow
import PySide6.QtAsyncio as QtAsyncio

from .config import PendulumConfig, TrainingConfig, VisualizationConfig
from .environment import CartPendulumEnv
from .renderer import PendulumScene, PendulumView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(path: str):
    """Load a trained PPO model (returns *None* when *path* is empty)."""
    if not path:
        return None
    from stable_baselines3 import PPO
    return PPO.load(path)


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

class PendulumWindow(QMainWindow):
    """Main window that hosts the simulation view."""

    def __init__(
        self,
        env: CartPendulumEnv,
        p_cfg: PendulumConfig,
        v: VisualizationConfig,
    ):
        super().__init__()
        self.env = env
        self.p_cfg = p_cfg
        self.v = v

        self.setWindowTitle("Inverted Pendulum")
        self.setFixedSize(v.width, v.height)

        self.obs, _ = env.reset()

        # Scene & view
        self._scene = PendulumScene(env, p_cfg, v)
        self._view = PendulumView(self._scene)
        self.setCentralWidget(self._view)

        # Initial sync
        self._scene.sync_from_state(env._state)

    # -- key handling -------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_R:
            self.obs, _ = self.env.reset()
        elif event.key() == Qt.Key.Key_F:
            self._scene._force_circle.toggle()
            # Sync force circle to current cursor position immediately
            cursor_pos = self._view.mapFromGlobal(self.cursor().pos())
            scene_pos = self._view.mapToScene(cursor_pos)
            self._scene._force_circle.update_position(scene_pos.x(), scene_pos.y())
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Async simulation loop
# ---------------------------------------------------------------------------

async def _async_run(window: PendulumWindow, model, p_cfg: PendulumConfig):
    """Async main loop — mirrors the original asyncio timing approach."""
    env = window.env
    limit_frame_duration = 1.0 / p_cfg.fps
    next_frame_target = 0.0

    while True:
        # --- Framerate limiter (same strategy as original Pygame version) ---
        this_frame = time.time()
        delay = next_frame_target - this_frame
        if delay > 0:
            await asyncio.sleep(delay)
        next_frame_target = time.time() + limit_frame_duration

        # --- action ---
        cart = window._scene._cart

        if cart.is_dragging:
            action = np.array([0.0], dtype=np.float32)
        elif model is not None:
            action, _ = model.predict(window.obs, deterministic=True)
        else:
            action = np.array([0.0])

        window.obs, _reward, terminated, truncated, _ = env.step(action)

        if (terminated or truncated) and not cart.is_dragging:
            window.obs, _ = env.reset()

        # --- sync scene items to physics state ---
        window._scene.sync_from_state(env._state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    model_path: str = "",
    pendulum_cfg: PendulumConfig | None = None,
    vis_cfg: VisualizationConfig | None = None,
):
    """Open a Qt window and run the pendulum simulation."""
    p_cfg = pendulum_cfg or PendulumConfig()
    v = vis_cfg or VisualizationConfig()

    t_cfg = TrainingConfig()
    env = CartPendulumEnv(pendulum_config=p_cfg, max_episode_steps=t_cfg.total_timesteps)
    model = _load_model(model_path)

    app = QApplication.instance() or QApplication(sys.argv)
    window = PendulumWindow(env, p_cfg, v)
    window.show()

    # Schedule the async loop once QtAsyncio's event loop is running
    QTimer.singleShot(0, lambda: asyncio.ensure_future(
        _async_run(window, model, p_cfg)
    ))
    # QtAsyncio.run() drives both the Qt and asyncio event loops
    QtAsyncio.run()


# ---- CLI entry point -------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Visualise pendulum")
    parser.add_argument("--model", type=str, default="",
                        help="Path to trained model (omit for random actions)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(model_path=args.model)