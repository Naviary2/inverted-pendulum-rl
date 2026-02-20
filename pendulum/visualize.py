# pendulum/visualize.py

"""PySide6 visualisation of the trained pendulum model.

Renders:
  • Dark background
  • White horizontal track
  • White-outlined cart with semi-transparent fill and drop shadow
  • White pendulum rod(s) with round caps
  • White nodes at each joint / tip

Usage:
    python -m pendulum.visualize                        # random actions
    python -m pendulum.visualize --model models/ppo_pendulum
"""

from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

from PySide6.QtCore import Qt, QTimer, QLineF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPen,
    QKeyEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QMainWindow,
)

from .config import PendulumConfig, TrainingConfig, VisualizationConfig
from .environment import CartPendulumEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb(t: tuple) -> QColor:
    """Convert an (r, g, b) or (r, g, b, a) tuple to a QColor."""
    return QColor(*t)


def _load_model(path: str):
    """Load a trained PPO model (returns *None* when *path* is empty)."""
    if not path:
        return None
    from stable_baselines3 import PPO
    return PPO.load(path)


# ---------------------------------------------------------------------------
# Custom QGraphicsItems
# ---------------------------------------------------------------------------

class CartItem(QGraphicsRectItem):
    """Draggable cart rectangle with a drop-shadow effect.

    The item's *local* coordinate origin is at the centre of the cart
    rectangle so that ``setPos(px, 0)`` places the cart correctly on the
    track.
    """

    def __init__(
        self,
        env: CartPendulumEnv,
        p_cfg: PendulumConfig,
        v: VisualizationConfig,
        parent=None,
    ):
        cart_w = v.cart_width * v.scale
        cart_h = v.cart_height * v.scale
        # Rect centred on local (0, 0)
        super().__init__(-cart_w / 2, -cart_h / 2, cart_w, cart_h, parent)

        self.env = env
        self.p_cfg = p_cfg
        self.v = v
        self.is_dragging = False

        # MuJoCo data handle
        self._mujoco_data = env._mujoco_env.unwrapped.data

        # Visual styling
        cart_rad = int(v.cart_rad * v.scale)
        fill_color = _rgb(v.node_fill_color)
        outline_color = _rgb(v.fg_color)

        self.setBrush(QBrush(fill_color))
        pen = QPen(outline_color, v.track_thick)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self.setPen(pen)

        # Drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setOffset(4, 4)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # -- mouse interaction --------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self._mujoco_data.eq_active = 1
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            scene_pos = event.scenePos()
            target_x = scene_pos.x() / self.v.scale
            half_track = self.p_cfg.track_length / 2.0
            target_x = float(np.clip(target_x, -half_track, half_track))
            self._mujoco_data.mocap_pos[0, 0] = target_x
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            self._mujoco_data.eq_active = 0
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)


class ForceCircleItem(QGraphicsEllipseItem):
    """A hollow circle that follows the mouse and interacts with MuJoCo bodies."""

    def __init__(
        self,
        env: CartPendulumEnv,
        p_cfg: PendulumConfig,
        v: VisualizationConfig,
        parent=None,
    ):
        radius_px = p_cfg.force_circle_radius * v.scale
        super().__init__(-radius_px, -radius_px, 2 * radius_px, 2 * radius_px, parent)

        self.env = env
        self.p_cfg = p_cfg
        self.v = v
        self.is_active = False

        # MuJoCo mocap handle
        mujoco_model = env._mujoco_env.unwrapped.model
        self._mujoco_data = env._mujoco_env.unwrapped.data
        body_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "force_circle_mocap")
        self._mocap_index = mujoco_model.body_mocapid[body_id]

        pen = QPen(_rgb(v.force_circle_color), v.force_circle_thickness)
        self.setPen(pen)
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setVisible(False)

    def toggle(self):
        self.is_active = not self.is_active
        self.setVisible(self.is_active)
        if not self.is_active:
            self._mujoco_data.mocap_pos[self._mocap_index, 2] = 100.0

    def update_position(self, scene_x: float, scene_y: float):
        if not self.is_active:
            return
        self.setPos(scene_x, scene_y)
        world_x = scene_x / self.v.scale
        world_z = -scene_y / self.v.scale  # scene Y down → MuJoCo Z up
        self._mujoco_data.mocap_pos[self._mocap_index, 0] = world_x
        self._mujoco_data.mocap_pos[self._mocap_index, 2] = world_z


# ---------------------------------------------------------------------------
# Scene, View, Window
# ---------------------------------------------------------------------------

class PendulumScene(QGraphicsScene):
    """QGraphicsScene that holds all simulation items."""

    def __init__(self, env, p_cfg, v, parent=None):
        super().__init__(parent)
        self.env = env
        self.p_cfg = p_cfg
        self.v = v

        self.setBackgroundBrush(QBrush(_rgb(v.bg_color)))

        # Scene rect centred on (0, 0) so cart x=0 is at centre
        self.setSceneRect(-v.width / 2, -v.height / 2, v.width, v.height)

        fg = _rgb(v.fg_color)

        # --- Track ---
        cart_w_px = v.cart_width * v.scale
        cart_node_px = v.cart_node_radius * v.scale
        track_len = p_cfg.track_length * v.scale + cart_w_px + cart_node_px
        track_h = v.track_h * v.scale
        track_rad = int(v.track_rad * v.scale)
        self._track = QGraphicsRectItem(-track_len / 2, -track_h / 2, track_len, track_h)
        pen_track = QPen(fg, v.track_thick)
        self._track.setPen(pen_track)
        self._track.setBrush(Qt.BrushStyle.NoBrush)
        self.addItem(self._track)

        # --- Cart ---
        self._cart = CartItem(env, p_cfg, v)
        self.addItem(self._cart)

        # --- Cart pivot node ---
        cnr = v.cart_node_radius * v.scale
        self._cart_node = QGraphicsEllipseItem(-cnr, -cnr, 2 * cnr, 2 * cnr)
        self._cart_node.setBrush(QBrush(fg))
        self._cart_node.setPen(QPen(Qt.PenStyle.NoPen))
        self.addItem(self._cart_node)

        # --- Pendulum links (lines) and tip nodes ---
        n = p_cfg.num_links
        node_rad = p_cfg.node_radius * v.scale
        node_inner = node_rad - v.node_outline_width * v.scale
        pend_w = v.pendulum_width * v.scale

        self._links: list[QGraphicsLineItem] = []
        self._nodes_outer: list[QGraphicsEllipseItem] = []
        self._nodes_inner: list[QGraphicsEllipseItem] = []

        for _ in range(n):
            line = QGraphicsLineItem()
            pen = QPen(fg, pend_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            line.setPen(pen)
            self.addItem(line)
            self._links.append(line)

            # outer node
            outer = QGraphicsEllipseItem(-node_rad, -node_rad, 2 * node_rad, 2 * node_rad)
            outer.setBrush(QBrush(fg))
            outer.setPen(QPen(Qt.PenStyle.NoPen))
            self.addItem(outer)
            self._nodes_outer.append(outer)

            # inner node
            inner = QGraphicsEllipseItem(-node_inner, -node_inner, 2 * node_inner, 2 * node_inner)
            inner.setBrush(QBrush(_rgb(v.node_fill_color)))
            inner.setPen(QPen(Qt.PenStyle.NoPen))
            self.addItem(inner)
            self._nodes_inner.append(inner)

        # --- Force circle ---
        self._force_circle = ForceCircleItem(env, p_cfg, v)
        self.addItem(self._force_circle)

    # ---------------------------------------------------------------

    def sync_from_state(self, state: np.ndarray):
        """Update item positions from MuJoCo state ``[x, θ, ẋ, θ̇]``."""
        v = self.v
        p_cfg = self.p_cfg
        half_length = p_cfg.track_length / 2.0
        cart_x = float(np.clip(state[0], -half_length, half_length))
        cart_px = cart_x * v.scale

        self._cart.setPos(cart_px, 0)
        self._cart_node.setPos(cart_px, 0)

        pivot_x, pivot_y = cart_px, 0.0

        for i in range(p_cfg.num_links):
            theta_i = state[1 + i]
            length_px = p_cfg.link_lengths[i] * v.scale
            end_x = pivot_x + length_px * np.sin(theta_i)
            end_y = pivot_y - length_px * np.cos(theta_i)

            self._links[i].setLine(QLineF(pivot_x, pivot_y, end_x, end_y))
            self._nodes_outer[i].setPos(end_x, end_y)
            self._nodes_inner[i].setPos(end_x, end_y)

            pivot_x, pivot_y = end_x, end_y


class PendulumView(QGraphicsView):
    """QGraphicsView with anti-aliasing and mouse-tracking for the force circle."""

    def __init__(self, scene: PendulumScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self._scene: PendulumScene = scene

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        self._scene._force_circle.update_position(scene_pos.x(), scene_pos.y())
        super().mouseMoveEvent(event)


class PendulumWindow(QMainWindow):
    """Main window that hosts the simulation view and drives the timer loop."""

    def __init__(
        self,
        env: CartPendulumEnv,
        model,
        p_cfg: PendulumConfig,
        v: VisualizationConfig,
    ):
        super().__init__()
        self.env = env
        self.model = model
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

        # Timer at config fps
        self._timer = QTimer(self)
        interval_ms = max(1, int(1000 / p_cfg.fps))
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # -- key handling -------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_R:
            self.obs, _ = self.env.reset()
        elif event.key() == Qt.Key.Key_F:
            self._scene._force_circle.toggle()
        super().keyPressEvent(event)

    # -- simulation tick ----------------------------------------------------

    def _tick(self):
        cart = self._scene._cart

        if cart.is_dragging:
            action = np.array([0.0], dtype=np.float32)
        elif self.model is not None:
            action, _ = self.model.predict(self.obs, deterministic=True)
        else:
            action = np.array([0.0])

        self.obs, _reward, terminated, truncated, _ = self.env.step(action)

        if (terminated or truncated) and not cart.is_dragging:
            self.obs, _ = self.env.reset()

        self._scene.sync_from_state(self.env._state)


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
    window = PendulumWindow(env, model, p_cfg, v)
    window.show()
    sys.exit(app.exec())


# ---- CLI entry point -------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Visualise pendulum")
    parser.add_argument("--model", type=str, default="",
                        help="Path to trained model (omit for random actions)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(model_path=args.model)