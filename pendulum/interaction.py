# pendulum/interaction.py

"""Custom QGraphicsItem subclasses that handle user interaction with the simulation."""

from __future__ import annotations

import mujoco
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem

from .config import PendulumConfig, VisualizationConfig
from .environment import CartPendulumEnv


def _rgb(t: tuple) -> QColor:
    """Convert an (r, g, b) or (r, g, b, a) tuple to a QColor."""
    return QColor(*t)


class CartItem(QGraphicsRectItem):
    """Draggable cart rectangle.

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
        self.is_locked = False

        # MuJoCo data handle
        self._mujoco_data = env._mujoco_env.unwrapped.data

        # Visual styling
        self._normal_color = _rgb(v.node_fill_color)
        self._locked_color = _rgb(v.cart_locked_color)
        outline_color = _rgb(v.fg_color)

        self.setBrush(QBrush(self._normal_color))
        pen = QPen(outline_color, v.track_thick)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self.setPen(pen)

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # -- mouse interaction --------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self._mujoco_data.eq_active = 1
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            scene_pos = event.scenePos()
            target_x = scene_pos.x() / self.v.scale
            half_track = self.p_cfg.track_length / 2.0
            target_x = float(np.clip(target_x, -half_track, half_track))
            self._mujoco_data.mocap_pos[0, 0] = target_x
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            if not self.is_locked:
                self._mujoco_data.eq_active = 0
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def toggle_lock(self):
        """Toggle the cart lock. When locked, the cart is held at its current position."""
        self.is_locked = not self.is_locked
        if self.is_locked:
            self._mujoco_data.mocap_pos[0, 0] = self._mujoco_data.qpos[0]
            self._mujoco_data.eq_active = 1
            self.setBrush(QBrush(self._locked_color))
        else:
            if not self.is_dragging:
                self._mujoco_data.eq_active = 0
            self.setBrush(QBrush(self._normal_color))


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
