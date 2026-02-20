# pendulum/interaction.py

"""Custom QGraphicsItem subclasses that handle user interaction with the simulation."""

from __future__ import annotations

import math
import os

import mujoco
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
)

from .config import PendulumConfig, VisualizationConfig
from .environment import CartPendulumEnv


def _rgb(t: tuple) -> QColor:
    """Convert an (r, g, b) or (r, g, b, a) tuple to a QColor."""
    return QColor(*t)


class CartItem(QGraphicsRectItem):
    """Cart with white body, side struts, rolling wheels, and a pivot node.

    Visual layers (back to front):
      1. Wheels (QGraphicsPixmapItem, behind parent via ItemStacksBehindParent)
      2. Struts (QGraphicsRectItem, behind parent via ItemStacksBehindParent)
      3. Cart body - the parent QGraphicsRectItem (white rectangle)
      4. Pivot node outer circle  (white, matches pendulum-node outline)
      5. Pivot node inner circle  (coloured, matches pendulum-node fill)

    The item's *local* coordinate origin is at the pivot (centre of the body)
    so that ``setPos(px, 0)`` places the cart correctly on the track.
    """

    def __init__(
        self,
        env: CartPendulumEnv,
        p_cfg: PendulumConfig,
        v: VisualizationConfig,
        parent=None,
    ):
        # ------------------------------------------------------------------
        # Derived pixel sizes
        # ------------------------------------------------------------------
        node_rad_px = p_cfg.node_radius * v.scale
        node_outline_px = v.node_outline_width * v.scale
        track_h_px = v.track_h * v.scale

        # Cart body: 3 node-diameters wide, track-height tall
        body_w = v.cart_body_width * v.scale
        body_h = track_h_px

        # Strut rectangles on the left/right sides of the body
        strut_w = v.cart_strut_width * v.scale
        strut_h = v.cart_strut_height * v.scale

        # Struts are angled toward the centre and their centres sit at the
        # bottom of the cart body, so the top and bottom wheel centres end up
        # at different distances from the two track rails → different radii.
        strut_angle_rad = math.radians(v.cart_strut_angle)
        cos_a = math.cos(strut_angle_rad)
        sin_a = math.sin(strut_angle_rad)

        # Bottom wheel sits just outside the track's bottom rail.
        #   bottom-wheel centre y = body_h/2 + strut_h/2 * cos_a
        #   track bottom rail   y = body_h/2
        #   radius = (body_h/2 + strut_h/2·cos_a) − body_h/2 = strut_h/2·cos_a
        wheel_rad_bottom = strut_h / 2 * cos_a
        # Top wheel sits just outside the track's top rail.
        #   top-wheel centre y = body_h/2 − strut_h/2·cos_a
        #   track top rail   y = −body_h/2
        #   radius = (−body_h/2) − (body_h/2 − strut_h/2·cos_a) = strut_h/2·cos_a − body_h
        wheel_rad_top = strut_h / 2 * cos_a - body_h
        if wheel_rad_top <= 0:
            raise ValueError(
                f"Top wheel radius is non-positive ({wheel_rad_top:.2f} px). "
                "Increase cart_strut_height or reduce track_h so that "
                "strut_height/2 * cos(strut_angle) > cart_body_height (= track_height)."
            )

        # ------------------------------------------------------------------
        # Base rect = cart body (white rectangle), centred on local (0, 0)
        # ------------------------------------------------------------------
        super().__init__(-body_w / 2, -body_h / 2, body_w, body_h, parent)

        self.env = env
        self.p_cfg = p_cfg
        self.v = v
        self.is_dragging = False
        self.is_locked = False

        # Wheel rotation is tracked per-wheel in rotate_wheels.
        # Each entry in _wheels is (pixmap_item, wheel_radius_in_metres).
        # _wheel_angles stores the current accumulated rotation (degrees) for each wheel.

        # MuJoCo data handle
        self._mujoco_data = env._mujoco_env.unwrapped.data

        fg = _rgb(v.fg_color)            # white
        node_color = _rgb(v.node_fill_color)

        # Cart body: white fill, no border
        self.setBrush(QBrush(fg))
        self.setPen(QPen(Qt.PenStyle.NoPen))

        # ------------------------------------------------------------------
        # Struts (drawn behind the body)
        # ------------------------------------------------------------------
        for sign in (-1, 1):
            strut = QGraphicsRectItem(-strut_w / 2, -strut_h / 2, strut_w, strut_h, self)
            strut.setPos(sign * (body_w / 2), body_h / 2)  # centre on bottom of cart body
            strut.setRotation(-sign * v.cart_strut_angle)   # tilt tops toward centre
            strut.setBrush(QBrush(fg))
            strut.setPen(QPen(Qt.PenStyle.NoPen))
            strut.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)
            strut.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        # ------------------------------------------------------------------
        # Wheels  (drawn behind the body)
        # ------------------------------------------------------------------
        wheel_path = os.path.join(os.path.dirname(__file__), "..", "res", "wheel.png")
        wheel_path = os.path.normpath(wheel_path)
        if not os.path.isfile(wheel_path):
            raise FileNotFoundError(f"Wheel image not found: {wheel_path!r}")
        raw_pixmap = QPixmap(wheel_path)
        wheel_pixmap_top = raw_pixmap.scaled(
            max(1, round(2 * wheel_rad_top)),
            max(1, round(2 * wheel_rad_top)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        wheel_pixmap_bottom = raw_pixmap.scaled(
            max(1, round(2 * wheel_rad_bottom)),
            max(1, round(2 * wheel_rad_bottom)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Each entry is (pixmap_item, wheel_radius_in_metres); see _wheel_angles.
        self._wheels: list[tuple[QGraphicsPixmapItem, float]] = []
        self._wheel_angles: list[float] = []   # accumulated rotation per wheel (degrees)
        for sx in (-1, 1):                          # left / right strut
            strut_cx = sx * (body_w / 2)
            # Cart-local positions of the rotated strut ends:
            #   top end  (0, −strut_h/2) rotated by −sx·angle about strut centre
            #   bot end  (0, +strut_h/2) rotated by −sx·angle about strut centre
            top_x = strut_cx - sx * strut_h / 2 * sin_a
            top_y = body_h / 2 - strut_h / 2 * cos_a
            bot_x = strut_cx + sx * strut_h / 2 * sin_a
            bot_y = body_h / 2 + strut_h / 2 * cos_a

            w_top = QGraphicsPixmapItem(wheel_pixmap_top, self)
            # Position so that the pixmap centre is at (top_x, top_y)
            w_top.setPos(top_x - wheel_rad_top, top_y - wheel_rad_top)
            w_top.setTransformOriginPoint(wheel_rad_top, wheel_rad_top)
            w_top.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)
            w_top.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._wheels.append((w_top, wheel_rad_top / v.scale))
            self._wheel_angles.append(0.0)

            w_bot = QGraphicsPixmapItem(wheel_pixmap_bottom, self)
            # Position so that the pixmap centre is at (bot_x, bot_y)
            w_bot.setPos(bot_x - wheel_rad_bottom, bot_y - wheel_rad_bottom)
            w_bot.setTransformOriginPoint(wheel_rad_bottom, wheel_rad_bottom)
            w_bot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent, True)
            w_bot.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._wheels.append((w_bot, wheel_rad_bottom / v.scale))
            self._wheel_angles.append(0.0)

        # ------------------------------------------------------------------
        # Pivot node (drawn on top of body)
        # Looks identical to the pendulum tip node: white outline + coloured fill
        # ------------------------------------------------------------------
        self._node_outer = QGraphicsEllipseItem(
            -node_rad_px, -node_rad_px, 2 * node_rad_px, 2 * node_rad_px, self
        )
        self._node_outer.setBrush(QBrush(fg))
        self._node_outer.setPen(QPen(Qt.PenStyle.NoPen))

        node_inner_rad = node_rad_px - node_outline_px
        self._node_inner = QGraphicsEllipseItem(
            -node_inner_rad, -node_inner_rad, 2 * node_inner_rad, 2 * node_inner_rad, self
        )
        self._node_inner.setBrush(QBrush(node_color))
        self._node_inner.setPen(QPen(Qt.PenStyle.NoPen))

        # Colours used when toggling the cart lock
        self._normal_color = node_color
        self._locked_color = _rgb(v.cart_locked_color)

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ------------------------------------------------------------------
    # Wheel animation
    # ------------------------------------------------------------------

    def rotate_wheels(self, delta_x_meters: float) -> None:
        """Rotate all wheels so they appear to roll along the track.

        ``delta_x_meters`` is the signed displacement of the cart since the
        last frame.  Positive = rightward motion.  Each wheel uses its own
        radius so that both top and bottom wheels roll without slipping.
        """
        for i, (w, rad_m) in enumerate(self._wheels):
            if rad_m > 0:
                self._wheel_angles[i] += math.degrees(delta_x_meters / rad_m)
                w.setRotation(self._wheel_angles[i])

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
            self._node_inner.setBrush(QBrush(self._locked_color))
        else:
            if not self.is_dragging:
                self._mujoco_data.eq_active = 0
            self._node_inner.setBrush(QBrush(self._normal_color))


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

        pen = QPen(_rgb(v.force_circle_color), v.force_circle_thickness * v.scale)
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
