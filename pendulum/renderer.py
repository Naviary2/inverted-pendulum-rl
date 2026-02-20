# pendulum/renderer.py

"""QGraphicsScene and QGraphicsView that render the pendulum simulation."""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QLineF
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from .config import PendulumConfig, VisualizationConfig
from .environment import CartPendulumEnv
from .interaction import CartItem, ForceCircleItem


def _rgb(t: tuple) -> QColor:
    """Convert an (r, g, b) or (r, g, b, a) tuple to a QColor."""
    return QColor(*t)


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
        # Width: physics track length + one full cart body width for visual margin
        body_w_px = v.cart_body_width_nd * 2 * p_cfg.node_radius * v.scale
        track_len = p_cfg.track_length * v.scale + body_w_px
        track_h = v.track_h * v.scale
        self._track = QGraphicsRectItem(-track_len / 2, -track_h / 2, track_len, track_h)
        pen_track = QPen(fg, v.track_thick)
        self._track.setPen(pen_track)
        self._track.setBrush(Qt.BrushStyle.NoBrush)
        self.addItem(self._track)

        # --- Cart (body + struts + wheels + pivot node all in one item) ---
        self._cart = CartItem(env, p_cfg, v)
        self.addItem(self._cart)

        # Track the previous cart x position so we can compute delta for wheel rotation.
        # None on the first frame to avoid a large spurious rotation.
        self._prev_cart_x: float | None = None

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
        cart_px = state[0] * v.scale

        # Move cart and rotate wheels by how much the cart has moved this frame.
        # Skip rotation on the first frame to avoid a spurious jump from 0.
        if self._prev_cart_x is not None:
            delta_x_meters = state[0] - self._prev_cart_x
            self._cart.rotate_wheels(delta_x_meters)
        self._prev_cart_x = state[0]
        self._cart.setPos(cart_px, 0)

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
