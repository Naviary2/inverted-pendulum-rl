# pendulum/renderer.py

"""QGraphicsScene and QGraphicsView that render the pendulum simulation."""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QLineF, QRectF
from PySide6.QtGui import QBrush, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
)

from .config import PendulumConfig, VisualizationConfig
from .environment import CartPendulumEnv
from .interaction import CartItem, ForceCircleItem
from .widgets import ForceWidget, CartLockWidget, PendulumWidget, StatusWidget, TickRulerItem, _rgb


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

        # --- Simulation widget (rounded-rect container) ---
        self._widget = PendulumWidget(p_cfg, v)
        self.addItem(self._widget)

        # The ForceWidget always sits to the left, so the pendulum widget is
        # shifted right to make visual room.
        _agent_w = ForceWidget._W
        _agent_gap = 24.0                      # px gap between the two widgets
        _pendulum_offset_x = (_agent_w + _agent_gap) / 2.0
        self._widget.setPos(_pendulum_offset_x, 0)

        # --- Track (child of widget) ---
        # Width: physics track length + one full cart body width for visual margin
        body_w_px = v.cart_body_width * v.scale
        track_len = p_cfg.track_length * v.scale + body_w_px * 1.6  # plus some constant for padding
        track_h = v.track_h * v.scale
        track_path = QPainterPath()
        track_rad_px = v.track_rad * v.scale
        track_path.addRoundedRect(-track_len / 2, -track_h / 2, track_len, track_h, track_rad_px, track_rad_px)
        self._track = QGraphicsPathItem(track_path, self._widget)
        pen_track = QPen(fg, v.track_thick * v.scale)
        self._track.setPen(pen_track)
        self._track.setBrush(Qt.BrushStyle.NoBrush)

        # --- Tick ruler (graduated marks below the track, child of widget) ---
        self._tick_ruler = TickRulerItem(v, parent=self._widget)

        # --- Pendulum links (lines) and tip nodes (children of widget) ---
        n = p_cfg.num_links
        node_rad = p_cfg.node_radius * v.scale
        node_inner = node_rad - v.node_outline_width * v.scale
        pend_w = v.pendulum_width * v.scale

        self._links: list[QGraphicsLineItem] = []
        self._nodes_outer: list[QGraphicsEllipseItem] = []
        self._nodes_inner: list[QGraphicsEllipseItem] = []

        for _ in range(n):
            line = QGraphicsLineItem(self._widget)
            pen = QPen(fg, pend_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            line.setPen(pen)
            self._links.append(line)

            # outer node
            outer = QGraphicsEllipseItem(-node_rad, -node_rad, 2 * node_rad, 2 * node_rad, self._widget)
            outer.setBrush(QBrush(fg))
            outer.setPen(QPen(Qt.PenStyle.NoPen))
            self._nodes_outer.append(outer)

            # inner node
            inner = QGraphicsEllipseItem(-node_inner, -node_inner, 2 * node_inner, 2 * node_inner, self._widget)
            inner.setBrush(QBrush(_rgb(PendulumWidget._THEME_COLOR)))
            inner.setPen(QPen(Qt.PenStyle.NoPen))
            self._nodes_inner.append(inner)


        # --- Cart (child of widget) ---
        self._cart = CartItem(env, p_cfg, v, PendulumWidget._THEME_COLOR, parent=self._widget)

        # Track the previous cart x position so we can compute delta for wheel rotation.
        # None on the first frame to avoid a large spurious rotation.
        self._prev_cart_x: float | None = None

        # --- Status HUD widget (top-left corner, on top of everything) ---
        margin_px = 20.0
        self._status_widget = StatusWidget()
        self._status_widget.setPos(
            -v.width / 2 + margin_px,
            -v.height / 2 + margin_px,
        )
        self.addItem(self._status_widget)

        # --- Cart lock widget (centred on the full screen, below the pendulum widget) ---
        lock_gap_px = 70.0
        pw_bottom = self._widget.rect.bottom()    # y of pendulum widget's bottom edge in scene
        lock_size = CartLockWidget._SIZE
        self._cart_lock_widget = CartLockWidget(self._cart)
        self._cart_lock_widget.setPos(-lock_size / 2, pw_bottom + lock_gap_px)
        self.addItem(self._cart_lock_widget)

        # --- Force widget (always visible, left of the pendulum widget, vertically centred) ---
        force_widget_x = _pendulum_offset_x + self._widget.rect.left() - _agent_gap - _agent_w
        force_widget_y = -ForceWidget._H / 2.0
        self._force_widget = ForceWidget(p_cfg.force_magnitude)
        self._force_widget.setPos(force_widget_x, force_widget_y)
        self.addItem(self._force_widget)

        # --- Force circle (top-level scene item, renders above all widgets) ---
        self._force_circle = ForceCircleItem(env, p_cfg, v, scene_offset_x=_pendulum_offset_x)
        self.addItem(self._force_circle)

    # ---------------------------------------------------------------

    def update_status(
        self,
        sim_time_secs: float,
        fps: float,
        physics_hz: int,
        agent_active: bool,
    ) -> None:
        """Forward status values to the HUD widget."""
        self._status_widget.update_status(sim_time_secs, fps, physics_hz, agent_active)

    def update_cart_lock(self) -> None:
        """Refresh the cart lock widget (call after toggling lock state)."""
        self._cart_lock_widget.refresh()

    def update_force_widget(self, force_newton: float, sim_time_secs: float) -> None:
        """Push a new force reading (Newtons) to the Force widget."""
        self._force_widget.update_force(force_newton, sim_time_secs)

    def reset_force_widget(self) -> None:
        """Clear the Force widget history (call on simulation reset)."""
        self._force_widget.reset()

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

