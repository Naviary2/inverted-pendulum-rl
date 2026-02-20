# pendulum/renderer.py

"""QGraphicsScene and QGraphicsView that render the pendulum simulation."""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QLineF, QRectF
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
)

from .config import PendulumConfig, VisualizationConfig
from .environment import CartPendulumEnv
from .interaction import CartItem, ForceCircleItem


def _rgb(t: tuple) -> QColor:
    """Convert an (r, g, b) or (r, g, b, a) tuple to a QColor."""
    return QColor(*t)


def _draw_rounded_rect_shadow(
    painter: QPainter,
    rect: QRectF,
    radius_px: float,
    spread: float,
    layers: int = 8,
) -> None:
    """Draw a layered soft drop shadow for a rounded rectangle.

    Paints ``layers`` semi-transparent rounded rects that grow from the
    rect outward.  The outermost layer always ends exactly at ``spread``
    pixels from ``rect``, regardless of ``layers``; more layers just pack
    them more densely together within that same spread distance.
    """
    painter.setPen(QPen(Qt.PenStyle.NoPen))
    for i in range(layers, 0, -1):
        t = i / layers                      # 1.0 (outermost) → 1/layers (innermost)
        alpha = int(10 * (1 - t) * t * 4)  # bell-curve alpha, peak in the middle
        expand = spread * t
        shadow_rect = rect.adjusted(-expand, -expand, expand, expand)
        painter.setBrush(QBrush(QColor(0, 0, 0, alpha)))
        painter.drawRoundedRect(shadow_rect, radius_px + expand, radius_px + expand)


class SimulationWidget(QGraphicsObject):
    """Rounded-rect widget that visually contains the track, cart, and pendulums.

    Constants (all lengths in metres):
        widget_padding_x      - left / right padding inside the rounded rect
        widget_padding_y      - top / bottom padding inside the rounded rect
        widget_bg_color       - background fill colour (slightly darker than scene bg)
        widget_border_radius  - corner radius of the rounded rect
        widget_theme_color    - outline / accent colour
        widget_outline_width  - stroke width of the outline
        widget_shadow_blur    - blur radius of the drop shadow
    """

    def __init__(self, p_cfg, v, parent=None):
        super().__init__(parent)
        self._v = v

        # Widget half-width: half the physics track + cart-body margin + h-padding
        # The margin uses 0.8 × cart_body_width because the full track visual width
        # adds 1.6 × body_w (one full body width split equally on each side).
        half_w = (
            p_cfg.track_length / 2
            + v.cart_body_width * 0.8
            + v.widget_padding_x
        ) * v.scale

        # Widget top edge: total pendulum height above cart + node width + v-padding
        total_link_len = sum(p_cfg.link_lengths) + p_cfg.node_radius
        top = -(total_link_len + v.widget_padding_y) * v.scale

        # Widget bottom edge: total pendulum height below cart + node width + v-padding
        bottom = -top

        self._rect = QRectF(-half_w, top, 2 * half_w, bottom - top)

        # Shadow spread in pixels, pre-computed once from the config constant.
        self._shadow_spread = v.widget_shadow_blur * v.scale

    def boundingRect(self) -> QRectF:
        # Expand by half the outline pen width so the stroke is never clipped,
        # and by the full shadow spread so the shadow is never clipped either.
        half_pen = self._v.widget_outline_width * self._v.scale / 2
        spread = self._shadow_spread
        margin = half_pen + spread
        return self._rect.adjusted(-margin, -margin, margin, margin)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        v = self._v
        radius_px = v.widget_border_radius * v.scale

        # --- Shadow (painted statically; does NOT re-run when children move) ---
        # Layered semi-transparent rounded rects approximate a soft drop shadow
        # without using QGraphicsDropShadowEffect (which forces a full offscreen
        # re-render every frame when any child item changes position).
        _draw_rounded_rect_shadow(painter, self._rect, radius_px, self._shadow_spread)

        # --- Widget background + themed outline ---
        pen = QPen(_rgb(v.widget_theme_color), v.widget_outline_width * v.scale)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QBrush(_rgb(v.widget_bg_color)))
        painter.drawRoundedRect(self._rect, radius_px, radius_px)


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
        self._widget = SimulationWidget(p_cfg, v)
        self.addItem(self._widget)

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

        # --- Pendulum links (bars) and joints (children of widget) ---
        n = p_cfg.num_links
        pend_w = v.pendulum_width * v.scale
        joint_rad = v.joint_radius * v.scale

        self._links: list[QGraphicsLineItem] = []
        self._joints: list[QGraphicsEllipseItem] = []

        for _ in range(n):
            # Dark metal bar: rounded-cap thick line = visually a rounded rectangle
            line = QGraphicsLineItem(self._widget)
            pen = QPen(_rgb(v.pendulum_bar_color), pend_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            line.setPen(pen)
            self._links.append(line)

            # Joint circle at the far end of this link (drawn on top of the bar)
            joint = QGraphicsEllipseItem(-joint_rad, -joint_rad, 2 * joint_rad, 2 * joint_rad, self._widget)
            joint.setBrush(QBrush(_rgb(v.joint_color)))
            joint.setPen(QPen(Qt.PenStyle.NoPen))
            self._joints.append(joint)


        # --- Cart (child of widget) ---
        self._cart = CartItem(env, p_cfg, v, parent=self._widget)

        # Track the previous cart x position so we can compute delta for wheel rotation.
        # None on the first frame to avoid a large spurious rotation.
        self._prev_cart_x: float | None = None

        # --- Force circle (child of widget) ---
        self._force_circle = ForceCircleItem(env, p_cfg, v, parent=self._widget)

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
            self._joints[i].setPos(end_x, end_y)

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
