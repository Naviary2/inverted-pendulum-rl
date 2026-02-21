# pendulum/renderer.py

"""QGraphicsScene and QGraphicsView that render the pendulum simulation."""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QLineF, QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
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


class TickRulerItem(QGraphicsItem):
    """Graduated tick marks drawn below the track.

    Renders ticks from ``-tick_range`` m to ``+tick_range`` m centred on x = 0.
    Three levels of prominence:
        * integer positions  - tallest, most opaque
        * half-integer positions - medium
        * tenth-step positions - shortest, most transparent

    The item's ``boundingRect`` covers only the drawn ticks so it never
    influences the layout of the parent ``SimulationWidget``.
    """

    def __init__(self, v, parent=None):
        super().__init__(parent)
        self._v = v

    def boundingRect(self) -> QRectF:
        v = self._v
        half_w = v.tick_range * v.scale
        y_mid = (v.track_h / 2 + v.tick_gap) * v.scale
        max_half_h = v.tick_zero_height / 2 * v.scale
        y_top = y_mid - max_half_h
        # Reserve space for the number labels below the ticks
        label_area = (v.tick_label_gap + v.tick_label_height) * v.scale
        y_bot = y_mid + max_half_h + label_area
        return QRectF(-half_w, y_top, 2 * half_w, y_bot - y_top)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        v = self._v
        y_mid = (v.track_h / 2 + v.tick_gap) * v.scale
        r, g, b = v.fg_color

        steps = round(v.tick_range * 10)
        for i in range(-steps, steps + 1):
            x_px = (i / 10) * v.scale

            if i == 0:             # zero tick – most prominent
                h_px = v.tick_zero_height * v.scale
                alpha = v.tick_zero_alpha
                width = v.tick_zero_width * v.scale
            elif i % 10 == 0:      # integer tick
                h_px = v.tick_int_height * v.scale
                alpha = v.tick_int_alpha
                width = v.tick_int_width * v.scale
            elif i % 5 == 0:       # half-integer tick
                h_px = v.tick_half_height * v.scale
                alpha = v.tick_half_alpha
                width = v.tick_half_width * v.scale
            else:                  # tenth-step tick
                h_px = v.tick_tenth_height * v.scale
                alpha = v.tick_tenth_alpha
                width = v.tick_tenth_width * v.scale

            pen = QPen(QColor(r, g, b, alpha), width)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            painter.drawLine(QLineF(x_px, y_mid - h_px / 2, x_px, y_mid + h_px / 2))

        # Draw number labels below each integer tick
        font = QFont(v.tick_label_font_family)
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSizeF(v.tick_label_font_size)
        painter.setFont(font)
        label_y = y_mid + v.tick_zero_height / 2 * v.scale + v.tick_label_gap * v.scale
        label_w_px = 60.0  # wide enough for "-3"
        label_h_px = v.tick_label_height * v.scale
        for i in range(-v.tick_range, v.tick_range + 1):
            x_px = i * v.scale
            alpha = v.tick_zero_alpha if i == 0 else v.tick_int_alpha
            painter.setPen(QPen(QColor(r, g, b, alpha)))
            painter.drawText(
                QRectF(x_px - label_w_px / 2, label_y, label_w_px, label_h_px),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                str(i),
            )


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
            inner.setBrush(QBrush(_rgb(v.widget_theme_color)))
            inner.setPen(QPen(Qt.PenStyle.NoPen))
            self._nodes_inner.append(inner)


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
