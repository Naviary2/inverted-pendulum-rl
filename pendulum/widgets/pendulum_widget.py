# pendulum/widgets/pendulum_widget.py

"""PendulumWidget and TickRulerItem for the pendulum simulation scene."""

from __future__ import annotations

from PySide6.QtCore import Qt, QLineF, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem

from .widget_base import BaseWidget


class PendulumWidget(BaseWidget):
    """Rounded-rect widget that visually contains the track, cart, and pendulums.

    Geometry is derived from the physics and visualisation configs; the only
    styling parameter is ``theme_color``, inherited from ``BaseWidget``.

    Padding around the track content is owned by this widget as constants;
    the base class receives only the final dimensions.
    """

    # Padding around the track content, in metres (constant for this widget type)
    _PADDING_X: float = 1.0   # m  left / right padding
    _PADDING_Y: float = 0.25  # m  top / bottom padding
    _THEME_COLOR: tuple = (70, 140, 255)  # blue accent color
    # _THEME_COLOR: tuple = (50, 160, 30)  # green

    def __init__(self, p_cfg, v, parent=None):
        # Base content half-width: half the physics track + cart-body visual margin
        # The margin uses 0.8 × cart_body_width because the full track visual width
        # adds 1.6 × body_w (one full body width split equally on each side).
        base_half_w = (p_cfg.track_length / 2 + v.cart_body_width * 0.8) * v.scale

        # Base content half-height: total pendulum height above/below cart + node radius
        base_half_h = (sum(p_cfg.link_lengths) + p_cfg.node_radius) * v.scale

        # Full widget dimensions = base content + padding
        padding_x_px = self._PADDING_X * v.scale
        padding_y_px = self._PADDING_Y * v.scale
        half_w = base_half_w + padding_x_px
        half_h = base_half_h + padding_y_px

        rect = QRectF(-half_w, -half_h, 2 * half_w, 2 * half_h)
        super().__init__(rect, self._THEME_COLOR, parent)


class TickRulerItem(QGraphicsItem):
    """Graduated tick marks drawn below the track.

    Renders ticks from ``-tick_range`` m to ``+tick_range`` m centred on x = 0.
    Three levels of prominence:
        * integer positions  - tallest, most opaque
        * half-integer positions - medium
        * tenth-step positions - shortest, most transparent

    The item's ``boundingRect`` covers only the drawn ticks so it never
    influences the layout of the parent ``PendulumWidget``.
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
