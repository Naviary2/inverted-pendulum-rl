# pendulum/widgets/widget_base.py

"""Base class for rounded-rect display widgets.

All concrete widgets inherit from ``BaseWidget``, which owns the shadow,
background fill, and themed outline.  Only the per-widget ``theme_color`` is
required at construction time; everything else is derived from the widget's
geometry or is constant across all widgets.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsObject


# ---- Helpers ----------------------------------------------------------------

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


# ---- Base widget ------------------------------------------------------------

class BaseWidget(QGraphicsObject):
    """Rounded-rect widget with drop shadow, background fill, and themed outline.

    The border radius and drop shadow spread are each proportional to the
    widget's shortest side (linear scaling).  The border/outline size is a
    single constant shared by all widgets.  The only per-widget styling
    parameter required at construction time is ``theme_color``.

    Subclasses supply their own ``rect`` (computed from their own geometry)
    and call ``super().__init__(rect, theme_color, scale, parent)``.

    Class attributes (override in subclasses to customise):
        _BORDER_SIZE          - outline stroke width in metres (constant)
        _BG_COLOR             - background fill colour
        _BORDER_RADIUS_FACTOR - border_radius = factor * shortest_side (px)
        _SHADOW_SPREAD_FACTOR - shadow_spread  = factor * shortest_side (px)
    """

    _BORDER_SIZE: float = 0.02           # m  outline stroke width (all widgets)
    _BG_COLOR: tuple = (35, 35, 35)      # background fill colour
    _BORDER_RADIUS_FACTOR: float = 0.0435  # border_radius ∝ shortest side
    _SHADOW_SPREAD_FACTOR: float = 0.0435  # shadow_spread  ∝ shortest side

    def __init__(
        self,
        rect: QRectF,
        theme_color: tuple,
        scale: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._rect = rect
        self._theme_color = theme_color

        shortest_side = min(rect.width(), rect.height())
        self._radius_px: float = shortest_side * self._BORDER_RADIUS_FACTOR
        self._shadow_spread: float = shortest_side * self._SHADOW_SPREAD_FACTOR
        self._border_size_px: float = self._BORDER_SIZE * scale

    def boundingRect(self) -> QRectF:
        # Expand by half the outline pen so the stroke is never clipped,
        # and by the full shadow spread so the shadow is never clipped either.
        half_pen = self._border_size_px / 2
        margin = half_pen + self._shadow_spread
        return self._rect.adjusted(-margin, -margin, margin, margin)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        # --- Shadow ---
        _draw_rounded_rect_shadow(
            painter, self._rect, self._radius_px, self._shadow_spread
        )

        # --- Background + themed outline ---
        pen = QPen(_rgb(self._theme_color), self._border_size_px)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QBrush(_rgb(self._BG_COLOR)))
        painter.drawRoundedRect(self._rect, self._radius_px, self._radius_px)
