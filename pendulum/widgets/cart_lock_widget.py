# pendulum/widgets/cart_lock_widget.py

"""CartLockWidget: a square button that displays and toggles the cart lock state."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtSvg import QSvgRenderer

from .widget_base import BaseWidget


class CartLockWidget(BaseWidget):
    """Roughly-square button widget for viewing and toggling the cart lock.

    Layout (top → bottom):
    ┌──────────────┐
    │  CART LOCK   │  ← muted header label
    │   🔒 / 🔓    │  ← lock.svg or unlock.svg, centred
    │  Locked/Free │  ← white when locked, grey when free
    └──────────────┘

    Clicking anywhere on the widget calls ``cart_item.toggle_lock()``.
    Hovering shows a semi-transparent highlight and a pointer cursor.
    """

    _THEME_COLOR: tuple = (220, 160, 40)    # amber
    _BG_COLOR: tuple = (35, 35, 35)
    _BG_COLOR_LOCKED: tuple = (195, 115, 20)  # amber background when cart is locked

    # Geometry
    _SIZE: float = 150.0   # width == height → roughly square

    # Layout (pixels)
    _PAD_TOP: float = 10.0
    _PAD_SIDE: float = 0.0    # header/status span full width and are centred
    _HEADER_H: float = 22.0
    _GAP: float = 3.0
    _ICON_H: float = 76.0
    _STATUS_H: float = 22.0
    _PAD_BOT: float = 10.0
    # Total: 10 + 22 + 5 + 76 + 5 + 22 + 10 = 150 → fills widget exactly

    # Typography
    _HEADER_FONT_SIZE: float = 14.0
    _STATUS_FONT_SIZE: float = 18.0

    # Colours
    _COL_HEADER: QColor = QColor(115, 118, 130)     # muted blue-grey
    _COL_FREE: QColor = QColor(235, 237, 242)       # grey when free
    _COL_LOCKED: QColor = QColor(235, 237, 242)     # near-white when locked
    _COL_HOVER: QColor = QColor(255, 255, 255, 18)  # subtle white tint on hover

    def __init__(self, cart_item, parent=None):
        rect = QRectF(0.0, 0.0, self._SIZE, self._SIZE)
        super().__init__(rect, self._THEME_COLOR, parent)

        self._cart_item = cart_item
        self._hovered: bool = False

        # Load SVG renderers once; these are cheap to keep around
        lock_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "res", "lock")
        )
        lock_path = os.path.join(lock_dir, "lock.svg")
        unlock_path = os.path.join(lock_dir, "unlock.svg")
        if not os.path.isfile(lock_path):
            raise FileNotFoundError(f"lock.svg not found: {lock_path!r}")
        if not os.path.isfile(unlock_path):
            raise FileNotFoundError(f"unlock.svg not found: {unlock_path!r}")
        self._lock_svg = QSvgRenderer(lock_path)
        self._unlock_svg = QSvgRenderer(unlock_path)

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Schedule a repaint (called externally when lock state changes)."""
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        super().paint(painter, option, widget)

        locked = self._cart_item.is_locked

        # Amber background overlay when locked — inset by half the border width so
        # the fill sits entirely inside the border stroke and doesn't bleed into it.
        if locked:
            half_pen = self._border_size_px / 2
            inner_rect = self._rect.adjusted(half_pen, half_pen, -half_pen, -half_pen)
            painter.setPen(QPen(Qt.PenStyle.NoPen))
            painter.setBrush(QBrush(QColor(*self._BG_COLOR_LOCKED)))
            painter.drawRoundedRect(inner_rect, self._radius_px, self._radius_px * 0.5)

        # Hover highlight overlay
        if self._hovered:
            painter.setPen(QPen(Qt.PenStyle.NoPen))
            painter.setBrush(QBrush(self._COL_HOVER))
            painter.drawRoundedRect(self._rect, self._radius_px, self._radius_px * 0.5)

        # ── 1. Header ────────────────────────────────────────────────
        header_font = QFont()
        header_font.setPointSizeF(self._HEADER_FONT_SIZE)
        painter.setFont(header_font)
        header_color = self._COL_LOCKED if locked else self._COL_HEADER
        painter.setPen(QPen(header_color))
        painter.drawText(
            QRectF(0.0, self._PAD_TOP, self._SIZE, self._HEADER_H),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "CART LOCK",
        )

        # ── 2. SVG icon ───────────────────────────────────────────────
        icon_y = self._PAD_TOP + self._HEADER_H + self._GAP
        icon_x = (self._SIZE - self._ICON_H) / 2.0
        icon_rect = QRectF(icon_x, icon_y, self._ICON_H, self._ICON_H)
        renderer = self._lock_svg if locked else self._unlock_svg
        renderer.render(painter, icon_rect)

        # ── 3. Status text ───────────────────────────────────────────
        status_y = icon_y + self._ICON_H + self._GAP
        if locked:
            painter.setPen(QPen(self._COL_LOCKED))
            status_text = "Locked"
        else:
            painter.setPen(QPen(self._COL_FREE))
            status_text = "Free"
        status_font = QFont()
        status_font.setPointSizeF(self._STATUS_FONT_SIZE)
        painter.setFont(status_font)
        painter.drawText(
            QRectF(0.0, status_y, self._SIZE, self._STATUS_H),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            status_text,
        )

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def hoverEnterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._cart_item.toggle_lock()
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)
