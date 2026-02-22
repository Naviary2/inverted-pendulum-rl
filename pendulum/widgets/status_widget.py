# pendulum/widgets/status_widget.py

"""StatusWidget: top-left HUD overlay displaying live simulation diagnostics."""

from __future__ import annotations

from PySide6.QtCore import Qt, QLineF, QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from .widget_base import BaseWidget


class StatusWidget(BaseWidget):
    """HUD panel shown in the top-left corner of the simulation window.

    Layout (top → bottom):
    ┌──────────────────────────────┐  ← green themed border
    │         1.5 s               │  ← runtime in seconds, large bold font
    │ ─────────────────────────── │  ← thin divider
    │  FPS        PHYSICS         │  ← grey label row
    │  238        240 Hz          │  ← bright value row
    │                             │
    │  AGENT      CART LOCK       │  ← grey label row
    │  YES      [ LOCKED ]        │  ← value / coloured badge
    └──────────────────────────────┘

    All values are updated each frame via ``update_status()``.
    """

    _THEME_COLOR: tuple[int, int, int] = (50, 180, 50)   # green accent
    _BORDER_SIZE: float = 2.5                              # px (scale passed as 1.0)
    _BG_COLOR: tuple = (28, 30, 34)                        # dark blue-tinted background

    # --- Layout constants (pixels) ---
    _W: float = 280.0
    _H: float = 220.0
    _PAD_X: float = 16.0    # horizontal outer padding

    # Timer section
    _TIMER_Y: float = 10.0
    _TIMER_H: float = 58.0

    # Divider
    _DIV_Y: float = 76.0

    # Stat row 1 (FPS / Physics)
    _ROW1_LABEL_Y: float = 88.0
    _ROW1_VALUE_Y: float = 106.0

    # Stat row 2 (Agent / Cart Lock)
    _ROW2_LABEL_Y: float = 150.0
    _ROW2_VALUE_Y: float = 168.0
    _BADGE_H: float = 34.0    # height of the cart-lock badge

    # --- Typography ---
    _TIMER_FONT_SIZE: float = 34.0
    _LABEL_FONT_SIZE: float = 9.5
    _VALUE_FONT_SIZE: float = 17.0
    _BADGE_FONT_SIZE: float = 12.0

    # --- Colours ---
    _COL_LABEL: QColor = QColor(115, 118, 130)      # muted blue-grey for labels
    _COL_VALUE: QColor = QColor(235, 237, 242)      # near-white for values
    _COL_TIMER_POS: QColor = QColor(235, 237, 242)  # white when running
    _COL_TIMER_NEG: QColor = QColor(195, 80, 75)    # muted red during warmup
    _COL_AGENT_YES: QColor = QColor(90, 200, 100)   # green when agent active
    _COL_AGENT_NO: QColor = QColor(115, 118, 130)   # muted when no agent
    _COL_LOCK_ON_BG: QColor = QColor(195, 115, 20)  # amber fill when locked
    _COL_LOCK_ON_FG: QColor = QColor(255, 240, 215) # warm white text when locked
    _COL_LOCK_OFF_BG: QColor = QColor(48, 50, 56)   # dark fill when free
    _COL_LOCK_OFF_FG: QColor = QColor(115, 118, 130) # grey text when free

    def __init__(self, parent=None):
        rect = QRectF(0.0, 0.0, self._W, self._H)
        super().__init__(rect, self._THEME_COLOR, 1.0, parent)
        self._sim_time_secs: float = -0.5
        self._fps: float = 0.0
        self._physics_hz: int = 0
        self._agent_active: bool = False
        self._cart_locked: bool = False

    def update_status(
        self,
        sim_time_secs: float,
        fps: float,
        physics_hz: int,
        agent_active: bool,
        cart_locked: bool,
    ) -> None:
        """Refresh all displayed values and schedule a repaint."""
        self._sim_time_secs = sim_time_secs
        self._fps = fps
        self._physics_hz = physics_hz
        self._agent_active = agent_active
        self._cart_locked = cart_locked
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        super().paint(painter, option, widget)

        w = self._W
        pad = self._PAD_X
        inner_w = w - 2.0 * pad    # usable width between left/right pads
        col_w = inner_w / 2.0      # width of each of the two stat columns

        # ── 1. Runtime ──────────────────────────────────────────────────────
        timer_font = QFont()
        timer_font.setPointSizeF(self._TIMER_FONT_SIZE)
        timer_font.setBold(True)
        painter.setFont(timer_font)
        timer_color = self._COL_TIMER_NEG if self._sim_time_secs < 0 else self._COL_TIMER_POS
        painter.setPen(QPen(timer_color))
        painter.drawText(
            QRectF(pad, self._TIMER_Y, inner_w, self._TIMER_H),
            Qt.AlignmentFlag.AlignCenter,
            f"{self._sim_time_secs:.1f} s",
        )

        # ── 2. Divider ───────────────────────────────────────────────────────
        painter.setPen(QPen(QColor(60, 62, 68), 1.0))
        painter.drawLine(QLineF(pad, self._DIV_Y, w - pad, self._DIV_Y))

        # ── 3. Stat row 1 — FPS (left), Physics (right) ─────────────────────
        self._draw_stat(
            painter, pad, self._ROW1_LABEL_Y, self._ROW1_VALUE_Y, col_w,
            "FPS", f"{self._fps:.0f}", self._COL_VALUE,
        )
        self._draw_stat(
            painter, pad + col_w, self._ROW1_LABEL_Y, self._ROW1_VALUE_Y, col_w,
            "PHYSICS", f"{self._physics_hz} Hz", self._COL_VALUE,
        )

        # ── 4. Stat row 2 — Agent (left), Cart Lock badge (right) ───────────
        agent_color = self._COL_AGENT_YES if self._agent_active else self._COL_AGENT_NO
        self._draw_stat(
            painter, pad, self._ROW2_LABEL_Y, self._ROW2_VALUE_Y, col_w,
            "AGENT", "YES" if self._agent_active else "NO", agent_color,
        )
        self._draw_label(painter, pad + col_w, self._ROW2_LABEL_Y, col_w, "CART LOCK")
        self._draw_lock_badge(painter, pad + col_w, self._ROW2_VALUE_Y, col_w - 8.0)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_label(
        self,
        painter: QPainter,
        x: float,
        y: float,
        w: float,
        text: str,
    ) -> None:
        """Draw a grey small-caps-style label."""
        font = QFont()
        font.setPointSizeF(self._LABEL_FONT_SIZE)
        painter.setFont(font)
        painter.setPen(QPen(self._COL_LABEL))
        painter.drawText(
            QRectF(x, y, w, 20.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def _draw_stat(
        self,
        painter: QPainter,
        x: float,
        label_y: float,
        value_y: float,
        col_w: float,
        label: str,
        value: str,
        value_color: QColor,
    ) -> None:
        """Draw a label + value pair stacked vertically."""
        self._draw_label(painter, x, label_y, col_w, label)

        font = QFont()
        font.setPointSizeF(self._VALUE_FONT_SIZE)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(value_color))
        painter.drawText(
            QRectF(x, value_y, col_w - 4.0, 32.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            value,
        )

    def _draw_lock_badge(
        self,
        painter: QPainter,
        x: float,
        y: float,
        badge_w: float,
    ) -> None:
        """Draw a coloured pill badge showing LOCKED / FREE."""
        badge_rect = QRectF(x, y, badge_w, self._BADGE_H)
        radius = 7.0

        if self._cart_locked:
            bg = self._COL_LOCK_ON_BG
            fg = self._COL_LOCK_ON_FG
            label = "LOCKED"
        else:
            bg = self._COL_LOCK_OFF_BG
            fg = self._COL_LOCK_OFF_FG
            label = "FREE"

        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.drawRoundedRect(badge_rect, radius, radius)

        font = QFont()
        font.setPointSizeF(self._BADGE_FONT_SIZE)
        font.setBold(self._cart_locked)
        painter.setFont(font)
        painter.setPen(QPen(fg))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, label)
