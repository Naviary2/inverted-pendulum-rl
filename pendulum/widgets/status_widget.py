# pendulum/widgets/status_widget.py

"""StatusWidget: top-left HUD overlay displaying live simulation diagnostics."""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen
from .widget_base import BaseWidget


class StatusWidget(BaseWidget):
    """HUD panel shown in the top-left corner of the simulation window.

    Layout (top → bottom):
    ┌──────────────────────────────┐  ← white themed border
    │           5.2s              │  ← runtime, large centered monospace font
    │  FPS             238        │  ← title right-aligned | value left-aligned
    │  PHYSICS         240 Hz     │
    │  AGENT           Enabled    │
    └──────────────────────────────┘

    All values are updated each frame via ``update_status()``.
    """

    # White theme to match the physics widget border style
    _THEME_COLOR: tuple[int, int, int] = (255, 255, 255)
    _BG_COLOR: tuple = (28, 30, 34)   # dark blue-tinted background

    # --- Layout constants (pixels) ---
    _W: float = 270.0
    _H: float = 170.0
    _PAD_X: float = 16.0    # horizontal outer padding
    _PAD_TOP: float = 8.0   # top padding

    # Runtime section
    _TIMER_H: float = 52.0

    # Stats section — row height for each label+value pair
    _STAT_ROW_H: float = 32.0

    # Two-column split: left = titles (right-aligned), right = values (left-aligned)
    # Proportions of inner_w (W - 2*PAD_X)
    _TITLE_COL_FRAC: float = 0.45   # left column fraction
    _COL_GAP: float = 12.0          # gap between the two columns

    # --- Typography ---
    _TIMER_FONT_SIZE: float = 34.0
    _LABEL_FONT_SIZE: float = 13.0
    _VALUE_FONT_SIZE: float = 18.0

    # --- Colours ---
    _COL_LABEL: QColor = QColor(115, 118, 130)      # muted blue-grey for titles
    _COL_VALUE: QColor = QColor(235, 237, 242)      # near-white for values
    _COL_TIMER: QColor = QColor(235, 237, 242)      # white timer
    _COL_TIMER_NEG: QColor = QColor(130, 132, 140)  # grey when warmup (negative)

    def __init__(self, parent=None):
        rect = QRectF(0.0, 0.0, self._W, self._H)
        # Pass scale=200.0 so _BORDER_SIZE (0.02 m) yields the same 4 px border
        # as the PendulumWidget, keeping border width consistent across all widgets.
        super().__init__(rect, self._THEME_COLOR, 200.0, parent)
        self._sim_time_secs: float = -0.5
        self._fps: float = 0.0
        self._physics_hz: int = 0
        self._agent_active: bool = False

    def update_status(
        self,
        sim_time_secs: float,
        fps: float,
        physics_hz: int,
        agent_active: bool,
    ) -> None:
        """Refresh all displayed values and schedule a repaint."""
        self._sim_time_secs = sim_time_secs
        self._fps = fps
        self._physics_hz = physics_hz
        self._agent_active = agent_active
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        super().paint(painter, option, widget)

        pad = self._PAD_X
        inner_w = self._W - 2.0 * pad
        title_col_w = inner_w * self._TITLE_COL_FRAC
        val_col_x = pad + title_col_w + self._COL_GAP
        val_col_w = inner_w - title_col_w - self._COL_GAP

        # ── 1. Runtime (system monospace, bold, centered) ───────────────────
        timer_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        timer_font.setPointSizeF(self._TIMER_FONT_SIZE)
        timer_font.setBold(True)
        painter.setFont(timer_font)
        timer_color = self._COL_TIMER_NEG if self._sim_time_secs < 0 else self._COL_TIMER
        painter.setPen(QPen(timer_color))
        painter.drawText(
            QRectF(pad, self._PAD_TOP, inner_w, self._TIMER_H),
            Qt.AlignmentFlag.AlignCenter,
            f"{self._sim_time_secs:.1f}s",
        )

        # ── 2. Stat rows ─────────────────────────────────────────────────────
        stats = [
            ("FPS",     f"{self._fps:.0f}",       self._COL_VALUE),
            ("PHYSICS", f"{self._physics_hz} Hz", self._COL_VALUE),
            ("AGENT",   "Enabled" if self._agent_active else "Disabled",
                        self._COL_VALUE if self._agent_active else self._COL_LABEL),
        ]
        stat_y_start = self._PAD_TOP + self._TIMER_H

        for i, (title, value, val_color) in enumerate(stats):
            row_y = stat_y_start + i * self._STAT_ROW_H

            # Title — right-aligned in left column
            label_font = QFont()
            label_font.setPointSizeF(self._LABEL_FONT_SIZE)
            painter.setFont(label_font)
            painter.setPen(QPen(self._COL_LABEL))
            painter.drawText(
                QRectF(pad, row_y, title_col_w, self._STAT_ROW_H),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                title,
            )

            # Value — left-aligned in right column
            val_font = QFont()
            val_font.setPointSizeF(self._VALUE_FONT_SIZE)
            painter.setFont(val_font)
            painter.setPen(QPen(val_color))
            painter.drawText(
                QRectF(val_col_x, row_y, val_col_w, self._STAT_ROW_H),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                value,
            )


