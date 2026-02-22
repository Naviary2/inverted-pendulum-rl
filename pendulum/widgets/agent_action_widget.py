# pendulum/widgets/agent_action_widget.py

"""AgentActionWidget: scrolling force-history graph showing the agent's actions."""

from __future__ import annotations

import time
from collections import deque

from PySide6.QtCore import Qt, QLineF, QRectF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)

from .widget_base import BaseWidget


class AgentActionWidget(BaseWidget):
    """Scrolling line graph showing the agent's recent force actions.

    Layout (top → bottom):
    ┌──────────────────────┐  ← teal themed border
    │ AGENT ACTION (N)  +42│  ← title left-aligned, current force right-aligned
    ├──────────────────────┤
    │      │               │  ← graph: now at top, 2 s ago at bottom
    │      │╲              │      center vertical line = 0 force
    │      │ ╲___          │      force line connects successive readings
    │      │     ╲         │      area between center and line is shaded
    └──────────────────────┘

    Call ``update_force(force_newton)`` each simulation tick to push a new
    sample; the widget prunes readings older than ``_HISTORY_SECS`` and
    schedules a repaint automatically.
    """

    _THEME_COLOR: tuple = (60, 200, 170)   # teal accent

    # Widget dimensions (pixels)
    _W: float = 200.0
    _H: float = 440.0

    # Layout (pixels)
    _TITLE_H: float = 44.0       # height of the title row
    _PAD_X: float = 10.0         # horizontal outer padding
    _PAD_BOT: float = 8.0        # gap between last tick and widget bottom edge
    _PAD_TITLE_TOP: float = 4.0  # top padding inside the title row

    # Graph time window
    _HISTORY_SECS: float = 2.0

    # Typography
    _TITLE_FONT_SIZE: float = 9.5
    _FORCE_FONT_SIZE: float = 12.0

    # Appearance
    _LINE_ALPHA: int = 230          # opacity of the force line stroke
    _FILL_ALPHA_MAX: int = 155      # max opacity for the shaded fill (at max force)
    _CENTER_LINE_ALPHA: int = 45    # opacity of the center (0-force) guide line
    _SEP_ALPHA: int = 60            # opacity of the separator below the title

    # Minimum force guard (avoids division by zero when max_force is 0)
    _MIN_FORCE: float = 1.0
    # Extra pixel margin added to force-text measurement for the column split
    _FORCE_TEXT_PADDING: float = 4.0

    def __init__(self, max_force: float, parent=None):
        """
        Parameters
        ----------
        max_force:
            The maximum absolute force the agent can apply (Newtons).
            Used to scale the X axis of the graph.
        """
        rect = QRectF(0.0, 0.0, self._W, self._H)
        super().__init__(rect, self._THEME_COLOR, parent)

        self._max_force: float = max(max_force, self._MIN_FORCE)   # guard against division by zero
        self._force_history: deque[tuple[float, float]] = deque()
        self._current_force: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_force(self, force_newton: float) -> None:
        """Append a new force reading (Newtons) and schedule a repaint.

        Readings older than ``_HISTORY_SECS`` are automatically pruned.
        """
        now = time.perf_counter()
        self._force_history.append((now, force_newton))
        self._current_force = force_newton

        cutoff = now - self._HISTORY_SECS
        while self._force_history and self._force_history[0][0] < cutoff:
            self._force_history.popleft()

        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        super().paint(painter, option, widget)

        r, g, b = self._THEME_COLOR
        now = time.perf_counter()

        pad_x = self._PAD_X
        inner_w = self._W - 2.0 * pad_x
        graph_top = self._TITLE_H
        graph_bot = self._H - self._PAD_BOT
        graph_h = graph_bot - graph_top
        center_x = self._W / 2.0
        half_graph_w = inner_w / 2.0
        graph_left = pad_x
        graph_right = self._W - pad_x

        # ── 1. Title row ─────────────────────────────────────────────
        title_font = QFont()
        title_font.setPointSizeF(self._TITLE_FONT_SIZE)
        force_font = QFont()
        force_font.setPointSizeF(self._FORCE_FONT_SIZE)
        force_font.setBold(True)

        force_text = f"{self._current_force:+.0f}"

        # Measure actual rendered widths to split the row without overlap.
        force_text_w = QFontMetricsF(force_font).horizontalAdvance(force_text) + self._FORCE_TEXT_PADDING
        title_col_w = inner_w - force_text_w

        painter.setPen(QPen(QColor(r, g, b)))

        # Left-aligned label in the left column
        painter.setFont(title_font)
        painter.drawText(
            QRectF(pad_x, self._PAD_TITLE_TOP, title_col_w, self._TITLE_H - self._PAD_TITLE_TOP),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "AGENT ACTION (N)",
        )

        # Right-aligned current force value in the right column
        painter.setFont(force_font)
        painter.drawText(
            QRectF(pad_x + title_col_w, self._PAD_TITLE_TOP, force_text_w, self._TITLE_H - self._PAD_TITLE_TOP),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            force_text,
        )

        # Separator below title
        painter.setPen(QPen(QColor(r, g, b, self._SEP_ALPHA), 1.0))
        painter.drawLine(QLineF(graph_left, graph_top, graph_right, graph_top))

        # ── 2. Center vertical guide line (0 force) ──────────────────
        painter.setPen(QPen(QColor(r, g, b, self._CENTER_LINE_ALPHA), 1.0))
        painter.drawLine(QLineF(center_x, graph_top, center_x, graph_bot))

        # ── 3. Build list of (x, y) display points ───────────────────
        history = list(self._force_history)
        if not history:
            return

        def t_to_y(t: float) -> float:
            return graph_top + ((now - t) / self._HISTORY_SECS) * graph_h

        def f_to_x(f: float) -> float:
            return center_x + (f / self._max_force) * half_graph_w

        # Iterate from most-recent (last in deque) to oldest (first),
        # stopping when we fall outside the graph window.
        points: list[tuple[float, float]] = []
        for ts, f in reversed(history):
            y = t_to_y(ts)
            if y > graph_bot:
                break   # older than the visible window
            points.append((f_to_x(f), y))

        if not points:
            return

        top_y = points[0][1]
        bot_y = points[-1][1]

        # ── 4. Shaded fill between the force line and the center ──────
        # Horizontal gradient: transparent at center, near-theme at edges.
        # The fill color is thus proportional to force magnitude.
        gradient = QLinearGradient(graph_left, 0.0, graph_right, 0.0)
        gradient.setColorAt(0.0, QColor(r, g, b, self._FILL_ALPHA_MAX))  # left (max −force)
        gradient.setColorAt(0.5, QColor(r, g, b, 0))                      # center (0 force)
        gradient.setColorAt(1.0, QColor(r, g, b, self._FILL_ALPHA_MAX))  # right (max +force)

        fill_path = QPainterPath()
        fill_path.moveTo(center_x, top_y)
        fill_path.lineTo(points[0][0], top_y)
        for px, py in points[1:]:
            fill_path.lineTo(px, py)
        fill_path.lineTo(center_x, bot_y)
        fill_path.closeSubpath()

        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QBrush(gradient))
        painter.drawPath(fill_path)

        # ── 5. Force line ────────────────────────────────────────────
        pen = QPen(QColor(r, g, b, self._LINE_ALPHA), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        line_path = QPainterPath()
        line_path.moveTo(points[0][0], points[0][1])
        for px, py in points[1:]:
            line_path.lineTo(px, py)
        painter.drawPath(line_path)
