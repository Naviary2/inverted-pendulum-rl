# pendulum/widgets/agent_action_widget.py

"""ForceWidget: scrolling force-history graph (renamed class lives here)."""

from __future__ import annotations

import math
import time
from collections import deque

from PySide6.QtCore import Qt, QLineF, QRectF, QTimer
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
from PySide6.QtWidgets import QGraphicsItem

from .widget_base import BaseWidget


class ForceWidget(BaseWidget):
    """Scrolling line graph showing the force applied to the cart each frame.

    Layout (top → bottom):
    ┌──────────────────────┐  ← teal themed border
    │ Force (Newtons)  +42 │  ← title left-aligned (white), current force right-aligned (white)
    │  -200      0    +200 │  ← fixed x-axis labels
    │      │               │  ← graph: now at top, 2 s ago at bottom
    │      │╲  ----0.0s    │      scrolling gridlines with sim-time labels
    │      │ ╲___          │      center vertical line = 0 force
    │      │     ╲  -0.5s  │      force line + shaded fill
    └──────────────────────┘

    Call ``update_force(force_newton, sim_time_secs)`` each tick; history older
    than ``_HISTORY_SECS`` is pruned automatically.  Call ``reset()`` when the
    simulation resets.
    """

    _THEME_COLOR: tuple = (60, 200, 170)   # teal accent

    # Widget dimensions (pixels)
    _W: float = 220.0
    _H: float = 440.0

    # Layout (pixels)
    _TITLE_H: float = 50.0        # height of the title row (taller to fit larger fonts)
    _AXIS_LABEL_H: float = 20.0   # fixed x-axis label strip just above the graph
    _PAD_X: float = 20.0          # horizontal outer padding
    _PAD_BOT: float = 20.0        # gap between graph bottom and widget bottom edge (matches PAD_X)
    _PAD_TITLE_TOP: float = 4.0   # top padding inside the title row

    # Graph time window
    _HISTORY_SECS: float = 2.0
    _GRID_INTERVAL_SECS: float = 0.5   # one horizontal gridline per this many seconds

    # Typography
    _TITLE_FONT_SIZE: float = 12.0
    _FORCE_FONT_SIZE: float = 14.0
    _AXIS_LABEL_FONT_SIZE: float = 11.0
    _GRID_LABEL_FONT_SIZE: float = 10.0

    # Appearance
    _LINE_ALPHA: int = 230          # opacity of the force line stroke
    _FILL_ALPHA_MAX: int = 155      # max opacity for the shaded fill (at max force)
    _CENTER_LINE_ALPHA: int = 45    # opacity of the center (0-force) guide line
    _GRID_ALPHA: int = 40           # opacity of horizontal gridlines
    _GRID_LABEL_ALPHA: int = 120    # opacity of scrolling time labels
    _GRID_LABEL_H: float = 15.0     # pixel height reserved for each scrolling time label
    _AXIS_LABEL_ALPHA: int = 180    # opacity of fixed x-axis labels

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

        # Smooth sim-clock: updated only when sim_time_secs jumps (e.g. each 0.1 s tick or
        # on reset), NOT every physics frame.  This lets paint() extrapolate a smooth sim_now
        # from wall-clock time instead of producing a staircase that jumps every 0.1 s.
        self._last_sim_time: float = 0.0
        self._last_perf_time: float | None = None   # None → force sync on first update_force call

        # Minimum change in sim_time_secs that triggers a clock re-sync (~one rounded tick).
        self._SIM_SYNC_THRESHOLD: float = 0.05

        # Cache the rendered image so that force-circle movements (which overlap with this
        # widget's scene area) don't trigger expensive full repaints.
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

        # Continuous animation timer: ensures smooth gridline scrolling at ~60 fps even
        # when the tick rate is lower than the display refresh rate.
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)   # ~60 fps
        self._anim_timer.timeout.connect(self.update)
        self._anim_timer.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_force(self, force_newton: float, sim_time_secs: float) -> None:
        """Append a new force reading (Newtons) and schedule a repaint.

        Readings older than ``_HISTORY_SECS`` are automatically pruned.
        ``sim_time_secs`` drives the scrolling gridlines.
        """
        now = time.perf_counter()
        self._force_history.append((now, force_newton))
        self._current_force = force_newton

        # Sync the smooth sim clock only when sim_time_secs has moved by more
        # than a rounding step (0.1 s) away from our extrapolated value.  This
        # means the epoch updates ~10 times/sec (on each sim-clock tick) rather
        # than 240 times/sec, so paint() can smoothly extrapolate between ticks.
        if self._last_perf_time is None or abs(sim_time_secs - self._last_sim_time) > self._SIM_SYNC_THRESHOLD:
            self._last_sim_time = sim_time_secs
            self._last_perf_time = now

        cutoff = now - self._HISTORY_SECS
        while self._force_history and self._force_history[0][0] < cutoff:
            self._force_history.popleft()

        self.update()

    def reset(self) -> None:
        """Clear the force history (call when the simulation resets)."""
        self._force_history.clear()
        self._current_force = 0.0
        self._last_perf_time = None   # force epoch re-sync on next update_force call
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
        graph_top = self._TITLE_H + self._AXIS_LABEL_H
        graph_bot = self._H - self._PAD_BOT
        graph_h = graph_bot - graph_top
        center_x = self._W / 2.0
        half_graph_w = inner_w / 2.0
        graph_left = pad_x
        graph_right = self._W - pad_x

        # Smooth sim clock: extrapolate continuously from the last epoch pair.
        if self._last_perf_time is not None:
            sim_now = self._last_sim_time + (now - self._last_perf_time)
        else:
            sim_now = 0.0

        # ── 1. Title row (white text) ─────────────────────────────────
        title_font = QFont()
        title_font.setPointSizeF(self._TITLE_FONT_SIZE)
        force_font = QFont()
        force_font.setPointSizeF(self._FORCE_FONT_SIZE)
        force_font.setBold(True)

        force_text = f"{self._current_force:+.0f}"

        # Measure rendered widths to split the row without overlap.
        force_text_w = QFontMetricsF(force_font).horizontalAdvance(force_text) + self._FORCE_TEXT_PADDING
        title_col_w = inner_w - force_text_w

        white = QColor(255, 255, 255)
        painter.setPen(QPen(white))

        # Left-aligned title label
        painter.setFont(title_font)
        painter.drawText(
            QRectF(pad_x, self._PAD_TITLE_TOP, title_col_w, self._TITLE_H - self._PAD_TITLE_TOP),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "Force (Newtons)",
        )

        # Right-aligned current force value
        painter.setFont(force_font)
        painter.drawText(
            QRectF(pad_x + title_col_w, self._PAD_TITLE_TOP, force_text_w, self._TITLE_H - self._PAD_TITLE_TOP),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            force_text,
        )

        # ── 2. Fixed x-axis labels (just above the graph area) ───────
        axis_font = QFont()
        axis_font.setPointSizeF(self._AXIS_LABEL_FONT_SIZE)
        painter.setFont(axis_font)
        painter.setPen(QPen(QColor(r, g, b, self._AXIS_LABEL_ALPHA)))

        ayl = self._TITLE_H       # y of axis label strip top
        ayh = self._AXIS_LABEL_H  # height of axis label strip

        # Left edge: –max_force
        painter.drawText(
            QRectF(graph_left, ayl, half_graph_w, ayh),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            f"-{self._max_force:.0f}",
        )
        # Centre: 0
        painter.drawText(
            QRectF(graph_left, ayl, inner_w, ayh),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            "0",
        )
        # Right edge: +max_force
        painter.drawText(
            QRectF(center_x, ayl, half_graph_w, ayh),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            f"+{self._max_force:.0f}",
        )

        # ── 3. Center vertical guide line (0 force) ──────────────────
        painter.setPen(QPen(QColor(r, g, b, self._CENTER_LINE_ALPHA), 1.0))
        painter.drawLine(QLineF(center_x, graph_top, center_x, graph_bot))

        # ── 4. Scrolling horizontal gridlines with time labels ────────
        grid_font = QFont()
        grid_font.setPointSizeF(self._GRID_LABEL_FONT_SIZE)

        step = self._GRID_INTERVAL_SECS
        idx_start = math.floor((sim_now - self._HISTORY_SECS) / step)
        idx_end   = math.floor(sim_now / step)

        painter.save()
        painter.setClipRect(QRectF(0.0, graph_top, self._W, graph_bot - graph_top))

        for idx in range(idx_start, idx_end + 1):
            t = idx * step                         # sim time of this gridline (multiples of step)
            age = sim_now - t                      # seconds ago (0 = now, positive = in the past)
            y = graph_top + (age / self._HISTORY_SECS) * graph_h
            if graph_top <= y <= graph_bot:
                # Horizontal gridline
                painter.setPen(QPen(QColor(r, g, b, self._GRID_ALPHA), 1.0))
                painter.drawLine(QLineF(graph_left, y, graph_right, y))
                # Time label immediately below, right-aligned
                label = f"{t:g}s"
                painter.setFont(grid_font)
                painter.setPen(QPen(QColor(r, g, b, self._GRID_LABEL_ALPHA)))
                painter.drawText(
                    QRectF(graph_left, y + 1.0, inner_w, self._GRID_LABEL_H),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                    label,
                )

        painter.restore()

        # ── 5. Build force line + fill from history ───────────────────
        history = list(self._force_history)
        if not history:
            return

        def t_to_y(t: float) -> float:
            return graph_top + ((now - t) / self._HISTORY_SECS) * graph_h

        def f_to_x(f: float) -> float:
            return center_x + (f / self._max_force) * half_graph_w

        # Collect visible points (most-recent first).
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

        # ── 6. Shaded fill between the force line and the center ──────
        # Horizontal gradient: transparent at center, near-theme at edges.
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

        # ── 7. Force line ─────────────────────────────────────────────
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


# Backward-compatible alias so any stale imports keep working.
AgentActionWidget = ForceWidget
