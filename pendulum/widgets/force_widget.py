# pendulum/widgets/force_widget.py

"""ForceWidget: scrolling force-history graph showing forces applied to the cart."""

from __future__ import annotations

import math
import time
from collections import deque

from PySide6.QtCore import Qt, QLineF, QRectF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)

from .widget_base import BaseWidget

# Grey RGB used for all data labels and gridlines.
_GREY = (150, 150, 150)


class ForceWidget(BaseWidget):
    """Scrolling line graph showing the force applied to the cart each frame.

    History is stored as ``(perf_time, sim_time, force)`` triples so that
    gridlines are always exactly aligned with the data points at every paint
    call — no epoch extrapolation errors across the warmup/post-warmup boundary.

    Layout (top → bottom):
    ┌──────────────────────┐  ← teal themed border
    │ Force (Newtons)   42 │  ← title (white), current force (white monospace)
    │  -200      0    +200 │  ← fixed x-axis labels (grey); expand when force > max
    │      │               │  ← graph: now at top, 2.5 s ago at bottom
    │  0.0s │╲             │      left-aligned grey gridline labels
    │       │ ╲___         │      grey center vertical line
    │ -0.5s │     ╲        │      teal force line + shaded fill
    └──────────────────────┘

    Call ``update_force(force_newton, sim_time_exact)`` each tick; history older
    than ``_HISTORY_SECS`` is pruned automatically.  Call ``reset()`` on reset.
    """

    _THEME_COLOR: tuple = (60, 200, 170)   # teal accent

    # Widget width (pixels) — height is supplied at construction to match the pendulum widget.
    _W: float = 220.0
    _H: float = 440.0   # default fallback only; pass height= in renderer

    # Layout (pixels)
    _TITLE_H: float = 50.0        # height of the title row
    _AXIS_LABEL_H: float = 20.0   # fixed x-axis label strip just above the graph
    _PAD_X: float = 20.0          # horizontal outer padding
    _PAD_BOT: float = 20.0        # bottom padding (matches _PAD_X)
    _PAD_TITLE_TOP: float = 4.0   # top padding inside the title row

    # Graph time window
    _HISTORY_SECS: float = 2.5
    _GRID_INTERVAL_SECS: float = 0.5   # one horizontal gridline per this many seconds

    # Typography
    _TITLE_FONT_SIZE: float = 12.0
    _FORCE_FONT_SIZE: float = 14.0
    _AXIS_LABEL_FONT_SIZE: float = 11.0
    _GRID_LABEL_FONT_SIZE: float = 10.0

    # Appearance — force line/fill use theme color, everything else uses grey
    _LINE_ALPHA: int = 230
    _FILL_ALPHA_MAX: int = 155
    _CENTER_LINE_ALPHA: int = 45
    _GRID_ALPHA: int = 60           # opacity of grey horizontal gridlines
    _GRID_LABEL_ALPHA: int = 160    # opacity of grey scrolling time labels
    _GRID_LABEL_H: float = 15.0     # pixel height reserved for each scrolling time label
    _AXIS_LABEL_ALPHA: int = 190    # opacity of grey fixed x-axis labels

    # Guard against division by zero when max_force is 0
    _MIN_FORCE: float = 1.0
    # Extra pixel margin added to force-text measurement for the column split
    _FORCE_TEXT_PADDING: float = 4.0

    def __init__(self, max_force: float, height: float | None = None, parent=None):
        """
        Parameters
        ----------
        max_force:
            The default maximum absolute force (sets the initial X-axis range).
            The axis auto-expands when visible forces exceed this value.
        height:
            Widget height in pixels.  Defaults to ``_H`` when omitted; pass the
            pendulum widget's height to make the two widgets the same size.
        """
        h = height if height is not None else self._H
        rect = QRectF(0.0, 0.0, self._W, h)
        super().__init__(rect, self._THEME_COLOR, parent)

        self._max_force: float = max(max_force, self._MIN_FORCE)
        self._force_history: deque[tuple[float, float]] = deque()   # (perf_time, force)
        self._current_force: float = 0.0

        # Sim-clock reference pair updated on every update_force call.
        # Storing the exact (unrounded) sim time lets paint() extrapolate
        # sim_now continuously between ticks with no stairstepping.
        self._last_sim_time: float = 0.0
        self._last_perf_time: float | None = None   # None until first update_force

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_force(self, force_newton: float, sim_time_exact: float) -> None:
        """Append a new force reading and schedule a repaint.

        ``sim_time_exact`` must be the unrounded simulation time (seconds):
        negative during warmup, non-negative after warmup starts.
        Readings older than ``_HISTORY_SECS`` are automatically pruned.
        """
        now = time.perf_counter()
        self._force_history.append((now, force_newton))
        self._current_force = force_newton
        # Always sync: sim_time_exact is continuous so we never stairStep.
        self._last_sim_time = sim_time_exact
        self._last_perf_time = now

        cutoff = now - self._HISTORY_SECS
        while self._force_history and self._force_history[0][0] < cutoff:
            self._force_history.popleft()

        self.update()

    def reset(self) -> None:
        """Clear the force history (call when the simulation resets)."""
        self._force_history.clear()
        self._current_force = 0.0
        self._last_perf_time = None   # re-sync on next update_force call
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        super().paint(painter, option, widget)

        r, g, b = self._THEME_COLOR
        now = time.perf_counter()
        gr, gg, gb = _GREY

        # Widget dimensions from the stored rect (set at construction time).
        w = self._rect.width()
        h = self._rect.height()

        pad_x = self._PAD_X
        inner_w = w - 2.0 * pad_x
        graph_top = self._TITLE_H + self._AXIS_LABEL_H
        graph_bot = h - self._PAD_BOT
        graph_h = graph_bot - graph_top
        center_x = w / 2.0
        half_graph_w = inner_w / 2.0
        graph_left = pad_x
        graph_right = w - pad_x

        # Snapshot history once so the whole paint call is consistent.
        history = list(self._force_history)   # [(perf_time, force), ...]

        # Extrapolate sim_now from the last update_force reference pair.
        if self._last_perf_time is not None:
            sim_now = self._last_sim_time + (now - self._last_perf_time)
        else:
            sim_now = 0.0

        # Dynamic X-axis scale: expand if any history point exceeds max_force.
        x_scale = self._max_force
        if history:
            max_abs = max(abs(f) for _, f in history)
            if max_abs > x_scale:
                x_scale = max_abs

        # ── 1. Title row (white text) ─────────────────────────────────
        title_font = QFont()
        title_font.setPointSizeF(self._TITLE_FONT_SIZE)
        # Use the system monospace font for the live force reading.
        force_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        force_font.setPointSizeF(self._FORCE_FONT_SIZE)
        force_font.setBold(True)

        # Show "0" (not "+0" or "-0") when the rounded value is zero.
        raw_text = f"{self._current_force:+.0f}"
        force_text = "0" if raw_text in ("+0", "-0") else raw_text

        force_text_w = QFontMetricsF(force_font).horizontalAdvance(force_text) + self._FORCE_TEXT_PADDING
        title_col_w = inner_w - force_text_w

        white = QColor(255, 255, 255)
        painter.setPen(QPen(white))

        painter.setFont(title_font)
        painter.drawText(
            QRectF(pad_x, self._PAD_TITLE_TOP, title_col_w, self._TITLE_H - self._PAD_TITLE_TOP),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "Force (Newtons)",
        )

        painter.setFont(force_font)
        painter.drawText(
            QRectF(pad_x + title_col_w, self._PAD_TITLE_TOP, force_text_w, self._TITLE_H - self._PAD_TITLE_TOP),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            force_text,
        )

        # ── 2. Fixed x-axis labels (grey, auto-scale) ─────────────────
        axis_font = QFont()
        axis_font.setPointSizeF(self._AXIS_LABEL_FONT_SIZE)
        painter.setFont(axis_font)
        painter.setPen(QPen(QColor(gr, gg, gb, self._AXIS_LABEL_ALPHA)))

        ayl = self._TITLE_H
        ayh = self._AXIS_LABEL_H

        painter.drawText(
            QRectF(graph_left, ayl, half_graph_w, ayh),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            f"-{x_scale:.0f}",
        )
        painter.drawText(
            QRectF(graph_left, ayl, inner_w, ayh),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            "0",
        )
        painter.drawText(
            QRectF(center_x, ayl, half_graph_w, ayh),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            f"+{x_scale:.0f}",
        )

        # ── 3. Center vertical guide line (grey) ──────────────────────
        painter.setPen(QPen(QColor(gr, gg, gb, self._CENTER_LINE_ALPHA), 1.0))
        painter.drawLine(QLineF(center_x, graph_top, center_x, graph_bot))

        # ── 4. Scrolling horizontal gridlines with time labels (grey, left-aligned) ─
        grid_font = QFont()
        grid_font.setPointSizeF(self._GRID_LABEL_FONT_SIZE)

        step = self._GRID_INTERVAL_SECS
        idx_start = math.floor((sim_now - self._HISTORY_SECS) / step)
        idx_end   = math.floor(sim_now / step)

        painter.save()
        painter.setClipRect(QRectF(0.0, graph_top, w, graph_bot - graph_top))

        for idx in range(idx_start, idx_end + 1):
            t = idx * step
            # delta_t = how far this gridline's sim-time is behind sim_now.
            # Positive → gridline is in the past (lower on screen).
            # Negative → gridline is ahead of the current instant (above graph_top, clipped).
            delta_t = sim_now - t
            y = graph_top + (delta_t / self._HISTORY_SECS) * graph_h
            if graph_top <= y <= graph_bot:
                painter.setPen(QPen(QColor(gr, gg, gb, self._GRID_ALPHA), 1.0))
                painter.drawLine(QLineF(graph_left, y, graph_right, y))
                label = f"{t:g}s"
                painter.setFont(grid_font)
                painter.setPen(QPen(QColor(gr, gg, gb, self._GRID_LABEL_ALPHA)))
                painter.drawText(
                    QRectF(graph_left, y + 1.0, inner_w, self._GRID_LABEL_H),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                    label,
                )

        painter.restore()

        # ── 5. Force line + fill from history ─────────────────────────
        if not history:
            return

        def t_to_y(ts: float) -> float:
            """Map a perf_counter timestamp to a vertical pixel coordinate."""
            return graph_top + ((now - ts) / self._HISTORY_SECS) * graph_h

        def f_to_x(f: float) -> float:
            return center_x + (f / x_scale) * half_graph_w

        # Collect visible points (most-recent first).
        points: list[tuple[float, float]] = []
        for ts, f in reversed(history):
            y = t_to_y(ts)
            if y > graph_bot:
                break
            points.append((f_to_x(f), y))

        if not points:
            return

        top_y = points[0][1]
        bot_y = points[-1][1]

        # ── 6. Shaded fill (teal gradient, transparent at center) ─────
        gradient = QLinearGradient(graph_left, 0.0, graph_right, 0.0)
        gradient.setColorAt(0.0, QColor(r, g, b, self._FILL_ALPHA_MAX))
        gradient.setColorAt(0.5, QColor(r, g, b, 0))
        gradient.setColorAt(1.0, QColor(r, g, b, self._FILL_ALPHA_MAX))

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

        # ── 7. Force line (teal) ───────────────────────────────────────
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
