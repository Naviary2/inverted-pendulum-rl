# pendulum/renderer.py

"""Renders the scene objects (track, cart, pendulum links and nodes) to a pygame surface."""

from __future__ import annotations

import math

import numpy as np
import pygame
import pygame.gfxdraw

from .config import PendulumConfig, VisualizationConfig
from .interaction import ForceCircleController


def _draw_thick_aaline(surface, p1, p2, width, color):
    """Draws a thick anti-aliased line (simulated via polygon)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    if length == 0:
        return

    # Normalized direction vector components
    ux = dx / length
    uy = dy / length

    # Perpendicular vector ((-uy, ux) corresponds to rotation by 90 degrees)
    # Scaled by half width to get offset from center line
    half_width = width / 2
    px = -uy * half_width
    py = ux * half_width

    # Calculate 4 corners of the rectangle
    corners = [
        (int(x1 + px), int(y1 + py)),
        (int(x1 - px), int(y1 - py)),
        (int(x2 - px), int(y2 - py)),
        (int(x2 + px), int(y2 + py)),
    ]

    # Draw filled polygon (the body)
    pygame.gfxdraw.filled_polygon(surface, corners, color)
    # Draw anti-aliased outline (the smooth edges)
    pygame.gfxdraw.aapolygon(surface, corners, color)


class SceneRenderer:
    """Draws the full simulation scene: background, track, cart, and pendulum."""

    def __init__(self, p_cfg: PendulumConfig, v: VisualizationConfig):
        self.p_cfg = p_cfg
        self.v = v

    def draw(
        self,
        screen: pygame.Surface,
        state: np.ndarray,
        cx: int,
        cy: int,
        mouse_pos: tuple,
        force_circle_controller: ForceCircleController,
    ) -> None:
        """Render one frame of the simulation scene."""
        v = self.v
        p_cfg = self.p_cfg

        # Background
        screen.fill(v.bg_color)

        # Pre-compute pixel sizes from metre-based config values
        cart_w_px      = int(v.cart_width        * v.scale)
        cart_h_px      = int(v.cart_height       * v.scale)
        cart_rad_px    = int(v.cart_rad          * v.scale)
        cart_node_px   = int(v.cart_node_radius  * v.scale)
        node_rad_px    = int(p_cfg.node_radius    * v.scale)
        node_out_px    = int(v.node_outline_width * v.scale)
        pend_w_px      = int(v.pendulum_width    * v.scale)
        track_h_px     = int(v.track_h           * v.scale)
        track_rad_px   = int(v.track_rad         * v.scale)

        # Track
        track_len_px = int(p_cfg.track_length * v.scale) + cart_w_px + cart_node_px
        # Create a rectangle centered at (cx, cy)
        track_rect = pygame.Rect(
            cx - track_len_px // 2,  # Left
            cy - track_h_px // 2,    # Top
            track_len_px,            # Width
            track_h_px               # Height
        )
        # Draw hollow rounded rectangle
        pygame.draw.rect(
            screen,
            v.fg_color,
            track_rect,
            v.track_thick,              # Thickness (makes it hollow)
            border_radius=track_rad_px  # Roundness
        )

        # Define the hard limit based on the track length
        half_length = p_cfg.track_length / 2.0
        # Clamp the cart's x position so it visually never leaves the track
        # (Even if MuJoCo calculates a slight penetration of the wall)
        cart_x = float(np.clip(state[0], -half_length, half_length))
        # Use this clamped value to calculate pixels
        cart_x_px = cx + int(cart_x * v.scale)

        # Cart rectangle (outlined + translucent fill)
        cart_rect = pygame.Rect(
            cart_x_px - cart_w_px // 2,
            cy - cart_h_px // 2,
            cart_w_px,
            cart_h_px,
        )

        # Draw Fill
        pygame.draw.rect(
            screen,
            v.node_fill_color,
            cart_rect,
            border_radius=cart_rad_px  # Roundness
        )
        # Draw hollow rounded rectangle (Outline)
        pygame.draw.rect(
            screen,
            v.fg_color,
            cart_rect,
            v.track_thick,              # Thickness (makes it hollow)
            border_radius=cart_rad_px   # Roundness
        )

        # --- pendulum links & nodes ---
        n = p_cfg.num_links
        lengths = p_cfg.link_lengths

        # First node sits at the cart pivot (top-centre of cart)
        pivot_x, pivot_y = cart_x_px, cy

        # Draw first node
        pygame.gfxdraw.aacircle(screen, pivot_x, pivot_y, cart_node_px, v.fg_color) ## AA outline
        pygame.gfxdraw.filled_circle(screen, pivot_x, pivot_y, cart_node_px, v.fg_color) # Filled circle

        for i in range(n):
            theta_i = state[1 + i]
            end_x = pivot_x + int(lengths[i] * v.scale * np.sin(theta_i))
            end_y = pivot_y - int(lengths[i] * v.scale * np.cos(theta_i))

            # Using custom helper for AA thick line,
            # Since gfxdraw.line doesn't support width and pygame.draw.line isn't anti-aliased.
            _draw_thick_aaline(
                screen,
                (pivot_x, pivot_y),
                (end_x, end_y),
                pend_w_px,
                v.fg_color
            )

            # --- Draw Tip Node ---
            # Using gfxdraw for AA

            # 1. Draw Outline (Outer Circle)
            pygame.gfxdraw.aacircle(screen, end_x, end_y, node_rad_px, v.fg_color) ## AA outline
            pygame.gfxdraw.filled_circle(screen, end_x, end_y, node_rad_px, v.fg_color) # Filled circle

            # 2. Draw Fill (Inner Circle)
            inner_radius = node_rad_px - node_out_px
            pygame.gfxdraw.aacircle(screen, end_x, end_y, inner_radius, v.node_fill_color) # AA outline
            pygame.gfxdraw.filled_circle(screen, end_x, end_y, inner_radius, v.node_fill_color) # Filled circle

            # Next pivot is this tip
            pivot_x, pivot_y = end_x, end_y

        # --- Draw Force Circle ---
        force_circle_controller.draw(screen, mouse_pos)
