# pendulum/interaction.py

"""Handles manual user interaction, such as dragging the cart with the mouse."""

from __future__ import annotations

import numpy as np
import pygame

from .config import PendulumConfig, VisualizationConfig
from .environment import CartPendulumEnv


class CartDragController:
    """Encapsulates the logic for dragging the cart via MuJoCo mocap constraints."""

    def __init__(self, env: CartPendulumEnv, p_cfg: PendulumConfig, v_cfg: VisualizationConfig):
        self.env = env
        self.p_cfg = p_cfg
        self.v = v_cfg
        
        # --- Manual Interaction State ---
        self.is_dragging = False
        self.cursor_set_to_hand = False
        
        # Access internal MuJoCo handles for direct manipulation
        self.mujoco_data = env._mujoco_env.unwrapped.data

    def update(self, events: list, mouse_pos: tuple, cx: int, cy: int) -> np.ndarray | None:
        """
        Processes mouse events, updates the cursor, and manipulates the mocap body.
        
        Returns:
            np.ndarray: A zero-action array if the cart is being dragged (overriding AI).
            None: If the cart is not being dragged.
        """
        # Get current cart pixel position for hit testing
        state = self.env._state
        current_cart_x_px = cx + int(state[0] * self.v.scale)
        cart_hitbox = pygame.Rect(
            current_cart_x_px - self.v.cart_width // 2,
            cy - self.v.cart_height // 2,
            self.v.cart_width,
            self.v.cart_height
        )

        is_hovering = cart_hitbox.collidepoint(mouse_pos)

        # Handle Mouse Interactions from the event queue
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and is_hovering:
                    self.is_dragging = True
                    # Activate the weld constraint to grab the cart
                    self.mujoco_data.eq_active = 1
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.is_dragging = False
                    # Deactivate the weld constraint to release the cart
                    self.mujoco_data.eq_active = 0

        # --- Cursor Logic ---
        if self.is_dragging:
            if self.cursor_set_to_hand:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)
                self.cursor_set_to_hand = False
        elif is_hovering:
            if not self.cursor_set_to_hand:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                self.cursor_set_to_hand = True
        else:
            if self.cursor_set_to_hand:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                self.cursor_set_to_hand = False

        # --- Action & Mocap Update ---
        if self.is_dragging:
            # 1. Convert Mouse X (pixels) -> World X (meters)
            mouse_x = mouse_pos[0]
            target_x = (mouse_x - cx) / self.v.scale
            
            # Clamp to track limits
            half_track = self.p_cfg.track_length / 2.0
            target_x = float(np.clip(target_x, -half_track, half_track))
            
            # 2. Update the position of the mocap body
            # The weld constraint will pull the cart towards this position.
            self.mujoco_data.mocap_pos[0, 0] = target_x
            
            # 3. Apply zero action while dragging
            return np.array([0.0], dtype=np.float32)

        return None