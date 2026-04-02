"""
Ball Tracker — Smooths noisy detections into a clean trajectory using a Kalman Filter.

HOW IT WORKS:
  A Kalman Filter is a recursive algorithm that estimates the "true" state
  of a moving object by combining:
    1. A PREDICTION step — "where should the ball be based on its velocity?"
    2. An UPDATE step    — "here's where the detector actually found it"

  The filter balances trust between the model (physics) and the measurement
  (noisy detector). Result: smooth trajectories even when the detector misses
  a frame or gives a slightly wrong position.

  State vector: [cx, cy, vx, vy]
    cx, cy = ball center position in pixels
    vx, vy = ball velocity in pixels/frame

  We also store the last N positions as a "trail" for visualization.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class KalmanBallTracker:
    """
    Wraps a simple 2D constant-velocity Kalman Filter for the tennis ball.

    Usage:
        tracker = KalmanBallTracker()
        for frame_detections in video:
            cx, cy, conf = detector.detect(frame)
            smoothed_x, smoothed_y = tracker.update(cx, cy)
    """

    def __init__(self, trail_length: int = 15, process_noise: float = 1.0, measurement_noise: float = 5.0):
        """
        Args:
            trail_length:      How many past positions to remember for drawing the trail.
            process_noise:     How much we trust the motion model (higher = more reactive).
            measurement_noise: How much we trust the detector (higher = smoother but laggier).
        """
        self.trail: deque[tuple[float, float]] = deque(maxlen=trail_length)
        self.initialized = False

        # --- Kalman Filter matrices ---
        # State: [cx, cy, vx, vy]  (position + velocity)
        dt = 1.0  # 1 frame timestep

        # State transition: next_state = F @ state
        # (position += velocity * dt, velocity unchanged)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        # Measurement matrix: we only observe position (cx, cy), not velocity
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise covariance (uncertainty in our motion model)
        self.Q = np.eye(4) * process_noise

        # Measurement noise covariance (uncertainty in detector output)
        self.R = np.eye(2) * measurement_noise

        # State estimate and covariance
        self.x = np.zeros((4, 1))          # initial state
        self.P = np.eye(4) * 100.0         # initial covariance (high = uncertain)

    # ------------------------------------------------------------------
    def update(self, cx: Optional[float], cy: Optional[float]) -> tuple[float, float]:
        """
        Feed one frame's detection into the filter.

        Args:
            cx, cy: Detected ball center (pixels). Pass None if no detection.

        Returns:
            Smoothed (cx, cy) estimate.
        """
        if not self.initialized:
            if cx is None:
                return 0.0, 0.0
            self.x = np.array([[cx], [cy], [0.0], [0.0]])
            self.initialized = True

        # --- PREDICT ---
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # --- UPDATE (only if we have a measurement) ---
        if cx is not None:
            z = np.array([[cx], [cy]])
            S = self.H @ P_pred @ self.H.T + self.R        # innovation covariance
            K = P_pred @ self.H.T @ np.linalg.inv(S)       # Kalman gain
            y = z - self.H @ x_pred                         # innovation
            self.x = x_pred + K @ y
            self.P = (np.eye(4) - K @ self.H) @ P_pred
        else:
            # No detection: just trust the prediction
            self.x = x_pred
            self.P = P_pred

        smoothed_cx = float(self.x[0])
        smoothed_cy = float(self.x[1])
        self.trail.append((smoothed_cx, smoothed_cy))

        return smoothed_cx, smoothed_cy

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated velocity in pixels/frame."""
        return float(self.x[2]), float(self.x[3])

    def reset(self):
        """Reset the tracker (use between rallies)."""
        self.initialized = False
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 100.0
        self.trail.clear()
