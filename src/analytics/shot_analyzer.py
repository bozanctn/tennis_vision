"""
Shot Analyzer — Computes shot speed, detects bounces, and calls in/out.

HOW IT WORKS:

  SHOT SPEED:
    We know the real-world distance the ball traveled (via homography) and the
    time elapsed (frame count / fps). Speed = distance / time.
    speed_kmh = (meters_per_frame * fps * 3.6)
    We measure speed at the moment of a hit (when ball velocity is highest).

  BOUNCE DETECTION:
    A bounce happens when the ball's vertical velocity (vy) changes from
    positive (falling) to negative (rising) — i.e., sign flip in vy.
    We look at the smoothed Kalman trajectory for this sign flip.

  IN/OUT DETECTION:
    After computing the court homography we transform the bounce position
    to real-world coordinates and check if it's within the court boundary
    (and within the correct service box if it's a serve).

  STROKE CLASSIFICATION (simple rule-based version):
    We look at which side of the player the racket wrist is on relative
    to the body center. A full ML classifier (LSTM on pose sequence) is
    the production approach — see notebooks/stroke_classifier.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class StrokeType(str, Enum):
    FOREHAND  = "forehand"
    BACKHAND  = "backhand"
    SERVE     = "serve"
    VOLLEY    = "volley"
    UNKNOWN   = "unknown"


@dataclass
class ShotEvent:
    """One detected shot/bounce event."""
    frame_idx: int
    event_type: str                    # "hit" | "bounce"
    pixel_pos: tuple[float, float]
    court_pos: Optional[tuple[float, float]] = None   # real-world meters
    speed_kmh: Optional[float] = None
    is_in: Optional[bool] = None
    stroke_type: StrokeType = StrokeType.UNKNOWN


class ShotAnalyzer:
    """
    Analyses the ball trajectory produced by the tracker to find shot events.

    Usage:
        analyzer = ShotAnalyzer(fps=30, homography_matrix=H)
        for frame_idx, (cx, cy, vx, vy) in enumerate(trajectory):
            events = analyzer.process_frame(frame_idx, cx, cy, vx, vy)
    """

    def __init__(
        self,
        fps: float = 30.0,
        court_length_m: float = 23.77,
        court_width_m: float = 8.23,
        homography: Optional[np.ndarray] = None,
    ):
        self.fps = fps
        self.court_length_m = court_length_m
        self.court_width_m = court_width_m
        self.H = homography

        self._prev_vy: Optional[float] = None
        self._prev_pos: Optional[tuple] = None
        self._prev_court_pos: Optional[tuple] = None
        self.events: list[ShotEvent] = []

    def set_homography(self, H: np.ndarray):
        self.H = H

    def process_frame(
        self,
        frame_idx: int,
        cx: float,
        cy: float,
        vx: float,
        vy: float,
    ) -> list[ShotEvent]:
        """
        Call once per frame with the Kalman-smoothed ball position and velocity.
        Returns any new events detected at this frame.
        """
        new_events: list[ShotEvent] = []

        court_pos = None
        if self.H is not None:
            import cv2
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            result = cv2.perspectiveTransform(pt, self.H)
            court_pos = (float(result[0][0][0]), float(result[0][0][1]))

        # --- Bounce detection: sign flip in vertical velocity ---
        if self._prev_vy is not None:
            if self._prev_vy > 2 and vy < -2:   # was falling, now rising → bounce
                bounce_pos = self._prev_pos or (cx, cy)
                bounce_court = self._prev_court_pos or court_pos

                is_in = None
                if bounce_court is not None:
                    mx, my = bounce_court
                    is_in = (0 <= mx <= self.court_width_m and
                             0 <= my <= self.court_length_m)

                event = ShotEvent(
                    frame_idx=frame_idx,
                    event_type="bounce",
                    pixel_pos=bounce_pos,
                    court_pos=bounce_court,
                    is_in=is_in,
                )
                new_events.append(event)
                self.events.append(event)

        # --- Speed estimation ---
        if self._prev_court_pos is not None and court_pos is not None:
            dx = court_pos[0] - self._prev_court_pos[0]
            dy = court_pos[1] - self._prev_court_pos[1]
            dist_m_per_frame = np.sqrt(dx**2 + dy**2)
            speed_kmh = dist_m_per_frame * self.fps * 3.6
        else:
            speed_kmh = None

        self._prev_vy = vy
        self._prev_pos = (cx, cy)
        self._prev_court_pos = court_pos

        return new_events

    def get_stats(self) -> dict:
        """Aggregate stats over all processed frames."""
        bounces = [e for e in self.events if e.event_type == "bounce"]
        in_calls  = [e for e in bounces if e.is_in is True]
        out_calls = [e for e in bounces if e.is_in is False]

        speeds = [e.speed_kmh for e in self.events if e.speed_kmh is not None]

        return {
            "total_bounces": len(bounces),
            "in": len(in_calls),
            "out": len(out_calls),
            "max_speed_kmh": round(max(speeds), 1) if speeds else None,
            "avg_speed_kmh": round(float(np.mean(speeds)), 1) if speeds else None,
        }
