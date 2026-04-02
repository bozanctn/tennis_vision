"""
Visualization — Draws all overlays onto frames.

Draws:
  - Ball position (circle)
  - Ball trail (fading dotted line)
  - Player bounding boxes
  - Pose skeleton
  - Court keypoints
  - Speed text
  - In/Out call banner
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np

# MediaPipe pose connections (pairs of landmark indices to draw as skeleton)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),   # left arm
    (12, 14), (14, 16),              # right arm
    (11, 23), (12, 24),              # torso
    (23, 24), (23, 25), (25, 27),   # left leg
    (24, 26), (26, 28),              # right leg
]


def draw_ball(frame: np.ndarray, cx: float, cy: float, conf: float) -> None:
    """Draw ball circle. Greener = higher confidence."""
    if cx is None:
        return
    color = (0, int(255 * conf), int(255 * (1 - conf)))   # green→red with conf
    cv2.circle(frame, (int(cx), int(cy)), 8, color, -1)
    cv2.circle(frame, (int(cx), int(cy)), 8, (255, 255, 255), 1)


def draw_trail(frame: np.ndarray, trail: deque, color: tuple = (0, 255, 255)) -> None:
    """Draw fading trail of past ball positions."""
    pts = list(trail)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        # Fade alpha based on age: older = more transparent
        alpha = i / len(pts)
        thickness = max(1, int(3 * alpha))
        faded_color = tuple(int(c * alpha) for c in color)
        cv2.line(
            frame,
            (int(pts[i - 1][0]), int(pts[i - 1][1])),
            (int(pts[i][0]),     int(pts[i][1])),
            faded_color,
            thickness,
        )


def draw_players(frame: np.ndarray, players: list[dict]) -> None:
    """Draw bounding boxes around detected players."""
    for i, player in enumerate(players):
        x1, y1, x2, y2 = [int(v) for v in player["bbox"]]
        color = (255, 128, 0) if i == 0 else (0, 128, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"P{i+1} {player['conf']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_pose(frame: np.ndarray, pose, color: tuple = (0, 255, 0)) -> None:
    """Draw pose skeleton on the frame."""
    if pose is None:
        return
    h, w = frame.shape[:2]
    lm = pose.landmarks  # (33, 3) normalized

    # Draw connections
    for a, b in POSE_CONNECTIONS:
        if pose.is_visible(a) and pose.is_visible(b):
            x1 = int(lm[a][0] * w)
            y1 = int(lm[a][1] * h)
            x2 = int(lm[b][0] * w)
            y2 = int(lm[b][1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    # Draw landmark points
    for i in range(33):
        if pose.is_visible(i):
            x = int(lm[i][0] * w)
            y = int(lm[i][1] * h)
            cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)


def draw_speed(frame: np.ndarray, speed_kmh: Optional[float]) -> None:
    """Overlay shot speed in the top-right corner."""
    if speed_kmh is None:
        return
    text = f"{speed_kmh:.0f} km/h"
    h, w = frame.shape[:2]
    cv2.putText(frame, text, (w - 180, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                (0, 200, 255), 3, cv2.LINE_AA)


def draw_in_out(frame: np.ndarray, is_in: Optional[bool]) -> None:
    """Draw a large IN/OUT banner at the bottom of the frame."""
    if is_in is None:
        return
    h, w = frame.shape[:2]
    if is_in:
        text, color = "IN", (0, 220, 0)
    else:
        text, color = "OUT", (0, 0, 220)

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2.5, 5)[0]
    tx = (w - text_size[0]) // 2
    cv2.putText(frame, text, (tx, h - 18), cv2.FONT_HERSHEY_DUPLEX, 2.5, color, 5, cv2.LINE_AA)
