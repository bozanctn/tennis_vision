"""
Pose Estimator — Extracts 33 body keypoints per player using MediaPipe Pose.

HOW IT WORKS:
  MediaPipe Pose uses a two-stage pipeline:
    1. BlazePose Detector  — finds the person bounding box in the frame
    2. BlazePose Landmark  — runs on the cropped person region, outputs 33
                             3D landmarks (x, y, z, visibility)

  The 33 landmarks cover: face, shoulders, elbows, wrists, hips, knees, ankles, feet.
  For stroke analysis we mainly care about:
    - Wrist (landmark 15/16)   → racket hand position
    - Elbow (13/14)            → arm extension
    - Shoulder (11/12)         → rotation / shoulder turn
    - Hip (23/24)              → weight transfer
    - Knee (25/26)             → leg bend (loading)

  These keypoints form a time-series that a classifier (LSTM) can use to
  identify stroke type: forehand, backhand, serve, volley.

  WHY MEDIAPIPE INSTEAD OF YOLO POSE?
    MediaPipe runs fully on CPU and is optimized for mobile, making it easy
    to deploy on a phone app. YOLOv8-pose is slightly more accurate but
    requires more compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseLandmarks:
    """Container for one player's pose result."""
    landmarks: np.ndarray   # shape (33, 3) — x, y, z (normalized 0-1)
    visibility: np.ndarray  # shape (33,)  — confidence per landmark
    bbox: tuple             # (x1, y1, x2, y2) in pixels

    # Convenient named landmark indices (MediaPipe Pose)
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13
    RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15
    RIGHT_WRIST    = 16
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26

    def get(self, landmark_id: int) -> tuple[float, float, float]:
        """Return (x, y, z) for a given landmark index."""
        return tuple(self.landmarks[landmark_id])

    def is_visible(self, landmark_id: int, threshold: float = 0.5) -> bool:
        return self.visibility[landmark_id] > threshold


class PoseEstimator:
    """
    Runs MediaPipe Pose on one or more player crops.

    Usage:
        estimator = PoseEstimator()
        # Full frame
        poses = estimator.estimate(frame)

        # Or pass cropped bounding boxes
        poses = estimator.estimate_from_bbox(frame, (x1, y1, x2, y2))
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,          # treat input as video stream → faster
            model_complexity=1,               # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,            # temporal smoothing across frames
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate(self, frame: np.ndarray) -> Optional[PoseLandmarks]:
        """
        Run pose estimation on the full frame.
        Use this when the frame already shows a single player.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if result.pose_landmarks is None:
            return None

        lm = result.pose_landmarks.landmark
        landmarks = np.array([[l.x, l.y, l.z] for l in lm])
        visibility = np.array([l.visibility for l in lm])

        return PoseLandmarks(
            landmarks=landmarks,
            visibility=visibility,
            bbox=(0, 0, w, h),
        )

    def estimate_from_bbox(
        self, frame: np.ndarray, bbox: tuple
    ) -> Optional[PoseLandmarks]:
        """
        Crop the player region from the frame, run pose, then map landmarks
        back to full-frame pixel coordinates.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        result = self.estimate(crop)
        if result is None:
            return None

        # Landmarks from crop are normalized 0-1 relative to the crop.
        # Convert to normalized coords relative to full frame.
        crop_h, crop_w = crop.shape[:2]
        full_h, full_w = frame.shape[:2]

        result.landmarks[:, 0] = (result.landmarks[:, 0] * crop_w + x1) / full_w
        result.landmarks[:, 1] = (result.landmarks[:, 1] * crop_h + y1) / full_h
        result.bbox = (x1, y1, x2, y2)
        return result

    def close(self):
        self.pose.close()
