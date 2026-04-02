"""
Ball Detector — Uses YOLOv8 to find the tennis ball in each frame.

HOW IT WORKS:
  YOLOv8 is a one-stage object detector. It divides the image into a grid,
  predicts bounding boxes + confidence scores + class probabilities at each
  grid cell simultaneously (hence "one-stage" — no separate proposal step).

  For a tennis ball (small, fast, yellow) we use the nano variant (YOLOv8n)
  because it is fast enough for real-time use. We fine-tune it on tennis-specific
  datasets so it learns what a tennis ball looks like at different distances,
  lighting conditions, and motion blur levels.

  Output per frame: list of (x_center, y_center, width, height, confidence)
  We take the highest-confidence detection as "the ball".
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ultralytics provides a clean API over PyTorch YOLOv8
from ultralytics import YOLO


class BallDetector:
    """
    Detects the tennis ball in a single frame using YOLOv8.

    Usage:
        detector = BallDetector("models/ball_detector.pt")
        cx, cy, conf = detector.detect(frame)
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.35, device: str = "cpu"):
        """
        Args:
            model_path:      Path to a .pt weights file.
                             If the file doesn't exist we fall back to the
                             pretrained YOLOv8n as a starting point.
            conf_threshold:  Minimum confidence to accept a detection.
            device:          "cpu", "cuda", or "mps" (Apple Silicon).
        """
        self.conf_threshold = conf_threshold
        self.device = device

        if Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            print(f"[BallDetector] Model not found at {model_path}. "
                  "Loading base YOLOv8n — run fine-tuning before production use.")
            self.model = YOLO("yolov8n.pt")  # downloads automatically on first run

        self.model.to(device)

    def detect(self, frame: np.ndarray) -> tuple[Optional[float], Optional[float], float]:
        """
        Run inference on one BGR frame (OpenCV format).

        Returns:
            (cx, cy, confidence) in pixel coordinates, or (None, None, 0.0)
            if no ball is found above the threshold.
        """
        # YOLOv8 expects RGB; OpenCV gives BGR — swap channels
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model(rgb, conf=self.conf_threshold, verbose=False)

        best_conf = 0.0
        best_cx, best_cy = None, None

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    # xyxy format → compute center
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_cx = (x1 + x2) / 2
                    best_cy = (y1 + y2) / 2

        return best_cx, best_cy, best_conf

    def detect_batch(self, frames: list[np.ndarray]) -> list[tuple]:
        """
        Run inference on a list of frames at once (more GPU-efficient).
        Returns a list of (cx, cy, conf) tuples, one per frame.
        """
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        results_list = self.model(rgb_frames, conf=self.conf_threshold, verbose=False)

        detections = []
        for result in results_list:
            best_conf = 0.0
            best_cx, best_cy = None, None
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        best_cx = (x1 + x2) / 2
                        best_cy = (y1 + y2) / 2
            detections.append((best_cx, best_cy, best_conf))

        return detections
