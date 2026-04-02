"""
Ball Detector — Unified interface that supports both YOLOv8 and TrackNet.

WHICH MODEL TO USE?

  YOLOv8  → good starting point, works without training data, single-frame
  TrackNet → much better for fast tennis balls, uses 3 frames, needs training

  Auto-selection logic (from_config):
    - If tracknet weights exist   → use TrackNet
    - Otherwise                   → fall back to YOLOv8

HOW EACH WORKS:

  YOLOv8:
    Single frame → grid predictions → highest confidence box center = ball
    Fast, general-purpose, but struggles with motion blur

  TrackNet:
    3 frames stacked (9 channels) → encoder-decoder CNN → heatmap
    Brightest pixel in heatmap = ball center
    Designed specifically for fast small objects (shuttlecock, tennis ball)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class BallDetector:
    """
    Unified ball detector. Automatically uses TrackNet if weights exist,
    otherwise falls back to YOLOv8.

    Usage:
        detector = BallDetector.from_config("config/config.yaml")
        cx, cy, conf = detector.detect(frame)
    """

    def __init__(self, backend: str, model):
        """
        Don't call directly. Use BallDetector.from_config() or
        BallDetector.yolo() / BallDetector.tracknet().
        """
        self.backend = backend   # "yolo" or "tracknet"
        self._model  = model

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "BallDetector":
        """
        Build from config dict. Prefers TrackNet if weights exist.
        """
        m = config["models"]
        d = config["detection"]

        tracknet_path = m.get("tracknet_model_path", "models/tracknet.pt")
        yolo_path     = m.get("ball_model_path",     "models/ball_detector.pt")
        device        = config.get("device", "cpu")

        if Path(tracknet_path).exists():
            print(f"[BallDetector] Using TrackNet ({tracknet_path})")
            return cls.tracknet(
                tracknet_path,
                conf_threshold=d.get("ball_conf_threshold", 0.5),
                device=device,
            )
        else:
            print(f"[BallDetector] TrackNet not found — using YOLOv8 ({yolo_path})")
            return cls.yolo(
                yolo_path,
                conf_threshold=d.get("ball_conf_threshold", 0.35),
                device=device,
            )

    @classmethod
    def yolo(cls, model_path: str, conf_threshold: float = 0.35, device: str = "cpu") -> "BallDetector":
        from ultralytics import YOLO

        if Path(model_path).exists():
            model = YOLO(model_path)
        else:
            print(f"[BallDetector/YOLO] Weights not found at {model_path} — loading base YOLOv8n.")
            model = YOLO("yolov8n.pt")

        model.to(device)

        # Wrap with metadata
        wrapper = _YOLOWrapper(model, conf_threshold)
        return cls("yolo", wrapper)

    @classmethod
    def tracknet(cls, model_path: str, conf_threshold: float = 0.5, device: str = "cpu") -> "BallDetector":
        from src.detection.tracknet import TrackNetDetector
        detector = TrackNetDetector(model_path, device=device, conf_threshold=conf_threshold)
        return cls("tracknet", detector)

    # ------------------------------------------------------------------
    # Unified detect interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> tuple[Optional[float], Optional[float], float]:
        """
        Run detection on one BGR frame.

        Returns:
            (cx, cy, confidence) — pixel coordinates + confidence in [0, 1]
            Returns (None, None, 0.0) if no ball found.
        """
        return self._model.detect(frame)

    def reset(self):
        """Reset internal state (call between points/rallies)."""
        if hasattr(self._model, "reset"):
            self._model.reset()

    @property
    def is_tracknet(self) -> bool:
        return self.backend == "tracknet"


# ---------------------------------------------------------------------------
# Internal wrappers — not public API
# ---------------------------------------------------------------------------

class _YOLOWrapper:
    """Thin wrapper to give YOLOv8 the same .detect() signature as TrackNet."""

    def __init__(self, model, conf_threshold: float):
        self.model = model
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> tuple[Optional[float], Optional[float], float]:
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
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_cx = (x1 + x2) / 2
                    best_cy = (y1 + y2) / 2

        return best_cx, best_cy, best_conf
