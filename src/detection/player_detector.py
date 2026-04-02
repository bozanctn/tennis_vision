"""
Player Detector — Finds all players (persons) in each frame using YOLOv8.

HOW IT WORKS:
  Standard YOLOv8 trained on COCO dataset can already detect people (class 0).
  We just filter to class=0 (person) and return the bounding boxes.
  In a tennis match there are at most 4 people (2 players + 2 coaches/ball kids).
  We sort by bounding box area and take the top N (usually 2).
"""

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np
from ultralytics import YOLO


class PlayerDetector:
    """
    Detects players (persons) in a frame.

    Returns bounding boxes as (x1, y1, x2, y2, confidence).
    """

    PERSON_CLASS_ID = 0  # COCO class 0 = person

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        max_players: int = 2,
        device: str = "cpu",
    ):
        """
        Args:
            model_path:    YOLOv8 weights. Uses base COCO weights by default.
            conf_threshold: Min confidence for a valid person detection.
            max_players:   Return only the N largest detections (players closest
                           to camera are biggest → avoids picking up crowd).
            device:        "cpu" / "cuda" / "mps"
        """
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.max_players = max_players

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect players in one frame.

        Returns:
            List of dicts: {"bbox": (x1,y1,x2,y2), "conf": float, "center": (cx,cy)}
            Sorted by bounding-box area descending (largest = main players).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(
            rgb,
            conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        players = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                area = (x2 - x1) * (y2 - y1)
                players.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "area": area,
                })

        # Sort by area descending, return top N
        players.sort(key=lambda p: p["area"], reverse=True)
        return players[: self.max_players]
