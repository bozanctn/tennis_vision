"""
Court Detector — Finds the 4 key court corners + net midpoint in each frame.

HOW IT WORKS:
  A tennis court has a well-defined shape (rectangle with inner lines).
  We use YOLOv8-pose (keypoint detection mode) trained on court images.
  It outputs 14 keypoints — the intersections that define the court layout:

        0 ──── 1 ──── 2 ──── 3
        |      |      |      |
        4 ──── 5 ──── 6 ──── 7   ← net line
        |      |      |      |
        8 ──── 9 ─── 10 ─── 11
       12 ─────────────────── 13  ← baselines

  Once we have these keypoints we compute a HOMOGRAPHY MATRIX.
  This is a 3×3 matrix that maps any pixel (x, y) in the image to a real-world
  coordinate on the court surface (in meters). This is essential for:
    - Deciding if a ball is in or out
    - Measuring where shots land
    - Normalizing player positions

  Alternative (simpler) approach used here as fallback:
    Classical Hough Line detection + RANSAC to find the court rectangle
    when the learned model is not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# Real-world court keypoints in meters (origin = top-left baseline corner)
# Matches the 14-keypoint layout described above
COURT_REAL_WORLD_POINTS = np.array([
    [0.0,    0.0],    # 0  top-left corner
    [4.115,  0.0],    # 1  top singles sideline left
    [4.115 + 8.23/2, 0.0],   # 2  top net center
    [8.23,   0.0],    # 3  top-right corner
    [0.0,    11.885], # 4  service line left
    [4.115,  11.885], # 5  service T left
    [4.115 + 8.23/2, 11.885],# 6  service T center (net)
    [8.23,   11.885], # 7  service line right
    [0.0,    11.885], # 8  (same as 4, doubles court)
    [4.115,  11.885], # 9
    [4.115 + 8.23/2, 11.885],#10
    [8.23,   11.885], #11
    [0.0,    23.77],  #12 bottom-left baseline corner
    [8.23,   23.77],  #13 bottom-right baseline corner
], dtype=np.float32)


class CourtDetector:
    """
    Detects court keypoints and computes the image→court homography.

    Usage:
        detector = CourtDetector()
        keypoints, H = detector.detect(frame)
        # H is a 3x3 homography matrix
        court_pos = CourtDetector.to_court_coords(H, pixel_x, pixel_y)
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path and Path(model_path).exists():
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"[CourtDetector] Loaded model from {model_path}")
        else:
            print("[CourtDetector] No model found — using classical line detection fallback.")

    def detect(self, frame: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect court in frame.

        Returns:
            keypoints: (N, 2) array of pixel coordinates, or None
            H:         3x3 homography matrix (image → real court), or None
        """
        if self.model is not None:
            return self._detect_with_model(frame)
        return self._detect_classical(frame)

    # ------------------------------------------------------------------
    # Model-based detection
    # ------------------------------------------------------------------
    def _detect_with_model(self, frame: np.ndarray):
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb, verbose=False)

        for result in results:
            if result.keypoints is None:
                continue
            kps = result.keypoints.xy[0].cpu().numpy()  # (14, 2)
            H = self._compute_homography(kps)
            return kps, H

        return None, None

    # ------------------------------------------------------------------
    # Classical fallback: Hough lines → find court rectangle
    # ------------------------------------------------------------------
    def _detect_classical(self, frame: np.ndarray):
        """
        Simple fallback when no ML model is available.
        Detects the 4 outer court corners using:
          1. Convert to HSV, threshold the court color (green/blue/red clay)
          2. Canny edge detection
          3. Probabilistic Hough Line Transform
          4. Cluster lines into horizontal/vertical groups
          5. Find 4 intersection corners
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=frame.shape[1] // 6,
            maxLineGap=30,
        )

        if lines is None:
            return None, None

        h_lines, v_lines = self._classify_lines(lines)
        corners = self._find_corners(h_lines, v_lines, frame.shape)

        if corners is None or len(corners) < 4:
            return None, None

        corners_arr = np.array(corners, dtype=np.float32)
        H = self._compute_homography_from_corners(corners_arr)
        return corners_arr, H

    def _classify_lines(self, lines):
        """Split detected lines into roughly horizontal and vertical groups."""
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle < 20:      # near-horizontal
                h_lines.append(line[0])
            elif angle > 70:    # near-vertical
                v_lines.append(line[0])
        return h_lines, v_lines

    def _find_corners(self, h_lines, v_lines, shape):
        """Intersect the outermost horizontal and vertical lines."""
        if not h_lines or not v_lines:
            return None

        def line_to_params(l):
            x1, y1, x2, y2 = l
            if x2 - x1 == 0:
                return None
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return m, b

        def intersect(l1, l2):
            p1, p2 = line_to_params(l1), line_to_params(l2)
            if p1 is None or p2 is None:
                return None
            m1, b1 = p1
            m2, b2 = p2
            if abs(m1 - m2) < 1e-6:
                return None
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
            return x, y

        # pick top/bottom horizontal lines and left/right vertical lines
        h_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        v_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)

        top_h = h_sorted[0]
        bot_h = h_sorted[-1]
        lft_v = v_sorted[0]
        rgt_v = v_sorted[-1]

        corners = [
            intersect(top_h, lft_v),
            intersect(top_h, rgt_v),
            intersect(bot_h, rgt_v),
            intersect(bot_h, lft_v),
        ]
        corners = [c for c in corners if c is not None]
        return corners if len(corners) == 4 else None

    # ------------------------------------------------------------------
    # Homography helpers
    # ------------------------------------------------------------------
    def _compute_homography(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """Compute homography from 14 detected keypoints to real-world coords."""
        if keypoints.shape[0] < 4:
            return None
        H, mask = cv2.findHomography(
            keypoints[:, :2].astype(np.float32),
            COURT_REAL_WORLD_POINTS[:len(keypoints)],
            cv2.RANSAC,
            5.0,
        )
        return H

    def _compute_homography_from_corners(self, corners: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute homography from just the 4 outer corners.
        corners order: [top-left, top-right, bottom-right, bottom-left]
        """
        real_corners = np.array([
            [0.0,   0.0],
            [8.23,  0.0],
            [8.23,  23.77],
            [0.0,   23.77],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(corners, real_corners, cv2.RANSAC, 5.0)
        return H

    # ------------------------------------------------------------------
    # Coordinate conversion (static — use anywhere once you have H)
    # ------------------------------------------------------------------
    @staticmethod
    def to_court_coords(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
        """
        Map a pixel (px, py) to real-world court position (meters).

        Maths: homogeneous coords [x, y, 1]  →  H @ [x, y, 1]  →  divide by w
        """
        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H)
        mx, my = result[0][0]
        return float(mx), float(my)

    @staticmethod
    def is_in_bounds(mx: float, my: float,
                     court_length: float = 23.77,
                     court_width: float = 8.23) -> bool:
        """Return True if the real-world coordinate is inside the court."""
        return 0 <= mx <= court_width and 0 <= my <= court_length
