"""
Video Pipeline — Orchestrates all modules to process a tennis video end-to-end.

PIPELINE FLOW (per frame):
  1. Read frame from video
  2. Detect ball → (cx, cy, conf)
  3. Track ball with Kalman filter → smoothed (cx, cy, vx, vy)
  4. Detect court (first frame only, then re-use H matrix)
  5. Detect players → bounding boxes
  6. Estimate pose for each player
  7. Analyze shots (bounce detection, speed, in/out)
  8. Draw all overlays onto the frame
  9. Write output frame to video file / yield for streaming

HOW TO USE:
    pipeline = VideoPipeline.from_config("config/config.yaml")
    pipeline.process("input.mp4", "output.mp4")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import yaml

from src.detection.ball_detector import BallDetector
from src.detection.court_detector import CourtDetector
from src.detection.player_detector import PlayerDetector
from src.tracking.ball_tracker import KalmanBallTracker
from src.pose.pose_estimator import PoseEstimator
from src.analytics.shot_analyzer import ShotAnalyzer
from src.utils.visualization import (
    draw_ball, draw_trail, draw_players, draw_pose, draw_speed, draw_in_out
)


class VideoPipeline:
    """
    Full end-to-end tennis analysis pipeline.

    Args:
        ball_detector:   BallDetector instance
        court_detector:  CourtDetector instance
        player_detector: PlayerDetector instance
        tracker:         KalmanBallTracker instance
        pose_estimator:  PoseEstimator instance
        analyzer:        ShotAnalyzer instance
        config:          dict of pipeline settings
    """

    def __init__(
        self,
        ball_detector: BallDetector,
        court_detector: CourtDetector,
        player_detector: PlayerDetector,
        tracker: KalmanBallTracker,
        pose_estimator: PoseEstimator,
        analyzer: ShotAnalyzer,
        config: dict,
    ):
        self.ball_detector   = ball_detector
        self.court_detector  = court_detector
        self.player_detector = player_detector
        self.tracker         = tracker
        self.pose_estimator  = pose_estimator
        self.analyzer        = analyzer
        self.config          = config

        self._court_H: Optional[np.ndarray] = None  # cached homography
        self._last_in_out: Optional[bool]   = None
        self._in_out_display_frames: int    = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str) -> "VideoPipeline":
        """Build a pipeline from a YAML config file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        m = cfg["models"]
        d = cfg["detection"]
        t = cfg["tracking"]
        p = cfg["pose"]
        a = cfg["analytics"]
        v = cfg["visualization"]

        ball_detector   = BallDetector.from_config(cfg)
        court_detector  = CourtDetector(m.get("court_model_path"))
        player_detector = PlayerDetector(conf_threshold=d["player_conf_threshold"])
        tracker         = KalmanBallTracker(trail_length=v["ball_trail_length"])
        pose_estimator  = PoseEstimator(
            min_detection_confidence=p["min_detection_confidence"],
            min_tracking_confidence=p["min_tracking_confidence"],
        )
        analyzer = ShotAnalyzer(
            fps=a["fps"],
            court_length_m=a["court_length_meters"],
            court_width_m=a["court_width_meters"],
        )

        return cls(
            ball_detector, court_detector, player_detector,
            tracker, pose_estimator, analyzer, cfg,
        )

    # ------------------------------------------------------------------
    # Main processing methods
    # ------------------------------------------------------------------
    def process(self, input_path: str, output_path: Optional[str] = None) -> dict:
        """
        Process a video file.

        Args:
            input_path:  Path to input MP4 / AVI.
            output_path: If given, write annotated video here.

        Returns:
            Stats dict from ShotAnalyzer.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing {Path(input_path).name}  [{width}x{height} @ {fps:.1f}fps, {total} frames]")

        frame_idx = 0
        t0 = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated = self.process_frame(frame, frame_idx)

                if writer:
                    writer.write(annotated)

                frame_idx += 1

                if frame_idx % 100 == 0:
                    elapsed = time.time() - t0
                    fps_actual = frame_idx / elapsed
                    print(f"  Frame {frame_idx}/{total}  ({fps_actual:.1f} fps)")
        finally:
            cap.release()
            if writer:
                writer.release()

        stats = self.analyzer.get_stats()
        print(f"\nDone. Stats: {stats}")
        return stats

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Process a single frame. Returns the annotated frame.
        This method is also used by the API for streaming.
        """
        out = frame.copy()

        # Step 1: Detect ball
        cx, cy, ball_conf = self.ball_detector.detect(frame)

        # Step 2: Track + smooth
        s_cx, s_cy = self.tracker.update(cx, cy)
        vx, vy = self.tracker.velocity

        # Step 3: Court detection (only needed once; H is stable for fixed camera)
        if self._court_H is None or frame_idx % 150 == 0:
            _, H = self.court_detector.detect(frame)
            if H is not None:
                self._court_H = H
                self.analyzer.set_homography(H)

        # Step 4: Detect players
        players = self.player_detector.detect(frame)

        # Step 5: Pose estimation on each player
        poses = []
        if self.config["visualization"]["show_pose"]:
            for player in players:
                pose = self.pose_estimator.estimate_from_bbox(frame, player["bbox"])
                poses.append(pose)

        # Step 6: Shot analysis
        new_events = self.analyzer.process_frame(frame_idx, s_cx, s_cy, vx, vy)
        if new_events:
            last_event = new_events[-1]
            if last_event.event_type == "bounce":
                self._last_in_out = last_event.is_in
                self._in_out_display_frames = 60   # show for 2 seconds

        # Step 7: Draw overlays
        if self.config["visualization"]["show_ball_trail"]:
            draw_trail(out, self.tracker.trail)

        draw_ball(out, s_cx, s_cy, ball_conf if ball_conf else 0.5)
        draw_players(out, players)

        for pose in poses:
            draw_pose(out, pose)

        if self.config["visualization"]["show_speed"]:
            speed = self._estimate_current_speed()
            draw_speed(out, speed)

        if self._in_out_display_frames > 0:
            draw_in_out(out, self._last_in_out)
            self._in_out_display_frames -= 1

        return out

    def _estimate_current_speed(self) -> Optional[float]:
        """Rough speed estimate from current velocity vector."""
        if not self.tracker.initialized or self._court_H is None:
            return None
        vx, vy = self.tracker.velocity
        pixel_speed = np.sqrt(vx**2 + vy**2)
        # Approximate: 1 pixel ≈ court_length_m / frame_height
        h = 720   # approximate; real app would use actual frame height
        m_per_pixel = self.config["analytics"]["court_length_meters"] / h
        fps = self.config["analytics"]["fps"]
        speed_kmh = pixel_speed * m_per_pixel * fps * 3.6
        return round(speed_kmh, 1) if speed_kmh > 5 else None

    def stream_frames(self, input_path: str) -> Generator[np.ndarray, None, None]:
        """
        Generator version of process() — yields annotated frames one by one.
        Used by the FastAPI streaming endpoint.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield self.process_frame(frame, frame_idx)
                frame_idx += 1
        finally:
            cap.release()
