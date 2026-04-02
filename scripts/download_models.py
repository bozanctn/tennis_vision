"""
Download and set up base models.

Run this FIRST before anything else:
    python scripts/download_models.py

What it does:
  1. Downloads YOLOv8n weights (for ball + player detection base)
  2. Downloads MediaPipe Pose model (auto-downloaded by mediapipe library)
  3. Creates the models/ directory

For production you should fine-tune the ball detector on a tennis dataset.
See notebooks/fine_tune_ball_detector.ipynb for the fine-tuning workflow.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_yolo_base():
    print("Downloading YOLOv8n base weights...")
    from ultralytics import YOLO
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Download YOLOv8n (nano) — smallest + fastest
    model = YOLO("yolov8n.pt")
    dest = models_dir / "yolov8n.pt"
    # ultralytics saves to ~/.ultralytics by default; copy here for clarity
    import shutil, os
    yolo_cache = Path.home() / ".cache" / "ultralytics" / "yolov8n.pt"
    if yolo_cache.exists():
        shutil.copy(yolo_cache, dest)
    print(f"  Saved to {dest}")
    return model


def verify_mediapipe():
    print("Verifying MediaPipe Pose...")
    import mediapipe as mp
    import numpy as np

    # mp.solutions may not be directly accessible on some versions —
    # access it via the full module path as a fallback.
    try:
        mp_pose = mp.solutions.pose
    except AttributeError:
        import mediapipe.python.solutions.pose as mp_pose_mod
        mp_pose = mp_pose_mod

    with mp_pose.Pose(static_image_mode=True) as pose:
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        pose.process(dummy)
    print("  MediaPipe Pose OK.")


def print_next_steps():
    print("""
=== Setup Complete ===

Next steps:
  1. Install dependencies:
       pip install -r requirements.txt

  2. (Optional) Fine-tune the ball detector on tennis data:
       - Download dataset from Roboflow: https://universe.roboflow.com
       - Run: python notebooks/fine_tune_ball_detector.ipynb

  3. Process a video:
       python scripts/process_video.py --input your_match.mp4

  4. Or start the API server:
       uvicorn api.main:app --host 0.0.0.0 --port 8000

  5. Send a video to the API:
       curl -X POST http://localhost:8000/analyze \\
            -F "file=@your_match.mp4"
""")


if __name__ == "__main__":
    download_yolo_base()
    verify_mediapipe()
    print_next_steps()
