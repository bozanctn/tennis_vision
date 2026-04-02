# Tennis Vision

A SwingVision-style tennis analysis app using computer vision and AI.

## What It Does

- **Ball tracking** — Detects the tennis ball in every frame using YOLOv8, smoothed with a Kalman filter
- **Court detection** — Finds court keypoints and computes a homography for real-world coordinates
- **Player detection** — Tracks up to 2 players per frame
- **Pose estimation** — 33-point body skeleton per player using MediaPipe
- **Shot analysis** — Bounce detection, in/out calls, shot speed in km/h

---

## Project Structure

```
tennis_vision/
├── config/config.yaml          ← All settings (thresholds, fps, court dims)
├── src/
│   ├── detection/
│   │   ├── ball_detector.py    ← YOLOv8 ball detection
│   │   ├── court_detector.py   ← Court keypoints + homography
│   │   └── player_detector.py  ← YOLOv8 person detection
│   ├── tracking/
│   │   └── ball_tracker.py     ← Kalman filter trajectory smoothing
│   ├── pose/
│   │   └── pose_estimator.py   ← MediaPipe 33-keypoint pose
│   ├── analytics/
│   │   └── shot_analyzer.py    ← Speed, bounce, in/out detection
│   ├── utils/
│   │   └── visualization.py    ← Drawing overlays on frames
│   └── pipeline/
│       └── video_pipeline.py   ← Ties everything together
├── api/main.py                 ← FastAPI HTTP server
├── scripts/
│   ├── download_models.py      ← Run this first!
│   └── process_video.py        ← CLI entrypoint
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download base models

```bash
python scripts/download_models.py
```

### 3. Process a video

```bash
python scripts/process_video.py --input my_match.mp4 --output annotated.mp4
```

With live preview window:

```bash
python scripts/process_video.py --input my_match.mp4 --preview
```

### 4. Run the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then send a video:

```bash
# Submit video
curl -X POST http://localhost:8000/analyze \
     -F "file=@my_match.mp4"
# Returns: {"job_id": "abc-123", "status": "queued"}

# Poll for results
curl http://localhost:8000/jobs/abc-123

# Download annotated video
curl http://localhost:8000/jobs/abc-123/download -o result.mp4
```

---

## How the Pipeline Works

```
Video Frame
    │
    ├─► Ball Detector (YOLOv8)
    │       └─► Kalman Tracker  ──► smoothed position + velocity
    │
    ├─► Court Detector ──► Homography matrix (pixels → meters)
    │
    ├─► Player Detector (YOLOv8)
    │       └─► Pose Estimator (MediaPipe) ──► 33 body keypoints
    │
    └─► Shot Analyzer
            ├── bounce detection (velocity sign flip)
            ├── in/out (court coords via homography)
            └── speed (meters/frame × fps × 3.6)
                    │
                    ▼
              Annotated Frame + Stats JSON
```

---

## Fine-tuning the Ball Detector (Important for accuracy)

The base YOLOv8n model doesn't know what a tennis ball looks like. You need to fine-tune it:

1. Download a tennis ball dataset from [Roboflow Universe](https://universe.roboflow.com) (search "tennis ball")
2. Export in YOLOv8 format
3. Run training:

```bash
yolo train \
  model=yolov8n.pt \
  data=path/to/dataset/data.yaml \
  epochs=50 \
  imgsz=640 \
  project=models \
  name=ball_detector
```

4. Update `config/config.yaml`:
```yaml
models:
  ball_model_path: "models/ball_detector/weights/best.pt"
```

---

## Next Steps (Roadmap)

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Ball detection + tracking | Done |
| 1 | Court detection | Done |
| 1 | Player detection | Done |
| 1 | Pose estimation | Done |
| 1 | Shot speed + bounce detection | Done |
| 2 | Stroke classification (LSTM) | TODO |
| 2 | Serve speed + placement | TODO |
| 3 | Mobile app (React Native) | TODO |
| 3 | Shot heatmaps | TODO |
| 4 | Real-time streaming | TODO |
