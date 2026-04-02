"""
FastAPI Server — HTTP API to submit videos and get analysis results.

ENDPOINTS:
  POST /analyze          — Upload a video file, get back stats JSON + output video path
  GET  /health           — Health check
  GET  /stream/{job_id}  — Stream annotated video frames as MJPEG

HOW TO RUN:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

HOW IT WORKS:
  1. Client POSTs a video file (multipart/form-data)
  2. Server saves it to /tmp, runs the pipeline asynchronously
  3. Returns a job_id
  4. Client polls GET /jobs/{job_id} for completion + stats
"""

import asyncio
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

from src.pipeline.video_pipeline import VideoPipeline

app = FastAPI(title="Tennis Vision API", version="0.1.0")

# In-memory job store (use Redis/DB in production)
JOBS: dict[str, dict] = {}

# Singleton pipeline (loaded once at startup)
_pipeline: Optional[VideoPipeline] = None


@app.on_event("startup")
async def load_pipeline():
    global _pipeline
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    print(f"Loading pipeline from {config_path} ...")
    _pipeline = VideoPipeline.from_config(config_path)
    print("Pipeline ready.")


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": _pipeline is not None}


@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a tennis video and start analysis.
    Returns a job_id you can use to poll for results.
    """
    job_id = str(uuid.uuid4())

    # Save the uploaded file to a temp directory
    tmp_dir = Path(tempfile.mkdtemp())
    input_path  = tmp_dir / f"input_{job_id}{Path(file.filename).suffix}"
    output_path = tmp_dir / f"output_{job_id}.mp4"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {
        "status": "queued",
        "input": str(input_path),
        "output": str(output_path),
        "stats": None,
        "error": None,
    }

    background_tasks.add_task(_run_pipeline, job_id, str(input_path), str(output_path))

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll for job completion and stats."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs/{job_id}/download")
async def download_output(job_id: str):
    """Download the annotated output video."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    return FileResponse(job["output"], media_type="video/mp4", filename=f"tennis_{job_id}.mp4")


# ------------------------------------------------------------------
# Background task
# ------------------------------------------------------------------
def _run_pipeline(job_id: str, input_path: str, output_path: str):
    """Run in background thread (FastAPI handles the thread pool)."""
    JOBS[job_id]["status"] = "processing"
    try:
        stats = _pipeline.process(input_path, output_path)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["stats"] = stats
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        raise
