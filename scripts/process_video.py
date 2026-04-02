"""
CLI script — Process a single tennis video from the command line.

Usage:
    # Local file
    python scripts/process_video.py --input my_match.mp4 --output annotated.mp4

    # YouTube / YouTube Shorts URL  (downloads automatically)
    python scripts/process_video.py --input "https://www.youtube.com/shorts/XXXX"
    python scripts/process_video.py --input "https://youtu.be/XXXX"

Options:
    --input   Path to local video file OR a YouTube URL (required)
    --output  Path for annotated output video (default: input_annotated.mp4)
    --config  Path to config YAML (default: config/config.yaml)
    --device  cpu | cuda | mps  (default: cpu)
    --preview Show a live preview window while processing (requires display)
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Make sure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from src.pipeline.video_pipeline import VideoPipeline


def is_youtube_url(s: str) -> bool:
    return s.startswith("http") and ("youtube.com" in s or "youtu.be" in s)


def download_youtube(url: str, out_path: str) -> str:
    """
    Download a YouTube or YouTube Shorts video using yt-dlp.
    Returns the path to the downloaded file.
    """
    try:
        import yt_dlp
    except ImportError:
        print("yt-dlp not found. Install it with: pip install yt-dlp")
        sys.exit(1)

    print(f"Downloading YouTube video: {url}")
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_path,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # yt-dlp may append .mp4 if not already there
    if not Path(out_path).exists() and Path(out_path + ".mp4").exists():
        return out_path + ".mp4"
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Tennis Vision — Analyze a tennis video")
    parser.add_argument("--input",   required=True, help="Local video path or YouTube URL")
    parser.add_argument("--output",  default=None,  help="Output video path")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--device",  default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--preview", action="store_true", help="Show live preview")
    args = parser.parse_args()

    # --- Handle YouTube URL ---
    if is_youtube_url(args.input):
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_video = str(tmp_dir / "downloaded_video.mp4")
        input_file = download_youtube(args.input, tmp_video)
        input_path = Path(input_file)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: input file not found: {input_path}")
            sys.exit(1)

    output_path = args.output or str(input_path.parent / (input_path.stem + "_annotated.mp4"))

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")

    pipeline = VideoPipeline.from_config(args.config)

    if args.preview:
        # Preview mode: display each frame in a window
        for annotated_frame in pipeline.stream_frames(str(input_path)):
            cv2.imshow("Tennis Vision Preview", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Preview stopped by user.")
                break
        cv2.destroyAllWindows()
    else:
        stats = pipeline.process(str(input_path), output_path)
        print("\n=== Match Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
