"""
CLI script — Process a single tennis video from the command line.

Usage:
    python scripts/process_video.py --input my_match.mp4 --output annotated.mp4

Options:
    --input   Path to input video file (required)
    --output  Path for annotated output video (default: input_annotated.mp4)
    --config  Path to config YAML (default: config/config.yaml)
    --device  cpu | cuda | mps  (default: cpu)
    --preview Show a live preview window while processing (requires display)
"""

import argparse
import sys
from pathlib import Path

# Make sure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from src.pipeline.video_pipeline import VideoPipeline


def main():
    parser = argparse.ArgumentParser(description="Tennis Vision — Analyze a tennis video")
    parser.add_argument("--input",   required=True, help="Input video path")
    parser.add_argument("--output",  default=None,  help="Output video path")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--device",  default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--preview", action="store_true", help="Show live preview")
    args = parser.parse_args()

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
