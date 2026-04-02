"""
TrackNet Training Script

HOW IT WORKS:
  1. Reads a labeled CSV file with columns: file, frame, cx, cy
     (one row per frame — cx/cy are pixel coords, empty if ball not visible)
  2. For each sample, loads 3 consecutive frames from the video
  3. Stacks them into a 9-channel tensor
  4. Generates a Gaussian heatmap target at the ball position
  5. Trains TrackNet with weighted BCE loss

DATASET FORMAT (CSV):
  video_path,frame_idx,cx,cy
  videos/rally1.mp4,0,,          <- no ball visible
  videos/rally1.mp4,1,320,240
  videos/rally1.mp4,2,335,238
  ...

HOW TO LABEL YOUR DATA:
  Option A — Use your Shorts video + manual labeling tool:
    pip install labelme  OR  use CVAT (free at cvat.ai)

  Option B — Use Roboflow to export frame-by-frame annotations
    (see colab_run.ipynb Section C)

  Option C — Use TrackNetV2 public dataset
    https://github.com/nttcom/TrackNetV2

USAGE:
  python scripts/train_tracknet.py \
    --csv data/labels.csv \
    --epochs 30 \
    --output models/tracknet.pt \
    --device cuda
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.detection.tracknet import TrackNet, TrackNetLoss, make_heatmap

INPUT_W = 640
INPUT_H = 360


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrackNetDataset(Dataset):
    """
    Loads triplets of consecutive frames + heatmap targets from a labeled CSV.
    """

    def __init__(self, csv_path: str, augment: bool = True):
        self.augment = augment
        self.samples = []   # list of (video_path, [frame_idx-2, frame_idx-1, frame_idx], cx, cy)

        # Group rows by video
        by_video: dict[str, list] = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row["video_path"]
                if v not in by_video:
                    by_video[v] = []
                cx = float(row["cx"]) if row["cx"] else None
                cy = float(row["cy"]) if row["cy"] else None
                by_video[v].append((int(row["frame_idx"]), cx, cy))

        # Build triplets: need at least 3 consecutive frames
        for video_path, rows in by_video.items():
            rows.sort(key=lambda r: r[0])
            for i in range(2, len(rows)):
                f0, _, _  = rows[i - 2]
                f1, _, _  = rows[i - 1]
                fi, cx, cy = rows[i]
                # Only use triplets with consecutive frame indices
                if fi - f0 == 2 and fi - f1 == 1:
                    self.samples.append((video_path, [f0, f1, fi], cx, cy))

        print(f"[TrackNetDataset] {len(self.samples)} triplets from {len(by_video)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_indices, cx, cy = self.samples[idx]

        cap = cv2.VideoCapture(video_path)
        frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (INPUT_W, INPUT_H))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # Scale ball coords to input resolution
        if cx is not None and cy is not None:
            orig_h = frames[2].shape[0]
            orig_w = frames[2].shape[1]
            cx_scaled = cx * INPUT_W / orig_w
            cy_scaled = cy * INPUT_H / orig_h
        else:
            cx_scaled, cy_scaled = None, None

        # Optional: horizontal flip augmentation
        if self.augment and np.random.rand() > 0.5:
            frames = [np.fliplr(f).copy() for f in frames]
            if cx_scaled is not None:
                cx_scaled = INPUT_W - cx_scaled

        # Stack into 9-channel tensor
        stacked = np.concatenate(frames, axis=2)  # (H, W, 9)
        tensor = torch.from_numpy(stacked).float() / 255.0
        tensor = tensor.permute(2, 0, 1)           # (9, H, W)

        # Build target heatmap
        heatmap = make_heatmap(cx_scaled, cy_scaled, INPUT_H, INPUT_W, sigma=5)
        heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0)  # (1, H, W)

        return tensor, heatmap_tensor


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    print(f"Training on: {device}")

    dataset = TrackNetDataset(args.csv, augment=True)
    if len(dataset) == 0:
        print("ERROR: No samples found. Check your CSV format.")
        sys.exit(1)

    # 90/10 train/val split
    val_size  = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    model     = TrackNet().to(device)
    criterion = TrackNetLoss(pos_weight=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")
    output_path   = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, str(output_path))
            print(f"  Saved best model → {output_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrackNet for tennis ball detection")
    parser.add_argument("--csv",        required=True,              help="Path to labels CSV")
    parser.add_argument("--epochs",     type=int, default=30,       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,        help="Batch size")
    parser.add_argument("--output",     default="models/tracknet.pt", help="Output weights path")
    parser.add_argument("--device",     default="cuda",             help="cpu / cuda")
    args = parser.parse_args()
    train(args)
