"""
TrackNet — Tennis Ball Detection via Sequential Frame Heatmap Prediction.

HOW IT WORKS vs YOLOv8:

  YOLOv8 (single frame):
    frame → CNN → bounding boxes
    Problem: Tennis ball at 200 km/h = heavy motion blur → detector misses it

  TrackNet (3 frames):
    [frame_t-2, frame_t-1, frame_t] → stacked as 9-channel input → CNN → heatmap
    The model SEES the ball's movement trail across frames → much harder to miss
    Output: a grayscale heatmap where bright pixels = ball location (Gaussian blob)

ARCHITECTURE:
  Encoder: VGG16-like (Conv → BN → ReLU blocks, MaxPool to downsample)
  Decoder: Symmetric upsampling blocks (Conv → BN → ReLU, Upsample)
  Skip connections: encoder features added to decoder (like U-Net)
  Final layer: sigmoid → values 0-1 (heatmap)

TRAINING TARGET:
  For each frame, draw a filled Gaussian circle (sigma=5px) at the ball center.
  Loss = Binary Cross Entropy between predicted heatmap and target heatmap.
  Positive pixels (ball) are very rare → use weighted BCE (weight=2.0 for ball pixels).

REFERENCE: TrackNetV2 (Huang et al., 2019)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv → BN → ReLU → Conv → BN → ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TrackNet(nn.Module):
    """
    TrackNet: 9-channel input (3 RGB frames) → 1-channel heatmap output.

    Input shape:  (B, 9, H, W)   — 3 frames × 3 channels stacked
    Output shape: (B, 1, H, W)   — ball probability heatmap
    """

    def __init__(self):
        super().__init__()

        # --- Encoder (downsample x4) ---
        self.enc1 = _conv_block(9,   64)   # 9 = 3 frames × 3 RGB channels
        self.enc2 = _conv_block(64,  128)
        self.enc3 = _conv_block(128, 256)
        self.enc4 = _conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        self.bottleneck = _conv_block(512, 512)

        # --- Decoder (upsample x4 with skip connections) ---
        self.dec4 = _conv_block(512 + 512, 256)
        self.dec3 = _conv_block(256 + 256, 128)
        self.dec2 = _conv_block(128 + 128, 64)
        self.dec1 = _conv_block(64  + 64,  64)

        # --- Output: 1-channel heatmap ---
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)               # (B, 64,  H,   W)
        e2 = self.enc2(self.pool(e1))   # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))   # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))   # (B, 512, H/8, W/8)

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))  # (B, 512, H/16, W/16)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([F.interpolate(b,  size=e4.shape[2:], mode='bilinear', align_corners=False), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))   # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Inference Wrapper
# ---------------------------------------------------------------------------

class TrackNetDetector:
    """
    Uses a trained TrackNet model to detect the tennis ball in a video stream.

    Keeps a rolling buffer of the last 3 frames.
    On each call to detect(), it stacks the buffer and runs one forward pass.

    Usage:
        detector = TrackNetDetector("models/tracknet.pt", device="cuda")
        for frame in video_frames:
            cx, cy, conf = detector.detect(frame)
    """

    # Input size TrackNet was trained on (must match training)
    INPUT_W = 640
    INPUT_H = 360

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.5,
    ):
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.model = TrackNet().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        # Support both raw state_dict and checkpoint dicts
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()

        # Rolling buffer: stores last 3 preprocessed frames (H, W, 3) uint8
        self._buffer: list[np.ndarray] = []

    def detect(self, frame: np.ndarray) -> tuple[Optional[float], Optional[float], float]:
        """
        Feed one BGR frame. Returns (cx, cy, confidence) or (None, None, 0).
        Needs at least 3 frames before it can produce a detection.
        """
        resized = cv2.resize(frame, (self.INPUT_W, self.INPUT_H))
        self._buffer.append(resized)
        if len(self._buffer) > 3:
            self._buffer.pop(0)

        if len(self._buffer) < 3:
            return None, None, 0.0   # not enough frames yet

        # Stack 3 frames into 9-channel tensor: (9, H, W)
        stacked = np.concatenate([
            self._buffer[0],   # oldest frame (RGB channels 0-2)
            self._buffer[1],   # middle frame  (RGB channels 3-5)
            self._buffer[2],   # newest frame  (RGB channels 6-8)
        ], axis=2)             # (H, W, 9)

        # Normalize to [0, 1] and convert to tensor
        tensor = torch.from_numpy(stacked).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 9, H, W)

        with torch.no_grad():
            heatmap = self.model(tensor)  # (1, 1, H, W)

        heatmap_np = heatmap.squeeze().cpu().numpy()  # (H, W)

        # Find the brightest point in the heatmap
        max_conf = float(heatmap_np.max())
        if max_conf < self.conf_threshold:
            return None, None, max_conf

        # Get location of max value
        hy, hx = np.unravel_index(heatmap_np.argmax(), heatmap_np.shape)

        # Scale back to original frame size
        orig_h, orig_w = frame.shape[:2]
        cx = hx * orig_w / self.INPUT_W
        cy = hy * orig_h / self.INPUT_H

        return float(cx), float(cy), max_conf

    def reset(self):
        """Clear the frame buffer (call between rallies)."""
        self._buffer.clear()


# ---------------------------------------------------------------------------
# Training Helpers
# ---------------------------------------------------------------------------

def make_heatmap(
    cx: Optional[float],
    cy: Optional[float],
    height: int,
    width: int,
    sigma: int = 5,
) -> np.ndarray:
    """
    Create a training target heatmap: a Gaussian blob at (cx, cy).
    If cx/cy is None (no ball visible), returns all-zeros map.

    Args:
        cx, cy:  Ball center in pixels (at the heatmap resolution)
        height, width: Heatmap size (should match INPUT_H, INPUT_W)
        sigma:   Gaussian radius in pixels (larger = softer target)
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    if cx is None or cy is None:
        return heatmap

    x, y = int(round(cx)), int(round(cy))
    if 0 <= x < width and 0 <= y < height:
        cv2.circle(heatmap, (x, y), sigma * 2, 1.0, -1)
        heatmap = cv2.GaussianBlur(heatmap, (sigma * 4 + 1, sigma * 4 + 1), sigma)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()   # normalize to [0, 1]

    return heatmap


class TrackNetLoss(nn.Module):
    """
    Weighted Binary Cross Entropy for heatmap prediction.
    Ball pixels are rare (maybe 50 out of 230,000) so we up-weight them.
    """

    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 1, H, W)
        weight = torch.where(target > 0.5,
                             torch.full_like(target, self.pos_weight),
                             torch.ones_like(target))
        return F.binary_cross_entropy(pred, target, weight=weight)
