"""
piv_nn.py — Neural Network PIV Model and Dataset
=================================================
Architecture: PIV-UNet
  - Siamese encoder: shared ConvBlock feature extraction for each frame
  - Correlation layer: computes cross-correlation feature volume (like FlowNet)
  - Decoder: progressive upsampling with skip connections → dense (u, v) flow

Dataset: PIVDataset
  - Loads img1, img2, (u, v) from PIV_dataset .tif + .flo files
  - Supports all subsets: uniform, backstep, cylinder, SQG, DNS_turbulence,
    JHTDB_channel, JHTDB_channel_hd, JHTDB_isotropic1024_hd, JHTDB_mhd1024_hd
"""

from __future__ import annotations

import os
import glob
import struct
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _read_flo_fast(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        magic, w, h = struct.unpack("fii", f.read(12))
        if abs(magic - 202021.25) > 1.0:
            raise ValueError(f"Bad .flo magic in {path}")
        data = np.frombuffer(f.read(h * w * 8), dtype=np.float32)
    flow = data.reshape(h, w, 2)
    return flow[:, :, 0], flow[:, :, 1]


class PIVDataset(Dataset):
    """PyTorch Dataset that loads synthetic PIV image pairs + ground-truth flow.

    Each item is a dict:
        img_pair : (2, H, W) float32 tensor — stacked frame1 and frame2
        flow     : (2, H, W) float32 tensor — (u, v) ground-truth displacement
    """

    def __init__(
        self,
        data_root: str,
        subsets: Optional[List[str]] = None,
        img_size: int = 256,
        max_samples: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        data_root : path to  …/PIV-genImages/data
        subsets   : list of sub-folder names to include; None → all found
        img_size  : resize target (both H and W)
        max_samples : cap on total samples (useful for quick debugging)
        """
        self.img_size = img_size
        self.samples: List[Tuple[str, str, str]] = []  # (img1, img2, flo)

        if subsets is None:
            subsets = [
                d for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))
            ]

        for subset in subsets:
            folder = os.path.join(data_root, subset, subset)
            if not os.path.isdir(folder):
                # Try one level deeper
                folder = os.path.join(data_root, subset)
            flo_paths = sorted(glob.glob(os.path.join(folder, "*_flow.flo")))
            for flo in flo_paths:
                base = flo.replace("_flow.flo", "")
                img1 = base + "_img1.tif"
                img2 = base + "_img2.tif"
                if os.path.exists(img1) and os.path.exists(img2):
                    self.samples.append((img1, img2, flo))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        import cv2

        img1_path, img2_path, flo_path = self.samples[idx]

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if self.img_size != img1.shape[0] or self.img_size != img1.shape[1]:
            img1 = cv2.resize(img1, (self.img_size, self.img_size))
            img2 = cv2.resize(img2, (self.img_size, self.img_size))

        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        u, v = _read_flo_fast(flo_path)
        if u.shape[0] != self.img_size:
            u = cv2.resize(u, (self.img_size, self.img_size))
            v = cv2.resize(v, (self.img_size, self.img_size))

        img_pair = torch.from_numpy(np.stack([img1, img2], axis=0))  # (2, H, W)
        flow = torch.from_numpy(np.stack([u, v], axis=0))            # (2, H, W)

        return {"img_pair": img_pair, "flow": flow}


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class _EncBlock(nn.Module):
    """Double conv + downsample."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn_relu(in_ch, out_ch),
            _conv_bn_relu(out_ch, out_ch, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecBlock(nn.Module):
    """Upsample + double conv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            _conv_bn_relu(out_ch + skip_ch, out_ch),
            _conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad in case of size mismatch
        if x.shape != skip.shape:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([x, skip], dim=1))


class _CorrelationLayer(nn.Module):
    """Lightweight correlation: computes dot-product between spatially-shifted
    feature maps within a local window.  Output channels = (2*max_disp+1)^2.
    """
    def __init__(self, max_disp: int = 4):
        super().__init__()
        self.max_disp = max_disp

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f1.shape
        d = self.max_disp
        pad_f2 = F.pad(f2, [d, d, d, d])
        out = []
        for dy in range(-d, d + 1):
            for dx in range(-d, d + 1):
                shifted = pad_f2[:, :, d + dy: d + dy + H, d + dx: d + dx + W]
                corr = (f1 * shifted).mean(dim=1, keepdim=True)   # (B,1,H,W)
                out.append(corr)
        return torch.cat(out, dim=1)   # (B, (2d+1)^2, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# PIV-UNet Model
# ─────────────────────────────────────────────────────────────────────────────

class PIVUNet(nn.Module):
    """PIV-UNet: Siamese encoder + correlation + U-Net decoder.

    Input:  (B, 2, H, W) — frame1 and frame2 stacked
    Output: (B, 2, H, W) — (u, v) dense displacement field
    """

    def __init__(self, max_disp: int = 4, base_ch: int = 32):
        super().__init__()
        d = max_disp
        corr_ch = (2 * d + 1) ** 2   # e.g. 81 for d=4
        ch1 = base_ch          # 32
        ch2 = base_ch * 2      # 64
        ch3 = base_ch * 4      # 128

        # Siamese encoder (shared weights via single module called twice)
        self.enc1 = _EncBlock(1, ch1)     # → ch1  at H/2
        self.enc2 = _EncBlock(ch1, ch2)   # → ch2  at H/4
        self.enc3 = _EncBlock(ch2, ch3)   # → ch3  at H/8

        # Correlation layer at H/8 resolution
        self.corr = _CorrelationLayer(max_disp=d)

        # Bottleneck: corr_ch + ch3 (f1) + ch3 (f2) → ch3*2
        self.bottleneck = nn.Sequential(
            _conv_bn_relu(corr_ch + ch3 + ch3, ch3 * 2),
            _conv_bn_relu(ch3 * 2, ch3 * 2),
        )

        # Decoder using frame-1 skip connections only
        # dec3: (ch3*2) upsample, skip=ch2 (f1-enc2) → ch3
        self.dec3 = _DecBlock(ch3 * 2, ch2, ch3)
        # dec2: ch3 upsample, skip=ch1 (f1-enc1) → ch2
        self.dec2 = _DecBlock(ch3, ch1, ch2)
        # dec1: ch2 upsample, skip=1 (raw f1) → ch1
        self.dec1 = _DecBlock(ch2, 1, ch1)

        # Final flow head
        self.flow_head = nn.Conv2d(ch1, 2, 1)

    def _encode(self, x: torch.Tensor):
        s1 = self.enc1(x)   # H/2
        s2 = self.enc2(s1)  # H/4
        s3 = self.enc3(s2)  # H/8
        return s1, s2, s3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = x[:, 0:1, :, :]
        f2 = x[:, 1:2, :, :]

        # Siamese encoding
        s1_a, s2_a, s3_a = self._encode(f1)
        _s1_b, _s2_b, s3_b = self._encode(f2)

        # Correlation at deepest scale (H/8)
        corr_feat = self.corr(s3_a, s3_b)

        # Fuse: corr + f1-enc3 + f2-enc3
        fused = torch.cat([corr_feat, s3_a, s3_b], dim=1)
        bottleneck = self.bottleneck(fused)

        # Decode with frame-1 skip connections
        d3 = self.dec3(bottleneck, s2_a)   # skip: ch2 from f1-enc2
        d2 = self.dec2(d3, s1_a)           # skip: ch1 from f1-enc1
        d1 = self.dec1(d2, f1)             # skip: 1ch raw f1

        flow = self.flow_head(d1)
        if flow.shape[-2:] != x.shape[-2:]:
            flow = F.interpolate(flow, size=x.shape[-2:],
                                 mode='bilinear', align_corners=False)
        return flow


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class EPELoss(nn.Module):
    """End-Point Error (EPE) loss — mean Euclidean distance between
    predicted and ground-truth flow vectors."""
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        epe = torch.sqrt(((pred - gt) ** 2).sum(dim=1) + 1e-8)
        return epe.mean()


class MultiscaleEPELoss(nn.Module):
    """EPE computed at multiple scales for more stable training."""
    def __init__(self, weights=(0.32, 0.08, 0.02, 0.01)):
        super().__init__()
        self.weights = weights
        self.epe = EPELoss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss = self.epe(pred, gt) * self.weights[0]
        for i, w in enumerate(self.weights[1:], 1):
            scale = 1 / (2 ** i)
            gt_ds = F.interpolate(gt, scale_factor=scale,
                                  mode='bilinear', align_corners=False) * scale
            pred_ds = F.interpolate(pred, scale_factor=scale,
                                    mode='bilinear', align_corners=False)
            loss = loss + w * self.epe(pred_ds, gt_ds)
        return loss
