"""
piv_system.py — Core PIV/PTV Algorithm Library
================================================
Contains implementations of:
  1. Classical PIV  — FFT-based cross-correlation window method
  2. Particle Detection — Laplacian of Gaussian (LoG) blob detector
  3. Classical PTV  — Nearest-neighbour particle matching
  4. Hybrid PIV-Guided PTV — PIV velocity used to predict particle
                              positions before matching (improves accuracy
                              at high seeding densities)
  5. Utility helpers — .flo reader, RMSE, interpolation to regular grid
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import maximum_filter, label
from scipy.interpolate import griddata
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# I/O Utilities
# ─────────────────────────────────────────────────────────────────────────────

def read_flo(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Middlebury .flo optical-flow file.

    Returns
    -------
    u : (H, W) float32 — horizontal velocity component
    v : (H, W) float32 — vertical velocity component
    """
    with open(path, "rb") as f:
        magic = np.frombuffer(f.read(4), dtype=np.float32)[0]
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo magic number in {path}")
        w = np.frombuffer(f.read(4), dtype=np.int32)[0]
        h = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(h * w * 8), dtype=np.float32)
    flow = data.reshape((h, w, 2))
    return flow[:, :, 0], flow[:, :, 1]


def load_image_gray(path: str) -> np.ndarray:
    """Load an image as float32 grayscale in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Classical PIV — FFT Cross-Correlation
# ─────────────────────────────────────────────────────────────────────────────

def _subpixel_peak(corr: np.ndarray, peak_row: int, peak_col: int) -> Tuple[float, float]:
    """Gaussian sub-pixel interpolation around the correlation peak."""
    rows, cols = corr.shape
    r, c = peak_row, peak_col

    # Clamp to avoid boundary issues
    r = np.clip(r, 1, rows - 2)
    c = np.clip(c, 1, cols - 2)

    def _gauss(a, b, cc):
        with np.errstate(divide='ignore', invalid='ignore'):
            num = np.log(np.maximum(a, 1e-10)) - np.log(np.maximum(cc, 1e-10))
            den = 2 * (np.log(np.maximum(a, 1e-10)) + np.log(np.maximum(cc, 1e-10))
                       - 2 * np.log(np.maximum(b, 1e-10)))
            return np.where(np.abs(den) > 1e-12, num / den, 0.0)

    dr = _gauss(corr[r - 1, c], corr[r, c], corr[r + 1, c])
    dc = _gauss(corr[r, c - 1], corr[r, c], corr[r, c + 1])
    return float(dr), float(dc)


def classical_piv(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 32,
    overlap: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Window-based FFT cross-correlation PIV.

    Parameters
    ----------
    img1, img2 : (H, W) float32
    window_size : interrogation window edge length (pixels)
    overlap     : overlap between windows (pixels), step = window_size - overlap

    Returns
    -------
    x, y : 1-D centre coordinates of each interrogation window (pixels)
    u, v : displacement arrays, shape (len(y), len(x)) (pixels/frame)
    """
    H, W = img1.shape
    step = window_size - overlap

    xs = np.arange(window_size // 2, W - window_size // 2 + 1, step)
    ys = np.arange(window_size // 2, H - window_size // 2 + 1, step)

    u = np.zeros((len(ys), len(xs)), dtype=np.float32)
    v = np.zeros((len(ys), len(xs)), dtype=np.float32)

    hw = window_size // 2

    for j, cy in enumerate(ys):
        for i, cx in enumerate(xs):
            r0, r1 = cy - hw, cy + hw
            c0, c1 = cx - hw, cx + hw

            w1 = img1[r0:r1, c0:c1]
            w2 = img2[r0:r1, c0:c1]

            # Zero-mean windows
            w1 = w1 - w1.mean()
            w2 = w2 - w2.mean()

            # FFT cross-correlation
            F1 = np.fft.fft2(w1)
            F2 = np.fft.fft2(w2)
            corr = np.fft.ifft2(np.conj(F1) * F2).real
            corr = np.fft.fftshift(corr)

            peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
            pr, pc = peak_idx

            dr, dc = _subpixel_peak(corr, pr, pc)
            pr_sub = pr + dr
            pc_sub = pc + dc

            # Displacement relative to centre
            v[j, i] = pr_sub - hw
            u[j, i] = pc_sub - hw

    return xs, ys, u, v


# ─────────────────────────────────────────────────────────────────────────────
# 2. Particle Detection — LoG Blob Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_particles(
    img: np.ndarray,
    min_sigma: float = 1.0,
    max_sigma: float = 4.0,
    num_sigma: int = 5,
    threshold: float = 0.03,
    min_distance: int = 4,
) -> np.ndarray:
    """Detect particle centroids via Laplacian of Gaussian (LoG).

    Returns
    -------
    particles : (N, 2) array of (col, row) = (x, y) centroids
    """
    sigmas = np.linspace(min_sigma, max_sigma, num_sigma)
    img_pad = img.astype(np.float32)

    # Multi-scale LoG response
    log_stack = []
    for sigma in sigmas:
        blurred = cv2.GaussianBlur(img_pad, (0, 0), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_32F) * (sigma ** 2)
        log_stack.append(np.abs(log))

    # Max response across scales
    log_max = np.max(log_stack, axis=0)

    # Non-maximum suppression
    nms = maximum_filter(log_max, size=min_distance)
    mask = (log_max == nms) & (log_max > threshold)

    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Sub-pixel refinement: centroid in a small neighbourhood
    particles = []
    h, w = img.shape
    for r, c in zip(rows, cols):
        r0, r1 = max(0, r - 2), min(h, r + 3)
        c0, c1 = max(0, c - 2), min(w, c + 3)
        patch = img[r0:r1, c0:c1]
        total = patch.sum()
        if total < 1e-8:
            particles.append([float(c), float(r)])
        else:
            dr_grid, dc_grid = np.mgrid[r0:r1, c0:c1]
            sub_r = float((dr_grid * patch).sum() / total)
            sub_c = float((dc_grid * patch).sum() / total)
            particles.append([sub_c, sub_r])   # (x, y)

    return np.array(particles, dtype=np.float32)


def detect_particles_binarized(
    img: np.ndarray,
    threshold: float = 0.4,
    min_area: float = 20.0,
    max_area: float = 5000.0,
    invert: bool = False,
) -> np.ndarray:
    """Detect large particle centroids and orientation angles via binarization and central moments.

    Parameters
    ----------
    img : (H, W) float32 in [0, 1]
    threshold : binarization threshold in [0, 1]
    min_area : minimum contour area in pixels to keep
    max_area : maximum contour area in pixels to keep
    invert : if True, invert the thresholding (objects are darker than background)

    Returns
    -------
    particles : (N, 3) array of (col, row, theta) = (x, y, theta) centroids and angles in radians
    """
    # Convert image from float32 in [0, 1] to uint8 in [0, 255]
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # Apply thresholding
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, thresh_img = cv2.threshold(img_uint8, int(threshold * 255), 255, thresh_type)

    # Find contours
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    particles = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            M = cv2.moments(c)
            if M["m00"] > 1e-5:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                
                # Orientation angle via central moments:
                mu20 = M["mu20"]
                mu02 = M["mu02"]
                mu11 = M["mu11"]
                if abs(mu20 - mu02) > 1e-5 or abs(mu11) > 1e-5:
                    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
                else:
                    theta = 0.0
                    
                particles.append([cx, cy, theta])

    if not particles:
        return np.empty((0, 3), dtype=np.float32)

    return np.array(particles, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Classical PTV — Nearest-Neighbour Matching
# ─────────────────────────────────────────────────────────────────────────────

def classical_ptv(
    particles1: np.ndarray,
    particles2: np.ndarray,
    max_distance: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match particles between two frames using nearest-neighbour search.

    Parameters
    ----------
    particles1 : (N, 2) — (x, y) in frame 1
    particles2 : (M, 2) — (x, y) in frame 2
    max_distance : maximum allowed displacement for a valid match (pixels)

    Returns
    -------
    src : (K, 2) matched positions in frame 1
    dst : (K, 2) matched positions in frame 2
    """
    if len(particles1) == 0 or len(particles2) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    matched_src, matched_dst = [], []
    used = np.zeros(len(particles2), dtype=bool)

    # For each particle in frame 1, find closest in frame 2
    for p1 in particles1:
        dists = np.linalg.norm(particles2 - p1, axis=1)
        dists[used] = np.inf
        idx = np.argmin(dists)
        if dists[idx] <= max_distance:
            matched_src.append(p1)
            matched_dst.append(particles2[idx])
            used[idx] = True

    if not matched_src:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    return np.array(matched_src, dtype=np.float32), np.array(matched_dst, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hybrid PIV-Guided PTV
# ─────────────────────────────────────────────────────────────────────────────

def _interpolate_piv(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    query_xy: np.ndarray,
) -> np.ndarray:
    """Bi-linear interpolation of PIV velocity at arbitrary positions."""
    if len(query_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Build grid points
    xx, yy = np.meshgrid(x_grid, y_grid)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    u_vals = u.ravel()
    v_vals = v.ravel()

    uq = griddata(points, u_vals, query_xy, method='linear', fill_value=0.0)
    vq = griddata(points, v_vals, query_xy, method='linear', fill_value=0.0)
    return np.column_stack([uq, vq]).astype(np.float32)


def guided_ptv(
    particles1: np.ndarray,
    particles2: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    piv_u: np.ndarray,
    piv_v: np.ndarray,
    search_radius: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """PIV-Guided PTV: use the PIV velocity field to predict each particle's
    next position, then search only within a small radius around that prediction.

    Parameters
    ----------
    particles1   : (N, 2) positions in frame 1
    particles2   : (M, 2) positions in frame 2
    x_grid, y_grid : 1-D PIV grid coordinates
    piv_u, piv_v : (Ny, Nx) PIV displacement maps
    search_radius : radius (pixels) around the predicted position to search

    Returns
    -------
    src : (K, 2) matched positions in frame 1
    dst : (K, 2) matched positions in frame 2
    """
    if len(particles1) == 0 or len(particles2) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    # Predict positions of frame-1 particles in frame 2
    vel = _interpolate_piv(x_grid, y_grid, piv_u, piv_v, particles1)
    predicted2 = particles1 + vel  # shape (N, 2)

    matched_src, matched_dst = [], []
    used = np.zeros(len(particles2), dtype=bool)

    for p1, pred in zip(particles1, predicted2):
        # Residual distances from prediction
        dists = np.linalg.norm(particles2 - pred, axis=1)
        dists[used] = np.inf
        idx = np.argmin(dists)
        if dists[idx] <= search_radius:
            matched_src.append(p1)
            matched_dst.append(particles2[idx])
            used[idx] = True

    if not matched_src:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    return np.array(matched_src, dtype=np.float32), np.array(matched_dst, dtype=np.float32)


def nn_guided_ptv(
    particles1: np.ndarray,
    particles2: np.ndarray,
    piv_u: np.ndarray,
    piv_v: np.ndarray,
    search_radius: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """PIV-UNet Guided PTV: use dense PIV velocity field to predict each particle's
    next position, then search only within a small radius around that prediction.

    Parameters
    ----------
    particles1   : (N, 2) positions in frame 1
    particles2   : (M, 2) positions in frame 2
    piv_u, piv_v : (H, W) dense PIV displacement maps
    search_radius : radius (pixels) around the predicted position to search

    Returns
    -------
    src : (K, 2) matched positions in frame 1
    dst : (K, 2) matched positions in frame 2
    """
    if len(particles1) == 0 or len(particles2) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    H, W = piv_u.shape
    x = np.clip(particles1[:, 0], 0, W - 1)
    y = np.clip(particles1[:, 1], 0, H - 1)

    # Fast bilinear interpolation on dense grid
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    uq = (piv_u[y0, x0] * wa +
          piv_u[y1, x0] * wb +
          piv_u[y0, x1] * wc +
          piv_u[y1, x1] * wd)

    vq = (piv_v[y0, x0] * wa +
          piv_v[y1, x0] * wb +
          piv_v[y0, x1] * wc +
          piv_v[y1, x1] * wd)

    vel = np.column_stack([uq, vq])
    predicted2 = particles1 + vel  # shape (N, 2)

    matched_src, matched_dst = [], []
    used = np.zeros(len(particles2), dtype=bool)

    for p1, pred in zip(particles1, predicted2):
        dists = np.linalg.norm(particles2 - pred, axis=1)
        dists[used] = np.inf
        idx = np.argmin(dists)
        if dists[idx] <= search_radius:
            matched_src.append(p1)
            matched_dst.append(particles2[idx])
            used[idx] = True

    if not matched_src:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    return np.array(matched_src, dtype=np.float32), np.array(matched_dst, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rmse_piv(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_gt: np.ndarray,
    v_gt: np.ndarray,
) -> float:
    """RMSE of velocity magnitude between predicted and ground-truth fields."""
    err_u = u_pred - u_gt
    err_v = v_pred - v_gt
    return float(np.sqrt(np.mean(err_u ** 2 + err_v ** 2)))


def rmse_ptv(
    src: np.ndarray,
    dst: np.ndarray,
    u_gt: np.ndarray,
    v_gt: np.ndarray,
    img_shape: Tuple[int, int],
) -> float:
    """RMSE of PTV displacement vs ground-truth at matched particle positions."""
    if len(src) == 0:
        return float('nan')

    H, W = img_shape
    disp_pred = dst - src   # (K, 2): (dx, dy)

    # Sample ground-truth at particle positions (nearest neighbour)
    xs = np.clip(np.round(src[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(src[:, 1]).astype(int), 0, H - 1)

    gt_u = u_gt[ys, xs]
    gt_v = v_gt[ys, xs]

    err_u = disp_pred[:, 0] - gt_u
    err_v = disp_pred[:, 1] - gt_v
    return float(np.sqrt(np.mean(err_u ** 2 + err_v ** 2)))


def ptv_to_grid(
    src: np.ndarray,
    dst: np.ndarray,
    img_shape: Tuple[int, int],
    grid_step: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate sparse PTV vectors onto a regular grid for visualisation."""
    H, W = img_shape
    disp = dst - src   # (K, 2)

    xs = np.arange(0, W, grid_step, dtype=float)
    ys = np.arange(0, H, grid_step, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    query = np.column_stack([xx.ravel(), yy.ravel()])

    u_interp = griddata(src, disp[:, 0], query, method='linear', fill_value=0.0)
    v_interp = griddata(src, disp[:, 1], query, method='linear', fill_value=0.0)

    return xs, ys, u_interp.reshape(xx.shape), v_interp.reshape(yy.shape)
