"""
process_video_ptv.py — Track Large Particles in Video using PIV-UNet Guided PTV
================================================================================
Detects large particles (e.g. rocks) via binarization and contour centroids,
runs PIV-UNet to predict the dense fluid flow, and tracks the centroids across
frames using PIV-UNet Guided PTV. Exports a side-by-side comparison video.

Usage:
    python3 script/process_video_ptv.py \
        [--input_video PATH] \
        [--threshold 0.4] \
        [--min_area 20] \
        [--max_area 5000] \
        [--invert] \
        [--search_radius 12.0] \
        [--max_frames N]
"""

import sys
import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from piv_nn import PIVUNet
from piv_system import detect_particles_binarized, nn_guided_ptv

# ─────────────────────────────────────────────────────────────────────────────
# Config & Directories
# ─────────────────────────────────────────────────────────────────────────────
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULT_DIR = os.path.join(WORKSPACE, 'result')

def get_next_deploy_dir(base_dir):
    deploy_base = os.path.join(base_dir, 'deploy')
    os.makedirs(deploy_base, exist_ok=True)
    existing = [d for d in os.listdir(deploy_base) if d.startswith('run') and os.path.isdir(os.path.join(deploy_base, d))]
    nums = []
    for d in existing:
        try:
            nums.append(int(d[3:]))
        except ValueError:
            pass
    next_num = max(nums) + 1 if nums else 1
    run_dir = os.path.join(deploy_base, f'run{next_num}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Find the trained model path
train_base = os.path.join(RESULT_DIR, 'train')
CKPT_PATH = os.path.join(train_base, 'run1', 'checkpoints', 'best_model.pt')
if not os.path.exists(CKPT_PATH):
    import glob
    ckpt_candidates = sorted(glob.glob(os.path.join(train_base, 'run*', 'checkpoints', 'best_model.pt')))
    if ckpt_candidates:
        CKPT_PATH = ckpt_candidates[-1]
    else:
        CKPT_PATH = os.path.join(RESULT_DIR, 'checkpoints', 'best_model.pt')

# Setup target deploy directory
DEPLOY_DIR = get_next_deploy_dir(RESULT_DIR)
OUTPUT_VIDEO = os.path.join(DEPLOY_DIR, 'large_particle_tracking.mp4')

# ─────────────────────────────────────────────────────────────────────────────
# Argparse Setup
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Track large particles in video.")
parser.add_argument('--input_video', type=str, default=None,
                    help="Path to the input video. Defaults to default tracer video.")
parser.add_argument('--threshold', type=float, default=0.4,
                    help="Binarization threshold value (0.0 to 1.0).")
parser.add_argument('--min_area', type=float, default=20.0,
                    help="Minimum area of particle contour to track.")
parser.add_argument('--max_area', type=float, default=5000.0,
                    help="Maximum area of particle contour to track.")
parser.add_argument('--invert', action='store_true',
                    help="Invert binarization (for dark particles on light background).")
parser.add_argument('--search_radius', type=float, default=12.0,
                    help="Search radius (in pixels) for matching guided particles.")
parser.add_argument('--max_frames', type=int, default=None,
                    help="Max frames to process (useful for testing).")
parser.add_argument('--max_flow', type=float, default=8.0,
                    help="Max flow magnitude for color mapping scale.")
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Default video path if not specified
if args.input_video is None:
    args.input_video = os.path.join(WORKSPACE, 'data', 'PTV_dataset', 'laminar_jet_with_tracer_particles_for_piv.mp4')

# ─────────────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────────────
def flow_to_colormap(u: np.ndarray, v: np.ndarray, max_flow: float = 8.0) -> np.ndarray:
    """Map dense flow magnitude to a jet/RdYlBu_r-like colormap."""
    mag = np.sqrt(u**2 + v**2)
    norm_mag = np.clip(mag / max_flow, 0.0, 1.0)
    colormap = matplotlib.colormaps['RdYlBu_r']
    colored = colormap(norm_mag)[..., :3]
    bgr = (colored * 255).astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)

def draw_quiver(img: np.ndarray, u: np.ndarray, v: np.ndarray, step: int = 20, scale: float = 4.0, color: tuple = (0, 0, 0)) -> np.ndarray:
    """Draw vector arrows on top of an image indicating displacement."""
    H, W = img.shape[:2]
    out = img.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            dx = int(u[y, x] * scale)
            dy = int(v[y, x] * scale)
            if abs(dx) < 1 and abs(dy) < 1:
                continue
            cv2.arrowedLine(out, (x, y), (x + dx, y + dy), color, 1, line_type=cv2.LINE_AA, tipLength=0.25)
    return out

def draw_ptv_tracking(
    img: np.ndarray, 
    src: np.ndarray, 
    dst: np.ndarray, 
    p1: np.ndarray,
    src_angles: np.ndarray,
    dst_angles: np.ndarray,
    arrow_scale: float = 3.0
) -> np.ndarray:
    """Draw detected particles, their tracked matching vectors, and their translation/rotation speeds."""
    out = img.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        
    # Draw all detected particles in frame 1 as green dots
    for pt in p1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(out, (x, y), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        
    # Draw matched displacements as magenta arrow lines and translation/rotation speed texts
    for i, (s, d) in enumerate(zip(src, dst)):
        x1, y1 = int(round(s[0])), int(round(s[1]))
        # Apply visual scale to displacement vectors
        dx = d[0] - s[0]
        dy = d[1] - s[1]
        x2 = int(round(s[0] + dx * arrow_scale))
        y2 = int(round(s[1] + dy * arrow_scale))
        cv2.arrowedLine(out, (x1, y1), (x2, y2), (255, 0, 255), 2, line_type=cv2.LINE_AA, tipLength=0.3)
        
        # Calculate translation speed in pixels per frame
        speed = float(np.linalg.norm(d - s))
        
        # Calculate rotation speed (angular velocity) in degrees per frame
        d_theta = dst_angles[i] - src_angles[i]
        d_theta_deg = d_theta * 180.0 / np.pi
        # Wrap orientation angle differences to [-90, 90] due to 180-degree symmetry of central moments
        omega_deg = (d_theta_deg + 90.0) % 180.0 - 90.0
        
        # Format text strings
        text_v = f"v: {speed:.2f} px/f"
        text_w = f"w: {omega_deg:.1f} deg/f"
        
        # Calculate text bounding boxes to create background rect
        (w_v, h_v), baseline_v = cv2.getTextSize(text_v, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        (w_w, h_w), baseline_w = cv2.getTextSize(text_w, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        
        box_w = max(w_v, w_w) + 6
        box_h = h_v + h_w + baseline_v + baseline_w + 6
        
        # Shift text positions higher (35 pixels above particle centroid)
        offset_y = 35
        text_y_v = y1 - offset_y
        text_y_w = text_y_v + h_v + baseline_v + 3
        
        box_x1 = x1 + 6
        box_y1 = text_y_v - h_v - 3
        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h
        
        # Keep box coordinates within image boundaries
        H, W = out.shape[:2]
        box_x1 = max(0, min(box_x1, W - 1))
        box_y1 = max(0, min(box_y1, H - 1))
        box_x2 = max(0, min(box_x2, W - 1))
        box_y2 = max(0, min(box_y2, H - 1))
        
        # Draw semi-transparent white background (80% opacity)
        if box_x2 > box_x1 and box_y2 > box_y1:
            sub_img = out[box_y1:box_y2, box_x1:box_x2]
            rect = np.full_like(sub_img, 255) # White fill
            out[box_y1:box_y2, box_x1:box_x2] = cv2.addWeighted(sub_img, 0.2, rect, 0.8, 0)
        
        # Draw translation speed (v) and angular speed (w) in black over the white background
        cv2.putText(out, text_v, (x1 + 8, text_y_v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(out, text_w, (x1 + 8, text_y_w),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at {CKPT_PATH}. Please train the model first.")

print(f"Loading PIV-UNet from {CKPT_PATH}...")
model = PIVUNet(max_disp=4, base_ch=32).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print("Model loaded.")

# Open Video
cap = cv2.VideoCapture(args.input_video)
if not cap.isOpened():
    raise IOError(f"Cannot open input video {args.input_video}")

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video Properties: {orig_w}x{orig_h} @ {fps} FPS, Total Frames: {total_frames}")

# Setup video writer
# We write only the tracking overlay video (Left side only)
out_w = orig_w
out_h = orig_h
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (out_w, out_h))
print(f"Writing tracking output to: {OUTPUT_VIDEO}")

ret, prev_frame = cap.read()
if not ret:
    print("Failed to read first frame. Exiting.")
    sys.exit(1)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# Detect initial particles in first frame
p1 = detect_particles_binarized(
    prev_gray.astype(np.float32) / 255.0,
    threshold=args.threshold,
    min_area=args.min_area,
    max_area=args.max_area,
    invert=args.invert
)

frame_idx = 0
try:
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. PIV-UNet Flow Estimation
        img1_res = cv2.resize(prev_gray, (256, 256)).astype(np.float32) / 255.0
        img2_res = cv2.resize(curr_gray, (256, 256)).astype(np.float32) / 255.0
        
        input_tensor = torch.from_numpy(np.stack([img1_res, img2_res], axis=0)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_flow = model(input_tensor).squeeze(0).cpu().numpy() # (2, 256, 256)
            
        # Bilinear upscale flow to original resolution
        u_256, v_256 = pred_flow[0], pred_flow[1]
        u_orig = cv2.resize(u_256, (orig_w, orig_h)) * (orig_w / 256.0)
        v_orig = cv2.resize(v_256, (orig_w, orig_h)) * (orig_h / 256.0)
        
        # 2. Large Particle Detection
        p2 = detect_particles_binarized(
            curr_gray.astype(np.float32) / 255.0,
            threshold=args.threshold,
            min_area=args.min_area,
            max_area=args.max_area,
            invert=args.invert
        )
        
        # 3. PIV-UNet Guided PTV Tracking
        # Pass only the 2D spatial coordinates for matching
        src, dst = nn_guided_ptv(p1[:, :2], p2[:, :2], piv_u=u_orig, piv_v=v_orig, search_radius=args.search_radius)
        
        # Retrieve angles corresponding to matched source and destination particles
        src_angles = []
        dst_angles = []
        if len(src) > 0:
            for s, d in zip(src, dst):
                idx1 = np.argmin(np.linalg.norm(p1[:, :2] - s, axis=1))
                idx2 = np.argmin(np.linalg.norm(p2[:, :2] - d, axis=1))
                src_angles.append(p1[idx1, 2])
                dst_angles.append(p2[idx2, 2])
        src_angles = np.array(src_angles)
        dst_angles = np.array(dst_angles)
        
        # 4. Render Visualizations
        # Raw current frame overlaid with detected particle centroids, tracking vectors, and speeds
        tracking_overlay = draw_ptv_tracking(prev_frame, src, dst, p1, src_angles, dst_angles)
        
        # Write to video
        out_writer.write(tracking_overlay)
        
        # Prepare for next frame
        prev_gray = curr_gray
        prev_frame = curr_frame.copy()
        p1 = p2 # Particles detected in curr become p1 for the next pair
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{total_frames - 1} | Found {len(p2)} particles, matched {len(src)} trajectories")
            
        if args.max_frames and frame_idx >= args.max_frames:
            print(f"Reached --max_frames cap of {args.max_frames}. Stopping.")
            break
finally:
    cap.release()
    out_writer.release()
    print(f"\nProcessing complete! Output saved to: {OUTPUT_VIDEO}")
