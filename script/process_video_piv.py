"""
process_video_piv.py — Process Video using PIV-UNet
===================================================
Applies the trained PIV-UNet model to estimate optical flow on the experimental video
'laminar_jet_with_tracer_particles_for_piv.mp4'. Saves the result as a side-by-side
video containing a quiver plot overlay and a scientific color-coded flow magnitude map
overlaid with dense black vector arrows.

Usage:
    python3 script/process_video_piv.py [--max_frames N]
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

# ─────────────────────────────────────────────────────────────────────────────
# Config & Directories
# ─────────────────────────────────────────────────────────────────────────────
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_VIDEO = os.path.join(WORKSPACE, 'data', 'PTV_dataset', 'laminar_jet_with_tracer_particles_for_piv.mp4')
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
OUTPUT_VIDEO = os.path.join(DEPLOY_DIR, 'laminar_jet_flow_prediction.mp4')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--max_frames', type=int, default=None, help="Stop after N frames (useful for testing)")
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def flow_to_colormap(u: np.ndarray, v: np.ndarray, max_flow: float = 12.0) -> np.ndarray:
    """Map dense flow magnitude to a jet/RdYlBu_r-like colormap."""
    mag = np.sqrt(u**2 + v**2)
    # Normalize to [0, 1]
    norm_mag = np.clip(mag / max_flow, 0.0, 1.0)
    colormap = matplotlib.colormaps['RdYlBu_r']
    colored = colormap(norm_mag)[..., :3] # Keep RGB, drop alpha
    # Convert RGB to BGR for OpenCV
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
            # Draw arrow vector (thick=1, clean arrow tip)
            cv2.arrowedLine(out, (x, y), (x + dx, y + dy), color, 1, line_type=cv2.LINE_AA, tipLength=0.25)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Main Processing
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at {CKPT_PATH}")

print(f"Loading PIV-UNet from {CKPT_PATH}...")
model = PIVUNet(max_disp=4, base_ch=32).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print("Model loaded.")

# Open Video
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise IOError(f"Cannot open input video {INPUT_VIDEO}")

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video Properties: {orig_w}x{orig_h} @ {fps} FPS, Total Frames: {total_frames}")

# Setup video writer
# We write a side-by-side video: Left = Raw frame with black Quiver, Right = Colormap with black Quiver
out_w = orig_w * 2
out_h = orig_h
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (out_w, out_h))
print(f"Writing side-by-side output to: {OUTPUT_VIDEO}")

ret, prev_frame = cap.read()
if not ret:
    print("Failed to read first frame. Exiting.")
    sys.exit(1)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_idx = 0
try:
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Prepare inputs for PIV-UNet (resize to 256x256 and scale to [0, 1])
        img1_res = cv2.resize(prev_gray, (256, 256)).astype(np.float32) / 255.0
        img2_res = cv2.resize(curr_gray, (256, 256)).astype(np.float32) / 255.0
        
        input_tensor = torch.from_numpy(np.stack([img1_res, img2_res], axis=0)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_flow = model(input_tensor).squeeze(0).cpu().numpy() # (2, 256, 256)
            
        # Bilinear upscale flow to original resolution
        u_256, v_256 = pred_flow[0], pred_flow[1]
        u_orig = cv2.resize(u_256, (orig_w, orig_h))
        v_orig = cv2.resize(v_256, (orig_w, orig_h))
        
        # Adjust flow vectors for scale change
        u_orig = u_orig * (orig_w / 256.0)
        v_orig = v_orig * (orig_h / 256.0)
        
        # Visualizations (Green quivers for raw data, black quivers for colormap)
        quiver_overlay = draw_quiver(prev_frame, u_orig, v_orig, step=20, scale=3.5, color=(0, 255, 0))
        cmap_flow = flow_to_colormap(u_orig, v_orig, max_flow=8.0)
        cmap_quiver = draw_quiver(cmap_flow, u_orig, v_orig, step=20, scale=3.5, color=(0, 0, 0))
        
        # Combine side-by-side
        combined = np.hstack([quiver_overlay, cmap_quiver])
        
        # Write to video
        out_writer.write(combined)
        
        prev_gray = curr_gray
        prev_frame = curr_frame.copy()
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{total_frames - 1}")
            
        if args.max_frames and frame_idx >= args.max_frames:
            print(f"Reached --max_frames cap of {args.max_frames}. Stopping.")
            break
finally:
    cap.release()
    out_writer.release()
    print(f"\nProcessing complete! Output saved to: {OUTPUT_VIDEO}")
