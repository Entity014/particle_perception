"""
evaluate_nn_ptv.py — Evaluation & Comparison Script:
PIV-UNet vs Classical PIV, Classical PTV, PIV-Guided PTV, and PIV-UNet Guided PTV
================================================================================
Loads a trained PIV-UNet model, runs it alongside classical PIV/PTV algorithms
on validation samples from multiple flow subsets (uniform, cylinder, backstep, etc.),
calculates RMSE metrics and computational times, and generates a visual report.

Usage:
    python3 script/evaluate_nn_ptv.py [--subsets uniform cylinder backstep DNS_turbulence] [--samples 3]
"""

import sys
import os
import time
import argparse
import glob
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from piv_nn import PIVUNet
from piv_system import (
    read_flo,
    load_image_gray,
    classical_piv,
    detect_particles,
    classical_ptv,
    guided_ptv,
    nn_guided_ptv,
    rmse_piv,
    rmse_ptv,
    ptv_to_grid,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config & Directories
# ─────────────────────────────────────────────────────────────────────────────
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(WORKSPACE, 'data', 'PIV_dataset', 'PIV-genImages', 'data')
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
    # Try finding any checkpoint under train/runX/checkpoints/best_model.pt
    ckpt_candidates = sorted(glob.glob(os.path.join(train_base, 'run*', 'checkpoints', 'best_model.pt')))
    if ckpt_candidates:
        CKPT_PATH = ckpt_candidates[-1]
    else:
        # Fallback to old path if any
        CKPT_PATH = os.path.join(RESULT_DIR, 'checkpoints', 'best_model.pt')

# Setup target deploy directory
DEPLOY_DIR = get_next_deploy_dir(RESULT_DIR)
print(f"Deployment outputs will be saved to: {DEPLOY_DIR}")

# Default parameter configurations
WIN_SIZE     = 32       # PIV window size
OVERLAP      = 16       # PIV window overlap
MAX_DIST_PTV = 8.0      # Classical PTV: max search distance
SEARCH_RAD   = 4.0      # Guided PTV: search radius

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--subsets', nargs='+', default=['uniform', 'cylinder', 'backstep', 'DNS_turbulence'])
parser.add_argument('--samples', type=int, default=3, help="Number of samples to evaluate per subset")
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Evaluation Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Trained model checkpoint not found at {CKPT_PATH}. Please run training first.")

print(f"Loading PIV-UNet from {CKPT_PATH}...")
model = PIVUNet(max_disp=4, base_ch=32).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print("Model loaded successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Loop
# ─────────────────────────────────────────────────────────────────────────────
results = {}

for subset in args.subsets:
    subset_dir = os.path.join(DATA_ROOT, subset, subset)
    if not os.path.isdir(subset_dir):
        subset_dir = os.path.join(DATA_ROOT, subset)
    
    if not os.path.isdir(subset_dir):
        print(f"Warning: Subset directory {subset_dir} not found. Skipping.")
        continue

    flo_files = sorted(glob.glob(os.path.join(subset_dir, "*_flow.flo")))
    if not flo_files:
        print(f"Warning: No .flo files found in {subset_dir}. Skipping.")
        continue
    
    selected_flos = flo_files[:args.samples]
    print(f"\nEvaluating subset: {subset} ({len(selected_flos)} samples)...")
    
    subset_results = []
    
    for idx, flo_path in enumerate(selected_flos):
        base = flo_path.replace("_flow.flo", "")
        img1_path = base + "_img1.tif"
        img2_path = base + "_img2.tif"
        
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            continue
            
        print(f"  Sample {idx+1}/{len(selected_flos)}: {os.path.basename(base)}")
        
        # Load Data
        img1 = load_image_gray(img1_path)
        img2 = load_image_gray(img2_path)
        gt_u, gt_v = read_flo(flo_path)
        H, W = img1.shape
        
        # Particle Detection
        p1 = detect_particles(img1, threshold=0.025)
        p2 = detect_particles(img2, threshold=0.025)
        
        # --- CLASSICAL PIV (FFT) ---
        t0 = time.perf_counter()
        xs_piv, ys_piv, u_piv, v_piv = classical_piv(img1, img2, window_size=WIN_SIZE, overlap=OVERLAP)
        t_piv = time.perf_counter() - t0
        
        # Interpolate Classical PIV to full grid for RMSE
        _interp_u = RegularGridInterpolator((ys_piv, xs_piv), u_piv, method='linear', bounds_error=False, fill_value=0.0)
        _interp_v = RegularGridInterpolator((ys_piv, xs_piv), v_piv, method='linear', bounds_error=False, fill_value=0.0)
        yy_full, xx_full = np.mgrid[0:H, 0:W]
        pts_full = np.column_stack([yy_full.ravel(), xx_full.ravel()])
        u_piv_full = _interp_u(pts_full).reshape(H, W)
        v_piv_full = _interp_v(pts_full).reshape(H, W)
        rmse_piv_val = rmse_piv(u_piv_full, v_piv_full, gt_u, gt_v)
        
        # --- PIV-UNet (Deep Learning) ---
        t0 = time.perf_counter()
        input_tensor = torch.from_numpy(np.stack([img1, img2], axis=0)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_flow = model(input_tensor).squeeze(0).cpu().numpy() # (2, H, W)
        t_nn = time.perf_counter() - t0
        u_nn, v_nn = pred_flow[0], pred_flow[1]
        rmse_nn_val = rmse_piv(u_nn, v_nn, gt_u, gt_v)
        
        # --- CLASSICAL PTV (NN) ---
        t0 = time.perf_counter()
        src_ptv, dst_ptv = classical_ptv(p1, p2, max_distance=MAX_DIST_PTV)
        t_ptv = time.perf_counter() - t0
        rmse_ptv_val = rmse_ptv(src_ptv, dst_ptv, gt_u, gt_v, img1.shape)
        
        # --- CLASSICAL PIV GUIDED PTV ---
        t0 = time.perf_counter()
        src_guided, dst_guided = guided_ptv(p1, p2, x_grid=xs_piv, y_grid=ys_piv, piv_u=u_piv, piv_v=v_piv, search_radius=SEARCH_RAD)
        t_guided = time.perf_counter() - t0
        rmse_guided_val = rmse_ptv(src_guided, dst_guided, gt_u, gt_v, img1.shape)
        
        # --- PIV-UNet GUIDED PTV ---
        t0 = time.perf_counter()
        src_nn_guided, dst_nn_guided = nn_guided_ptv(p1, p2, piv_u=u_nn, piv_v=v_nn, search_radius=SEARCH_RAD)
        t_nn_guided = time.perf_counter() - t0
        rmse_nn_guided_val = rmse_ptv(src_nn_guided, dst_nn_guided, gt_u, gt_v, img1.shape)
        
        sample_res = {
            'name': os.path.basename(base),
            'img1': img1, 'img2': img2, 'gt_u': gt_u, 'gt_v': gt_v,
            'p1': p1, 'p2': p2,
            'piv_u': u_piv_full, 'piv_v': v_piv_full, # dense interpolated
            'nn_u': u_nn, 'nn_v': v_nn,
            'src_ptv': src_ptv, 'dst_ptv': dst_ptv,
            'src_guided': src_guided, 'dst_guided': dst_guided,
            'src_nn_guided': src_nn_guided, 'dst_nn_guided': dst_nn_guided,
            'metrics': {
                'piv': {'rmse': rmse_piv_val, 'time': t_piv},
                'nn': {'rmse': rmse_nn_val, 'time': t_nn},
                'ptv': {'rmse': rmse_ptv_val, 'matches': len(src_ptv), 'time': t_ptv},
                'guided': {'rmse': rmse_guided_val, 'matches': len(src_guided), 'time': t_guided},
                'nn_guided': {'rmse': rmse_nn_guided_val, 'matches': len(src_nn_guided), 'time': t_nn_guided}
            }
        }
        subset_results.append(sample_res)
        
    results[subset] = subset_results

# ─────────────────────────────────────────────────────────────────────────────
# Generate Metrics Report (Markdown)
# ─────────────────────────────────────────────────────────────────────────────
summary_path = os.path.join(DEPLOY_DIR, 'evaluation_summary.md')
print(f"\nWriting summary report to {summary_path}...")

with open(summary_path, 'w') as f:
    f.write("# PIV-UNet Evaluation & Comparison Report\n\n")
    f.write("This report presents the comparative metrics of Classical PIV, Classical PTV, PIV-Guided PTV, and **PIV-UNet / PIV-UNet Guided PTV**.\n\n")
    
    for subset, samples in results.items():
        if not samples:
            continue
        f.write(f"## Flow Subset: `{subset}`\n\n")
        f.write("| Method | PIV Flow RMSE (px) | PTV Tracking RMSE (px) | PTV Matches | Processing Time (s) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        
        # Calculate averages
        avg_piv_rmse = np.mean([s['metrics']['piv']['rmse'] for s in samples])
        avg_piv_time = np.mean([s['metrics']['piv']['time'] for s in samples])
        
        avg_nn_rmse = np.mean([s['metrics']['nn']['rmse'] for s in samples])
        avg_nn_time = np.mean([s['metrics']['nn']['time'] for s in samples])
        
        ptv_rmses = [s['metrics']['ptv']['rmse'] for s in samples if not np.isnan(s['metrics']['ptv']['rmse'])]
        avg_ptv_rmse = np.mean(ptv_rmses) if ptv_rmses else float('nan')
        avg_ptv_matches = np.mean([s['metrics']['ptv']['matches'] for s in samples])
        avg_ptv_time = np.mean([s['metrics']['ptv']['time'] for s in samples])
        
        guided_rmses = [s['metrics']['guided']['rmse'] for s in samples if not np.isnan(s['metrics']['guided']['rmse'])]
        avg_guided_rmse = np.mean(guided_rmses) if guided_rmses else float('nan')
        avg_guided_matches = np.mean([s['metrics']['guided']['matches'] for s in samples])
        avg_guided_time = np.mean([s['metrics']['guided']['time'] for s in samples])
        
        nnguided_rmses = [s['metrics']['nn_guided']['rmse'] for s in samples if not np.isnan(s['metrics']['nn_guided']['rmse'])]
        avg_nnguided_rmse = np.mean(nnguided_rmses) if nnguided_rmses else float('nan')
        avg_nnguided_matches = np.mean([s['metrics']['nn_guided']['matches'] for s in samples])
        avg_nnguided_time = np.mean([s['metrics']['nn_guided']['time'] for s in samples])
        
        f.write(f"| **Classical PIV (FFT)** | {avg_piv_rmse:.4f} | - | - | {avg_piv_time:.4f}s |\n")
        f.write(f"| **PIV-UNet (Ours)** | **{avg_nn_rmse:.4f}** | - | - | **{avg_nn_time:.4f}s** |\n")
        f.write(f"| **Classical PTV (NN)** | - | {avg_ptv_rmse:.4f} | {avg_ptv_matches:.1f} | {avg_ptv_time:.4f}s |\n")
        f.write(f"| **Classical-Guided PTV** | - | {avg_guided_rmse:.4f} | {avg_guided_matches:.1f} | {avg_guided_time:.4f}s |\n")
        f.write(f"| **PIV-UNet Guided PTV** | - | **{avg_nnguided_rmse:.4f}** | **{avg_nnguided_matches:.1f}** | **{avg_nnguided_time:.4f}s** |\n\n")

with open(summary_path, 'r') as f:
    print(f.read())

# ─────────────────────────────────────────────────────────────────────────────
# Generate Comparison Plot (Scientific Style - colormaps, black quiver, streamlines)
# ─────────────────────────────────────────────────────────────────────────────
viz_subset = next(iter(results.keys()))
viz_sample = results[viz_subset][0]

print(f"Generating enhanced scientific comparison plot for {viz_subset} / {viz_sample['name']}...")

# Switch to a light background theme
plt.style.use('default')
fig = plt.figure(figsize=(24, 15), facecolor='white')

# Title
fig.text(0.5, 0.96,
         f"PIV-UNet Scientific Flow Analysis — Subset: {viz_subset.upper()} ({viz_sample['name']})",
         ha='center', va='top', fontsize=22, fontweight='bold',
         color='#111111', family='sans-serif')

gs = GridSpec(2, 3, figure=fig, left=0.04, right=0.97, bottom=0.05, top=0.90, hspace=0.25, wspace=0.25)

# Dense arrow sampling function for matplotlib plotting
def add_quiver_layer(ax, u, v, step=12, scale=120, color='black'):
    H, W = u.shape
    y, x = np.mgrid[step//2:H:step, step//2:W:step]
    ax.quiver(x, y, u[y, x], v[y, x], color=color, scale=scale, width=0.003, headwidth=4.5, headaxislength=4)

# 1. Overlay Particle Image Pair
ax0 = fig.add_subplot(gs[0, 0])
overlay = np.stack([viz_sample['img1'], viz_sample['img2'], viz_sample['img1']*0.5 + viz_sample['img2']*0.5], axis=-1)
ax0.imshow(np.clip(overlay, 0, 1), origin='upper')
ax0.set_title("Particle Image Pair Overlay\n(Frame 1=Red, Frame 2=Green)", color='#111111', fontsize=13, fontweight='bold')
ax0.tick_params(colors='#333333'); ax0.set_facecolor('white')

# 2. Ground Truth Flow (Colormap + Black Dense Quiver)
ax1 = fig.add_subplot(gs[0, 1])
gt_mag = np.sqrt(viz_sample['gt_u']**2 + viz_sample['gt_v']**2)
im1 = ax1.imshow(gt_mag, cmap='RdYlBu_r', origin='upper') # Jet/RdYlBu_r scientific colormap
add_quiver_layer(ax1, viz_sample['gt_u'], viz_sample['gt_v'], step=14, color='black')
plt.colorbar(im1, ax=ax1, label='Displacement Magnitude (px/frame)')
ax1.set_title("Ground Truth Flow Field", color='#111111', fontsize=13, fontweight='bold')
ax1.tick_params(colors='#333333')

# 3. Classical PIV (FFT) (Colormap + Black Dense Quiver)
ax2 = fig.add_subplot(gs[0, 2])
piv_mag = np.sqrt(viz_sample['piv_u']**2 + viz_sample['piv_v']**2)
im2 = ax2.imshow(piv_mag, cmap='RdYlBu_r', origin='upper')
add_quiver_layer(ax2, viz_sample['piv_u'], viz_sample['piv_v'], step=14, color='black')
plt.colorbar(im2, ax=ax2, label='Displacement Magnitude (px/frame)')
ax2.set_title(f"Classical PIV (FFT)\nRMSE: {viz_sample['metrics']['piv']['rmse']:.4f} px | t: {viz_sample['metrics']['piv']['time']:.3f}s", color='#b30000', fontsize=13, fontweight='bold')
ax2.tick_params(colors='#333333')

# 4. PIV-UNet (Colormap + Black Dense Quiver)
ax3 = fig.add_subplot(gs[1, 0])
nn_mag = np.sqrt(viz_sample['nn_u']**2 + viz_sample['nn_v']**2)
im3 = ax3.imshow(nn_mag, cmap='RdYlBu_r', origin='upper')
add_quiver_layer(ax3, viz_sample['nn_u'], viz_sample['nn_v'], step=14, color='black')
plt.colorbar(im3, ax=ax3, label='Displacement Magnitude (px/frame)')
ax3.set_title(f"PIV-UNet (Ours)\nRMSE: {viz_sample['metrics']['nn']['rmse']:.4f} px | t: {viz_sample['metrics']['nn']['time']:.3f}s", color='#0066cc', fontsize=13, fontweight='bold')
ax3.tick_params(colors='#333333')

# 5. PIV-UNet Streamline Visualization (Streamplot on colormap)
ax4 = fig.add_subplot(gs[1, 1])
im4 = ax4.imshow(nn_mag, cmap='RdYlBu_r', origin='upper', alpha=0.95)
# Generate regular grid vectors for streamplot
xs = np.arange(W)
ys = np.arange(H)
# We flip or match coords: streamplot needs X, Y, U, V
# Since image origin is 'upper', streamplot handles coords differently. We grid them cleanly.
ax4.streamplot(xs, ys, viz_sample['nn_u'], viz_sample['nn_v'], color='#111111', linewidth=1.1, density=1.4, arrowstyle='->', arrowsize=1.2)
plt.colorbar(im4, ax=ax4, label='Velocity Magnitude (px/frame)')
ax4.set_xlim(0, W); ax4.set_ylim(H, 0) # Maintain image orientation
ax4.set_title("PIV-UNet Flow Streamlines", color='#111111', fontsize=13, fontweight='bold')
ax4.tick_params(colors='#333333')

# 6. PIV-UNet Guided PTV Tracking Results
ax5 = fig.add_subplot(gs[1, 2])
src_nng, dst_nng = viz_sample['src_nn_guided'], viz_sample['dst_nn_guided']
ax5.imshow(viz_sample['img1'], cmap='gray', origin='upper')
if len(src_nng):
    # Vector displacement lines connecting matched frames
    ax5.quiver(src_nng[:, 0], src_nng[:, 1], dst_nng[:, 0] - src_nng[:, 0], dst_nng[:, 1] - src_nng[:, 1], color='#e60073', scale=80, width=0.003, alpha=0.9)
    ax5.scatter(src_nng[:, 0], src_nng[:, 1], s=5, c='#00e676', alpha=0.8, edgecolors='none')
ax5.set_xlim(0, W); ax5.set_ylim(H, 0)
ax5.set_title(f"PIV-UNet Guided PTV Tracking\nRMSE: {viz_sample['metrics']['nn_guided']['rmse']:.4f} px | Matches: {len(src_nng)}", color='#e60073', fontsize=13, fontweight='bold')
ax5.tick_params(colors='#333333')

plot_path = os.path.join(DEPLOY_DIR, 'nn_ptv_comparison.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved enhanced scientific visualization to {plot_path}")
