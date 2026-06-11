"""
compare_piv_ptv.py — Comparison Script: Classical PIV vs Classical PTV vs Hybrid PIV-Guided PTV
================================================================================================
Loads a pair of synthetic particle images from the PIV_dataset (uniform flow)
along with the ground-truth .flo file, runs three algorithms, computes RMSE
against ground truth, and saves a multi-panel comparison figure.

Usage:
    python3 script/compare_piv_ptv.py

Outputs:
    result/piv_ptv_comparison.png
"""

import sys, os, time

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from piv_system import (
    read_flo,
    load_image_gray,
    classical_piv,
    detect_particles,
    classical_ptv,
    guided_ptv,
    rmse_piv,
    rmse_ptv,
    ptv_to_grid,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR  = os.path.join(WORKSPACE, 'data', 'PIV_dataset',
                         'PIV-genImages', 'data', 'uniform', 'uniform')
RESULT_DIR = os.path.join(WORKSPACE, 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

SAMPLE_IDX   = 1        # which sample pair to load (1-based)
WIN_SIZE     = 32       # PIV window size
OVERLAP      = 16       # PIV window overlap
MAX_DIST_PTV = 8.0      # Classical PTV: max search distance
SEARCH_RAD   = 4.0      # Hybrid PTV: guided search radius

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
prefix   = os.path.join(DATA_DIR, f'uniform_{SAMPLE_IDX:05d}')
img1_path = prefix + '_img1.tif'
img2_path = prefix + '_img2.tif'
flo_path  = prefix + '_flow.flo'

print(f"Loading sample {SAMPLE_IDX}…")
img1 = load_image_gray(img1_path)
img2 = load_image_gray(img2_path)
gt_u, gt_v = read_flo(flo_path)
H, W = img1.shape
print(f"  Image size: {W}x{H}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Classical PIV
# ─────────────────────────────────────────────────────────────────────────────
print("Running Classical PIV (FFT cross-correlation)…")
t0 = time.perf_counter()
xs_piv, ys_piv, u_piv, v_piv = classical_piv(img1, img2,
                                               window_size=WIN_SIZE,
                                               overlap=OVERLAP)
t_piv = time.perf_counter() - t0

# Interpolate PIV to full image grid for RMSE
from scipy.interpolate import RegularGridInterpolator
_interp_u = RegularGridInterpolator((ys_piv, xs_piv), u_piv,
                                     method='linear',
                                     bounds_error=False, fill_value=0.0)
_interp_v = RegularGridInterpolator((ys_piv, xs_piv), v_piv,
                                     method='linear',
                                     bounds_error=False, fill_value=0.0)
yy_full, xx_full = np.mgrid[0:H, 0:W]
pts_full = np.column_stack([yy_full.ravel(), xx_full.ravel()])
u_piv_full = _interp_u(pts_full).reshape(H, W)
v_piv_full = _interp_v(pts_full).reshape(H, W)

rmse_piv_val = rmse_piv(u_piv_full, v_piv_full, gt_u, gt_v)
print(f"  Classical PIV  RMSE={rmse_piv_val:.4f} px  time={t_piv:.3f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Classical PTV
# ─────────────────────────────────────────────────────────────────────────────
print("Detecting particles…")
t0 = time.perf_counter()
p1 = detect_particles(img1, threshold=0.025)
p2 = detect_particles(img2, threshold=0.025)
t_detect = time.perf_counter() - t0
print(f"  Detected {len(p1)} particles in frame 1, {len(p2)} in frame 2  ({t_detect:.3f}s)")

print("Running Classical PTV (Nearest Neighbour)…")
t0 = time.perf_counter()
src_ptv, dst_ptv = classical_ptv(p1, p2, max_distance=MAX_DIST_PTV)
t_ptv = time.perf_counter() - t0
rmse_ptv_val = rmse_ptv(src_ptv, dst_ptv, gt_u, gt_v, img1.shape)
print(f"  Classical PTV  matched={len(src_ptv)}  RMSE={rmse_ptv_val:.4f} px  time={t_ptv:.3f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid PIV-Guided PTV
# ─────────────────────────────────────────────────────────────────────────────
print("Running Hybrid PIV-Guided PTV…")
t0 = time.perf_counter()
src_hpt, dst_hpt = guided_ptv(p1, p2,
                               x_grid=xs_piv, y_grid=ys_piv,
                               piv_u=u_piv, piv_v=v_piv,
                               search_radius=SEARCH_RAD)
t_hpt = time.perf_counter() - t0
rmse_hpt_val = rmse_ptv(src_hpt, dst_hpt, gt_u, gt_v, img1.shape)
print(f"  Hybrid PIV-PTV  matched={len(src_hpt)}  RMSE={rmse_hpt_val:.4f} px  time={t_hpt:.3f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Interpolate PTV results onto grids for visualisation
# ─────────────────────────────────────────────────────────────────────────────
if len(src_ptv) > 3:
    xs_ptv, ys_ptv_g, u_ptv_grid, v_ptv_grid = ptv_to_grid(src_ptv, dst_ptv, img1.shape, grid_step=16)
else:
    xs_ptv, ys_ptv_g = xs_piv, ys_piv
    u_ptv_grid = np.zeros_like(u_piv)
    v_ptv_grid = np.zeros_like(v_piv)

if len(src_hpt) > 3:
    xs_hpt, ys_hpt_g, u_hpt_grid, v_hpt_grid = ptv_to_grid(src_hpt, dst_hpt, img1.shape, grid_step=16)
else:
    xs_hpt, ys_hpt_g = xs_piv, ys_piv
    u_hpt_grid = np.zeros_like(u_piv)
    v_hpt_grid = np.zeros_like(v_piv)

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — 5-panel comparison figure
# ─────────────────────────────────────────────────────────────────────────────
print("Generating comparison figure…")

plt.style.use('dark_background')
ACCENT  = '#00E5FF'
C_PIV   = '#FF6B6B'
C_PTV   = '#FFD166'
C_HPT   = '#06D6A0'
C_GT    = '#AAAAFF'

fig = plt.figure(figsize=(22, 14), facecolor='#0D1117')

# Title
fig.text(0.5, 0.97,
         'PIV  vs  PTV  vs  Hybrid PIV-Guided PTV — Comparison',
         ha='center', va='top', fontsize=18, fontweight='bold',
         color='white', family='monospace')

gs = GridSpec(2, 3, figure=fig,
              left=0.04, right=0.97, bottom=0.07, top=0.92,
              hspace=0.35, wspace=0.28)

# ── Row-0 left: raw particle image pair overlay ──────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
overlay = np.stack([img1, img2, img1 * 0.5 + img2 * 0.5], axis=-1)
overlay = np.clip(overlay, 0, 1)
ax0.imshow(overlay, origin='upper', aspect='equal')
ax0.set_title('Particle Image Pair\n(Frame 1=Red, Frame 2=Green)', color='white', fontsize=10)
ax0.tick_params(colors='white'); ax0.set_facecolor('#0D1117')
for s in ax0.spines.values(): s.set_color('#333')

# ── Row-0 middle: ground-truth velocity magnitude ────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
gt_mag = np.sqrt(gt_u ** 2 + gt_v ** 2)
im1 = ax1.imshow(gt_mag, cmap='plasma', origin='upper', aspect='equal')
step_gt = max(W // 20, 1)
xx_q = np.arange(step_gt // 2, W, step_gt)
yy_q = np.arange(step_gt // 2, H, step_gt)
xx_qg, yy_qg = np.meshgrid(xx_q, yy_q)
ax1.quiver(xx_qg, yy_qg,
           gt_u[yy_qg, xx_qg], gt_v[yy_qg, xx_qg],
           color=C_GT, scale=150, width=0.003, alpha=0.8)
plt.colorbar(im1, ax=ax1, label='px / frame').ax.yaxis.label.set_color('white')
ax1.set_title('Ground Truth Flow', color='white', fontsize=10)
ax1.tick_params(colors='white'); ax1.set_facecolor('#0D1117')
for s in ax1.spines.values(): s.set_color('#333')

# ── Row-0 right: detected particles ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.imshow(img1, cmap='gray', origin='upper', aspect='equal', vmin=0, vmax=1)
if len(p1):
    ax2.scatter(p1[:, 0], p1[:, 1], s=8, c=ACCENT, marker='o',
                linewidths=0, alpha=0.8, label=f'Frame 1 ({len(p1)})')
if len(p2):
    ax2.scatter(p2[:, 0], p2[:, 1], s=8, c='#FF9F1C', marker='o',
                linewidths=0, alpha=0.8, label=f'Frame 2 ({len(p2)})')
ax2.legend(fontsize=7, loc='lower right', framealpha=0.3)
ax2.set_title('Detected Particles (LoG)', color='white', fontsize=10)
ax2.tick_params(colors='white'); ax2.set_facecolor('#0D1117')
for s in ax2.spines.values(): s.set_color('#333')

# ── Row-1 left: Classical PIV ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
piv_mag = np.sqrt(u_piv ** 2 + v_piv ** 2)
xx_piv, yy_piv = np.meshgrid(xs_piv, ys_piv)
sc3 = ax3.scatter(xx_piv, yy_piv, c=piv_mag, cmap='plasma',
                  s=30, vmin=0, vmax=gt_mag.max(), zorder=3)
ax3.quiver(xx_piv, yy_piv, u_piv, v_piv,
           color=C_PIV, scale=150, width=0.004, alpha=0.9)
plt.colorbar(sc3, ax=ax3, label='px / frame').ax.yaxis.label.set_color('white')
ax3.set_xlim(0, W); ax3.set_ylim(H, 0)
ax3.set_title(f'Classical PIV (FFT)\nRMSE={rmse_piv_val:.3f} px  |  t={t_piv:.3f}s',
              color=C_PIV, fontsize=10, fontweight='bold')
ax3.tick_params(colors='white'); ax3.set_facecolor('#0D1117')
for s in ax3.spines.values(): s.set_color('#333')

# ── Row-1 middle: Classical PTV ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
if len(src_ptv):
    disp_ptv = np.linalg.norm(dst_ptv - src_ptv, axis=1)
    ax4.quiver(src_ptv[:, 0], src_ptv[:, 1],
               dst_ptv[:, 0] - src_ptv[:, 0],
               dst_ptv[:, 1] - src_ptv[:, 1],
               color=C_PTV, scale=150, width=0.003, alpha=0.85)
    ax4.scatter(src_ptv[:, 0], src_ptv[:, 1], s=4, c=C_PTV, alpha=0.5)
ax4.set_xlim(0, W); ax4.set_ylim(H, 0)
ax4.set_title(f'Classical PTV (NN)\nRMSE={rmse_ptv_val:.3f} px  |  matched={len(src_ptv)}  |  t={t_ptv:.3f}s',
              color=C_PTV, fontsize=10, fontweight='bold')
ax4.tick_params(colors='white'); ax4.set_facecolor('#0D1117')
for s in ax4.spines.values(): s.set_color('#333')

# ── Row-1 right: Hybrid PIV-Guided PTV ────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
if len(src_hpt):
    ax5.quiver(src_hpt[:, 0], src_hpt[:, 1],
               dst_hpt[:, 0] - src_hpt[:, 0],
               dst_hpt[:, 1] - src_hpt[:, 1],
               color=C_HPT, scale=150, width=0.003, alpha=0.85)
    ax5.scatter(src_hpt[:, 0], src_hpt[:, 1], s=4, c=C_HPT, alpha=0.5)
ax5.set_xlim(0, W); ax5.set_ylim(H, 0)
ax5.set_title(f'Hybrid PIV-Guided PTV\nRMSE={rmse_hpt_val:.3f} px  |  matched={len(src_hpt)}  |  t={t_hpt:.3f}s',
              color=C_HPT, fontsize=10, fontweight='bold')
ax5.tick_params(colors='white'); ax5.set_facecolor('#0D1117')
for s in ax5.spines.values(): s.set_color('#333')

# ── Footer stats bar ────────────────────────────────────────────────────────
stats_text = (
    f"Sample: uniform_{SAMPLE_IDX:05d}   |   "
    f"Win={WIN_SIZE}px  Overlap={OVERLAP}px   |   "
    f"Particles detected: {len(p1)}  |   "
    f"RMSE ▶  PIV: {rmse_piv_val:.4f}   Classical PTV: {rmse_ptv_val:.4f}   "
    f"Hybrid PTV: {rmse_hpt_val:.4f} px"
)
fig.text(0.5, 0.02, stats_text, ha='center', va='bottom', fontsize=9,
         color='#AAAAAA', family='monospace')

out_path = os.path.join(RESULT_DIR, 'piv_ptv_comparison.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.close(fig)
print(f"\nSaved comparison figure → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print(f"{'Method':<26} {'RMSE (px)':>10} {'Matches':>10} {'Time (s)':>10}")
print("─" * 60)
print(f"{'Classical PIV':<26} {rmse_piv_val:>10.4f} {'(dense grid)':>10} {t_piv:>10.3f}")
print(f"{'Classical PTV (NN)':<26} {rmse_ptv_val:>10.4f} {len(src_ptv):>10} {t_ptv:>10.3f}")
print(f"{'Hybrid PIV-Guided PTV':<26} {rmse_hpt_val:>10.4f} {len(src_hpt):>10} {t_hpt:>10.3f}")
print("═" * 60)
