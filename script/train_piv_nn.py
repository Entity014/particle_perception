"""
train_piv_nn.py — Training Script for PIV-UNet
================================================
Trains the PIVUNet model on the full PIV_dataset (all subsets).

Usage:
    python3 script/train_piv_nn.py [--epochs N] [--batch B] [--lr LR]

Checkpoints → result/checkpoints/
Training log (CSV) → result/training_log.csv
Loss curve plot    → result/training_curve.png
"""

import sys, os, time, argparse, csv
from pathlib import Path

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from piv_nn import PIVUNet, PIVDataset, EPELoss, MultiscaleEPELoss


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
WORKSPACE   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT   = os.path.join(WORKSPACE, 'data', 'PIV_dataset',
                           'PIV-genImages', 'data')
RESULT_DIR  = os.path.join(WORKSPACE, 'result')
CKPT_DIR    = os.path.join(RESULT_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',     type=int,   default=30)
parser.add_argument('--batch',      type=int,   default=8)
parser.add_argument('--lr',         type=float, default=1e-3)
parser.add_argument('--img_size',   type=int,   default=256)
parser.add_argument('--val_split',  type=float, default=0.1)
parser.add_argument('--max_samples', type=int,  default=None,
                    help='Cap dataset size (e.g. 500 for quick smoke test)')
parser.add_argument('--resume',     type=str,   default=None,
                    help='Path to checkpoint to resume from')
parser.add_argument('--workers',    type=int,   default=4)
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset & DataLoader
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nBuilding dataset from {DATA_ROOT} …")
dataset = PIVDataset(DATA_ROOT, img_size=args.img_size,
                     max_samples=args.max_samples)
print(f"  Total samples: {len(dataset)}")

val_n  = max(1, int(len(dataset) * args.val_split))
trn_n  = len(dataset) - val_n
trn_ds, val_ds = random_split(
    dataset, [trn_n, val_n],
    generator=torch.Generator().manual_seed(42)
)
print(f"  Train: {trn_n}   Val: {val_n}")

trn_loader = DataLoader(trn_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)

# ─────────────────────────────────────────────────────────────────────────────
# Model, Loss, Optimiser, Scheduler
# ─────────────────────────────────────────────────────────────────────────────
model = PIVUNet(max_disp=4, base_ch=32).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nPIV-UNet  trainable params: {n_params:,}")

criterion = EPELoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.lr,
    steps_per_epoch=len(trn_loader), epochs=args.epochs,
    pct_start=0.1
)

start_epoch = 0
best_val_epe = float('inf')

if args.resume and os.path.exists(args.resume):
    print(f"\nResuming from {args.resume} …")
    ckpt = torch.load(args.resume, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch'] + 1
    best_val_epe = ckpt.get('best_val_epe', float('inf'))
    print(f"  Resumed at epoch {start_epoch}, best_val_epe={best_val_epe:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
log_path = os.path.join(RESULT_DIR, 'training_log.csv')
log_exists = os.path.exists(log_path)
log_file = open(log_path, 'a', newline='')
log_writer = csv.writer(log_file)
if not log_exists:
    log_writer.writerow(['epoch', 'trn_epe', 'val_epe', 'lr', 'epoch_time_s'])

history_trn, history_val = [], []

# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"{'Epoch':>6}  {'Train EPE':>10}  {'Val EPE':>10}  {'LR':>10}  {'Time':>8}")
print(f"{'─'*65}")

for epoch in range(start_epoch, args.epochs):
    t0 = time.perf_counter()

    # ── Train ─────────────────────────────────────────────────────────────
    model.train()
    trn_epe_sum, trn_n_batch = 0.0, 0
    for batch in trn_loader:
        imgs = batch['img_pair'].to(DEVICE, non_blocking=True)
        gt   = batch['flow'].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(imgs)
        loss = criterion(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        trn_epe_sum  += loss.item()
        trn_n_batch  += 1

    trn_epe = trn_epe_sum / trn_n_batch

    # ── Validate ──────────────────────────────────────────────────────────
    model.eval()
    val_epe_sum, val_n_batch = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['img_pair'].to(DEVICE, non_blocking=True)
            gt   = batch['flow'].to(DEVICE, non_blocking=True)
            pred = model(imgs)
            val_epe_sum  += criterion(pred, gt).item()
            val_n_batch  += 1
    val_epe = val_epe_sum / val_n_batch

    dt = time.perf_counter() - t0
    lr_now = scheduler.get_last_lr()[0]

    history_trn.append(trn_epe)
    history_val.append(val_epe)
    log_writer.writerow([epoch + 1, f'{trn_epe:.5f}', f'{val_epe:.5f}',
                         f'{lr_now:.6f}', f'{dt:.1f}'])
    log_file.flush()

    print(f"{epoch+1:>6}  {trn_epe:>10.4f}  {val_epe:>10.4f}  "
          f"{lr_now:>10.2e}  {dt:>6.1f}s")

    # Save best checkpoint
    if val_epe < best_val_epe:
        best_val_epe = val_epe
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_epe': best_val_epe,
            'args': vars(args),
        }, os.path.join(CKPT_DIR, 'best_model.pt'))
        print(f"        ✓ Saved best model  (val_epe={best_val_epe:.4f})")

    # Periodic checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_epe': best_val_epe,
            'args': vars(args),
        }, os.path.join(CKPT_DIR, f'epoch_{epoch+1:04d}.pt'))

print(f"{'═'*65}")
log_file.close()

# ─────────────────────────────────────────────────────────────────────────────
# Plot training curve
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0D1117')
ax.set_facecolor('#0D1117')
epochs_range = range(start_epoch + 1, start_epoch + 1 + len(history_trn))
ax.plot(epochs_range, history_trn, color='#FF6B6B', lw=2, label='Train EPE')
ax.plot(epochs_range, history_val, color='#06D6A0', lw=2, label='Val EPE')
ax.axvline(np.argmin(history_val) + start_epoch + 1, color='#FFD166',
           ls='--', lw=1, label=f'Best Val EPE={min(history_val):.4f} px')
ax.set_xlabel('Epoch', color='white')
ax.set_ylabel('EPE (px)', color='white')
ax.set_title('PIV-UNet Training — End-Point Error', color='white', fontsize=13)
ax.legend(framealpha=0.3)
ax.tick_params(colors='white')
for s in ax.spines.values():
    s.set_color('#333')
curve_path = os.path.join(RESULT_DIR, 'training_curve.png')
fig.savefig(curve_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.close(fig)

print(f"\nTraining complete!")
print(f"  Best model  → {os.path.join(CKPT_DIR, 'best_model.pt')}")
print(f"  Best Val EPE → {best_val_epe:.4f} px")
print(f"  Training log → {log_path}")
print(f"  Loss curve   → {curve_path}")
