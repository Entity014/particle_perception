# Particle Perception: PIV-UNet & Guided PTV Framework

This repository contains a deep-learning-based framework for **Particle Image Velocimetry (PIV)** and **Particle Tracking Velocimetry (PTV)**. By leveraging a custom U-Net architecture with a Siamese encoder and a correlation layer (inspired by FlowNet), this project achieves high-accuracy, real-time dense flow prediction and guides PTV algorithms to track individual particles across complex fluid flows.

---

## 🌟 Key Features
1. **PIV-UNet Architecture**: 
   - A shared Siamese encoder extracts spatial features from consecutive frames.
   - A local cross-correlation layer computes dot-product displacement volumes at the bottleneck.
   - A decoder progressively reconstructs dense, pixel-level optical flow fields $(u, v)$ with skip connections.
2. **PIV-UNet Guided PTV**:
   - Uses the highly accurate dense flow predicted by PIV-UNet to project particle positions into the subsequent frame.
   - Reduces the nearest-neighbor search radius to a small window, significantly reducing tracking mismatches under high seeding densities.
3. **Advanced Scientific Visualization**:
   - Renders flow fields utilizing scientific color palettes (e.g., `'RdYlBu_r'`) with overlaid dense black quiver arrows and streamlines (`matplotlib.streamplot`).
   - Processes raw video streams to export side-by-side comparative visualizations.
4. **Structured Experiment Directories**:
   - Training outputs are cleanly separated into `result/train/run{N}/`.
   - Deployment, evaluation summary metrics, and output videos are automatically routed to incremented folders in `result/deploy/run{N}/`.

---

## 📁 Project Directory Structure

```directory
particle_perception/
├── src/
│   ├── piv_nn.py         # PyTorch PIVDataset loader and PIVUNet network
│   └── piv_system.py     # Classical PIV (FFT), PTV, and PIV-UNet guided PTV logic
├── script/
│   ├── download_dataset.py # Script to fetch synthetic PIV datasets
│   ├── train_piv_nn.py     # Training workflow for the PIV-UNet model
│   ├── evaluate_nn_ptv.py  # Benchmark suite comparing PIV, PIV-UNet, and PTV methods
│   └── process_video_piv.py# Script to infer and export flow animations on raw videos
├── result/
│   ├── train/
│   │   └── run1/         # Checkpoints, training logs, loss curves
│   └── deploy/
│       ├── run1/         # Initial baseline evaluation results
│       ├── run2/         # Scientific comparison plots & summary metrics
│       └── run4/         # Processed video showing green quivers & RdYlBu_r colormaps
└── requirements.txt      # Core Python library dependencies
```

---

## 🚀 Setup & Installation

Ensure you have Python 3.8+ and a CUDA-compatible GPU (recommended for fast training/inference).

1. Clone or navigate into the workspace:
   ```bash
   cd particle_perception
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage Guide

### 1. Download the Dataset
If you do not have the PIV synthetic image dataset, run the following download helper:
```bash
python3 script/download_dataset.py
```

### 2. Train the PIV-UNet Model
To train the neural network model on all flow subsets (e.g., cylinder, backstep, uniform, turbulence):
```bash
python3 script/train_piv_nn.py --epochs 30 --batch 8 --lr 1e-3
```
*Outputs will be saved under `result/train/run{N}/`.*

### 3. Evaluate & Compare Algorithms
To run a comprehensive comparison between Classical PIV, PIV-UNet, Classical PTV, PIV-Guided PTV, and PIV-UNet Guided PTV:
```bash
python3 script/evaluate_nn_ptv.py --subsets uniform cylinder backstep DNS_turbulence --samples 3
```
*Outputs (e.g., `evaluation_summary.md` and comparison plot `nn_ptv_comparison.png`) will be saved under `result/deploy/run{N}/`.*

### 4. Run Inference on a Video
To run PIV-UNet on an experimental video stream (generating side-by-side quiver/colormap frames):
```bash
python3 script/process_video_piv.py
```
*Outputs (e.g., `laminar_jet_flow_prediction.mp4`) will be saved under `result/deploy/run{N}/`.*

---

## 📊 Evaluation Highlights

### PIV Flow Estimation Accuracy (RMSE in pixels, lower is better)
| Flow Subset | Classical PIV (FFT) | PIV-UNet (Ours) | Error Reduction |
| :--- | :---: | :---: | :---: |
| `uniform` | 1.3653 px | **0.1773 px** | **~87%** |
| `cylinder` | 8.7665 px | **0.0609 px** | **~99%** |
| `backstep` | 1.2617 px | **0.1513 px** | **~88%** |
| `DNS_turbulence` | 0.7314 px | **0.1343 px** | **~81%** |

### PTV Particle Tracking Accuracy (RMSE in pixels, lower is better)
| Flow Subset | Classical PTV | Classical-Guided PTV | PIV-UNet Guided PTV (Ours) |
| :--- | :---: | :---: | :---: |
| `uniform` | 5.5831 px | 2.2408 px | **1.4461 px** |
| `cylinder` | 3.9621 px | 1.4393 px | **1.3534 px** |
| `backstep` | 4.8247 px | 1.7913 px | **1.2480 px** |
| `DNS_turbulence` | 3.9025 px | 1.4077 px | **1.3013 px** |
