# Spatio-Temporal Traffic Flow Prediction using DCRNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Traffic Flow Forecasting using Diffusion Convolutional Recurrent Neural Networks**

This repository implements a state-of-the-art deep learning model for multi-horizon traffic speed prediction on road networks. The model combines graph convolution (spatial) and recurrent neural networks (temporal) to capture complex spatio-temporal dependencies in traffic data.

---

## Project Overview

This project implements a production-ready deep learning system for predicting traffic speeds across a network of 325 sensors in the San Francisco Bay Area. The system combines graph neural networks (spatial modeling) with recurrent neural networks (temporal modeling) to accurately forecast traffic conditions up to 60 minutes ahead.

**Model:** Diffusion Convolutional Recurrent Neural Network (DCRNN)  
**Dataset:** PEMS-BAY (325 sensors, 6 months, 52,116 timesteps at 5-minute intervals)  
**Paper:** Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting" (ICLR 2018)

### Key Achievements

- ✅ **Performance:** 1.930 mph test MAE on 325-sensor network
- ✅ **Interactive Dashboard:** Real-time traffic visualization and prediction interface
- ✅ **Full-Scale Implementation:** 6 months of data, 52,116 timesteps, zero missing values
- ✅ **Custom Architecture:** Hand-coded diffusion convolution (not using PyTorch Geometric)
- ✅ **Systematic Optimization:** Trained and evaluated 5K, 10K, and 20K sample models
- ✅ **Perfect Generalization:** Val MAE 2.097 mph ≈ Test MAE 1.930 mph (no overfitting)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/vaish725/Spatio-Temporal-Traffic-Flow-Prediction.git
cd Spatio-Temporal-Traffic-Flow-Prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Dashboard

```bash
# Start the dashboard
streamlit run app.py

# Open browser to http://localhost:8501
```

The dashboard provides:
- **Network Overview:** Interactive map of all 325 sensors with real-time predictions
- **Sensor Details:** Individual sensor analysis with 1-hour ahead forecasts
- **Comparison View:** Side-by-side comparison of multiple sensors

### 3. Data Preparation

The preprocessed data is already included in the repository. If you need to regenerate it:

```bash
python scripts/prepare_data.py
```

### 4. Model Training

```bash
# Train with default settings (Colab-optimized)
python scripts/train_colab_safe.py

# Or use the flexible training script
python scripts/train.py --data_path data/pems_bay_processed.npz --epochs 100
```

---

## Model Performance

### Production Results (10K Optimal Model)

| Metric | Validation | Test | Improvement vs Baseline |
|--------|-----------|------|------------------------|
| **MAE (mph)** | 2.097 | 1.930 | **75.9%** |
| **RMSE (mph)** | 4.258 | 3.946 | **71.2%** |
| **MAPE (%)** | 4.76 | 4.48 | **73.8%** |

**Model Details:**
- Architecture: DCRNN with 2-layer encoder/decoder
- Parameters: 446,593
- Training: 10,000 samples, 100 epochs
- Checkpoint: `checkpoints_colab/best_model_optimized.pt`

### Performance by Prediction Horizon

| Horizon | MAE (mph) | RMSE (mph) | MAPE (%) |
|---------|-----------|------------|----------|
| 15 min (step 3) | 1.64 | 3.23 | 3.78 |
| 30 min (step 6) | 1.93 | 3.89 | 4.48 |
| 45 min (step 9) | 2.08 | 4.26 | 4.88 |
| 60 min (step 12) | 2.18 | 4.53 | 5.18 |

**Key Insight:** The model maintains accuracy even for long-term predictions (60 minutes ahead).

---

## 🏗️ Repository Structure

```
├── app.py                      # Interactive Streamlit dashboard (18KB)
├── run_dashboard.sh            # Dashboard launcher script
│
├── dashboard/                  # Dashboard components (3 files)
│   ├── predictor.py           # Model loading and inference
│   ├── utils.py               # Utility functions
│   └── visualization.py       # Plotly visualizations
│
├── models/                     # Neural network architectures (2 files)
│   ├── dcrnn.py               # DCRNN encoder-decoder implementation
│   └── diffusion_conv.py      # Custom diffusion convolution layer
│
├── src/                        # Core utilities (2 files)
│   ├── dataset.py             # PyTorch Dataset classes
│   └── metrics.py             # Evaluation metrics (MAE, RMSE, MAPE)
│
├── scripts/                    # Training and utility scripts (11 files)
│   ├── train_colab_safe.py    # Main training script (Colab-optimized)
│   ├── train.py               # Flexible training with CLI arguments
│   ├── train_simple.py        # Debug/verification training
│   ├── evaluate.py            # Model evaluation script
│   ├── prepare_data.py        # Data preprocessing pipeline
│   ├── test_dashboard.py      # Dashboard component testing
│   ├── download_pems_bay.py   # Dataset acquisition script
│   ├── verify_teacher_forcing.py  # Teacher forcing verification
│   ├── traffic_flow_setup.py  # Setup utility
│   ├── generate_presentation_plots.py  # Visualization generation
│   └── run_experiments.py     # Experiment automation
│
├── notebooks/                  # Jupyter notebooks (3 files)
│   ├── DCRNN_Training_Colab_Online.ipynb  # Colab training notebook
│   ├── Train_DCRNN_Simple.ipynb           # Simplified training notebook
│   └── 01_data_exploration.ipynb          # Exploratory data analysis
│
├── data/                       # Datasets (579MB)
│   ├── PEMS-BAY.csv           # Raw traffic data (52,116 timesteps × 326 columns)
│   ├── pems_bay_processed.npz # Preprocessed train/val/test splits
│   └── adj_mx_bay.pkl         # Spatial adjacency matrix (325×325)
│
├── checkpoints_colab/          # Trained models (16MB)
│   ├── best_model_optimized.pt         # 🏆 10K optimal model (RECOMMENDED)
│   ├── best_model_20k.pt               # 20K model (comparison)
│   ├── training_history_optimized.json # Training metrics for 10K model
│   └── training_history_20k.json       # Training metrics for 20K model
│
├── results/                    # Evaluation results (44MB)
│   ├── colab_evaluation/      # Latest evaluation outputs
│   ├── predictions.npy        # Model predictions (10,419 × 325 × 12)
│   ├── targets.npy            # Ground truth values (10,419 × 325 × 12)
│   ├── metrics.json           # Performance metrics (MAE, RMSE, MAPE)
│   └── *.png                  # Visualization plots
│
├── presentation_figures/       # Presentation visualizations
│   ├── 1_training_curves.png
│   ├── 2_performance_comparison.png
│   ├── 3_horizon_analysis.png
│   ├── 4_spatial_predictions.png
│   ├── 5_results_table.png
│   └── 6_model_comparison.png
│
├── docs/                       # Documentation and guides
│   ├── DASHBOARD_WALKTHROUGH.md       # Complete dashboard guide
│   ├── DASHBOARD_QUICK_REFERENCE.md   # One-page cheat sheet
│   ├── PRESENTATION_GUIDE.md          # Presentation structure
│   ├── QUICK_REFERENCE.md             # Key metrics reference
│   ├── PROJECT_SUMMARY.md             # Overall project summary
│   ├── RESULTS_ANALYSIS_AND_IMPROVEMENT.md  # Performance analysis
│   └── *.md                           # Various technical documents
│
├── tests/                      # Unit tests
│   └── smoke_test_dcrnn.py    # Model smoke test
│
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```
---

## 🎨 Interactive Dashboard Features

### Network Overview
- **Interactive Map:** Visualize all 325 sensors with predictions
- **Network Statistics:** Average speed, active sensors, prediction accuracy
- **Time Series:** Historical and predicted traffic patterns across the network

### Sensor Details
- **Individual Analysis:** Select any sensor for detailed inspection
- **Multi-Horizon Predictions:** View forecasts from 5 to 60 minutes ahead
- **Historical Context:** Compare predictions against actual traffic patterns
- **Performance Metrics:** MAE, RMSE, MAPE for each sensor

### Comparison View
- **Multi-Sensor Analysis:** Compare up to 4 sensors side-by-side
- **Correlation Detection:** Identify spatial relationships between sensors
- **Peak Hour Analysis:** Examine traffic patterns during rush hours
- **Custom Time Ranges:** Filter data by date and time

---

## 📈 Dataset: PEMS-BAY

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Number of Sensors | 325 |
| Temporal Coverage | 6 months (Jan-June 2017) |
| Total Timesteps | 52,116 |
| Sampling Rate | 5 minutes |
| Mean Speed | 63.26 mph |
| Std Deviation | 8.58 mph |
| Missing Values | 0 |

### Data Splits

| Split | Timesteps | Samples | Date Range |
|-------|-----------|---------|------------|
| **Train** | 36,481 | 36,469 | Jan 1 - May 13, 2017 |
| **Validation** | 5,209 | 5,197 | May 13 - May 31, 2017 |
| **Test** | 10,426 | 10,419 | June 1 - June 30, 2017 |

**Note:** Samples = Timesteps - 12 (input sequence length)

### Key Findings from EDA

- **Strong daily periodicity:** Morning rush (8:00 AM = 56.90 mph), Evening rush (5:00 PM = 54.15 mph)
- **Weekend effects:** ~6% higher average speeds on Sat/Sun compared to weekdays
- **Spatial correlation:** Mean correlation 0.359, max 0.970 (neighbors), validating graph-based approach
- **Sparse adjacency:** 97.45% sparsity, ~8.3 connections per sensor, efficient computation
- **Node degree:** Mean 8, Median 8, Max 23 (hub sensors), Min 1 (isolated sensors)

---

## Model Architecture

### DCRNN (Diffusion Convolutional Recurrent Neural Network)

```
Input Sequence (12 steps, 1 hour)
         ↓
    [ENCODER]
    DCGRU Cell 1 → DCGRU Cell 2 → Hidden State
    (Spatial diffusion + GRU temporal modeling)
         ↓
    [DECODER]
    DCGRU Cell 1 → DCGRU Cell 2 → Output
         ↓
Output Predictions (12 steps, 1 hour)
```

### Key Components

1. **Diffusion Convolution:** Captures spatial dependencies through graph structure
   - Bidirectional random walks on road network graph
   - Learns both upstream and downstream traffic influences
   - Custom implementation (not using PyTorch Geometric)

2. **GRU Cells:** Model temporal dynamics in time series
   - Update gate: Controls information flow
   - Reset gate: Determines how to combine new input with memory
   - Modified with diffusion convolution for spatial awareness

3. **Encoder-Decoder:** Sequence-to-sequence architecture for multi-step prediction
   - Encoder: Processes historical traffic data (12 timesteps)
   - Decoder: Generates future predictions (12 timesteps)
   - Teacher forcing during training for stable learning

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Sequence Length | 12 steps (1 hour) |
| Output Sequence Length | 12 steps (1 hour) |
| Encoder/Decoder Layers | 2 |
| Hidden Dimension | 64 |
| Diffusion Steps (K) | 2 |
| Total Parameters | 446,593 |
| Learning Rate | 0.01 |
| Batch Size | 64 |
| Optimizer | Adam |
| L2 Regularization | 1e-3 |

---

## Training & Optimization

### Training Strategy

1. **Data Sampling:** Experimented with 5K, 10K, and 20K training samples
2. **Early Stopping:** Monitor validation loss with patience=10
3. **Learning Rate Scheduling:** Reduce on plateau
4. **Teacher Forcing:** Gradually reduce ratio during training
5. **Gradient Clipping:** Prevent exploding gradients

### Optimization Results

| Model | Train Samples | Val MAE | Test MAE | Training Time |
|-------|--------------|---------|----------|--------------|
| 5K Model | 5,000 | 2.842 | 2.654 | ~20 min |
| **10K Model** | **10,000** | **2.097** | **1.930** | **~35 min** |
| 20K Model | 20,000 | 2.156 | 2.018 | ~65 min |

**Optimal Choice:** The 10K model provides the best balance of accuracy and training efficiency.

### Training Curves

The optimal 10K model shows:
- Smooth convergence without overfitting
- Val loss tracks train loss closely
---

## Usage Examples

### Load Model and Make Predictions

```python
import torch
import numpy as np
from dashboard.predictor import TrafficPredictor

# Initialize predictor
predictor = TrafficPredictor(
    model_path='checkpoints_colab/best_model_optimized.pt',
    data_path='data/pems_bay_processed.npz',
    adj_mx_path='data/adj_mx_bay.pkl'
)

# Load test data
data = np.load('data/pems_bay_processed.npz')
X_test = data['X_test']  # Shape: (10419, 12, 325, 1)

# Make predictions
predictions = predictor.predict(X_test[:100])  # Predict on first 100 samples
print(f"Predictions shape: {predictions.shape}")  # (100, 12, 325, 1)
```

### Evaluate Model Performance

```python
from src.metrics import masked_mae, masked_rmse, masked_mape

# Calculate metrics
mae = masked_mae(predictions, targets)
rmse = masked_rmse(predictions, targets)
mape = masked_mape(predictions, targets)

print(f"MAE: {mae:.3f} mph")
print(f"RMSE: {rmse:.3f} mph")
print(f"MAPE: {mape:.2f}%")
```

### Train Custom Model

```python
from models.dcrnn import DCRNNModel
from src.dataset import TrafficDataset
import torch.optim as optim

# Load data
dataset = TrafficDataset('data/pems_bay_processed.npz', 'train')
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model
model = DCRNNModel(
    num_nodes=325,
    input_dim=1,
    hidden_dim=64,
    output_dim=1,
    num_layers=2,
    adj_mx=adj_mx
)

# Training loop
optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = torch.nn.functional.l1_loss(predictions, batch_y)
        loss.backward()
        optimizer.step()
```

---

## 📦 Requirements

```
Python >= 3.8
numpy >= 1.22.0
pandas >= 1.5.0
torch >= 2.0.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
streamlit >= 1.28.0
plotly >= 5.17.0
scipy >= 1.10.0
tqdm >= 4.65.0
jupyter >= 1.0.0
```

See `requirements.txt` for the complete dependency list.

---

## 📚 Technical Details

### Graph Structure

- **Adjacency Matrix:** 325×325 sparse matrix (97.45% sparsity)
- **Distance Threshold:** Sensors connected if within certain distance
- **Normalization:** Row-normalized for stable diffusion
- **Bidirectional:** Captures both upstream and downstream traffic flow

### Loss Function

Mean Absolute Error (MAE) with masking for invalid values:

```python
def masked_mae(predictions, targets, null_val=0.0):
    mask = (targets != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    loss = torch.abs(predictions - targets)
    loss = loss * mask
    return torch.mean(loss)
```

### Prediction Strategy

- **Input:** 12 timesteps (1 hour) of historical traffic speeds
- **Output:** 12 timesteps (1 hour) of future predictions
- **Autoregressive:** During inference, use predicted values for multi-step ahead forecasting
- **Scheduled Sampling:** Gradually reduce teacher forcing during training

---

## Results Analysis

### Strengths

✅ **Excellent short-term predictions:** 15-min MAE of 1.64 mph  
✅ **Stable long-term forecasts:** 60-min MAE of 2.18 mph (only +0.54 degradation)  
✅ **No overfitting:** Test MAE actually better than validation MAE  
✅ **Spatial awareness:** Successfully captures network-wide traffic patterns  
✅ **Temporal modeling:** Learns daily and weekly periodicity

### Limitations

⚠️ **Peak hour challenges:** Slightly higher errors during rush hours  
⚠️ **Rare events:** May struggle with accidents or unusual congestion  
⚠️ **Computational cost:** Full 325-sensor prediction requires ~2 seconds per timestep

### Future Improvements

- Incorporate external factors (weather, events, holidays)
- Attention mechanisms for adaptive spatial weighting
- Multi-task learning (speed + flow + occupancy)
- Real-time model updates with online learning

---

## 🔗 References

1. **Original Paper:**  
   Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. *International Conference on Learning Representations (ICLR)*.  
   [arXiv:1707.01926](https://arxiv.org/abs/1707.01926)

2. **Dataset:**  
   PeMS (Caltrans Performance Measurement System)  
   [http://pems.dot.ca.gov/](http://pems.dot.ca.gov/)

3. **Related Work:**
   - Spatial-Temporal Graph Convolutional Networks (STGCN)
   - Graph WaveNet for Deep Spatial-Temporal Graph Modeling
   - Attention-based Spatial-Temporal Graph Convolutional Networks (ASTGCN)

---

## Author

**Vaishnavi Kamdi**  
George Washington University  
Advanced Machine Learning (Fall 2025)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, suggestions, or collaboration opportunities:
- GitHub: [@vaish725](https://github.com/vaish725)
- Repository: [Spatio-Temporal-Traffic-Flow-Prediction](https://github.com/vaish725/Spatio-Temporal-Traffic-Flow-Prediction)

---

**⭐ If you find this project useful, please consider giving it a star!**
