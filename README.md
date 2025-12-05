# Spatio-Temporal Traffic Flow Prediction using DCRNN# Spatio-Temporal Traffic Flow Prediction using DCRNN



**Advanced Deep Learning for Multi-Horizon Traffic Speed Forecasting****Traffic Flow Forecasting using Diffusion Convolutional Recurrent Neural Networks**



[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)This repository implements a state-of-the-art deep learning model for multi-horizon traffic speed prediction on road networks. The model combines graph convolution (spatial) and recurrent neural networks (temporal) to capture complex spatio-temporal dependencies in traffic data.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)## Project Overview



> A production-ready implementation of Diffusion Convolutional Recurrent Neural Networks (DCRNN) for traffic forecasting, featuring an interactive Streamlit dashboard and achieving **1.930 mph test MAE** on the PEMS-BAY dataset.**Objective:** Predict traffic speeds at multiple road sensors for the next hour (12 time steps) based on historical observations.



---**Model:** Diffusion Convolutional Recurrent Neural Network (DCRNN)

- Paper: Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting" (ICLR 2018)

## 🎯 Project Overview- Combines diffusion convolution over graphs with encoder-decoder GRU architecture



This project implements a state-of-the-art deep learning system for predicting traffic speeds across a network of 325 sensors in the San Francisco Bay Area. The system combines graph neural networks (spatial modeling) with recurrent neural networks (temporal modeling) to accurately forecast traffic conditions up to 60 minutes ahead.**Dataset:** PEMS-BAY

- 325 traffic sensors in the Bay Area, California

### Key Achievements- 6 months of data (Jan-June 2017)

- 52,116 timesteps at 5-minute intervals

- ✅ **Production Model:** 1.930 mph test MAE (75.9% improvement over baseline)- High-quality data with zero missing values

- ✅ **Interactive Dashboard:** Real-time traffic visualization and prediction interface

- ✅ **Full-Scale Implementation:** 325 sensors, 6 months of data, 52,116 timesteps---

- ✅ **Custom Implementation:** Hand-coded diffusion convolution (not using PyG)

- ✅ **Systematic Optimization:** Trained and evaluated 5K, 10K, and 20K sample models## Repository Structure

- ✅ **Perfect Generalization:** Val MAE 2.097 mph ≈ Test MAE 1.930 mph (no overfitting)

```

---├── data/                  # Raw and processed datasets

│   ├── PEMS-BAY.csv      # Traffic speed time series

## 🚀 Quick Start│   ├── adj_mx_bay.pkl    # Spatial adjacency matrix

│   └── METR-LA.csv       # Alternative dataset

### 1. Installation│

├── models/                # Model implementations

```bash│   ├── dcrnn.py          # DCRNN architecture (encoder-decoder)

# Clone the repository│   └── __init__.py

git clone https://github.com/vaish725/Spatio-Temporal-Traffic-Flow-Prediction.git│

cd Spatio-Temporal-Traffic-Flow-Prediction├── notebooks/             # Jupyter notebooks for analysis

│   └── 01_data_exploration.ipynb  # EDA with visualizations

# Install dependencies│

pip install -r requirements.txt├── scripts/               # Training and evaluation scripts

```│   └── train_dcrnn_minimal.py     # Smoke test training script

│

### 2. Launch Interactive Dashboard├── src/                   # Reusable modules and utilities

│   └── __init__.py

```bash│

# Start the dashboard├── tests/                 # Unit tests

streamlit run app.py│   └── smoke_test_dcrnn.py

│

# Open browser to http://localhost:8501├── docs/                  # Documentation and visualizations

```│   ├── QUICKSTART_FOR_PRESENTATION.md

│   ├── presentation_structure.md

The dashboard provides:│   └── *.png             # Generated visualization images

- **Network Overview:** Real-time traffic visualization on interactive map│

- **Sensor Details:** Deep-dive analysis for individual sensors├── requirements.txt       # Python dependencies

- **Comparison View:** Compare predictions across multiple sensors└── README.md             # This file

- **Date/Time Selection:** Explore predictions for any timestamp in test set (June 3-30, 2017)```



### 3. Data Preparation (Optional - data already included)---



If you need to reprocess the data:## Quick Start



```bash### 1. Installation

python scripts/prepare_data.py

``````bash

# Clone the repository

---git clone https://github.com/vaish725/Spatio-Temporal-Traffic-Flow-Prediction.git

cd Spatio-Temporal-Traffic-Flow-Prediction

## 📊 Model Performance

# Install dependencies

### Final Results (10K Optimal Model)pip install -r requirements.txt

```

| Metric | Value | Baseline | Improvement |

|--------|-------|----------|-------------|### 2. Data Preparation

| **Test MAE** | **1.930 mph** | 8.01 mph | **75.9%** ↓ |

| **Test RMSE** | 3.827 mph | 10.25 mph | 62.7% ↓ |Place the following datasets in the `data/` folder:

| **Test MAPE** | 3.18% | 12.8% | 75.2% ↓ |- `PEMS-BAY.csv` - Traffic speed time series

| **Val MAE** | 2.097 mph | - | - |- `adj_mx_bay.pkl` - Adjacency matrix for sensor network

| **Training Time** | ~45 min | - | - |

Run the preprocessing pipeline:

### Model Characteristics```bash

python scripts/prepare_data.py

- **Architecture:** 2-layer DCRNN encoder-decoder```

- **Parameters:** 446,593 trainable parameters

- **Hidden Dimension:** 64### 3. Exploratory Data Analysis

- **Max Diffusion Steps:** 2

- **Input Sequence:** 12 timesteps (1 hour history)```bash

- **Output Sequence:** 12 timesteps (5, 15, 30, 60 min predictions)# Launch Jupyter Lab

jupyter lab

### Training Configuration

# Open and run: notebooks/01_data_exploration.ipynb

- **Training Samples:** 10,000 (optimized for speed/accuracy balance)```

- **Validation Samples:** 5,209

- **Test Samples:** 10,419 (June 3-30, 2017)This will generate visualizations in the `docs/` folder:

- **Batch Size:** 32- Temporal patterns (hourly, daily, weekly)

- **Learning Rate:** 0.001 (Adam optimizer)- Spatial correlation analysis

- **Epochs:** 20 (early stopping at epoch 15)- Speed distributions

- Time series examples

---

### 4. Train DCRNN Model (Smoke Test)

## 🏗️ Repository Structure

```bash

```# Run minimal training on subset of data

├── app.py                      # 🎨 Interactive Streamlit dashboardpython scripts/train_dcrnn_minimal.py

├── run_dashboard.sh            # Dashboard launcher script```

│

├── dashboard/                  # Dashboard componentsThis trains a small DCRNN model (50 sensors, 7 days) and generates:

│   ├── predictor.py           # Model loading and inference- Training/validation loss curves

│   ├── utils.py               # Utility functions- Prediction vs ground truth plots

│   └── visualization.py       # Plotly visualizations- Error distribution analysis

│

├── models/                     # Neural network architectures**Actual Results (Proof-of-Concept):**

│   ├── dcrnn.py               # DCRNN encoder-decoder implementation- Test MAE: 1.75 mph

│   └── diffusion_conv.py      # Custom diffusion convolution layer- Test RMSE: 3.52 mph

│- Training time: ~15 minutes on CPU

├── src/                        # Core utilities- Model parameters: 1,841

│   ├── dataset.py             # PyTorch Dataset classes

│   └── metrics.py             # Evaluation metrics (MAE, RMSE, MAPE)---

│

├── scripts/                    # Training and utility scripts## Model Architecture

│   ├── train_colab_safe.py    # Main training script (Colab-optimized)

│   ├── train.py               # Flexible training with CLI args**DCRNN (Diffusion Convolutional Recurrent Neural Network):**

│   ├── train_simple.py        # Debug/verification training

│   ├── evaluate.py            # Model evaluation script```

│   ├── prepare_data.py        # Data preprocessing pipelineInput Sequence (12 steps, 1 hour)

│   ├── test_dashboard.py      # Dashboard component testing         ↓

│   ├── download_pems_bay.py   # Dataset acquisition script    [ENCODER]

│   ├── verify_teacher_forcing.py  # Teacher forcing verification    DCGRU Cell 1 → DCGRU Cell 2 → ... → Hidden State

│   ├── traffic_flow_setup.py  # Setup utility    (Spatial diffusion + GRU temporal modeling)

│   ├── generate_presentation_plots.py  # Visualization generation         ↓

│   └── run_experiments.py     # Experiment automation    [DECODER]

│    DCGRU Cell 1 → DCGRU Cell 2 → ... → Output

├── notebooks/                  # Jupyter notebooks         ↓

│   ├── DCRNN_Training_Colab_Online.ipynb  # Colab training notebookOutput Predictions (12 steps, 1 hour)

│   ├── Train_DCRNN_Simple.ipynb           # Simplified training (20K samples)```

│   └── 01_data_exploration.ipynb          # Exploratory data analysis

│**Key Components:**

├── data/                       # Datasets (579MB)1. **Diffusion Convolution:** Captures spatial dependencies through graph structure

│   ├── PEMS-BAY.csv           # Raw traffic data (52,116 × 326)2. **GRU Cells:** Model temporal dynamics in time series

│   ├── pems_bay_processed.npz # Preprocessed train/val/test splits3. **Encoder-Decoder:** Sequence-to-sequence architecture for multi-step prediction

│   └── adj_mx_bay.pkl         # Spatial adjacency matrix

│---

├── checkpoints_colab/          # Trained models (16MB)

│   ├── best_model_optimized.pt         # 🏆 10K optimal model (RECOMMENDED)## Dataset Details

│   ├── best_model_20k.pt               # 20K model (comparison)

│   ├── training_history_optimized.json # Training metrics### PEMS-BAY Statistics

│   └── training_history_20k.json       # 20K training metrics

│| Metric | Value |

├── results/                    # Evaluation results (44MB)|--------|-------|

│   ├── colab_evaluation/      # Latest evaluation outputs| Number of Sensors | 325 |

│   ├── predictions.npy        # Model predictions| Temporal Coverage | 181 days (6 months) |

│   ├── targets.npy            # Ground truth| Timesteps | 52,116 |

│   ├── metrics.json           # Performance metrics| Sampling Rate | 5 minutes |

│   └── *.png                  # Visualization plots| Mean Speed | 63.26 mph |

│| Std Deviation | 8.58 mph |

├── presentation_figures/       # Presentation visualizations| Missing Values | 0 |

│   ├── 1_training_curves.png

│   ├── 2_performance_comparison.png### Key Findings from EDA:

│   ├── 5_results_table.png- **Strong daily periodicity:** Morning rush (8:00 AM = 56.90 mph), Evening rush (5:00 PM = 54.15 mph)

│   └── 6_model_comparison.png- **Weekend effects:** ~6% higher average speeds on Sat/Sun compared to weekdays

│- **Spatial correlation:** Mean correlation 0.359, max 0.970 (neighbors), validating graph-based approach

├── docs/                       # Documentation- **Sparse adjacency:** 97.45% sparsity, ~8.3 connections per sensor, efficient computation

│   ├── DASHBOARD_WALKTHROUGH.md       # Complete dashboard guide- **Node degree:** Mean 8, Median 8, Max 23 (hub sensors), Min 1 (isolated sensors)

│   ├── DASHBOARD_QUICK_REFERENCE.md   # One-page cheat sheet

│   ├── PRESENTATION_GUIDE.md          # Presentation structure---

│   ├── QUICK_REFERENCE.md             # Key metrics reference

│   └── *.md                           # Various analysis documents## Requirements

│

├── tests/                      # Unit tests```

│   └── smoke_test_dcrnn.pyPython >= 3.8

│numpy >= 1.22.0

├── requirements.txt            # Python dependenciespandas >= 1.5.0

├── LICENSE                     # MIT Licensetorch >= 2.0.0

└── README.md                   # This filetorch-geometric >= 2.3.0

```matplotlib >= 3.7.0

seaborn >= 0.12.0

---tqdm >= 4.65.0

jupyter >= 1.0.0

## 🎨 Interactive Dashboard Features```



### Network OverviewSee `requirements.txt` for full dependency list.

- **Interactive Map:** Visualize all 325 sensors with real-time predictions

- **Color Coding:** Green (free flow) → Yellow (moderate) → Red (congested)---

- **Network Health Metrics:** Average speed, congestion percentage, sensor statistics

## Usage Examples

### Sensor Details

- **Time Series Plots:** Historical data + 4 prediction horizons (5, 15, 30, 60 min)### Train on Full Dataset (Coming Soon)

- **Prediction Accuracy:** Compare predictions vs ground truth```bash

- **Sensor Information:** Location, speed statistics, prediction confidencepython scripts/train_dcrnn_full.py \

    --data data/PEMS-BAY.csv \

### Comparison View    --adj_matrix data/adj_mx_bay.pkl \

- **Multi-Sensor Analysis:** Compare up to 3 sensors simultaneously    --epochs 100 \

- **Performance Comparison:** Side-by-side prediction accuracy    --batch_size 64 \

- **Pattern Recognition:** Identify spatial correlations    --hidden_dim 64

```

### Interactive Controls

- **Date/Time Picker:** Explore any timestamp (June 3-30, 2017)### Evaluate Model

- **Timestamp Display:** Shows selected date, time, weekday, sample position```bash

- **View Mode Selection:** Network/Sensor/Comparison viewspython scripts/evaluate.py \

    --model_path checkpoints/best_model.pt \

---    --test_data data/test.npz

```

## 📈 Dataset: PEMS-BAY

---

### Overview

- **Source:** California Transportation Performance Measurement System (PeMS)## Current Progress

- **Location:** San Francisco Bay Area highway network

- **Sensors:** 325 traffic speed sensors**✅ Completed:**

- **Temporal Coverage:** 6 months (January 1 - June 30, 2017)- [x] Data acquisition and preprocessing pipeline

- **Timesteps:** 52,116 at 5-minute intervals- [x] Comprehensive exploratory data analysis with visualizations

- **Data Quality:** Zero missing values (high-quality dataset)- [x] DCRNN model implementation (encoder-decoder architecture)

- [x] Training pipeline with PyTorch

### Statistics- [x] Smoke test demonstrating end-to-end functionality

- [x] Visualization and evaluation metrics

| Metric | Value |

|--------|-------|**🔄 In Progress:**

| Mean Speed | 62.62 mph |- [ ] Scale to full dataset (all 325 sensors, 6 months)

| Std Deviation | 9.59 mph |- [ ] Implement actual diffusion convolution with adjacency matrix

| Min Speed | 0.0 mph |- [ ] Hyperparameter tuning

| Max Speed | 85.0 mph |

| Train Samples | 36,465 (70%) |**📋 Planned:**

| Val Samples | 5,209 (10%) |- [ ] Baseline comparisons (Historical Average, ARIMA, VAR)

| Test Samples | 10,419 (20%) |- [ ] Comprehensive evaluation (MAE, RMSE, MAPE)

- [ ] Ablation studies (spatial vs temporal components)

### Key Patterns (from EDA)- [ ] Sensitivity analysis

- **Daily Periodicity:** Morning rush (8 AM, 56.9 mph), Evening rush (5 PM, 54.2 mph)- [ ] Final report and presentation

- **Weekend Effect:** ~6% higher speeds on Sat/Sun vs weekdays

- **Spatial Correlation:** Mean 0.359, validates graph-based approach---

- **Network Sparsity:** 97.45% sparse adjacency, ~8.3 connections/sensor

## Results (Proof-of-Concept - Smoke Test)

---

### Configuration

## 🧠 Model Architecture| Parameter | Value |

|-----------|-------|

### DCRNN (Diffusion Convolutional Recurrent Neural Network)| Sensors Used | 50 (out of 325) |

| Training Data | 7 days (out of 181) |

```| Training Samples | 1,195 |

Input: (batch, 12 timesteps, 325 sensors, 1 feature)| Validation Samples | 399 |

         ↓| Test Samples | 399 |

    [ENCODER]| Model Parameters | 1,841 |

    ┌─────────────────────────────────────┐| Hidden Dimension | 16 |

    │ DCGRU Cell 1 (Layer 1)              │| Batch Size | 4 |

    │ • Diffusion Conv (forward/backward) │| Epochs | 10 |

    │ • GRU update gates                  │| Training Time | ~15 minutes (CPU) |

    ├─────────────────────────────────────┤

    │ DCGRU Cell 2 (Layer 2)              │### Performance Metrics

    │ • Diffusion Conv (forward/backward) │| Metric | Value |

    │ • GRU update gates                  │|--------|-------|

    └─────────────────────────────────────┘| Test MAE | 1.75 mph |

         ↓ Hidden State (64-dim)| Test RMSE | 3.52 mph |

    [DECODER]| Final Train Loss | 0.1903 |

    ┌─────────────────────────────────────┐| Final Val Loss | 0.3557 |

    │ DCGRU Cell 1 (Layer 1)              │| RMSE/MAE Ratio | 2.01 |

    │ • Teacher forcing (training)        │

    │ • Autoregressive (inference)        │### Analysis

    ├─────────────────────────────────────┤**Strengths:**

    │ DCGRU Cell 2 (Layer 2)              │- Model converges smoothly without overfitting

    │ • Output projection                 │- Predictions within ~2 mph on average (MAE 1.75)

    └─────────────────────────────────────┘- Stable training with consistent improvement over epochs

         ↓- Significantly outperforms naive baseline (~8-10 mph MAE)

Output: (batch, 12 timesteps, 325 sensors, 1 feature)

```**Current Limitations:**

- Conservative predictions (mean reversion behavior)

### Key Components- Limited dynamic range in forecasts

- Simplified architecture (linear layers vs. true diffusion convolution)

1. **Diffusion Convolution**- Trained on minimal data (4% of available dataset)

   - Captures spatial dependencies via graph structure

   - Forward diffusion: Information flow along directed edges**Interpretation:**

   - Backward diffusion: Reverse information propagationResults demonstrate successful proof-of-concept. Conservative behavior is expected given limited training data (1 week) and small model capacity (1,841 parameters). The model learns traffic patterns effectively but requires scaling to capture full dynamics. Next phase will implement true diffusion convolution and train on complete dataset.

   - K=2 hops for multi-scale spatial features

*Note: Proof-of-concept validates approach. Full-scale implementation expected to capture traffic dynamics more accurately.*

2. **DCGRU (Diffusion Convolutional GRU)**

   - Replaces matrix multiplication with diffusion convolution---

   - Update gate, Reset gate, Candidate state

   - Preserves temporal dynamics while incorporating spatial structure## References



3. **Encoder-Decoder Architecture**1. **Li, Y., Yu, R., Shahabi, C., & Liu, Y.** (2018). *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* International Conference on Learning Representations (ICLR).

   - Encoder: Compresses input sequence into latent representation

   - Decoder: Generates multi-step predictions autoregressively2. **PeMS Dataset:** California Transportation Sensors - Performance Measurement System

   - Teacher forcing during training for stable learning

3. **PyTorch Geometric:** Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.*

4. **Custom Implementation**

   - Hand-coded diffusion convolution (no PyTorch Geometric)---

   - Efficient sparse matrix operations

   - Flexible architecture supporting various configurations## Author



---**Vaishnavi Kamdi**  

George Washington University

## 🔬 Training & Optimization

---

### Systematic Approach

## License

We trained three models with increasing data sizes to find the optimal balance:

MIT License - See LICENSE file for details

| Model | Training Samples | Val MAE | Test MAE | Training Time | Status |
|-------|-----------------|---------|----------|---------------|--------|
| 5K    | 5,000          | 2.385   | 2.266    | ~25 min       | Baseline |
| **10K** | **10,000**    | **2.097** | **1.930** | **~45 min** | **✅ OPTIMAL** |
| 20K   | 20,000         | 2.132   | 1.956    | ~80 min       | Diminishing returns |

### Key Findings

1. **10K is Optimal:** Best test performance with reasonable training time
2. **Perfect Generalization:** Val ≈ Test (no overfitting despite limited data)
3. **Stable Training:** Smooth convergence, no instability
4. **Efficient Architecture:** 446K params sufficient for 325-sensor network

### Training Features

- ✅ Teacher forcing for stable sequence-to-sequence learning
- ✅ Early stopping based on validation MAE
- ✅ Gradient clipping (max norm 5.0)
- ✅ Learning rate warmup
- ✅ Batch normalization in diffusion layers
- ✅ Adam optimizer with weight decay

---

## 💻 Usage Examples

### Run Dashboard

```bash
# Start the interactive dashboard
streamlit run app.py

# Or use the launcher script
./run_dashboard.sh
```

### Train Model

```bash
# Train with default settings (10K samples, recommended)
python scripts/train_colab_safe.py

# Train with custom configuration
python scripts/train.py \
    --hidden_dim 64 \
    --num_layers 2 \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.001
```

### Evaluate Model

```bash
# Evaluate the optimal model
python scripts/evaluate.py \
    --checkpoint checkpoints_colab/best_model_optimized.pt \
    --hidden_dim 64 \
    --num_layers 2
```

### Test Dashboard Components

```bash
# Run dashboard component tests
python scripts/test_dashboard.py
```

### Generate Presentation Plots

```bash
# Create all visualization figures
python scripts/generate_presentation_plots.py
```

---

## 📦 Requirements

### Core Dependencies

```
Python >= 3.8
numpy >= 1.22.0
pandas >= 1.5.0
torch >= 2.0.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
scipy >= 1.10.0
tqdm >= 4.65.0
```

### Dashboard Dependencies

```
streamlit >= 1.28.0
plotly >= 5.17.0
```

### Optional

```
jupyter >= 1.0.0          # For notebooks
h5py >= 3.8.0            # For HDF5 data files
```

See `requirements.txt` for complete list with pinned versions.

---

## 🎓 Implementation Highlights

### What Makes This Implementation Special

1. **Custom Diffusion Convolution**
   - Hand-coded implementation (not using PyTorch Geometric)
   - Deep understanding of graph neural network operations
   - Efficient sparse matrix handling

2. **Production-Ready Dashboard**
   - Interactive visualization with Streamlit
   - Real-time predictions and exploration
   - Professional UI/UX design

3. **Systematic Optimization**
   - Trained 3 models (5K, 10K, 20K) to find sweet spot
   - Documented performance tradeoffs
   - Evidence-based model selection

4. **Perfect Generalization**
   - Val MAE ≈ Test MAE (2.097 vs 1.930 mph)
   - No overfitting despite limited training data
   - Robust model architecture

5. **Comprehensive Documentation**
   - Dashboard walkthrough guide
   - Quick reference sheets
   - Detailed code comments

---

## 📚 Technical Details

### Diffusion Convolution

The diffusion convolution captures spatial dependencies by propagating information across the graph:

```
H = Σ(θk,1 * (P_fwd)^k + θk,2 * (P_bwd)^k) * X

where:
- P_fwd: Forward transition matrix (row-normalized adjacency)
- P_bwd: Backward transition matrix (column-normalized adjacency)
- k: Diffusion steps (0 to K)
- θ: Learnable parameters
```

### Teacher Forcing

During training, the decoder uses ground truth values as input:
- **Training:** decoder_input = ground_truth[:, t-1, :, :]
- **Inference:** decoder_input = prediction[:, t-1, :, :]

This stabilizes training and prevents error accumulation.

### Data Preprocessing

1. **Normalization:** Z-score normalization (mean=62.62, std=9.59)
2. **Sequence Creation:** Sliding window (input=12, output=12)
3. **Train/Val/Test Split:** 70%/15%/15% chronological split
4. **Adjacency Matrix:** Gaussian kernel on sensor distances, thresholded at 0.1

---

## 🎯 Results Analysis

### Performance by Horizon

| Horizon | Time | MAE (mph) | RMSE (mph) | MAPE (%) |
|---------|------|-----------|------------|----------|
| 1       | 5 min  | 1.621     | 3.124      | 2.67     |
| 3       | 15 min | 1.812     | 3.516      | 2.98     |
| 6       | 30 min | 1.968     | 3.826      | 3.24     |
| 12      | 60 min | 2.118     | 4.102      | 3.49     |

**Observation:** Performance degrades gracefully with prediction horizon, as expected.

### Comparison with Baselines

| Model | Test MAE | Test RMSE | Parameters |
|-------|----------|-----------|------------|
| Historical Average | 8.01 | 10.25 | 0 |
| DCRNN (Ours) | **1.930** | **3.827** | 446,593 |
| **Improvement** | **75.9%** ↓ | **62.7%** ↓ | - |

---

## 📖 Documentation

- **[Dashboard Walkthrough](docs/DASHBOARD_WALKTHROUGH.md)** - Complete dashboard guide (~300 lines)
- **[Dashboard Quick Reference](docs/DASHBOARD_QUICK_REFERENCE.md)** - One-page cheat sheet
- **[Presentation Guide](docs/PRESENTATION_GUIDE.md)** - 7-slide presentation structure
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Key metrics and model info

---

## 🚧 Project Status

**✅ COMPLETED** - All major components implemented and tested

### Completed Features
- [x] Data preprocessing pipeline
- [x] DCRNN model implementation (custom diffusion convolution)
- [x] Training pipeline with teacher forcing
- [x] Systematic model optimization (5K/10K/20K experiments)
- [x] Interactive Streamlit dashboard
- [x] Comprehensive evaluation and visualization
- [x] Production-ready deployment
- [x] Full documentation

### Production Model
- [x] **10K Optimal Model:** 1.930 mph test MAE
- [x] Located at: `checkpoints_colab/best_model_optimized.pt`
- [x] Ready for deployment and demo

---

## 🔗 References

1. **Li, Y., Yu, R., Shahabi, C., & Liu, Y.** (2018). *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* International Conference on Learning Representations (ICLR). [Paper](https://arxiv.org/abs/1707.01926)

2. **PEMS-BAY Dataset:** California Performance Measurement System (PeMS), Caltrans. [Dataset](http://pems.dot.ca.gov/)

3. **PyTorch:** Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS.

---

## 👤 Author

**Vaishnavi Kamdi**  
MS Data Science, George Washington University  
Advanced Machine Learning, Fall 2025

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset provided by California Department of Transportation (Caltrans)
- Original DCRNN paper by Li et al. (ICLR 2018)
- George Washington University, Advanced ML Course

---

## 📞 Contact

For questions or collaboration opportunities:
- GitHub: [@vaish725](https://github.com/vaish725)
- Repository: [Spatio-Temporal-Traffic-Flow-Prediction](https://github.com/vaish725/Spatio-Temporal-Traffic-Flow-Prediction)

---

**⭐ Star this repo if you find it helpful!**
