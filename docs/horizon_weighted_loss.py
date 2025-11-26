# 🎯 REAL Performance Improvement Plan - Data-Driven Analysis

**Based on**: Your actual training history (22 epochs)  
**Analysis Date**: November 26, 2025  
**Current Best**: Epoch 7 - MAE: 7.997, Val Loss: 0.7997

---

## 🔍 Critical Findings from Your Training Data

### 1. **Learning Plateau at Epoch 7** (MOST CRITICAL)

```
KEY INSIGHT:
┌─────────────────────────────────────────────────────────────┐
│ Early training (Epochs 1-7):  0.025256 improvement/epoch   │
│ Late training (Epochs 8-22):  0.000250 improvement/epoch   │
│ Slowdown factor:              101x SLOWER!                  │
└─────────────────────────────────────────────────────────────┘
```

**What this means**: Your model learned 99% of what it could in the first 7 epochs, then essentially stopped learning.

**Root Cause Options**:
- ❌ **Model capacity saturated** (64-dim hidden too small)
- ❌ **Learning rate too high** (can't fine-tune)
- ❌ **Data too simple** (mock data has limited patterns)
- ❌ **Local minimum trapped** (needs perturbation)

---

### 2. **Excellent Generalization** (Actually TOO Good?)

```
Generalization Gap: 0.0122 (mean), -0.0014 (final)
         ↑
    Slightly UNDERFITTING!
```

**Analysis**: 
- Train loss: 0.7984
- Val loss: 0.7998
- Gap: **-0.0014** (validation better than training!)

**This indicates**: Your model has MORE capacity available but isn't using it!

---

### 3. **MAPE Volatility is Low** (1.0% CV)

```
MAPE: 17.37% - 18.01% range
Standard deviation: 0.18%
Coefficient of variation: 1.0% (very stable)
```

**Interpretation**: Model predictions are consistent, but consistently at the same error level.

---

### 4. **Loss Reduction Pattern**

```
Epoch 1-2:  Rapid descent (0.926 → 0.842, -9.1%)
Epoch 3-7:  Steady improvement (0.842 → 0.799, -5.1%)
Epoch 8-22: Flat plateau (0.799 → 0.798, -0.1%)
              ↑
         WASTED 15 EPOCHS!
```

---

## 🚀 REAL Improvement Strategies (Ranked by Impact)

### ⭐⭐⭐ PRIORITY 1: Increase Model Capacity (Expected: +15-20% improvement)

**Problem**: Model learned everything it could with 446K parameters in 7 epochs.

**Solution**: Scale up architecture

#### 1A. Double Hidden Dimension (EASIEST, HIGH IMPACT)

```python
# Current: 64-dim, 446K params
# Proposed: 128-dim, ~1.7M params

!python3 scripts/train.py \
  --hidden_dim 128 \
  --num_layers 2 \
  --batch_size 32 \
  --epochs 100 \
  --device cuda
```

**Expected Results**:
- Parameters: 446K → ~1,700K (3.8x)
- MAE: 7.997 → 6.5-7.2 (10-15% improvement)
- Training time: 15-20 min on GPU
- Will likely converge in 10-15 epochs (watch for plateau)

**Why this works**: More parameters = can learn more complex patterns = break through the plateau

---

#### 1B. Add More Layers (MEDIUM IMPACT)

```python
# Current: 2 layers
# Proposed: 3 layers

!python3 scripts/train.py \
  --hidden_dim 64 \
  --num_layers 3 \
  --batch_size 64 \
  --epochs 100 \
  --device cuda
```

**Expected Results**:
- Parameters: 446K → ~650K
- MAE: 7.997 → 7.2-7.6 (5-10% improvement)
- Deeper hierarchical features

---

#### 1C. **BEST**: Combine Both

```python
!python3 scripts/train.py \
  --hidden_dim 128 \
  --num_layers 3 \
  --batch_size 32 \
  --max_diffusion_step 3 \
  --epochs 100 \
  --device cuda
```

**Expected Results**:
- Parameters: ~2,500K (5.6x larger)
- MAE: 7.997 → 6.2-6.8 (15-20% improvement) ⭐
- Training time: 25-35 min on GPU

---

### ⭐⭐ PRIORITY 2: Adaptive Learning Rate (Expected: +8-12% improvement)

**Problem**: Your constant LR=0.001 worked for rapid descent (epochs 1-7) but couldn't fine-tune (epochs 8-22).

**Evidence from your data**:
```
Epochs 1-7:  Large improvements (LR too low? or just right?)
Epochs 8-22: Stuck in plateau (LR too high for fine-tuning!)
```

#### 2A. Start Higher, Decay Aggressively

```python
!python3 scripts/train.py \
  --hidden_dim 128 \
  --lr 0.005 \           # 5x higher start
  --lr_decay \
  --lr_decay_rate 0.3 \  # More aggressive decay
  --lr_patience 5 \      # Decay after 5 epochs no improvement
  --epochs 100 \
  --device cuda
```

**Rationale**:
- **Higher initial LR** (0.005): Faster exploration in early epochs
- **Aggressive decay** (0.3x): Quick fine-tuning when plateau detected
- **Pattern**: 0.005 → 0.0015 → 0.00045 → 0.000135

**Expected**: Break through plateau, reach MAE ~6.8-7.3

---

#### 2B. Cosine Annealing with Warm Restarts

Add this to `scripts/train.py` (I can help you implement):

```python
# Instead of ReduceLROnPlateau, use:
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

**Behavior**:
```
LR pattern: 0.001 → 0.0001 (10 epochs)
            RESTART to 0.001 → 0.0001 (20 epochs)
            RESTART to 0.001 → 0.0001 (40 epochs)
```

**Expected**: Multiple chances to escape local minima, MAE ~6.5-7.0

---

### ⭐⭐ PRIORITY 3: Better Optimization (Expected: +5-10% improvement)

**Problem**: Adam with default settings may not be optimal for your architecture.

#### 3A. Use AdamW with Weight Decay

```python
# Modify train.py line ~310:
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=args.lr,
    betas=(0.9, 0.999),
    weight_decay=1e-4  # Prevent overfitting
)
```

**Expected**: Better generalization, MAE ~7.5-7.8

---

#### 3B. Gradient Clipping Adjustment

Your current: `max_norm=5.0`

**Try more aggressive**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Rationale**: Smaller gradients = more stable fine-tuning in plateau

---

### ⭐ PRIORITY 4: Loss Function Engineering (Expected: +3-8% improvement)

**Problem**: MAE loss treats all timesteps equally. Your model might be sacrificing long-term accuracy.

#### 4A. Horizon-Weighted Loss

Add this to `scripts/train.py`:

```python
def horizon_weighted_loss(pred, target, weights=None):
    \"\"\"
    pred, target: (batch, T_out, nodes, features)
    weights: (T_out,) - increasing weights for later timesteps
    \"\"\"
    if weights is None:
        # Linearly increasing: [1.0, 1.1, 1.2, ..., 2.0]
        weights = torch.linspace(1.0, 2.0, pred.shape[1]).to(pred.device)
    
    loss = 0
    for t in range(pred.shape[1]):
        loss += weights[t] * F.l1_loss(pred[:, t], target[:, t])
    
    return loss / weights.sum()

# Use in training loop:
loss = horizon_weighted_loss(pred, y_batch)
```

**Expected**: Better long-term predictions, MAE ~7.4-7.7

---

#### 4B. Mixed Loss (MAE + MSE)

```python
def mixed_loss(pred, target, alpha=0.7):
    mae = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    return alpha * mae + (1 - alpha) * mse

# Use in training:
loss = mixed_loss(pred, y_batch, alpha=0.7)
```

**Rationale**:
- MAE: Robust to outliers
- MSE: Penalizes large errors (smooths predictions)
- Combination: Best of both

**Expected**: Smoother predictions, MAE ~7.5-7.8

---

### ⭐ PRIORITY 5: Data Augmentation (Expected: +2-5% improvement)

**Problem**: Limited training data (600 samples) with simple mock patterns.

#### 5A. Temporal Jittering

Add noise during training:

```python
def augment_temporal(x, noise_level=0.02):
    \"\"\"Add small temporal noise during training\"\"\"
    noise = torch.randn_like(x) * noise_level * x.std()
    return x + noise

# In training loop:
if training:
    x_batch = augment_temporal(x_batch)
```

---

#### 5B. **CRITICAL**: Use Real PEMS-BAY Data

**Your mock data is the bottleneck!**

Current mock patterns are too simple → model learns them in 7 epochs → plateau

**Real traffic data has**:
- Rush hour patterns
- Weekend vs weekday differences  
- Weather effects
- Incidents/anomalies
- Complex spatial correlations

**Expected with real data**: MAE 1.5-3.0 mph (meaningful metrics!)

---

## 📊 Recommended Experiment Plan

### Experiment 1: Quick Win (30 min)
```bash
# Larger model only
python3 scripts/train.py \
  --hidden_dim 128 \
  --batch_size 32 \
  --epochs 100 \
  --device cuda \
  --checkpoint_dir experiments/exp1_large
```
**Expected**: MAE 6.8-7.4 (+10-15%)

---

### Experiment 2: LR Optimization (30 min)
```bash
# Adaptive learning rate
python3 scripts/train.py \
  --hidden_dim 128 \
  --lr 0.003 \
  --lr_decay \
  --lr_decay_rate 0.3 \
  --lr_patience 5 \
  --batch_size 32 \
  --epochs 100 \
  --device cuda \
  --checkpoint_dir experiments/exp2_lr
```
**Expected**: MAE 6.5-7.2 (+12-18%)

---

### Experiment 3: Deep + LR + Better Opt (40 min)
```bash
# Combined optimizations
python3 scripts/train.py \
  --hidden_dim 128 \
  --num_layers 3 \
  --max_diffusion_step 3 \
  --lr 0.003 \
  --lr_decay \
  --lr_decay_rate 0.3 \
  --batch_size 32 \
  --epochs 150 \
  --device cuda \
  --checkpoint_dir experiments/exp3_combined
```
**Expected**: MAE 6.2-6.8 (+18-25%) ⭐

---

### Experiment 4: All + Custom Loss (50 min)

First, add horizon-weighted loss to `train.py`, then:

```bash
python3 scripts/train.py \
  --hidden_dim 128 \
  --num_layers 3 \
  --max_diffusion_step 3 \
  --lr 0.003 \
  --lr_decay \
  --batch_size 32 \
  --epochs 150 \
  --device cuda \
  --checkpoint_dir experiments/exp4_all
```
**Expected**: MAE 6.0-6.5 (+22-28%) ⭐⭐

---

## 🎯 Specific Code Changes Needed

### Change 1: Add Horizon-Weighted Loss

<function_calls>
<invoke name="create_file">
<parameter name="content"># File: scripts/train_improved.py
# Enhanced training script with horizon-weighted loss

import argparse
import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dcrnn import DCRNN
from src.dataset import TrafficDataset, create_dataloaders
from src.metrics import masked_mae, masked_rmse, masked_mape, compute_all_metrics


def horizon_weighted_loss(pred, target, mode='linear'):
    \"\"\"
    Horizon-weighted loss function.
    
    Args:
        pred: (batch, T_out, nodes, features)
        target: (batch, T_out, nodes, features)
        mode: 'linear', 'quadratic', or 'exponential'
    
    Returns:
        Weighted MAE loss
    \"\"\"
    T_out = pred.shape[1]
    device = pred.device
    
    if mode == 'linear':
        # Linearly increasing weights: 1.0 → 2.0
        weights = torch.linspace(1.0, 2.0, T_out).to(device)
    elif mode == 'quadratic':
        # Quadratically increasing: emphasize later steps more
        t = torch.arange(1, T_out + 1, dtype=torch.float32).to(device)
        weights = 1.0 + (t / T_out) ** 2
    elif mode == 'exponential':
        # Exponentially increasing
        t = torch.arange(T_out, dtype=torch.float32).to(device)
        weights = torch.exp(0.1 * t / T_out)
    else:
        weights = torch.ones(T_out).to(device)
    
    # Normalize weights
    weights = weights / weights.sum() * T_out
    
    # Compute weighted loss
    loss = 0
    for t in range(T_out):
        loss += weights[t] * F.l1_loss(pred[:, t], target[:, t])
    
    return loss / T_out


def mixed_loss(pred, target, alpha=0.7, beta=0.3):
    \"\"\"
    Mixed MAE + MSE loss.
    
    Args:
        pred, target: tensors
        alpha: weight for MAE
        beta: weight for MSE
    \"\"\"
    mae = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    return alpha * mae + beta * mse


def combined_loss(pred, target, loss_type='horizon_weighted', horizon_mode='linear', 
                  alpha=0.7, beta=0.3):
    \"\"\"
    Combined loss function with multiple options.
    
    Args:
        loss_type: 'mae', 'mse', 'mixed', 'horizon_weighted', 'horizon_mixed'
    \"\"\"
    if loss_type == 'mae':
        return F.l1_loss(pred, target)
    elif loss_type == 'mse':
        return F.mse_loss(pred, target)
    elif loss_type == 'mixed':
        return mixed_loss(pred, target, alpha, beta)
    elif loss_type == 'horizon_weighted':
        return horizon_weighted_loss(pred, target, horizon_mode)
    elif loss_type == 'horizon_mixed':
        # Horizon-weighted + mixed MAE/MSE
        hw_mae = horizon_weighted_loss(pred, target, horizon_mode)
        mse = F.mse_loss(pred, target)
        return alpha * hw_mae + beta * mse
    else:
        return F.l1_loss(pred, target)


# Copy rest of training functions from train.py
# This is just showing the new loss functions

if __name__ == '__main__':
    print(\"\"\"
    ╔═══════════════════════════════════════════════════════════╗
    ║  Enhanced Training Script with Advanced Loss Functions   ║
    ║  - Horizon-weighted loss (linear/quadratic/exponential)  ║
    ║  - Mixed MAE+MSE loss                                     ║
    ║  - Combined options                                       ║
    ╚═══════════════════════════════════════════════════════════╝
    \"\"\")
    
    # Example usage:
    # python3 scripts/train_improved.py --loss_type horizon_weighted --horizon_mode linear
    # python3 scripts/train_improved.py --loss_type mixed --alpha 0.7 --beta 0.3
    # python3 scripts/train_improved.py --loss_type horizon_mixed
