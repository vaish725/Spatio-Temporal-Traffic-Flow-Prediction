"""
Minimal DCRNN Training Script - Smoke Test for Mid-Semester Presentation

This script demonstrates end-to-end DCRNN training on a small subset of PEMS-BAY data.
Purpose: Proof-of-concept showing the model can learn and make predictions.

Author: Vaishnavi Kamdi
Date: October 30, 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our DCRNN model
from models.dcrnn import DCRNN

print("="*70)
print("DCRNN MINIMAL TRAINING - SMOKE TEST")
print("="*70)


# ============================================================================
# 1. CONFIGURATION
# ============================================================================
class Config:
    """Configuration parameters for the smoke test"""
    # Data paths
    DATA_PATH = 'data/PEMS-BAY.csv'
    ADJ_MATRIX_PATH = 'data/adj_mx_bay.pkl'
    
    # Model hyperparameters
    INPUT_DIM = 1           # Traffic speed (single feature)
    HIDDEN_DIM = 16         # Small hidden dimension for quick training
    OUTPUT_DIM = 1          # Predict traffic speed
    NUM_LAYERS = 1          # Single layer for simplicity
    
    # Sequence parameters
    INPUT_SEQ_LEN = 12      # 1 hour of history (12 * 5 min)
    OUTPUT_SEQ_LEN = 12     # Predict next 1 hour
    
    # Training parameters
    BATCH_SIZE = 4          # Small batch for quick iteration
    NUM_EPOCHS = 10         # Limited epochs for smoke test
    LEARNING_RATE = 0.001
    
    # Data subset (use only portion of data for speed)
    USE_SENSORS = 50        # Use only first 50 sensors
    USE_DAYS = 7            # Use only first 7 days of data
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()


# ============================================================================
# 2. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[STEP 1] Loading PEMS-BAY data...")

# Load traffic data
df = pd.read_csv(config.DATA_PATH)
print(f"  - Original shape: {df.shape}")

# Extract subset of data (first N sensors, first M days)
num_timesteps_per_day = 288  # 5-minute intervals
max_timesteps = config.USE_DAYS * num_timesteps_per_day
df_subset = df.iloc[:max_timesteps, :config.USE_SENSORS + 1]  # +1 for timestamp column

# Extract speed values (exclude timestamp)
speed_values = df_subset.iloc[:, 1:].values  # Shape: (T, N)
print(f"  - Subset shape: {speed_values.shape}")
print(f"  - Using {config.USE_SENSORS} sensors, {config.USE_DAYS} days")


# ============================================================================
# 3. NORMALIZATION
# ============================================================================
print("\n[STEP 2] Normalizing data (Z-score)...")

# Z-score normalization
mean = speed_values.mean()
std = speed_values.std()
speed_normalized = (speed_values - mean) / std

print(f"  - Mean: {mean:.2f} mph")
print(f"  - Std: {std:.2f} mph")


# ============================================================================
# 4. SEQUENCE CREATION
# ============================================================================
print("\n[STEP 3] Creating sequences...")

def create_sequences(data, input_len, output_len):
    """
    Create input-output sequence pairs using sliding window.
    
    Args:
        data: (T, N) array of normalized speeds
        input_len: Length of input sequence
        output_len: Length of output sequence
    
    Returns:
        X: (num_samples, input_len, N) - input sequences
        Y: (num_samples, output_len, N) - target sequences
    """
    T, N = data.shape
    X, Y = [], []
    
    for i in range(T - input_len - output_len + 1):
        X.append(data[i : i + input_len])
        Y.append(data[i + input_len : i + input_len + output_len])
    
    return np.array(X), np.array(Y)

X, Y = create_sequences(speed_normalized, config.INPUT_SEQ_LEN, config.OUTPUT_SEQ_LEN)
print(f"  - X shape: {X.shape}")  # (num_samples, T_in, N)
print(f"  - Y shape: {Y.shape}")  # (num_samples, T_out, N)


# ============================================================================
# 5. TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n[STEP 4] Splitting data (chronological)...")

num_samples = X.shape[0]
train_end = int(0.6 * num_samples)
val_end = int(0.8 * num_samples)

X_train, Y_train = X[:train_end], Y[:train_end]
X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
X_test, Y_test = X[val_end:], Y[val_end:]

print(f"  - Train: {X_train.shape[0]} samples")
print(f"  - Val: {X_val.shape[0]} samples")
print(f"  - Test: {X_test.shape[0]} samples")


# ============================================================================
# 6. CREATE DATALOADERS
# ============================================================================
class TrafficDataset(Dataset):
    """PyTorch Dataset for traffic speed sequences"""
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_dataset = TrafficDataset(X_train, Y_train)
val_dataset = TrafficDataset(X_val, Y_val)
test_dataset = TrafficDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


# ============================================================================
# 7. INITIALIZE MODEL
# ============================================================================
print("\n[STEP 5] Initializing DCRNN model...")

model = DCRNN(
    input_dim=config.INPUT_DIM,
    hidden_dim=config.HIDDEN_DIM,
    output_dim=config.OUTPUT_DIM,
    num_layers=config.NUM_LAYERS
).to(config.DEVICE)

# Count parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  - Model parameters: {num_params:,}")
print(f"  - Device: {config.DEVICE}")

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


# ============================================================================
# 8. TRAINING LOOP
# ============================================================================
print("\n[STEP 6] Training model...")
print("="*70)

train_losses = []
val_losses = []

for epoch in range(config.NUM_EPOCHS):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    
    for batch_X, batch_Y in train_loader:
        batch_X = batch_X.to(config.DEVICE)  # (batch, T_in, N)
        batch_Y = batch_Y.to(config.DEVICE)  # (batch, T_out, N)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch_X, T_out=config.OUTPUT_SEQ_LEN)  # (batch, T_out, N, 1)
        
        # Reshape for loss calculation
        output = output.squeeze(-1)  # (batch, T_out, N)
        
        # Calculate loss
        loss = criterion(output, batch_Y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
    
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    epoch_val_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(config.DEVICE)
            batch_Y = batch_Y.to(config.DEVICE)
            
            output = model(batch_X, T_out=config.OUTPUT_SEQ_LEN)
            output = output.squeeze(-1)
            
            loss = criterion(output, batch_Y)
            epoch_val_loss += loss.item()
    
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Print progress
    print(f"Epoch [{epoch+1:2d}/{config.NUM_EPOCHS}] | "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

print("="*70)
print("Training completed!")


# ============================================================================
# 9. EVALUATION ON TEST SET
# ============================================================================
print("\n[STEP 7] Evaluating on test set...")

model.eval()
predictions = []
ground_truth = []

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        batch_X = batch_X.to(config.DEVICE)
        
        output = model(batch_X, T_out=config.OUTPUT_SEQ_LEN)
        output = output.squeeze(-1).cpu().numpy()
        
        predictions.append(output)
        ground_truth.append(batch_Y.numpy())

predictions = np.concatenate(predictions, axis=0)
ground_truth = np.concatenate(ground_truth, axis=0)

# Calculate MAE (Mean Absolute Error) in original scale
predictions_original = predictions * std + mean
ground_truth_original = ground_truth * std + mean

mae = np.mean(np.abs(predictions_original - ground_truth_original))
rmse = np.sqrt(np.mean((predictions_original - ground_truth_original) ** 2))

print(f"  - Test MAE: {mae:.2f} mph")
print(f"  - Test RMSE: {rmse:.2f} mph")


# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n[STEP 8] Generating visualizations...")

# Create docs folder if it doesn't exist
os.makedirs('docs', exist_ok=True)

# Visualization 1: Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, config.NUM_EPOCHS + 1), train_losses, marker='o', label='Train Loss', linewidth=2)
plt.plot(range(1, config.NUM_EPOCHS + 1), val_losses, marker='s', label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title('DCRNN Training Progress', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/training_loss.png', dpi=300, bbox_inches='tight')
print("  - Saved: docs/training_loss.png")

# Visualization 2: Predictions vs Ground Truth (sample sensor, sample time)
# Select a random test sample and a random sensor
sample_idx = np.random.randint(0, len(predictions))
sensor_idx = np.random.randint(0, predictions.shape[2])

pred_sample = predictions_original[sample_idx, :, sensor_idx]
truth_sample = ground_truth_original[sample_idx, :, sensor_idx]
time_steps = np.arange(1, config.OUTPUT_SEQ_LEN + 1)

plt.figure(figsize=(12, 5))
plt.plot(time_steps, truth_sample, marker='o', label='Ground Truth', linewidth=2, color='#2E86AB')
plt.plot(time_steps, pred_sample, marker='s', label='DCRNN Prediction', linewidth=2, color='#F18F01')
plt.xlabel('Time Step (5-min intervals)', fontsize=12)
plt.ylabel('Traffic Speed (mph)', fontsize=12)
plt.title(f'DCRNN Prediction vs Ground Truth - Sensor {sensor_idx}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/prediction_vs_truth.png', dpi=300, bbox_inches='tight')
print("  - Saved: docs/prediction_vs_truth.png")

# Visualization 3: Error Distribution
errors = (predictions_original - ground_truth_original).flatten()
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.axvline(errors.mean(), color='orange', linestyle='--', linewidth=2, 
            label=f'Mean Error: {errors.mean():.2f} mph')
plt.xlabel('Prediction Error (mph)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('docs/error_distribution.png', dpi=300, bbox_inches='tight')
print("  - Saved: docs/error_distribution.png")

plt.show()


# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SMOKE TEST SUMMARY")
print("="*70)
print(f"Model Successfully Trained and Tested!")
print(f"\nConfiguration:")
print(f"  - Sensors: {config.USE_SENSORS}")
print(f"  - Days of data: {config.USE_DAYS}")
print(f"  - Training epochs: {config.NUM_EPOCHS}")
print(f"  - Model parameters: {num_params:,}")
print(f"\nPerformance:")
print(f"  - Final Train Loss: {train_losses[-1]:.4f}")
print(f"  - Final Val Loss: {val_losses[-1]:.4f}")
print(f"  - Test MAE: {mae:.2f} mph")
print(f"  - Test RMSE: {rmse:.2f} mph")
print(f"\nNext Steps for Full Project:")
print(f"  1. Scale to all 325 sensors")
print(f"  2. Train on full 6-month dataset")
print(f"  3. Implement actual diffusion convolution (currently using simple linear layers)")
print(f"  4. Add teacher forcing for decoder")
print(f"  5. Hyperparameter tuning")
print(f"  6. Compare with baseline models (HA, ARIMA, VAR)")
print("="*70)

