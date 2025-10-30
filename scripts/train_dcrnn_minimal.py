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
    # ~100k datapoints
    
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
num_timesteps_per_day = 288  # 5-minute intervals, 24 hr * 12 intervals/hr = 288 intervals/day
max_timesteps = config.USE_DAYS * num_timesteps_per_day #max_timesteps = 7 days * 288 intervals/day = 2016 intervals
df_subset = df.iloc[:max_timesteps, :config.USE_SENSORS + 1]  # +1 for timestamp column

# Extract speed values (exclude timestamp)
speed_values = df_subset.iloc[:, 1:].values  # Shape: (T, N)
print(f"  - Subset shape: {speed_values.shape}") #Result: (2016, 50) array of speeds
print(f"  - Using {config.USE_SENSORS} sensors, {config.USE_DAYS} days")


# ============================================================================
# 3. Z-SCORE NORMALIZATION (Standardization)
# ============================================================================
print("\n[STEP 2] Normalizing data (Z-score)...")

# Z-score normalization
mean = speed_values.mean() #avg speed across all datapoints
std = speed_values.std() #std deviation (how spread out speeds are)
speed_normalized = (speed_values - mean) / std #subtract mean to center data around 0, then divide by std which makes std dev = 1

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

    #Sliding window with input_len=12, output_len=12:
    # Sample 1:
        # Input: timesteps 0-11 (past hour)
        # Output: timesteps 12-23 (next hour to predict)
    # Sample 2:
        # Input: timesteps 1-12 (shift by 1)
        # Output: timesteps 13-24
    # Sample 3:
        # Input: timesteps 2-13
        # Output: timesteps 14-25
    # And so on...
    T, N = data.shape # T=2016 timesteps, N=50 sensors
    X, Y = [], []
    
    for i in range(T - input_len - output_len + 1): #1993 samples
        X.append(data[i : i + input_len]) #Shape: (12, 50) = 12 timesteps × 50 sensors
        Y.append(data[i + input_len : i + input_len + output_len]) #Shape: (12, 50)
    
    return np.array(X), np.array(Y)
    #X shape: (1993, 12, 50) = 1993 samples, each is 12×50
    #Y shape: (1993, 12, 50)

X, Y = create_sequences(speed_normalized, config.INPUT_SEQ_LEN, config.OUTPUT_SEQ_LEN)
print(f"  - X shape: {X.shape}")  # (num_samples, T_in, N) #Result: (1993, 12, 50)
print(f"  - Y shape: {Y.shape}")  # (num_samples, T_out, N) #Result: (1993, 12, 50)


# ============================================================================
# 5. TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n[STEP 4] Splitting data (chronological split, NOT random shuffle)...")

num_samples = X.shape[0] #1993 samples
train_end = int(0.6 * num_samples) #60% of data for training; 1193
val_end = int(0.8 * num_samples) #20% of data for validation; 1594

X_train, Y_train = X[:train_end], Y[:train_end]
X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
X_test, Y_test = X[val_end:], Y[val_end:]

print(f"  - Train: {X_train.shape[0]} samples") #Result: 1195
print(f"  - Val: {X_val.shape[0]} samples") #Result: 399
print(f"  - Test: {X_test.shape[0]} samples") #Result: 399


# ============================================================================
# 6. CREATE DATALOADERS
# ============================================================================
class TrafficDataset(Dataset):
    """PyTorch Dataset for traffic speed sequences"""
    def __init__(self, X, Y): #convert numpy arrays to PyTorch tensors
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx): #return the i-th sample
        return self.X[idx], self.Y[idx]

train_dataset = TrafficDataset(X_train, Y_train)
val_dataset = TrafficDataset(X_val, Y_val)
test_dataset = TrafficDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
# shuffle=True (train only):
# Randomize order each epoch
# Prevents model from memorizing sequence order
# DON'T shuffle val/test (want consistent evaluation)

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
# p.numel() = number of elements in each parameter tensor
# if p.requires_grad = only count trainable parameters

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  - Model parameters: {num_params:,}")
print(f"  - Device: {config.DEVICE}")

# Expected output:
# Probably around 10,000-50,000 parameters

# Loss function and optimizer
criterion = nn.MSELoss() #Formula: (prediction - truth)²
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE) #Adam Optimizer: Adjusts model weights to reduce loss
#lr = 0.001: learning rate


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
        batch_X = batch_X.to(config.DEVICE)  # (batch, T_in, N); batch_X: (4, 12, 50) = 4 samples, 12 timesteps, 50 sensors
        batch_Y = batch_Y.to(config.DEVICE)  # (batch, T_out, N); batch_Y: (4, 12, 50) = targets for those 4 samples
        
        # Forward pass

        # Clear gradients from previous batch
        # Must do this EVERY batch
        # Otherwise gradients accumulate incorrectly

        optimizer.zero_grad()
        output = model(batch_X, T_out=config.OUTPUT_SEQ_LEN)  # (batch, T_out, N, 1); Output shape: (4, 12, 50, 1)
        
        # Reshape for loss calculation
        output = output.squeeze(-1)  # (batch, T_out, N); Remove last dimension (1); now matches the shape of batch_Y: (4, 12, 50)
        #done because loss func expects same shapes

        # Calculate loss
        loss = criterion(output, batch_Y) #mse
        
        # Backward pass
        loss.backward() #∂loss/∂weight for every weight
        optimizer.step() #weight = weight - learning_rate × gradient

        epoch_train_loss += loss.item() #loss.item():
        # Get the loss value as a Python number
        # Accumulate it for this epoch
    
    avg_train_loss = epoch_train_loss / len(train_loader) #Average loss across all batches
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    epoch_val_loss = 0.0
    
    with torch.no_grad(): #Disables gradient computation
        # Same as training, but:
            # NO optimizer.zero_grad()
            # NO loss.backward(), no weights updated
            # NO optimizer.step()
        # Just forward pass and loss calculation
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
    
    #If val loss increases while train decreases = overfitting

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
        output = output.squeeze(-1).cpu().numpy() #cpu().numpy() = move to CPU and convert to numpy
        
        predictions.append(output)
        ground_truth.append(batch_Y.numpy())


predictions = np.concatenate(predictions, axis=0) 
ground_truth = np.concatenate(ground_truth, axis=0)

# Calculate MAE (Mean Absolute Error) in original scale

# DENORMALIZE predictions
# Convert from normalized scale back to mph
# Reverse the Z-score normalization
predictions_original = predictions * std + mean #Formula: mean(|prediction - truth|)

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

# Visualization 1: Training and Validation Loss Curve
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
# Blue = truth, Orange = prediction
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
# Histogram of all errors
# Should be centered near 0
# Shows most errors are small

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

