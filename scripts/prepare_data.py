"""
Quick data preparation script for the dashboard
Run this before launching the dashboard if you don't have processed data
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

print("🔧 Preparing data for dashboard...")
print("=" * 50)

# Check if already processed
if os.path.exists('data/pems_bay_processed.npz'):
    print("✅ Data already processed!")
    print("   Location: data/pems_bay_processed.npz")
    data = np.load('data/pems_bay_processed.npz')
    print(f"   Sensors: {data['P_fwd'].shape[0]}")
    print(f"   Train samples: {len(data['X_train'])}")
    print(f"   Val samples: {len(data['X_val'])}")
    print(f"   Test samples: {len(data['X_test'])}")
    sys.exit(0)

# Check if raw data exists
if not os.path.exists('data/PEMS-BAY.csv'):
    print("❌ Raw data not found!")
    print("   Please download PEMS-BAY.csv first")
    print("   Run: python3 scripts/download_pems_bay.py")
    sys.exit(1)

print("\n📊 Loading raw data...")
df = pd.read_csv('data/PEMS-BAY.csv')
speed_data = df.drop(columns=[df.columns[0]]).values.astype(np.float32)
print(f"   Shape: {speed_data.shape} (timesteps x sensors)")

print("\n🔧 Handling missing values...")
for i in tqdm(range(speed_data.shape[1]), desc="Interpolating"):
    mask = np.isnan(speed_data[:, i])
    if mask.any():
        speed_data[mask, i] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            speed_data[~mask, i]
        )

print("\n📐 Normalizing data...")
mean = speed_data.mean()
std = speed_data.std()
speed_data_norm = (speed_data - mean) / std
print(f"   Mean: {mean:.2f} mph")
print(f"   Std: {std:.2f} mph")

print("\n🔄 Creating sequences...")
T_in, T_out = 12, 12
num_samples = speed_data_norm.shape[0] - T_in - T_out + 1
num_nodes = speed_data_norm.shape[1]

X = np.zeros((num_samples, T_in, num_nodes, 1), dtype=np.float32)
y = np.zeros((num_samples, T_out, num_nodes, 1), dtype=np.float32)

for i in tqdm(range(num_samples), desc="Creating sequences"):
    X[i, :, :, 0] = speed_data_norm[i:i+T_in, :]
    y[i, :, :, 0] = speed_data_norm[i+T_in:i+T_in+T_out, :]

print("\n✂️ Splitting data...")
train_split = int(0.7 * num_samples)
val_split = int(0.8 * num_samples)

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

print(f"   Train: {len(X_train)} samples")
print(f"   Val: {len(X_val)} samples")
print(f"   Test: {len(X_test)} samples")

print("\n🗺️ Creating adjacency matrix...")
np.random.seed(42)

# Simulate sensor positions along Bay Area highways
positions = np.linspace(0, 100, num_nodes).reshape(-1, 1)
positions = np.hstack([positions, np.random.randn(num_nodes, 1) * 5])

# Gaussian kernel for adjacency
distances = cdist(positions, positions, metric='euclidean')
sigma = np.std(distances) * 0.1
adj_matrix = np.exp(-distances**2 / (sigma**2))
adj_matrix[adj_matrix < 0.1] = 0
np.fill_diagonal(adj_matrix, 1.0)

# Transition matrices for diffusion convolution
row_sum = adj_matrix.sum(axis=1, keepdims=True) + 1e-8
P_fwd = (adj_matrix / row_sum).astype(np.float32)

col_sum = adj_matrix.sum(axis=0, keepdims=True) + 1e-8
P_bwd = (adj_matrix / col_sum).T.astype(np.float32)

num_edges = int((adj_matrix > 0).sum() - num_nodes) / 2
print(f"   Nodes: {num_nodes}")
print(f"   Edges: {num_edges}")

print("\n💾 Saving processed data...")
np.savez_compressed(
    'data/pems_bay_processed.npz',
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    P_fwd=P_fwd, P_bwd=P_bwd,
    mean=mean, std=std,
    adj_matrix=adj_matrix
)

file_size = os.path.getsize('data/pems_bay_processed.npz') / 1e6
print(f"   ✅ Saved: data/pems_bay_processed.npz ({file_size:.1f} MB)")

print("\n" + "=" * 50)
print("✅ Data preparation complete!")
print("\nYou can now:")
print("  1. Train the model: python3 scripts/train_colab_safe.py")
print("  2. Test dashboard: python3 scripts/test_dashboard.py")
print("  3. Launch dashboard: streamlit run app.py")
print("=" * 50)
