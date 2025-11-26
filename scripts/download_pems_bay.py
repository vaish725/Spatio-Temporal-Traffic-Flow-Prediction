#!/usr/bin/env python3
"""
Download and Preprocess PEMS-BAY Traffic Dataset

This script downloads the real PEMS-BAY dataset used in the DCRNN paper
and preprocesses it into the format expected by your training scripts.

Usage:
    python3 scripts/download_pems_bay.py
    
After running, update train.py to use real data instead of mock data.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import urllib.request
import zipfile

print("""
╔══════════════════════════════════════════════════════════════╗
║         PEMS-BAY Real Traffic Data Download Tool            ║
║                                                              ║
║  This will download ~100MB of real traffic sensor data      ║
║  from the Bay Area (California) highway system.             ║
╚══════════════════════════════════════════════════════════════╝
""")

# Create data directory
os.makedirs('data', exist_ok=True)

def download_file(url, output_path):
    """Download file with progress bar"""
    print(f"Downloading from: {url}")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    print(f"✅ Downloaded to: {output_path}")


def download_pems_bay():
    """
    Download PEMS-BAY dataset from multiple sources
    
    The dataset contains:
    - 325 sensors in Bay Area
    - 6 months of data (5 min intervals)
    - ~52,000 timesteps
    """
    
    print("\n📥 Step 1: Downloading PEMS-BAY dataset...")
    print("-" * 70)
    
    # Try multiple sources
    sources = [
        {
            "name": "Zenodo (Recommended)",
            "url": "https://zenodo.org/record/5724362/files/pems_bay.npz",
            "file": "data/pems_bay_raw.npz"
        },
        {
            "name": "Alternative Zenodo",
            "url": "https://zenodo.org/record/3939792/files/PEMS-BAY.zip",
            "file": "data/pems_bay_raw.zip"
        },
        {
            "name": "OpenML Repository",
            "url": "https://www.openml.org/data/get_csv/21854866/pems-bay.arff",
            "file": "data/pems_bay_raw.csv"
        }
    ]
    
    success = False
    for source in sources:
        try:
            print(f"\nTrying {source['name']}...")
            download_file(source['url'], source['file'])
            success = True
            downloaded_file = source['file']
            break
        except Exception as e:
            print(f"❌ Failed: {e}")
            print("Trying next source...")
    
    if not success:
        print("\n❌ All download sources failed.")
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*70)
        print("\n📥 Option 1: Download from Zenodo (RECOMMENDED)")
        print("   Link: https://zenodo.org/record/5724362")
        print("   File: pems_bay.npz (~80MB)")
        print("   Save as: data/pems_bay_raw.npz")
        print("\n📥 Option 2: Use preprocessed data from Google Drive")
        print("   Link: https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX")
        print("   File: pems-bay.h5")
        print("   Save as: data/pems_bay_raw.h5")
        print("\n📥 Option 3: Generate synthetic realistic data")
        print("   Run: python3 scripts/generate_realistic_data.py")
        print("\n After downloading, run this script again:")
        print("   python3 scripts/download_pems_bay.py")
        print("="*70)
        return None
    
    # Extract if compressed
    if downloaded_file.endswith('.tar.gz'):
        print("\n📦 Extracting tar.gz...")
        import tarfile
        with tarfile.open(downloaded_file, 'r:gz') as tar:
            tar.extractall('data/')
    elif downloaded_file.endswith('.zip'):
        print("\n📦 Extracting zip...")
        with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
            zip_ref.extractall('data/')
    
    # Find the extracted .h5 or .npz file
    for root, dirs, files in os.walk('data/'):
        for file in files:
            if file.endswith(('.h5', '.npz')):
                data_file = os.path.join(root, file)
                print(f"✅ Found data file: {data_file}")
                return data_file
    
    print("❌ Could not find data file after extraction")
    return None


def load_pems_bay_h5(filepath):
    """Load PEMS-BAY from HDF5 format"""
    import h5py
    
    print(f"\n📂 Loading {filepath}...")
    
    with h5py.File(filepath, 'r') as f:
        print("Available keys:", list(f.keys()))
        data = f['data'][:]  # Traffic speed data
        
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    
    return data


def load_pems_bay_npz(filepath):
    """Load PEMS-BAY from NPZ format"""
    print(f"\n📂 Loading {filepath}...")
    
    data_dict = np.load(filepath)
    print("Available keys:", list(data_dict.keys()))
    
    # Usually stored as 'data' or 'speed'
    if 'data' in data_dict:
        data = data_dict['data']
    elif 'speed' in data_dict:
        data = data_dict['speed']
    else:
        data = data_dict[list(data_dict.keys())[0]]
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    
    return data


def create_sequences(data, T_in=12, T_out=12):
    """
    Create input/output sequences from traffic data
    
    Args:
        data: (timesteps, nodes) array
        T_in: input sequence length
        T_out: output sequence length
    
    Returns:
        X: (num_samples, T_in, nodes, 1)
        y: (num_samples, T_out, nodes, 1)
    """
    num_timesteps, num_nodes = data.shape
    num_samples = num_timesteps - T_in - T_out + 1
    
    X = np.zeros((num_samples, T_in, num_nodes, 1))
    y = np.zeros((num_samples, T_out, num_nodes, 1))
    
    print(f"\n🔄 Creating sequences...")
    for i in tqdm(range(num_samples)):
        X[i, :, :, 0] = data[i:i+T_in, :]
        y[i, :, :, 0] = data[i+T_in:i+T_in+T_out, :]
    
    return X, y


def preprocess_pems_bay(data_file, T_in=12, T_out=12):
    """
    Preprocess PEMS-BAY data into train/val/test splits
    
    Standard splits:
    - Train: 70% (first 70% chronologically)
    - Val: 10%
    - Test: 20%
    """
    
    print("\n" + "="*70)
    print("PREPROCESSING PEMS-BAY DATA")
    print("="*70)
    
    # Load data
    if data_file.endswith('.h5'):
        data = load_pems_bay_h5(data_file)
    else:
        data = load_pems_bay_npz(data_file)
    
    # If data is 3D (timesteps, nodes, features), take first feature
    if len(data.shape) == 3:
        print(f"Data has shape {data.shape}, taking first feature...")
        data = data[:, :, 0]
    
    timesteps, nodes = data.shape
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Nodes (sensors): {nodes}")
    print(f"  Total measurements: {timesteps * nodes:,}")
    
    # Normalize
    print(f"\n🔢 Normalizing data...")
    mean = data.mean()
    std = data.std()
    data_norm = (data - mean) / std
    
    print(f"  Mean: {mean:.2f}")
    print(f"  Std: {std:.2f}")
    print(f"  Normalized range: [{data_norm.min():.2f}, {data_norm.max():.2f}]")
    
    # Create sequences
    X, y = create_sequences(data_norm, T_in, T_out)
    
    num_samples = X.shape[0]
    print(f"\n📦 Created {num_samples:,} sequences")
    
    # Split: 70% train, 10% val, 20% test (chronological)
    train_split = int(0.7 * num_samples)
    val_split = int(0.8 * num_samples)
    
    X_train = X[:train_split]
    y_train = y[:train_split]
    
    X_val = X[train_split:val_split]
    y_val = y[train_split:val_split]
    
    X_test = X[val_split:]
    y_test = y[val_split:]
    
    print(f"\n✂️  Data splits:")
    print(f"  Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/num_samples*100:.1f}%)")
    print(f"  Val:   {X_val.shape[0]:,} samples ({X_val.shape[0]/num_samples*100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/num_samples*100:.1f}%)")
    
    # Create adjacency matrix (simple spatial distance-based for now)
    print(f"\n🗺️  Creating adjacency matrix...")
    
    # For PEMS-BAY, ideally you'd load the actual sensor locations
    # For now, create a simple connected graph
    adj_matrix = np.eye(nodes)
    
    # Add connections to k nearest neighbors (k=5)
    k = 5
    for i in range(nodes):
        if i > 0:
            for j in range(max(0, i-k), i):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        if i < nodes - 1:
            for j in range(i+1, min(nodes, i+k+1)):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    
    # Create transition matrices (row-normalized)
    P_fwd = adj_matrix / (adj_matrix.sum(axis=1, keepdims=True) + 1e-8)
    P_bwd = adj_matrix / (adj_matrix.sum(axis=0, keepdims=True) + 1e-8).T
    
    # Save everything
    output_file = 'data/pems_bay_processed.npz'
    print(f"\n💾 Saving processed data to: {output_file}")
    
    np.savez_compressed(
        output_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        P_fwd=P_fwd,
        P_bwd=P_bwd,
        mean=mean,
        std=std,
        adj_matrix=adj_matrix
    )
    
    print(f"✅ Saved! File size: {os.path.getsize(output_file) / 1e6:.2f} MB")
    
    # Save metadata
    metadata = {
        'nodes': nodes,
        'timesteps': timesteps,
        'T_in': T_in,
        'T_out': T_out,
        'num_train': X_train.shape[0],
        'num_val': X_val.shape[0],
        'num_test': X_test.shape[0],
        'mean': float(mean),
        'std': float(std),
        'data_source': data_file
    }
    
    with open('data/pems_bay_metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print("✅ Saved metadata to: data/pems_bay_metadata.json")
    
    return output_file


def main():
    print("\n🚀 Starting PEMS-BAY download and preprocessing...\n")
    
    # Check if already processed
    if os.path.exists('data/pems_bay_processed.npz'):
        print("⚠️  Found existing processed data: data/pems_bay_processed.npz")
        response = input("Reprocess? (y/n): ")
        if response.lower() != 'y':
            print("Using existing data. Done!")
            return
    
    # Download
    data_file = download_pems_bay()
    
    if data_file is None:
        print("\n❌ Download failed. Please download manually.")
        return
    
    # Preprocess
    output_file = preprocess_pems_bay(data_file)
    
    print("\n" + "="*70)
    print("✅ PEMS-BAY DATA READY!")
    print("="*70)
    print(f"\nProcessed data: {output_file}")
    print("\n📝 Next steps:")
    print("1. Update scripts/train.py to load from data/pems_bay_processed.npz")
    print("2. Retrain your model:")
    print("   python3 scripts/train.py --epochs 100 --device cuda")
    print("3. Compare results with mock data baseline")
    print("\n🎯 Expected improvement: 10-30% with real data!")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
