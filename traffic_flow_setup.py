# Modular data preprocessing utilities: z-score normalization, sequence creation, train-val-test split, and transition matrix computation for future full-scale DCRNN implementation

import numpy as np
import pandas as pd
import torch
import warnings
from torch_geometric.data import Data # Used for defining graph objects (future use)

# Suppress minor warnings often generated during dependency loading
warnings.filterwarnings("ignore") 

# --- 1. Environment Setup Check ---
print("--- 1. Environment Setup Check ---")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"PyTorch Version: {torch.__version__}")
try:
    from torch_geometric.nn import GCNConv
    print("PyTorch Geometric (torch_geometric) is installed and ready.")
except ImportError:
    print("WARNING: PyTorch Geometric is not installed. Run: pip install torch_geometric")

# --- 2. Define File Paths and Sequence Constants ---

# File Paths (Replace with your actual paths later)
TRAFFIC_DATA_PATH = "data/PEMS-BAY.h5"
ADJACENCY_MATRIX_PATH = "data/PEMS-BAY.csv"

# Constants for Sequence Creation (Standard for short-term traffic forecasting)
INPUT_SEQUENCE_LENGTH = 12  # T_in: Number of historical steps to observe (e.g., 2 hours)
PREDICTION_HORIZON = 12     # T_out: Number of future steps to predict (e.g., next 2 hours)


# --- 3. Mock Data Loading Functions ---

def load_traffic_data(file_path):
    """
    Mocks loading the temporal traffic speed data from a sensor network.
    
    Returns a NumPy array of shape (T, N): T (time steps) x N (sensors).
    """
    print(f"\n[INFO] Loading temporal data from: {file_path} (Mocking 288 days, 325 sensors)...")
    
    # Dimensions: 288 time steps per day * 100 days = 28800 total time steps (T)
    T, N = 288 * 100, 325 
    
    # Create mock traffic speed data with slight periodicity
    time_index = np.arange(T)
    base_speed = 50 + 15 * np.sin(time_index / 24) * np.cos(time_index / 288)
    noise = np.random.normal(0, 5, (T, N))
    
    data = (base_speed[:, np.newaxis] + noise)
    data = np.clip(data, 0, 70) # Speeds are non-negative

    print(f"[SHAPE] Raw Traffic Data: {data.shape}")
    return data

def load_adjacency_matrix(file_path):
    """
    Mocks loading the spatial connectivity/distance data (Adjacency Matrix W).
    
    Returns a NumPy array of shape (N, N).
    """
    print(f"[INFO] Loading adjacency matrix from: {file_path} (Mocking 325x325 matrix)...")
    N = 325
    
    # Create a mock weighted adjacency matrix A (N x N)
    A = np.eye(N)
    
    # Add some random directed connections
    num_edges = 2000
    indices = np.random.randint(0, N, size=(num_edges, 2))
    weights = np.random.uniform(0.1, 0.9, num_edges)
    
    for (i, j), w in zip(indices, weights):
        if i != j:
            A[i, j] = w
            
    print(f"[SHAPE] Adjacency Matrix W: {A.shape}")
    return A

# --- 4. Data Preprocessing Functions ---

def z_score_normalize(data):
    """
    Applies Z-score normalization (Standardization) to the traffic data.
    
    Args: data (np.ndarray): Traffic data of shape (T, N).
    Returns: tuple: Normalized data (np.ndarray), mean (float), std (float).
    """
    print("[INFO] Normalizing data using Z-score...")
    # Calculate mean and standard deviation for each sensor (column)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1.0e-5 
    
    normalized_data = (data - mean) / std
    
    print(f"[INFO] Data Normalized. Mean shape: {mean.shape}, Std shape: {std.shape}")
    return normalized_data, mean, std

def create_transition_matrix(A):
    """
    Calculates the Diffusion Convolutional Recurrent Neural Network (DCRNN) 
    transition matrices: $P_{fwd}$ and $P_{bwd}$.

    $P_{fwd} = D_{O}^{-1}W$ (Forward Diffusion)
    $P_{bwd} = D_{I}^{-1}W^{T}$ (Backward Diffusion)
    
    Reference: Li, Y. et al. (2018). Diffusion Convolutional Recurrent Neural Network: 
    Data-Driven Traffic Forecasting. ICLR.
    
    Args: A (np.ndarray): Weighted Adjacency Matrix W (N, N).
    Returns: tuple: P_fwd (np.ndarray), P_bwd (np.ndarray).
    """
    print("[INFO] Creating DCRNN Diffusion Transition Matrices...")
    
    # Calculate Out-Degree Matrix Inverse (D_O_inv)
    out_degree = np.sum(A, axis=1)
    out_degree[out_degree == 0] = 1 # Avoid division by zero
    D_O_inv = np.diag(1.0 / out_degree)
    
    # Calculate In-Degree Matrix Inverse (D_I_inv)
    in_degree = np.sum(A, axis=0)
    in_degree[in_degree == 0] = 1 # Avoid division by zero
    D_I_inv = np.diag(1.0 / in_degree)
    
    # Calculate Transition Matrices
    P_fwd = D_O_inv @ A
    P_bwd = D_I_inv @ A.T
    
    print(f"[SHAPE] P_fwd: {P_fwd.shape}, P_bwd: {P_bwd.shape}")
    return P_fwd, P_bwd

def create_sequences(data, seq_len, pred_len):
    """
    Creates input (X) and target (Y) sequences using a sliding window.
    
    Args:
        data (np.ndarray): Normalized traffic data (T, N).
        seq_len (int): Input sequence length (T_in).
        pred_len (int): Output prediction length (T_out).
        
    Returns:
        tuple: X_sequences (Num_Samples, T_in, N), Y_sequences (Num_Samples, T_out, N).
    """
    print(f"[INFO] Creating sequences (X: {seq_len}, Y: {pred_len})...")
    T, N = data.shape
    X, Y = [], []
    
    # Iterate through data to create sample windows
    for i in range(T - seq_len - pred_len + 1):
        # X: Input sequence (T_in steps)
        x_sequence = data[i : i + seq_len, :]
        
        # Y: Target sequence (T_out steps, starting after X ends)
        y_sequence = data[i + seq_len : i + seq_len + pred_len, :]
        
        X.append(x_sequence)
        Y.append(y_sequence)
        
    X_sequences = np.stack(X)
    Y_sequences = np.stack(Y)
    
    print(f"[SHAPE] X Sequences (Input): {X_sequences.shape}")
    print(f"[SHAPE] Y Sequences (Target): {Y_sequences.shape}")
    return X_sequences, Y_sequences


def train_val_test_split(X, Y, train_ratio=0.6, val_ratio=0.2):
    """
    Chronologically split sequence datasets into train/validation/test sets.

    Args:
        X (np.ndarray): Input sequences, shape (S, T_in, N)
        Y (np.ndarray): Target sequences, shape (S, T_out, N)
        train_ratio (float): Fraction of samples for training (default 0.6)
        val_ratio (float): Fraction of samples for validation (default 0.2)

    Returns:
        dict: {
            'train': (X_train, Y_train),
            'val': (X_val, Y_val),
            'test': (X_test, Y_test)
        }

    Notes:
        The split is chronological: earlier samples -> train, middle -> val, later -> test.
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
    S = X.shape[0]
    if S == 0:
        return {'train': (X, Y), 'val': (X, Y), 'test': (X, Y)}

    # Compute split indices
    train_end = int(np.floor(S * train_ratio))
    val_end = int(np.floor(S * (train_ratio + val_ratio)))

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    print(f"[SPLIT] Total samples: {S}")
    print(f"[SPLIT] Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    return {
        'train': (X_train, Y_train),
        'val': (X_val, Y_val),
        'test': (X_test, Y_test)
    }

# --- 5. Execution of Data Load and Preprocessing ---

if __name__ == "__main__":
    
    # Load Mock Data
    raw_traffic_data = load_traffic_data(TRAFFIC_DATA_PATH)
    raw_adj_matrix = load_adjacency_matrix(ADJACENCY_MATRIX_PATH)
    
    print("\n" + "="*50)
    print("BEGINNING DATA PREPROCESSING (Week 1 Focus)")
    print("="*50)
    
    # 1. Normalization
    normalized_data, data_mean, data_std = z_score_normalize(raw_traffic_data)
    
    # 2. Graph Construction
    P_fwd, P_bwd = create_transition_matrix(raw_adj_matrix)
    
    # 3. Sequence Creation
    X_sequences, Y_sequences = create_sequences(
        normalized_data, 
        seq_len=INPUT_SEQUENCE_LENGTH, 
        pred_len=PREDICTION_HORIZON
    )
    
    # 4. Chronological Train/Val/Test Split
    splits = train_val_test_split(X_sequences, Y_sequences, train_ratio=0.6, val_ratio=0.2)
    X_train, Y_train = splits['train']
    X_val, Y_val = splits['val']
    X_test, Y_test = splits['test']

    print(f"\n[SHAPES] X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"[SHAPES] X_val: {X_val.shape}, Y_val: {Y_val.shape}")
    print(f"[SHAPES] X_test: {X_test.shape}, Y_test: {Y_test.shape}")
    
    print("\n" + "="*50)
    print("DATA PREPROCESSING COMPLETE")
    print("="*50)
    print("Next step: Split the data into training/validation/test sets and implement the DCRNN model architecture.")
