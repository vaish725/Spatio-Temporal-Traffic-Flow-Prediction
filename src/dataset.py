"""
PyTorch Dataset and DataLoader utilities for traffic forecasting

This module provides:
1. TrafficDataset: Wraps preprocessed (X, Y) sequences
2. DataLoader factory functions for train/val/test with chronological ordering
3. Helper functions for converting numpy arrays to PyTorch tensors
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    """
    PyTorch Dataset for traffic forecasting sequences
    
    Wraps preprocessed input (X) and target (Y) sequences for batching.
    Sequences should be chronologically ordered (no shuffling in DataLoader).
    
    Args:
        X (np.ndarray): Input sequences, shape (num_samples, T_in, N, features)
                        or (num_samples, T_in, N) for single feature
        Y (np.ndarray): Target sequences, shape (num_samples, T_out, N, features)
                        or (num_samples, T_out, N) for single feature
        P_fwd (np.ndarray, optional): Forward transition matrix (N, N)
        P_bwd (np.ndarray, optional): Backward transition matrix (N, N)
        transform (callable, optional): Optional transform to apply to samples
        
    Returns:
        dict with keys:
            'x': torch.FloatTensor of shape (T_in, N, features)
            'y': torch.FloatTensor of shape (T_out, N, features)
            'P_fwd': torch.FloatTensor of shape (N, N) if provided
            'P_bwd': torch.FloatTensor of shape (N, N) if provided
    """
    
    def __init__(self, X, Y, P_fwd=None, P_bwd=None, transform=None):
        """Initialize dataset with sequences and optional transition matrices"""
        assert len(X) == len(Y), f"X and Y must have same length, got {len(X)} and {len(Y)}"
        
        # Convert to numpy if not already
        self.X = np.array(X) if not isinstance(X, np.ndarray) else X
        self.Y = np.array(Y) if not isinstance(Y, np.ndarray) else Y
        
        # Ensure 4D: (num_samples, T, N, features)
        if self.X.ndim == 3:
            self.X = self.X[..., np.newaxis]  # Add feature dimension
        if self.Y.ndim == 3:
            self.Y = self.Y[..., np.newaxis]
            
        self.num_samples, self.T_in, self.N, self.input_dim = self.X.shape
        _, self.T_out, _, self.output_dim = self.Y.shape
        
        # Store transition matrices
        self.P_fwd = P_fwd
        self.P_bwd = P_bwd
        
        self.transform = transform
        
        # Convert transition matrices to tensors once (shared across all samples)
        if self.P_fwd is not None:
            self.P_fwd_tensor = torch.FloatTensor(self.P_fwd)
        else:
            self.P_fwd_tensor = None
            
        if self.P_bwd is not None:
            self.P_bwd_tensor = torch.FloatTensor(self.P_bwd)
        else:
            self.P_bwd_tensor = None
    
    def __len__(self):
        """Return number of samples"""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: {'x': tensor, 'y': tensor, 'P_fwd': tensor, 'P_bwd': tensor}
        """
        # Get sequences for this sample
        x = torch.FloatTensor(self.X[idx])  # (T_in, N, input_dim)
        y = torch.FloatTensor(self.Y[idx])  # (T_out, N, output_dim)
        
        # Apply optional transform
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        
        # Build sample dict
        sample = {
            'x': x,
            'y': y,
        }
        
        # Add transition matrices if available
        if self.P_fwd_tensor is not None:
            sample['P_fwd'] = self.P_fwd_tensor
        if self.P_bwd_tensor is not None:
            sample['P_bwd'] = self.P_bwd_tensor
            
        return sample
    
    def get_dims(self):
        """Return dataset dimensions"""
        return {
            'num_samples': self.num_samples,
            'T_in': self.T_in,
            'T_out': self.T_out,
            'N': self.N,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }


def create_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                       P_fwd=None, P_bwd=None,
                       batch_size=64, num_workers=0):
    """
    Create train, validation, and test DataLoaders
    
    Args:
        X_train, Y_train: Training sequences
        X_val, Y_val: Validation sequences
        X_test, Y_test: Test sequences
        P_fwd, P_bwd: Transition matrices (optional)
        batch_size (int): Batch size for training (default: 64)
        num_workers (int): Number of workers for data loading (default: 0)
        
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    # Create datasets
    train_dataset = TrafficDataset(X_train, Y_train, P_fwd, P_bwd)
    val_dataset = TrafficDataset(X_val, Y_val, P_fwd, P_bwd)
    test_dataset = TrafficDataset(X_test, Y_test, P_fwd, P_bwd)
    
    # Create DataLoaders
    # IMPORTANT: shuffle=False to maintain chronological order
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep chronological order
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def load_data_from_splits(splits_dict, P_fwd=None, P_bwd=None, batch_size=64):
    """
    Convenience function to create DataLoaders from split dictionary
    
    Args:
        splits_dict (dict): Output from train_val_test_split() with keys:
                           'train': (X_train, Y_train)
                           'val': (X_val, Y_val)
                           'test': (X_test, Y_test)
        P_fwd, P_bwd: Transition matrices
        batch_size (int): Batch size
        
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    X_train, Y_train = splits_dict['train']
    X_val, Y_val = splits_dict['val']
    X_test, Y_test = splits_dict['test']
    
    return create_dataloaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        P_fwd=P_fwd, P_bwd=P_bwd, batch_size=batch_size
    )


def test_dataset():
    """Test the TrafficDataset and DataLoader"""
    print("Testing TrafficDataset...")
    
    # Create mock data
    num_samples = 100
    T_in, T_out = 12, 12
    N = 10
    
    X = np.random.randn(num_samples, T_in, N)
    Y = np.random.randn(num_samples, T_out, N)
    
    # Create transition matrices
    P_fwd = np.random.rand(N, N)
    P_bwd = np.random.rand(N, N)
    
    # Create dataset
    dataset = TrafficDataset(X, Y, P_fwd, P_bwd)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset dimensions: {dataset.get_dims()}")
    
    # Test __getitem__
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"x shape: {sample['x'].shape}")
    print(f"y shape: {sample['y'].shape}")
    print(f"P_fwd shape: {sample['P_fwd'].shape}")
    print(f"P_bwd shape: {sample['P_bwd'].shape}")
    
    # Test DataLoader
    print("\nTesting DataLoader...")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    batch = next(iter(loader))
    print(f"Batch x shape: {batch['x'].shape}")
    print(f"Batch y shape: {batch['y'].shape}")
    print(f"Batch P_fwd shape: {batch['P_fwd'].shape}")
    
    # Test split loading
    print("\nTesting create_dataloaders...")
    splits = {
        'train': (X[:60], Y[:60]),
        'val': (X[60:80], Y[60:80]),
        'test': (X[80:], Y[80:])
    }
    
    loaders = load_data_from_splits(splits, P_fwd, P_bwd, batch_size=16)
    
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    print("\n✓ All dataset tests passed!")


if __name__ == '__main__':
    test_dataset()
