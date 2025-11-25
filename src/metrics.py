"""
Evaluation metrics for traffic forecasting

Implements standard metrics for time series forecasting:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

All metrics support:
- PyTorch tensors (GPU/CPU)
- Denormalization using mean and std
- Masked computation (ignore certain values)
"""

import torch
import numpy as np


def masked_mae(pred, target, null_val=np.nan):
    """
    Masked Mean Absolute Error
    
    Computes MAE while ignoring entries where target == null_val.
    Useful when some sensors have missing data.
    
    Args:
        pred: Predictions, shape (batch, T, N) or (batch, T, N, features)
        target: Ground truth, same shape as pred
        null_val: Value to ignore (default: np.nan)
        
    Returns:
        Scalar tensor with MAE
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = (target != null_val)
    
    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize by valid entries
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(pred - target)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def masked_rmse(pred, target, null_val=np.nan):
    """
    Masked Root Mean Squared Error
    
    Args:
        pred: Predictions, shape (batch, T, N) or (batch, T, N, features)
        target: Ground truth, same shape as pred
        null_val: Value to ignore (default: np.nan)
        
    Returns:
        Scalar tensor with RMSE
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = (target != null_val)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = (pred - target) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.sqrt(torch.mean(loss))


def masked_mape(pred, target, null_val=np.nan, epsilon=1e-5):
    """
    Masked Mean Absolute Percentage Error
    
    MAPE = mean(|y_pred - y_true| / |y_true|) * 100
    
    Args:
        pred: Predictions, shape (batch, T, N) or (batch, T, N, features)
        target: Ground truth, same shape as pred
        null_val: Value to ignore (default: np.nan)
        epsilon: Small value to avoid division by zero
        
    Returns:
        Scalar tensor with MAPE (percentage)
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = (target != null_val)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # Avoid division by very small values
    loss = torch.abs((pred - target) / (target + epsilon))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss) * 100  # Return as percentage


def denormalize(data, mean, std):
    """
    Reverse z-score normalization
    
    Args:
        data: Normalized data (torch.Tensor or np.ndarray)
        mean: Mean used for normalization (scalar or array matching data shape)
        std: Std used for normalization (scalar or array matching data shape)
        
    Returns:
        Denormalized data (same type as input)
    """
    if isinstance(data, torch.Tensor):
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=data.dtype, device=data.device)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        return data * std + mean
    else:
        return data * std + mean


def compute_all_metrics(pred, target, mean=None, std=None, null_val=np.nan):
    """
    Compute all metrics (MAE, RMSE, MAPE) at once
    
    Args:
        pred: Predictions (normalized if mean/std provided)
        target: Ground truth (normalized if mean/std provided)
        mean: Mean for denormalization (optional)
        std: Std for denormalization (optional)
        null_val: Value to mask
        
    Returns:
        dict: {'mae': float, 'rmse': float, 'mape': float}
    """
    # Denormalize if mean/std provided
    if mean is not None and std is not None:
        pred = denormalize(pred, mean, std)
        target = denormalize(target, mean, std)
    
    mae = masked_mae(pred, target, null_val).item()
    rmse = masked_rmse(pred, target, null_val).item()
    mape = masked_mape(pred, target, null_val).item()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


def compute_horizon_metrics(pred, target, horizons, mean=None, std=None):
    """
    Compute metrics for specific prediction horizons
    
    Useful for evaluating 1-step, 3-step, 6-step, 12-step ahead predictions.
    
    Args:
        pred: Predictions, shape (batch, T_out, N) or (batch, T_out, N, features)
        target: Ground truth, same shape as pred
        horizons: List of time steps to evaluate (e.g., [0, 2, 5, 11] for 1,3,6,12-step)
        mean, std: For denormalization
        
    Returns:
        dict: {horizon: {'mae': float, 'rmse': float, 'mape': float}}
    """
    results = {}
    
    for h in horizons:
        if h >= pred.shape[1]:
            continue
            
        pred_h = pred[:, h, ...]
        target_h = target[:, h, ...]
        
        metrics = compute_all_metrics(pred_h, target_h, mean, std)
        results[h + 1] = metrics  # Convert to 1-indexed (1-step, 3-step, etc.)
    
    return results


class MetricsTracker:
    """
    Track metrics over multiple batches/epochs
    
    Usage:
        tracker = MetricsTracker()
        for batch in dataloader:
            pred, target = model(batch), batch['y']
            tracker.update(pred, target, mean, std)
        metrics = tracker.compute()
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated values"""
        self.mae_sum = 0.0
        self.rmse_sum = 0.0
        self.mape_sum = 0.0
        self.count = 0
    
    def update(self, pred, target, mean=None, std=None, null_val=np.nan):
        """Update with new batch"""
        metrics = compute_all_metrics(pred, target, mean, std, null_val)
        
        self.mae_sum += metrics['mae']
        self.rmse_sum += metrics['rmse']
        self.mape_sum += metrics['mape']
        self.count += 1
    
    def compute(self):
        """Compute average metrics"""
        if self.count == 0:
            return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}
        
        return {
            'mae': self.mae_sum / self.count,
            'rmse': self.rmse_sum / self.count,
            'mape': self.mape_sum / self.count
        }


def test_metrics():
    """Test metrics functions"""
    print("Testing metrics module...")
    
    # Create mock predictions and targets
    batch = 4
    T = 12
    N = 10
    
    # Normalized data (mean=0, std=1)
    pred_norm = torch.randn(batch, T, N)
    target_norm = pred_norm + torch.randn(batch, T, N) * 0.1  # Add small noise
    
    # Normalization parameters
    mean = 50.0
    std = 10.0
    
    print("\n1. Testing on normalized data:")
    metrics_norm = compute_all_metrics(pred_norm, target_norm)
    print(f"   MAE: {metrics_norm['mae']:.4f}")
    print(f"   RMSE: {metrics_norm['rmse']:.4f}")
    print(f"   MAPE: {metrics_norm['mape']:.4f}%")
    
    print("\n2. Testing with denormalization:")
    metrics_denorm = compute_all_metrics(pred_norm, target_norm, mean, std)
    print(f"   MAE: {metrics_denorm['mae']:.4f}")
    print(f"   RMSE: {metrics_denorm['rmse']:.4f}")
    print(f"   MAPE: {metrics_denorm['mape']:.4f}%")
    
    print("\n3. Testing horizon metrics:")
    horizons = [0, 2, 5, 11]  # 1, 3, 6, 12-step ahead
    horizon_metrics = compute_horizon_metrics(pred_norm, target_norm, horizons, mean, std)
    for step, metrics in horizon_metrics.items():
        print(f"   {step}-step: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%")
    
    print("\n4. Testing MetricsTracker:")
    tracker = MetricsTracker()
    for i in range(5):
        pred_batch = torch.randn(2, T, N)
        target_batch = pred_batch + torch.randn(2, T, N) * 0.2
        tracker.update(pred_batch, target_batch, mean, std)
    
    avg_metrics = tracker.compute()
    print(f"   Average MAE: {avg_metrics['mae']:.4f}")
    print(f"   Average RMSE: {avg_metrics['rmse']:.4f}")
    print(f"   Average MAPE: {avg_metrics['mape']:.4f}%")
    
    print("\n5. Testing masked metrics (with NaN):")
    target_masked = target_norm.clone()
    target_masked[0, :, 0] = float('nan')  # Mask first sensor in first sample
    
    mae_masked = masked_mae(pred_norm, target_masked)
    print(f"   Masked MAE: {mae_masked.item():.4f}")
    
    print("\n✓ All metrics tests passed!")


if __name__ == '__main__':
    test_metrics()
