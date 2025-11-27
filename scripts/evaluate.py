"""
Evaluation script for DCRNN traffic forecasting model

This script:
- Loads trained model from checkpoint
- Runs inference on test set
- Computes metrics (MAE, RMSE, MAPE)
- Evaluates multi-horizon predictions (1, 3, 6, 12-step)
- Generates visualization plots
- Saves predictions for analysis

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.dcrnn import DCRNN
from src.dataset import load_data_from_splits
from src.metrics import compute_all_metrics, compute_horizon_metrics, MetricsTracker


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate DCRNN model')
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Directory to save evaluation results')
    
    # Model parameters (should match training)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--max_diffusion_step', type=int, default=2)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='auto')
    
    # Evaluation options
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 3, 6, 12],
                        help='Prediction horizons to evaluate (1-indexed)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    
    return parser.parse_args()


def get_device(device_arg):
    """Get computation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_arg == 'cpu':
        # Force CPU even if CUDA is available
        device = torch.device('cpu')
    elif device_arg == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device(device_arg)
    print(f"Using device: {device}")
    return device


def load_preprocessed_data():
    """
    Load preprocessed data (same as training)
    
    Returns:
        splits, P_fwd, P_bwd, mean, std, metadata
    """
    print("Loading preprocessed data...")
    
    # TODO: Replace with actual data loading
    num_samples = 1000
    T_in, T_out = 12, 12
    N = 325
    
    X = np.random.randn(num_samples, T_in, N)
    Y = np.random.randn(num_samples, T_out, N)
    
    train_idx = int(0.6 * num_samples)
    val_idx = int(0.8 * num_samples)
    
    splits = {
        'train': (X[:train_idx], Y[:train_idx]),
        'val': (X[train_idx:val_idx], Y[train_idx:val_idx]),
        'test': (X[val_idx:], Y[val_idx:])
    }
    
    A = np.random.rand(N, N)
    A = (A > 0.7).astype(float)
    
    D_out = A.sum(axis=1, keepdims=True)
    D_out[D_out == 0] = 1
    P_fwd = A / D_out
    
    D_in = A.sum(axis=0, keepdims=True)
    D_in[D_in == 0] = 1
    P_bwd = (A.T / D_in.T)
    
    mean = 50.0
    std = 10.0
    
    metadata = {
        'N': N,
        'T_in': T_in,
        'T_out': T_out,
        'num_test': len(splits['test'][0])
    }
    
    print(f"Test samples: {metadata['num_test']}")
    print(f"Nodes: {N}, T_in: {T_in}, T_out: {T_out}")
    
    return splits, P_fwd, P_bwd, mean, std, metadata


def load_model(checkpoint_path, model, device):
    """Load model from checkpoint"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    if 'metrics' in checkpoint:
        print(f"  Val MAE: {checkpoint['metrics']['mae']:.4f}")
    
    return model


def evaluate_model(model, dataloader, device, mean, std):
    """Run inference and compute metrics"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    tracker = MetricsTracker()
    
    print("\nRunning inference...")
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            P_fwd = batch.get('P_fwd', None)
            P_bwd = batch.get('P_bwd', None)
            
            if P_fwd is not None:
                P_fwd = P_fwd[0].to(device)
            if P_bwd is not None:
                P_bwd = P_bwd[0].to(device)
            
            # Predict
            pred = model(x, P_fwd=P_fwd, P_bwd=P_bwd, T_out=y.shape[1])
            
            # Store predictions
            all_predictions.append(pred.cpu())
            all_targets.append(y.cpu())
            
            # Update metrics
            tracker.update(pred, y, mean, std)
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Overall metrics
    overall_metrics = tracker.compute()
    
    return predictions, targets, overall_metrics


def evaluate_horizons(predictions, targets, horizons, mean, std):
    """Evaluate metrics at specific horizons"""
    print("\n" + "="*70)
    print("Multi-Horizon Evaluation")
    print("="*70)
    
    # Convert horizons from 1-indexed to 0-indexed
    horizon_indices = [h - 1 for h in horizons if h <= predictions.shape[1]]
    
    horizon_metrics = compute_horizon_metrics(
        predictions, targets, horizon_indices, mean, std
    )
    
    results = {}
    for step, metrics in horizon_metrics.items():
        print(f"\n{step}-step ahead:")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        results[f"{step}_step"] = metrics
    
    return results


def plot_predictions(predictions, targets, mean, std, output_dir, num_samples=3):
    """Generate visualization plots"""
    print("\nGenerating plots...")
    
    # Denormalize
    from src.metrics import denormalize
    pred_denorm = denormalize(predictions, mean, std).numpy()
    target_denorm = denormalize(targets, mean, std).numpy()
    
    # Select random samples
    indices = np.random.choice(len(predictions), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Pick a random node
        node_idx = np.random.randint(0, pred_denorm.shape[2])
        
        # Plot
        T = pred_denorm.shape[1]
        time_steps = np.arange(1, T + 1)
        
        ax.plot(time_steps, target_denorm[idx, :, node_idx, 0], 
                'b-', label='Ground Truth', linewidth=2)
        ax.plot(time_steps, pred_denorm[idx, :, node_idx, 0], 
                'r--', label='Prediction', linewidth=2)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Traffic Speed')
        ax.set_title(f'Sample {idx}, Node {node_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'predictions.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved plot: {plot_path}")
    plt.close()


def plot_horizon_metrics(horizon_results, output_dir):
    """Plot metrics vs prediction horizon"""
    print("Generating horizon metrics plot...")
    
    steps = sorted([int(k.split('_')[0]) for k in horizon_results.keys()])
    mae_values = [horizon_results[f"{s}_step"]['mae'] for s in steps]
    rmse_values = [horizon_results[f"{s}_step"]['rmse'] for s in steps]
    mape_values = [horizon_results[f"{s}_step"]['mape'] for s in steps]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # MAE
    axes[0].plot(steps, mae_values, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Prediction Horizon (steps)')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('MAE vs Horizon')
    axes[0].grid(True, alpha=0.3)
    
    # RMSE
    axes[1].plot(steps, rmse_values, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Prediction Horizon (steps)')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE vs Horizon')
    axes[1].grid(True, alpha=0.3)
    
    # MAPE
    axes[2].plot(steps, mape_values, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Prediction Horizon (steps)')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].set_title('MAPE vs Horizon')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'horizon_metrics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved plot: {plot_path}")
    plt.close()


def save_results(predictions, targets, overall_metrics, horizon_results, output_dir):
    """Save evaluation results"""
    print("\nSaving results...")
    
    # Save metrics as JSON
    results = {
        'overall': overall_metrics,
        'horizons': horizon_results
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved metrics: {metrics_path}")
    
    # Save predictions as numpy
    pred_path = os.path.join(output_dir, 'predictions.npy')
    target_path = os.path.join(output_dir, 'targets.npy')
    
    np.save(pred_path, predictions.numpy())
    np.save(target_path, targets.numpy())
    print(f"  ✓ Saved predictions: {pred_path}")
    print(f"  ✓ Saved targets: {target_path}")


def evaluate(args):
    """Main evaluation function"""
    print("="*70)
    print("DCRNN Evaluation")
    print("="*70)
    
    # Device
    device = get_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    splits, P_fwd, P_bwd, mean, std, metadata = load_preprocessed_data()
    
    # Create test dataloader
    dataloaders = load_data_from_splits(
        splits, P_fwd=P_fwd, P_bwd=P_bwd, batch_size=args.batch_size
    )
    test_loader = dataloaders['test']
    
    # Initialize model
    print("\nInitializing model...")
    model = DCRNN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        max_diffusion_step=args.max_diffusion_step
    ).to(device)
    
    # Load checkpoint
    model = load_model(args.checkpoint, model, device)
    
    # Evaluate
    predictions, targets, overall_metrics = evaluate_model(
        model, test_loader, device, mean, std
    )
    
    # Print overall metrics
    print("\n" + "="*70)
    print("Overall Test Set Metrics")
    print("="*70)
    print(f"MAE:  {overall_metrics['mae']:.4f}")
    print(f"RMSE: {overall_metrics['rmse']:.4f}")
    print(f"MAPE: {overall_metrics['mape']:.2f}%")
    
    # Horizon evaluation
    horizon_results = evaluate_horizons(
        predictions, targets, args.horizons, mean, std
    )
    
    # Save results
    if args.save_predictions:
        save_results(predictions, targets, overall_metrics, horizon_results, args.output_dir)
    
    # Generate plots
    if args.plot:
        plot_predictions(predictions, targets, mean, std, args.output_dir)
        plot_horizon_metrics(horizon_results, args.output_dir)
    
    print("\n" + "="*70)
    print("Evaluation Complete")
    print("="*70)
    
    return overall_metrics, horizon_results


if __name__ == '__main__':
    args = get_args()
    evaluate(args)
