"""
Training script for DCRNN traffic forecasting model

This script handles:
- Model initialization and training
- Validation with early stopping
- Model checkpointing (save best model)
- Progress tracking and logging
- Learning rate scheduling

Usage:
    python scripts/train.py --data_dir data/ --epochs 100 --batch_size 64
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.dcrnn import DCRNN
from src.dataset import create_dataloaders, load_data_from_splits
from src.metrics import compute_all_metrics, MetricsTracker


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DCRNN for traffic forecasting')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Directory containing preprocessed data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='Directory to save model checkpoints')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden state dimension')
    parser.add_argument('--output_dim', type=int, default=1,
                        help='Output feature dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of DCGRU layers')
    parser.add_argument('--max_diffusion_step', type=int, default=2,
                        help='K-hop diffusion steps')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--clip_grad', type=float, default=5.0,
                        help='Gradient clipping value (0 to disable)')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                        help='Minimum improvement for early stopping')
    
    # Learning rate scheduler
    parser.add_argument('--lr_decay', action='store_true',
                        help='Use learning rate decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='LR decay rate')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='LR decay patience')
    
    # Loss function
    parser.add_argument('--loss', type=str, default='mae', choices=['mae', 'mse'],
                        help='Loss function: mae or mse')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validate every N epochs')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device(device_arg):
    """Get computation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    print(f"Using device: {device}")
    return device


def load_preprocessed_data():
    """
    Load preprocessed data from traffic_flow_setup.py
    
    This is a placeholder that should be replaced with actual loading logic.
    For now, it generates mock data matching the expected format.
    
    Returns:
        splits: dict with 'train', 'val', 'test' tuples
        P_fwd, P_bwd: Transition matrices
        mean, std: Normalization parameters
        metadata: dict with data dimensions
    """
    print("Loading preprocessed data...")
    
    # TODO: Replace with actual data loading from saved files
    # For now, generate mock data
    num_samples = 1000
    T_in, T_out = 12, 12
    N = 325  # PEMS-BAY has 325 sensors
    
    X = np.random.randn(num_samples, T_in, N)
    Y = np.random.randn(num_samples, T_out, N)
    
    # Split chronologically (60/20/20)
    train_idx = int(0.6 * num_samples)
    val_idx = int(0.8 * num_samples)
    
    splits = {
        'train': (X[:train_idx], Y[:train_idx]),
        'val': (X[train_idx:val_idx], Y[train_idx:val_idx]),
        'test': (X[val_idx:], Y[val_idx:])
    }
    
    # Mock transition matrices
    A = np.random.rand(N, N)
    A = (A > 0.7).astype(float)  # Sparse
    
    D_out = A.sum(axis=1, keepdims=True)
    D_out[D_out == 0] = 1
    P_fwd = A / D_out
    
    D_in = A.sum(axis=0, keepdims=True)
    D_in[D_in == 0] = 1
    P_bwd = (A.T / D_in.T)
    
    # Normalization params
    mean = 50.0
    std = 10.0
    
    metadata = {
        'N': N,
        'T_in': T_in,
        'T_out': T_out,
        'num_train': len(splits['train'][0]),
        'num_val': len(splits['val'][0]),
        'num_test': len(splits['test'][0])
    }
    
    print(f"Loaded {metadata['num_train']} train, {metadata['num_val']} val, {metadata['num_test']} test samples")
    print(f"Nodes: {N}, T_in: {T_in}, T_out: {T_out}")
    
    return splits, P_fwd, P_bwd, mean, std, metadata


def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad, log_interval):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        x = batch['x'].to(device)  # (batch, T_in, N, features)
        y = batch['y'].to(device)  # (batch, T_out, N, features)
        P_fwd = batch.get('P_fwd', None)
        P_bwd = batch.get('P_bwd', None)
        
        if P_fwd is not None:
            P_fwd = P_fwd[0].to(device)  # Remove batch dimension (shared matrix)
        if P_bwd is not None:
            P_bwd = P_bwd[0].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(x, P_fwd=P_fwd, P_bwd=P_bwd, T_out=y.shape[1])
        
        # Compute loss
        loss = criterion(pred, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / batch_count
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
    
    return total_loss / batch_count


def validate(model, dataloader, criterion, device, mean, std):
    """Validate the model"""
    model.eval()
    tracker = MetricsTracker()
    total_loss = 0.0
    
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
            
            # Forward pass
            pred = model(x, P_fwd=P_fwd, P_bwd=P_bwd, T_out=y.shape[1])
            
            # Loss
            loss = criterion(pred, y)
            total_loss += loss.item()
            
            # Metrics (with denormalization)
            tracker.update(pred, y, mean, std)
    
    avg_loss = total_loss / len(dataloader)
    metrics = tracker.compute()
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, val_loss, metrics, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"  ✓ Saved checkpoint: {filepath}")


def train(args):
    """Main training function"""
    print("="*70)
    print("DCRNN Training")
    print("="*70)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = get_device(args.device)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    splits, P_fwd, P_bwd, mean, std, metadata = load_preprocessed_data()
    
    # Create dataloaders
    dataloaders = load_data_from_splits(
        splits, 
        P_fwd=P_fwd, 
        P_bwd=P_bwd, 
        batch_size=args.batch_size
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = DCRNN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        max_diffusion_step=args.max_diffusion_step
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Loss function
    if args.loss == 'mae':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    print(f"Loss function: {args.loss.upper()}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = None
    if args.lr_decay:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=args.lr_decay_rate,
            patience=args.lr_patience,
            verbose=True
        )
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_mape': []
    }
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(
            model, 
            dataloaders['train'], 
            optimizer, 
            criterion, 
            device,
            args.clip_grad,
            args.log_interval
        )
        
        print(f"  Train Loss: {train_loss:.4f}")
        training_history['train_loss'].append(train_loss)
        
        # Validate
        if epoch % args.val_interval == 0:
            val_loss, val_metrics = validate(
                model, 
                dataloaders['val'], 
                criterion, 
                device, 
                mean, 
                std
            )
            
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val MAE: {val_metrics['mae']:.4f}, "
                  f"RMSE: {val_metrics['rmse']:.4f}, "
                  f"MAPE: {val_metrics['mape']:.2f}%")
            
            training_history['val_loss'].append(val_loss)
            training_history['val_mae'].append(val_metrics['mae'])
            training_history['val_rmse'].append(val_metrics['rmse'])
            training_history['val_mape'].append(val_metrics['mape'])
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # Check for improvement
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, best_path)
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epoch(s)")
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break
        
        epoch_time = time.time() - epoch_start
        print(f"  Epoch time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*70)
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, final_path)
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"\n✓ Saved training history: {history_path}")
    
    return model, training_history


if __name__ == '__main__':
    args = get_args()
    train(args)
