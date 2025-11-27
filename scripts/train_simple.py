"""
SIMPLIFIED DCRNN TRAINING - For Debugging

This trains a MINIMAL 1-layer DCRNN to verify:
1. Teacher forcing works
2. Model can learn (not stuck at mean)
3. Training setup is correct

If this fails, the problem is fundamental.
If this works, we need to fix the 2-layer model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm

from models.dcrnn import DCRNN
from src.dataset import TrafficDataset
from src.metrics import MetricsTracker


def train_simple():
    """Train a minimal 1-layer DCRNN"""
    
    print("="*70)
    print("SIMPLIFIED DCRNN TRAINING (1-LAYER)")
    print("="*70)
    print("Testing if minimal model can learn...\n")
    
    device = torch.device('cpu')
    
    # Load data
    print("Loading data...")
    data = np.load('data/pems_bay_processed.npz')
    X_train = data['X_train'][:600]  # Use full training data
    y_train = data['y_train'][:600]
    X_val = data['X_val']
    y_val = data['y_val']
    P_fwd = torch.FloatTensor(data['P_fwd'])
    P_bwd = torch.FloatTensor(data['P_bwd'])
    mean = data['mean']
    std = data['std']
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create dataloaders
    train_dataset = TrafficDataset(X_train, y_train, P_fwd, P_bwd)
    val_dataset = TrafficDataset(X_val, y_val, P_fwd, P_bwd)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize MINIMAL model
    print("\nInitializing MINIMAL 1-layer model...")
    model = DCRNN(
        input_dim=1,
        hidden_dim=32,  # SMALLER hidden dim
        output_dim=1,
        num_layers=1,   # ONLY 1 LAYER
        max_diffusion_step=2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # HIGHER LR
    criterion = nn.L1Loss()
    
    # Training
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    history = []
    best_val_mae = float('inf')
    
    for epoch in range(1, 31):  # 30 epochs max
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            P_fwd_b = batch['P_fwd'][0].to(device)
            P_bwd_b = batch['P_bwd'][0].to(device)
            
            optimizer.zero_grad()
            pred = model(x, P_fwd=P_fwd_b, P_bwd=P_bwd_b, T_out=12, 
                        labels=y, training=True)  # TEACHER FORCING
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        tracker = MetricsTracker()
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                P_fwd_b = batch['P_fwd'][0].to(device)
                P_bwd_b = batch['P_bwd'][0].to(device)
                
                pred = model(x, P_fwd=P_fwd_b, P_bwd=P_bwd_b, T_out=12, 
                            training=False)  # NO TEACHER FORCING
                loss = criterion(pred, y)
                val_loss += loss.item()
                
                tracker.update(pred, y, mean, std)
        
        val_loss /= len(val_loader)
        metrics = tracker.compute()
        val_mae = metrics['mae']
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mae': val_mae
        })
        
        # Print progress
        print(f"Epoch {epoch:2d}/30 | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
        
        # Check if improving
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            improvement = (7.99 - val_mae) / 7.99 * 100
            print(f"  → NEW BEST! Improvement: {improvement:.1f}% vs baseline (7.99)")
        
        # Early stopping if stuck
        if epoch > 10 and val_mae > 7.9:
            print(f"\n⚠️  STUCK at epoch {epoch}! Val MAE still {val_mae:.4f}")
            print("  Model is NOT learning - still predicting mean")
            break
        
        if epoch > 15 and val_mae < 5.0:
            print(f"\n✅ SUCCESS at epoch {epoch}! Val MAE: {val_mae:.4f}")
            print("  Model IS learning - breaking through mean prediction!")
            break
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Best Val MAE: {best_val_mae:.4f}")
    print(f"Baseline MAE: 7.99")
    
    if best_val_mae < 7.9:
        improvement = (7.99 - best_val_mae) / 7.99 * 100
        print(f"Improvement: {improvement:.1f}%")
        print("\n✅ MODEL CAN LEARN! The setup works.")
        print("   Problem might be in 2-layer configuration.")
    else:
        print(f"\n❌ MODEL STUCK AT MEAN! Even 1-layer doesn't work.")
        print("   Fundamental problem in:")
        print("   1. Diffusion convolution implementation")
        print("   2. Teacher forcing implementation")
        print("   3. Data normalization")
        print("   4. Loss computation")
    
    # Save history
    os.makedirs('checkpoints_simple', exist_ok=True)
    with open('checkpoints_simple/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nHistory saved to: checkpoints_simple/history.json")


if __name__ == '__main__':
    train_simple()
