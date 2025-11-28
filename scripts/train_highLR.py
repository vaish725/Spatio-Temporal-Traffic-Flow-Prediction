"""
QUICK FIX: Train with higher learning rate to escape local minimum
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm

from models.dcrnn import DCRNN
from src.dataset import TrafficDataset
from src.metrics import MetricsTracker


print("="*70)
print("QUICK FIX: High Learning Rate Training")
print("="*70)
print("\nFixes applied:")
print("  1. Learning rate: 0.001 → 0.01 (10x higher)")
print("  2. Cosine annealing LR schedule")
print("  3. No early stopping for first 20 epochs")
print("  4. Same 2-layer, 64-dim architecture")
print()

device = torch.device('cpu')

# Load data
print("Loading data...")
data = np.load('data/pems_bay_processed.npz')
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
P_fwd = torch.FloatTensor(data['P_fwd'])
P_bwd = torch.FloatTensor(data['P_bwd'])
mean = float(data['mean'])
std = float(data['std'])

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# Create dataloaders
train_dataset = TrafficDataset(X_train, y_train, P_fwd, P_bwd)
val_dataset = TrafficDataset(X_val, y_val, P_fwd, P_bwd)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize model
print("\nInitializing model...")
model = DCRNN(
    input_dim=1,
    hidden_dim=64,
    output_dim=1,
    num_layers=2,
    max_diffusion_step=2
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# Optimizer with HIGH learning rate
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)

# Cosine annealing scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)

criterion = nn.L1Loss()

# Training
print("\n" + "="*70)
print("TRAINING")
print("="*70)

history = []
best_val_mae = float('inf')
patience_counter = 0

for epoch in range(1, 51):  # 50 epochs
    # Train
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/50", leave=False):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        P_fwd_b = batch['P_fwd'][0].to(device)
        P_bwd_b = batch['P_bwd'][0].to(device)
        
        optimizer.zero_grad()
        pred = model(x, P_fwd=P_fwd_b, P_bwd=P_bwd_b, T_out=12, 
                    labels=y, training=True)
        loss = criterion(pred, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
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
                        training=False)
            loss = criterion(pred, y)
            val_loss += loss.item()
            
            tracker.update(pred, y, mean, std)
    
    val_loss /= len(val_loader)
    metrics = tracker.compute()
    val_mae = metrics['mae']
    
    # Step scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    history.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_mae': val_mae,
        'lr': current_lr
    })
    
    # Print progress
    print(f"Epoch {epoch:2d}/50 | LR: {current_lr:.6f} | "
          f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {val_mae:.4f}")
    
    # Check improvement
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        improvement = (7.99 - val_mae) / 7.99 * 100
        patience_counter = 0
        
        # Save checkpoint
        os.makedirs('checkpoints_highLR', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae
        }, 'checkpoints_highLR/best_model.pt')
        
        print(f"  → NEW BEST! Improvement: {improvement:.1f}% vs baseline")
    else:
        patience_counter += 1
    
    # Early stopping (but not before epoch 20)
    if epoch > 20 and patience_counter >= 15:
        print(f"\nEarly stopping at epoch {epoch}")
        break
    
    # Check if stuck
    if epoch == 10 and val_mae > 7.5:
        print(f"\n⚠️  Still stuck at epoch 10, MAE: {val_mae:.4f}")
        print("  Continuing anyway to see if it escapes...")

# Summary
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Best Val MAE: {best_val_mae:.4f}")
print(f"Baseline MAE: 7.99")

if best_val_mae < 6.0:
    improvement = (7.99 - best_val_mae) / 7.99 * 100
    print(f"Improvement: {improvement:.1f}%")
    print("\n✅ HIGH LR WORKED! Model escaped local minimum!")
elif best_val_mae < 7.5:
    print("\n✓ Some improvement, but not enough")
    print("  May need even higher LR or different optimizer")
else:
    print(f"\n❌ Still stuck even with high LR")
    print("  This indicates a fundamental problem in:")
    print("  - Model architecture (diffusion conv not working)")
    print("  - Data (already too averaged)")
    print("  - Loss landscape (very flat)")

# Save history
with open('checkpoints_highLR/history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"\nResults saved to: checkpoints_highLR/")
