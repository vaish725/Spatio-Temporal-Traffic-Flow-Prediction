"""
Memory-safe training for Colab (auto-detects GPU/CPU)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import gc

from models.dcrnn import DCRNN
from src.dataset import TrafficDataset


print("="*70)
print("MEMORY-SAFE TRAINING FOR COLAB")
print("="*70)
print("\nOptimizations:")
print("  • Batch size: 64 → 16 (4x less memory)")
print("  • Gradient accumulation: 4 steps (same effective batch)")
print("  • Learning rate: 0.01")
print("  • Cosine annealing")
print("  • Aggressive garbage collection")
print()

# Auto-detect GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Load data
print("Loading data...")
data = np.load('data/pems_bay_processed.npz')
X_train = data['X_train'][:5000]  # Limit to 5K samples
y_train = data['y_train'][:5000]
X_val = data['X_val'][:1000]  # Limit to 1K samples
y_val = data['y_val'][:1000]
P_fwd = torch.FloatTensor(data['P_fwd'])
P_bwd = torch.FloatTensor(data['P_bwd'])
mean = float(data['mean'])
std = float(data['std'])

print(f"Train: {len(X_train)}, Val: {len(X_val)}")
print("(Using subset to prevent memory issues)")

# Create dataloaders
train_dataset = TrafficDataset(X_train, y_train, P_fwd, P_bwd)
val_dataset = TrafficDataset(X_val, y_val, P_fwd, P_bwd)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

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

# Optimizer with high LR
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)
criterion = nn.L1Loss()

# Training loop
print("\nTraining...")
print("="*70)

best_val_mae = float('inf')
patience = 5
patience_counter = 0
history = {'epoch': [], 'train_loss': [], 'val_mae': [], 'lr': []}

accumulation_steps = 4  # Accumulate 4 batches

for epoch in range(10):  # 10 epochs
    # Training
    model.train()
    train_losses = []
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/10")):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        P_fwd_b = batch['P_fwd'][0].to(device)
        P_bwd_b = batch['P_bwd'][0].to(device)
        
        # Forward
        pred = model(x, P_fwd=P_fwd_b, P_bwd=P_bwd_b, T_out=12, 
                    labels=y, training=True)
        loss = criterion(pred, y)
        loss = loss / accumulation_steps
        
        # Backward
        loss.backward()
        
        # Update every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()
            
            del pred
            gc.collect()
        
        train_losses.append(loss.item() * accumulation_steps)
    
    # Final update if needed
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
    
    train_loss = np.mean(train_losses)
    
    # Validation
    model.eval()
    val_maes = []
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            P_fwd_b = batch['P_fwd'][0].to(device)
            P_bwd_b = batch['P_bwd'][0].to(device)
            
            pred = model(x, P_fwd=P_fwd_b, P_bwd=P_bwd_b, T_out=12, training=False)
            mae = torch.abs(pred - y).mean().item()
            val_maes.append(mae)
            
            del pred
    
    val_mae = np.mean(val_maes)
    val_mae_denorm = val_mae * std
    
    # Update scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(train_loss)
    history['val_mae'].append(val_mae_denorm)
    history['lr'].append(current_lr)
    
    print(f"\nEpoch {epoch+1}/10:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val MAE: {val_mae_denorm:.3f} mph")
    print(f"  LR: {current_lr:.6f}")
    
    # Save best model
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0
        
        os.makedirs('checkpoints_colab', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mae': val_mae,
            'mean': mean,
            'std': std
        }, 'checkpoints_colab/best_model.pt')
        
        print(f"  ✓ Best model saved! (MAE: {val_mae_denorm:.3f} mph)")
    else:
        patience_counter += 1
    
    # Early stopping (but not before epoch 5)
    if epoch >= 5 and patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break
    
    gc.collect()

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Best Val MAE: {best_val_mae * std:.3f} mph")

# Save history
os.makedirs('checkpoints_colab', exist_ok=True)
with open('checkpoints_colab/history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\nResults saved to checkpoints_colab/")
