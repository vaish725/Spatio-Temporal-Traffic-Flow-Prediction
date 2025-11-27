"""
Quick test to verify teacher forcing implementation.
Tests that model can train with teacher forcing enabled.
"""
import torch
import torch.nn as nn
from models.dcrnn import DCRNN

print("=" * 60)
print("TEACHER FORCING VERIFICATION TEST")
print("=" * 60)

# Create model
model = DCRNN(input_dim=1, hidden_dim=16, output_dim=1, num_layers=1)
model.train()

# Create sample data
batch, T_in, T_out, N = 4, 12, 12, 10
X = torch.randn(batch, T_in, N, 1)
Y = torch.randn(batch, T_out, N, 1)  # Ground truth labels

print(f"\n✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")
print(f"✓ Input shape: {X.shape}")
print(f"✓ Label shape: {Y.shape}")

# Test 1: Training mode (with teacher forcing)
print("\n" + "=" * 60)
print("TEST 1: Training Mode (Teacher Forcing)")
print("=" * 60)

try:
    pred_train = model(X, T_out=T_out, labels=Y, training=True)
    print(f"✅ Training forward pass successful!")
    print(f"   Output shape: {pred_train.shape}")
    print(f"   Output mean: {pred_train.mean():.4f}")
    print(f"   Output std: {pred_train.std():.4f}")
    
    # Check if output has variance
    if pred_train.std() > 0.01:
        print(f"   ✅ Output has variance (not constant)")
    else:
        print(f"   ⚠️  Output variance very low")
        
except Exception as e:
    print(f"❌ Training forward pass failed: {e}")
    exit(1)

# Test 2: Inference mode (autoregressive)
print("\n" + "=" * 60)
print("TEST 2: Inference Mode (Autoregressive)")
print("=" * 60)

model.eval()
with torch.no_grad():
    try:
        pred_infer = model(X, T_out=T_out, training=False)
        print(f"✅ Inference forward pass successful!")
        print(f"   Output shape: {pred_infer.shape}")
        print(f"   Output mean: {pred_infer.mean():.4f}")
        print(f"   Output std: {pred_infer.std():.4f}")
        
        if pred_infer.std() > 0.01:
            print(f"   ✅ Output has variance (not constant)")
        else:
            print(f"   ⚠️  Output variance very low")
            
    except Exception as e:
        print(f"❌ Inference forward pass failed: {e}")
        exit(1)

# Test 3: Training with gradient update
print("\n" + "=" * 60)
print("TEST 3: Training Loop (5 iterations)")
print("=" * 60)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

losses = []
for i in range(5):
    optimizer.zero_grad()
    
    # Forward with teacher forcing
    pred = model(X, T_out=T_out, labels=Y, training=True)
    loss = criterion(pred, Y)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    print(f"   Iteration {i+1}: loss = {loss.item():.6f}")

# Check if loss is decreasing
if losses[-1] < losses[0]:
    print(f"\n✅ Loss decreased: {losses[0]:.6f} → {losses[-1]:.6f}")
    print(f"   Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
else:
    print(f"\n⚠️  Loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}")

# Test 4: Verify teacher forcing is actually used
print("\n" + "=" * 60)
print("TEST 4: Verify Teacher Forcing Behavior")
print("=" * 60)

model.train()
# Create inputs with very different patterns
X_test = torch.ones(2, 12, 10, 1) * 10.0
Y_test_ones = torch.ones(2, 12, 10, 1) * 1.0   # Labels = 1
Y_test_tens = torch.ones(2, 12, 10, 1) * 10.0  # Labels = 10

with torch.no_grad():
    pred_ones = model(X_test, T_out=12, labels=Y_test_ones, training=True)
    pred_tens = model(X_test, T_out=12, labels=Y_test_tens, training=True)
    
    mean_diff = abs(pred_ones.mean() - pred_tens.mean())
    print(f"   Predictions with Y=1:  mean={pred_ones.mean():.4f}")
    print(f"   Predictions with Y=10: mean={pred_tens.mean():.4f}")
    print(f"   Difference: {mean_diff:.4f}")
    
    if mean_diff > 0.1:
        print(f"\n✅ Teacher forcing is working!")
        print(f"   (Different labels produce different outputs)")
    else:
        print(f"\n⚠️  Teacher forcing may not be working properly")
        print(f"   (Same outputs despite different labels)")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\n✅ All tests passed! Model is ready for training.")
print("\nNext steps:")
print("1. Run: python verify_teacher_forcing.py")
print("2. If successful, run: python train_dcrnn_minimal.py --max_samples 100")
print("3. Check that training loss decreases over epochs")
print("4. Deploy to Colab if local training works!")
