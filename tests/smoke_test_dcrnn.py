# Quick unit test to verify DCRNN model can perform forward pass without errors (no training)
"""
Smoke test for DCRNN with diffusion convolutions

This test verifies:
1. Model instantiation with DiffusionConv layers
2. Forward pass with transition matrices (P_fwd, P_bwd)
3. Output shape correctness
4. Basic gradient flow (backprop test)
"""

import torch
import numpy as np
from models.dcrnn import DCRNN


def create_transition_matrices(N, density=0.3):
    """
    Create mock transition matrices similar to preprocessing
    
    Args:
        N: Number of nodes
        density: Edge density (0-1)
        
    Returns:
        P_fwd, P_bwd: Transition matrices as torch tensors
    """
    # Random adjacency matrix
    A = torch.rand(N, N)
    A = (A < density).float()  # Sparse adjacency
    
    # Forward: row-normalized (out-degree)
    D_out = A.sum(dim=1, keepdim=True)
    D_out[D_out == 0] = 1  # Avoid division by zero
    P_fwd = A / D_out
    
    # Backward: column-normalized transpose (in-degree)
    D_in = A.sum(dim=0, keepdim=True)
    D_in[D_in == 0] = 1
    P_bwd = (A.T / D_in.T)
    
    return P_fwd, P_bwd


def test_basic_forward_pass():
    """Test 1: Basic forward pass with transition matrices"""
    print("\n=== Test 1: Basic Forward Pass ===")
    
    batch = 2
    T_in = 4
    T_out = 3
    N = 5
    input_dim = 1
    hidden_dim = 8
    output_dim = 1
    max_diffusion_step = 2

    # Random input (batch, T_in, N)
    X = torch.randn(batch, T_in, N)
    
    # Create transition matrices
    P_fwd, P_bwd = create_transition_matrices(N)
    
    print(f"Input shape: {X.shape}")
    print(f"P_fwd shape: {P_fwd.shape}")
    print(f"P_bwd shape: {P_bwd.shape}")

    # Create model with diffusion convolution
    model = DCRNN(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        output_dim=output_dim, 
        num_layers=1,
        max_diffusion_step=max_diffusion_step
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    out = model(X, P_fwd=P_fwd, P_bwd=P_bwd, T_out=T_out)

    print(f"Output shape: {out.shape}")
    assert out.shape == (batch, T_out, N, output_dim), f"Expected {(batch, T_out, N, output_dim)}, got {out.shape}"
    print("✓ Test 1 passed!")
    
    return model, X, P_fwd, P_bwd, out


def test_multi_layer():
    """Test 2: Multi-layer DCRNN"""
    print("\n=== Test 2: Multi-Layer DCRNN ===")
    
    batch = 3
    T_in = 6
    T_out = 6
    N = 10
    input_dim = 1
    hidden_dim = 16
    output_dim = 1
    num_layers = 2

    X = torch.randn(batch, T_in, N)
    P_fwd, P_bwd = create_transition_matrices(N)

    model = DCRNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        max_diffusion_step=2
    )
    
    print(f"Num layers: {num_layers}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    out = model(X, P_fwd=P_fwd, P_bwd=P_bwd, T_out=T_out)

    print(f"Output shape: {out.shape}")
    assert out.shape == (batch, T_out, N, output_dim)
    print("✓ Test 2 passed!")


def test_gradient_flow():
    """Test 3: Gradient flow through diffusion convolutions"""
    print("\n=== Test 3: Gradient Flow ===")
    
    batch = 2
    T_in = 3
    T_out = 3
    N = 8
    input_dim = 1
    hidden_dim = 4
    output_dim = 1

    X = torch.randn(batch, T_in, N)
    Y = torch.randn(batch, T_out, N, output_dim)  # Target
    P_fwd, P_bwd = create_transition_matrices(N)

    model = DCRNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=1,
        max_diffusion_step=2
    )

    # Forward pass
    out = model(X, P_fwd=P_fwd, P_bwd=P_bwd, T_out=T_out)
    
    # Loss
    loss = torch.nn.functional.mse_loss(out, Y)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"Parameters with gradients: {has_grads}/{total_params}")
    
    assert has_grads == total_params, "Not all parameters have gradients!"
    print("✓ Test 3 passed!")


def test_without_transition_matrices():
    """Test 4: Model works even without P_fwd/P_bwd (fallback)"""
    print("\n=== Test 4: Without Transition Matrices ===")
    
    batch = 2
    T_in = 3
    T_out = 3
    N = 5
    input_dim = 1
    hidden_dim = 8
    output_dim = 1

    X = torch.randn(batch, T_in, N)

    model = DCRNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=1,
        max_diffusion_step=2
    )

    # Forward pass without transition matrices
    out = model(X, P_fwd=None, P_bwd=None, T_out=T_out)

    print(f"Output shape: {out.shape}")
    assert out.shape == (batch, T_out, N, output_dim)
    print("✓ Test 4 passed!")


def run_all_tests():
    """Run all smoke tests"""
    print("="*60)
    print("DCRNN Smoke Tests with Diffusion Convolutions")
    print("="*60)
    
    test_basic_forward_pass()
    test_multi_layer()
    test_gradient_flow()
    test_without_transition_matrices()
    
    print("\n" + "="*60)
    print("✓ All smoke tests passed!")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
