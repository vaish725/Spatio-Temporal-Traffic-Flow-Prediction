"""
Diffusion Convolution for DCRNN

This module implements the diffusion convolution operation from:
Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). 
Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. 
ICLR 2018.

The diffusion convolution captures spatial dependencies by modeling information
propagation through the graph using random walk transition matrices.

Key Formula:
    Z = Σ_{k=0}^{K} [θ_{k,1} * (P_fwd)^k + θ_{k,2} * (P_bwd)^k] * X
    
Where:
    - P_fwd: Forward transition matrix (out-degree normalized adjacency)
    - P_bwd: Backward transition matrix (in-degree normalized adjacency transpose)
    - K: Maximum diffusion steps (typically 2 or 3)
    - θ: Learnable parameters
    - X: Input features (batch, N, features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionConv(nn.Module):
    """
    Diffusion Convolution Layer
    
    Performs K-hop diffusion convolution on graph-structured data using
    forward and backward random walk transition matrices.
    
    Args:
        in_features (int): Number of input features per node
        out_features (int): Number of output features per node
        max_diffusion_step (int): Maximum diffusion steps K (default: 2)
        bias (bool): If True, adds a learnable bias (default: True)
        
    Input shapes:
        x: (batch, N, in_features) - Node features
        P_fwd: (N, N) - Forward transition matrix (optional, can be sparse)
        P_bwd: (N, N) - Backward transition matrix (optional, can be sparse)
        
    Output shape:
        (batch, N, out_features)
    """
    
    def __init__(self, in_features, out_features, max_diffusion_step=2, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_diffusion_step = max_diffusion_step
        
        # Number of diffusion supports: 
        # 2 directions (fwd, bwd) * (K+1) steps (0 to K)
        num_supports = 2 * (max_diffusion_step + 1)
        
        # Learnable weight matrix for all diffusion supports
        # Shape: (in_features, out_features, num_supports)
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features, num_supports)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _calculate_diffusion_supports(self, x, P_fwd, P_bwd):
        """
        Calculate all K-hop diffusion supports.
        
        For each direction (forward/backward), compute:
            [X, P*X, P^2*X, ..., P^K*X]
        
        Args:
            x: (batch, N, in_features)
            P_fwd: (N, N)
            P_bwd: (N, N)
            
        Returns:
            List of tensors, each (batch, N, in_features)
        """
        supports = []
        
        # Forward diffusion: P_fwd^k * X for k=0,1,...,K
        if P_fwd is not None:
            x_fwd = x  # k=0: identity (P^0 = I)
            supports.append(x_fwd)
            
            for k in range(1, self.max_diffusion_step + 1):
                # x_fwd = P_fwd @ x_fwd
                # Handle batch dimension: (N, N) @ (batch, N, features)
                x_fwd = torch.einsum('nm,bmc->bnc', P_fwd, x_fwd)
                supports.append(x_fwd)
        else:
            # If no P_fwd, just use identity (x itself) for all steps
            supports.extend([x] * (self.max_diffusion_step + 1))
        
        # Backward diffusion: P_bwd^k * X for k=0,1,...,K
        if P_bwd is not None:
            x_bwd = x  # k=0: identity
            supports.append(x_bwd)
            
            for k in range(1, self.max_diffusion_step + 1):
                x_bwd = torch.einsum('nm,bmc->bnc', P_bwd, x_bwd)
                supports.append(x_bwd)
        else:
            supports.extend([x] * (self.max_diffusion_step + 1))
        
        return supports
    
    def forward(self, x, P_fwd=None, P_bwd=None):
        """
        Forward pass of diffusion convolution.
        
        Args:
            x: (batch, N, in_features)
            P_fwd: (N, N) - Forward transition matrix
            P_bwd: (N, N) - Backward transition matrix
            
        Returns:
            out: (batch, N, out_features)
        """
        batch_size, num_nodes, in_features = x.shape
        
        # Calculate all diffusion supports
        supports = self._calculate_diffusion_supports(x, P_fwd, P_bwd)
        
        # Stack supports: (num_supports, batch, N, in_features)
        supports = torch.stack(supports, dim=0)
        
        # Reshape for matrix multiplication
        # supports: (num_supports, batch*N, in_features)
        supports = supports.permute(0, 1, 2, 3).reshape(
            len(supports), batch_size * num_nodes, in_features
        )
        
        # Weight: (in_features, out_features, num_supports)
        # Apply weights: (batch*N, out_features)
        out = torch.zeros(batch_size * num_nodes, self.out_features, 
                         device=x.device, dtype=x.dtype)
        
        for i, support in enumerate(supports):
            # support: (batch*N, in_features)
            # weight[:, :, i]: (in_features, out_features)
            out += support @ self.weight[:, :, i]
        
        # Reshape back: (batch, N, out_features)
        out = out.reshape(batch_size, num_nodes, self.out_features)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'max_diffusion_step={self.max_diffusion_step}, '
                f'bias={self.bias is not None})')


def test_diffusion_conv():
    """
    Quick test to verify DiffusionConv works correctly
    """
    print("Testing DiffusionConv...")
    
    # Small test parameters
    batch_size = 2
    num_nodes = 5
    in_features = 3
    out_features = 4
    K = 2
    
    # Random input
    x = torch.randn(batch_size, num_nodes, in_features)
    
    # Create simple transition matrices (random walk)
    A = torch.rand(num_nodes, num_nodes)
    A = (A > 0.5).float()  # Sparse adjacency
    
    # Forward: row-normalized
    D_out = A.sum(dim=1, keepdim=True)
    D_out[D_out == 0] = 1
    P_fwd = A / D_out
    
    # Backward: column-normalized transpose
    D_in = A.sum(dim=0, keepdim=True)
    D_in[D_in == 0] = 1
    P_bwd = (A.T / D_in.T)
    
    # Create layer
    layer = DiffusionConv(in_features, out_features, max_diffusion_step=K)
    
    # Forward pass
    out = layer(x, P_fwd, P_bwd)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"P_fwd shape: {P_fwd.shape}")
    print(f"P_bwd shape: {P_bwd.shape}")
    
    assert out.shape == (batch_size, num_nodes, out_features), "Shape mismatch!"
    print("✓ DiffusionConv test passed!")
    
    return layer, x, P_fwd, P_bwd, out


if __name__ == '__main__':
    test_diffusion_conv()
