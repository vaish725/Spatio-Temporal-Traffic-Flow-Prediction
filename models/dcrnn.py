"""
DCRNN Implementation: Diffusion Convolutional Recurrent Neural Network

This implements the DCRNN architecture from:
Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). 
Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. 
ICLR 2018.

The key innovation is replacing standard convolutions with diffusion convolutions
that model spatial dependencies through graph structure (transition matrices).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.diffusion_conv import DiffusionConv


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU (DCGRU) Cell
    
    This is the core building block of DCRNN. It replaces the linear 
    transformations in a standard GRU with diffusion convolutions to 
    capture spatial dependencies through the graph structure.
    
    Architecture follows standard GRU:
        z_t = σ(Θ_z * [x_t, h_{t-1}])     # Update gate
        r_t = σ(Θ_r * [x_t, h_{t-1}])     # Reset gate  
        h̃_t = tanh(Θ_h * [x_t, r_t ⊙ h_{t-1}])  # Candidate
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t   # New hidden state
    
    where Θ_* are diffusion convolution operators instead of matrices.

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden state dimension
        max_diffusion_step (int): K-hop diffusion steps (default: 2)
        bias (bool): Add bias to convolutions (default: True)
        
    Inputs:
        x_t: (batch, N, input_dim) - Input at time t
        h_prev: (batch, N, hidden_dim) - Previous hidden state
        P_fwd: (N, N) - Forward transition matrix
        P_bwd: (N, N) - Backward transition matrix
        
    Output:
        h_t: (batch, N, hidden_dim) - New hidden state
    """

    def __init__(self, input_dim, hidden_dim, max_diffusion_step=2, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_diffusion_step = max_diffusion_step

        # Update gate: processes [x_t, h_{t-1}]
        self.conv_xz = DiffusionConv(
            input_dim, hidden_dim, max_diffusion_step, bias=bias
        )
        self.conv_hz = DiffusionConv(
            hidden_dim, hidden_dim, max_diffusion_step, bias=bias
        )

        # Reset gate: processes [x_t, h_{t-1}]
        self.conv_xr = DiffusionConv(
            input_dim, hidden_dim, max_diffusion_step, bias=bias
        )
        self.conv_hr = DiffusionConv(
            hidden_dim, hidden_dim, max_diffusion_step, bias=bias
        )

        # Candidate hidden state: processes [x_t, r_t ⊙ h_{t-1}]
        self.conv_xh = DiffusionConv(
            input_dim, hidden_dim, max_diffusion_step, bias=bias
        )
        self.conv_hh = DiffusionConv(
            hidden_dim, hidden_dim, max_diffusion_step, bias=bias
        )

    def forward(self, x_t, h_prev, P_fwd=None, P_bwd=None):
        """
        Forward pass through DCGRU cell.
        
        Args:
            x_t: (batch, N, input_dim)
            h_prev: (batch, N, hidden_dim)
            P_fwd: (N, N) - Forward diffusion matrix
            P_bwd: (N, N) - Backward diffusion matrix
            
        Returns:
            h_t: (batch, N, hidden_dim)
        """
        # Update gate: z_t = σ(Θ_z * [x_t, h_{t-1}])
        z = torch.sigmoid(
            self.conv_xz(x_t, P_fwd, P_bwd) + 
            self.conv_hz(h_prev, P_fwd, P_bwd)
        )
        
        # Reset gate: r_t = σ(Θ_r * [x_t, h_{t-1}])
        r = torch.sigmoid(
            self.conv_xr(x_t, P_fwd, P_bwd) + 
            self.conv_hr(h_prev, P_fwd, P_bwd)
        )
        
        # Candidate: h̃_t = tanh(Θ_h * [x_t, r_t ⊙ h_{t-1}])
        h_tilde = torch.tanh(
            self.conv_xh(x_t, P_fwd, P_bwd) + 
            self.conv_hh(r * h_prev, P_fwd, P_bwd)
        )
        
        # New hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        h_t = (1 - z) * h_prev + z * h_tilde
        
        return h_t


class Encoder(nn.Module):
    """
    DCRNN Encoder
    
    Processes input sequences using stacked DCGRU cells.
    
    Args:
        input_dim (int): Input feature dimension per node
        hidden_dim (int): Hidden state dimension
        num_layers (int): Number of stacked DCGRU layers
        max_diffusion_step (int): K-hop diffusion steps
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, max_diffusion_step=2):
        super().__init__()
        self.layers = nn.ModuleList([
            DCGRUCell(
                input_dim if i == 0 else hidden_dim, 
                hidden_dim,
                max_diffusion_step=max_diffusion_step
            )
            for i in range(num_layers)
        ])

    def forward(self, X, P_fwd=None, P_bwd=None):
        # X: (batch, T_in, N, input_dim) or (batch, T_in, N)
        if X.dim() == 3:
            # Assume (batch, T_in, N) -> add feature dim
            X = X.unsqueeze(-1)

        batch, T_in, N, input_dim = X.shape
        h = [torch.zeros(batch, N, layer.hidden_dim, device=X.device) for layer in self.layers]

        for t in range(T_in):
            x_t = X[:, t]
            for i, layer in enumerate(self.layers):
                h[i] = layer(x_t, h[i], P_fwd=P_fwd, P_bwd=P_bwd)
                x_t = h[i]

        # Return final hidden states
        return h


class Decoder(nn.Module):
    """
    DCRNN Decoder
    
    Generates output sequences using stacked DCGRU cells and a projection layer.
    
    Args:
        output_dim (int): Output feature dimension per node
        hidden_dim (int): Hidden state dimension
        num_layers (int): Number of stacked DCGRU layers
        max_diffusion_step (int): K-hop diffusion steps
    """
    def __init__(self, output_dim, hidden_dim, num_layers=1, max_diffusion_step=2):
        super().__init__()
        self.layers = nn.ModuleList([
            DCGRUCell(
                output_dim if i == 0 else hidden_dim, 
                hidden_dim,
                max_diffusion_step=max_diffusion_step
            )
            for i in range(num_layers)
        ])
        # Projection from hidden to output using diffusion convolution
        self.proj = DiffusionConv(hidden_dim, output_dim, max_diffusion_step)

    def forward(self, H, T_out, P_fwd=None, P_bwd=None, last_input=None):
        """
        Generate output sequence autoregressively.
        
        Args:
            H: List of hidden states [(batch, N, hidden_dim), ...]
            T_out: Number of output steps to generate
            P_fwd: (N, N) - Forward diffusion matrix
            P_bwd: (N, N) - Backward diffusion matrix
            last_input: (batch, N, output_dim) - Last encoder input (optional)
            
        Returns:
            outputs: (batch, T_out, N, output_dim)
        """
        # H: list of hidden states per layer [(batch, N, hidden_dim), ...]
        batch, N, _ = H[0].shape
        outputs = []
        
        # CRITICAL FIX: Initialize with last encoder input, not zeros
        if last_input is not None:
            input_t = last_input
        else:
            input_t = torch.zeros(batch, N, self.proj.out_features, device=H[0].device)

        for t in range(T_out):
            x_t = input_t
            for i, layer in enumerate(self.layers):
                H[i] = layer(x_t, H[i], P_fwd=P_fwd, P_bwd=P_bwd)
                x_t = H[i]
            out_t = self.proj(x_t, P_fwd, P_bwd)
            outputs.append(out_t)
            # Autoregressive: use previous output as input
            # Note: For training, can add teacher forcing option
            input_t = out_t

        # Stack: (T_out, batch, N, output_dim) -> (batch, T_out, N, output_dim)
        outputs = torch.stack(outputs, dim=1)
        return outputs


class DCRNN(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network (DCRNN)
    
    Full encoder-decoder architecture for spatio-temporal forecasting.
    
    Args:
        input_dim (int): Input feature dimension per node
        hidden_dim (int): Hidden state dimension
        output_dim (int): Output feature dimension per node
        num_layers (int): Number of stacked DCGRU layers
        max_diffusion_step (int): K-hop diffusion steps (default: 2)
        
    Input:
        X: (batch, T_in, N) or (batch, T_in, N, input_dim) - Input sequences
        P_fwd: (N, N) - Forward transition matrix
        P_bwd: (N, N) - Backward transition matrix
        T_out: int - Number of steps to forecast
        
    Output:
        (batch, T_out, N, output_dim) - Predicted sequences
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, max_diffusion_step=2):
        super().__init__()
        self.encoder = Encoder(
            input_dim, hidden_dim, 
            num_layers=num_layers, 
            max_diffusion_step=max_diffusion_step
        )
        self.decoder = Decoder(
            output_dim, hidden_dim, 
            num_layers=num_layers,
            max_diffusion_step=max_diffusion_step
        )

    def forward(self, X, P_fwd=None, P_bwd=None, T_out=12):
        """
        Forward pass: encode input sequence, decode to predictions.
        
        Args:
            X: (batch, T_in, N) or (batch, T_in, N, input_dim)
            P_fwd: (N, N) - Forward transition matrix
            P_bwd: (N, N) - Backward transition matrix
            T_out: Number of output steps
            
        Returns:
            predictions: (batch, T_out, N, output_dim)
        """
        # Encode input sequence
        H = self.encoder(X, P_fwd=P_fwd, P_bwd=P_bwd)
        
        # CRITICAL FIX: Pass last input timestep to decoder
        if X.dim() == 3:
            last_input = X[:, -1, :].unsqueeze(-1)  # (batch, N, 1)
        else:
            last_input = X[:, -1, :, :]  # (batch, N, input_dim)
        
        # Decode with proper initialization
        out = self.decoder(H, T_out=T_out, P_fwd=P_fwd, P_bwd=P_bwd, last_input=last_input)
        return out
