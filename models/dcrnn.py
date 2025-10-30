"""
Minimal DCRNN skeleton: DCGRU cell and Seq2Seq wrapper.
This file provides a CPU-friendly, documented skeleton to start implementing
the real diffusion convolution operations. It includes shape comments and
simple alternatives in case PyG or other graph libraries are not used.

Note: This is a structural skeleton for development and unit testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU (DCGRU) cell skeleton.

    Inputs:
        x_t: (batch, N, input_dim)
        h_prev: (batch, N, hidden_dim)
        P_fwd, P_bwd: optional transition matrices as torch tensors (N, N)
    Outputs:
        h_t: (batch, N, hidden_dim)

    This implementation uses simple learnable linear transforms in place of
    explicit diffusion convolution for now. Replace the linear layers with
    diffusion convolution ops later.
    """

    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Gates
        self.lin_xz = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.lin_hz = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.lin_xr = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.lin_hr = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Candidate
        self.lin_xh = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.lin_hh = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, x_t, h_prev, P_fwd=None, P_bwd=None):
        # x_t: (batch, N, input_dim)
        # h_prev: (batch, N, hidden_dim)

        z = torch.sigmoid(self.lin_xz(x_t) + self.lin_hz(h_prev))
        r = torch.sigmoid(self.lin_xr(x_t) + self.lin_hr(h_prev))
        h_tilde = torch.tanh(self.lin_xh(x_t) + self.lin_hh(r * h_prev))
        h_t = (1 - z) * h_prev + z * h_tilde
        return h_t


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            DCGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim)
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
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            DCGRUCell(output_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        # Simple projection from hidden to output
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, H, T_out, P_fwd=None, P_bwd=None):
        # H: list of hidden states per layer [(batch, N, hidden_dim), ...]
        batch, N, _ = H[0].shape
        outputs = []
        input_t = torch.zeros(batch, N, self.proj.out_features, device=H[0].device)

        for t in range(T_out):
            x_t = input_t
            for i, layer in enumerate(self.layers):
                H[i] = layer(x_t, H[i], P_fwd=P_fwd, P_bwd=P_bwd)
                x_t = H[i]
            out_t = self.proj(x_t)
            outputs.append(out_t)
            # For now use autoregressive zero-input; teacher forcing can be added later
            input_t = out_t

        # (T_out, batch, N, output_dim) -> (batch, T_out, N, output_dim)
        outputs = torch.stack(outputs, dim=1)
        return outputs


class DCRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers=num_layers)
        self.decoder = Decoder(output_dim, hidden_dim, num_layers=num_layers)

    def forward(self, X, P_fwd=None, P_bwd=None, T_out=12):
        # X: (batch, T_in, N) or (batch, T_in, N, input_dim)
        H = self.encoder(X, P_fwd=P_fwd, P_bwd=P_bwd)
        out = self.decoder(H, T_out=T_out, P_fwd=P_fwd, P_bwd=P_bwd)
        return out
