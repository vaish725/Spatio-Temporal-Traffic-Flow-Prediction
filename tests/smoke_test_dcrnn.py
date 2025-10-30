import torch
from models.dcrnn import DCRNN


def run_smoke_test():
    batch = 2
    T_in = 4
    T_out = 3
    N = 5
    input_dim = 1
    hidden_dim = 8
    output_dim = 1

    # Random input (batch, T_in, N)
    X = torch.randn(batch, T_in, N)

    model = DCRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=1)
    out = model(X, P_fwd=None, P_bwd=None, T_out=T_out)

    print("Output shape:", out.shape)
    assert out.shape == (batch, T_out, N, output_dim), "Unexpected output shape"
    print("Smoke test passed.")


if __name__ == '__main__':
    run_smoke_test()
