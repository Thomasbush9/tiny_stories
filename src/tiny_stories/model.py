"""Script with the models trained using the Tiny Stories dataset."""

import math

import torch
import torch.nn as nn


class AttentionModule:
    def __init__(
        self, input_dim: int, output_dim: int, hidden: int = 128, nheads: int = 5
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nheads = nheads
        self.hidden = hidden

        # define the linear projections
        self.projq = nn.Linear(input_dim, hidden)
        self.projk = nn.Linear(input_dim, hidden)
        self.projv = nn.Linear(input_dim, hidden)
        self.proj_out = nn.Linear(hidden, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.projq(x)
        K = self.projk(x)
        V = self.projv(x)
        # get attn values
        affinities = torch.einsum("...qj, ...ki->...qk", Q, K)
        attn_weights = torch.softmax(affinities, dim=-1)
        attn_values = torch.einsum("...qk, ...kc->...qc", attn_weights, V)
        return self.proj_out(attn_values)


if __name__ == "__main__":
    attention = AttentionModule(10, 10)
    x = torch.randn(100, 10)
    h = attention.forward(x)
    print(h.shape)
