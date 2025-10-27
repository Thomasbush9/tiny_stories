"""Script with the models trained using the Tiny Stories dataset."""

import math

import einops
import torch
import torch.nn as nn


class AttentionModule:
    def __init__(
        self, input_dim: int, output_dim: int, hidden: int = 128, nheads: int = 8
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert hidden % nheads == 0, (
            "Hidden dimension must be divisible by number of heads"
        )
        self.c = hidden // nheads
        self.nheads = nheads
        self.hidden = hidden

        # define the linear projections
        self.projq = nn.Linear(input_dim, self.hidden)
        self.projk = nn.Linear(input_dim, self.hidden)
        self.projv = nn.Linear(input_dim, self.hidden)
        self.proj_out = nn.Linear(self.hidden, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.shape[:2] + (self.nheads, self.c)
        Q = self.projq(x).reshape(new_shape) / math.sqrt(self.c)
        K = self.projk(x).reshape(new_shape)
        V = self.projv(x).reshape(new_shape)
        # get attn values
        affinities = torch.einsum("...hq, ...hk->...hqk", Q, K)
        attn_weights = torch.softmax(affinities, dim=-1)
        attn_values = torch.einsum("...hqk, ...hk->...hq", attn_weights, V)
        attn_values = einops.rearrange(
            attn_values, "... h c -> ... (h c)", h=self.nheads, c=self.c
        )
        return self.proj_out(attn_values)


class MLP:
    def __init__(self, input_dim: int, hidden_dim: int):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    attention = AttentionModule(10, 10)
    x = torch.randn(100, 5, 10)
    h = attention.forward(x)
    mlp = MLP(10, 64)
    h = mlp.forward(h)
