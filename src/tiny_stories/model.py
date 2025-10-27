"""Script with the models trained using the Tiny Stories dataset."""

import math

import einops
import torch
import torch.nn as nn


class AttentionModule:
    def __init__(self, input_dim: int, hidden: int = 128, nheads: int = 8):
        self.input_dim = input_dim
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
    def __init__(self, input_dim: int, hidden: int):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock:
    def __init__(self, input_dim: int, hidden: int, nheads: int):
        self.mha = AttentionModule(input_dim=input_dim, hidden=hidden, nheads=nheads)
        self.mlp = MLP(input_dim, hidden)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mha.forward(x)
        x = self.norm(out + x)
        out = self.mlp.forward(x)
        out = self.norm(out + x)
        return out


def positionalEncoding(x: torch.Tensor, n: int = 1000):
    """Applies positional encoding using sin, cos funcs"""
    seq_len = x.shape[1]
    d = x.shape[-1]
    col = torch.arange(seq_len).unsqueeze(-1)
    row = torch.arange(d).unsqueeze(0)
    exp = 2 * (row // 2) / d
    scaling_term = torch.pow(n, exp)
    mat = col / scaling_term
    even_ids = torch.arange(0, d, 2)
    odd_ids = even_ids + 1
    mat[..., even_ids] = torch.sin(mat[..., even_ids])
    mat[..., odd_ids] = torch.cos(mat[..., odd_ids])
    return mat


if __name__ == "__main__":
    x = torch.randn(100, 5, 10)
    print(positionalEncoding(x).shape)
    transformer_block = TransformerBlock(input_dim=10, hidden=128, nheads=8)
    print(transformer_block.forward(x).shape)
