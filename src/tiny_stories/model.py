"""Script with the models trained using the Tiny Stories dataset."""

import math
from typing import Dict

import einops
import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    def __init__(
        self, input_dim: int, hidden: int = 128, nheads: int = 8, with_mask: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        assert hidden % nheads == 0, (
            "Hidden dimension must be divisible by number of heads"
        )
        self.c = hidden // nheads
        self.nheads = nheads
        self.hidden = hidden
        self.with_mask = with_mask

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
        if self.with_mask:
            mask = torch.triu(torch.ones(affinities.shape[-2:]), 1)
            affinities = affinities + mask.masked_fill(mask == 1, -torch.inf)

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


class TransformerBlock(nn.Module):
    def __init__(self, input_dim: int, hidden: int, nheads: int):
        super().__init__()
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


class Encoder(nn.Module):
    def __init__(self, n: int, args):
        super().__init__()
        self.input_dim = args["input_dim"]
        self.hidden = args["hidden"]
        self.nheads = args["nheads"]
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(self.input_dim, self.hidden, self.nheads)
                for i in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_enc = positionalEncoding(x)
        x += pos_enc
        for i, block in enumerate(self.transformer_blocks):
            x = block.forward(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, hidden: int, nheads: int):
        super().__init__()
        self.block_1 = AttentionModule(input_dim, hidden, nheads)
        self.norm = nn.LayerNorm(input_dim)
        self.block_2 = TransformerBlock(input_dim, hidden, nheads)

    def forward(self, x: torch.Tensor, enc_h: torch.Tensor) -> torch.Tensor:
        pos_enc = positionalEncoding(x)
        x += pos_enc
        out = self.block_1.forward(x)
        out = self.norm(out)
        # add residual
        out += enc_h
        out = self.block_2.forward(out)
        return out


if __name__ == "__main__":
    x = torch.randn(100, 5, 10)
    args = {
        "input_dim": 10,
        "hidden": 128,
        "nheads": 8,
    }
    encoder = Encoder(5, args)
    decoderblock = DecoderBlock(10, 128, 8)
    enc_h = encoder.forward(x)
    print(decoderblock.forward(x, enc_h).shape)
