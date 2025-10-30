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


class CrossAttention(nn.Module):
    def __init__(self, dim_dec: int, dim_enc: int, d: int, nheads: int):
        super().__init__()
        assert d % nheads == 0, "Hidden dimension must be divisible by heads"
        self.hidden = d // nheads
        self.d = d
        self.nheads = nheads
        self.proj_q = nn.Linear(dim_dec, d)
        self.proj_k = nn.Linear(dim_enc, d)
        self.proj_v = nn.Linear(dim_enc, d)
        self.proj_out = nn.Linear(d, dim_dec)

    def forward(self, x_dec: torch.Tensor, h_enc: torch.Tensor) -> torch.Tensor:
        # projections
        new_shape_dec = x_dec.shape[:2] + (self.nheads, self.hidden)
        new_shape_enc = h_enc.shape[:2] + (self.nheads, self.hidden)
        Q = self.proj_q(x_dec).reshape(new_shape_dec).movedim(-2, -3)
        K = self.proj_k(h_enc).reshape(new_shape_enc).movedim(-2, -3)
        V = self.proj_v(h_enc).reshape(new_shape_enc).movedim(-2, -3)
        # scores
        # TODO add mask pad for encoder
        scores = torch.einsum("...qd, ...kd->...qk", Q, K) / math.sqrt(self.d)
        probabilities = torch.softmax(scores, dim=-1)
        scaled_values = torch.einsum("...sh, ...hd-> ...sd", probabilities, V)
        scaled_values = einops.rearrange(
            scaled_values, "... h l d-> ... l (h d)", h=self.nheads, d=self.hidden
        )
        return self.proj_out(scaled_values)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, d, nheads):
        super().__init__()
        self.mha = AttentionModule(input_dim, d, nheads)
        self.norm = nn.LayerNorm(input_dim)

        self.cross_att = CrossAttention(input_dim, input_dim, d, nheads)
        self.ff = nn.Sequential(nn.Linear(input_dim, d), nn.Linear(d, input_dim))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == h.shape[-1], "input dims must be the same"
        out = self.mha(x) + x
        out = self.norm(out)
        out = self.cross_att(out, h) + out
        out = self.norm(out)
        out = self.ff(out) + out
        out = self.norm(out)
        return out


if __name__ == "__main__":
    x = torch.randn(100, 5, 10)
    args = {
        "input_dim": 10,
        "hidden": 128,
        "nheads": 8,
    }
    encoder = Encoder(5, args)
    enc_h = encoder.forward(x)
    y = torch.randn(100, 7, 10)
    decoder_b = DecoderBlock(10, 128, 8)
    print(decoder_b(y, enc_h).shape)
