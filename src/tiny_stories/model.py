"""Script with the models trained using the Tiny Stories dataset."""

import torch
import torch.nn as nn


class AttentionModule:
    def __init__(self, config):
        self.config = config
        self.proj_q = nn.Linear(config.input_dim, config.hidden)
        self.proj_k = nn.Linear(config.input_dim, config.hidden)
        self.proj_v = nn.Linear(config.input_dim, config.hidden)
        self.proj_out = nn.Linear(config.hidden, config.out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
