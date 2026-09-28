"""Temporal landmark model shared by training and inference."""

from __future__ import annotations

import torch
from torch import nn


FEATURE_DIM = (33 + 21 + 21) * 3


class TemporalLandmarkClassifier(nn.Module):
    def __init__(self, classes: int = 100, feature_dim: int = FEATURE_DIM, hidden: int = 192):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.encoder = nn.GRU(
            feature_dim,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )
        self.attention = nn.Sequential(nn.Linear(hidden * 2, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(self.input_norm(sequence))
        weights = torch.softmax(self.attention(encoded), dim=1)
        pooled = torch.sum(encoded * weights, dim=1)
        return self.head(pooled)
