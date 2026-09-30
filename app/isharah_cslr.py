"""Continuous Saudi sign recognition: frame features -> gloss sequence with CTC.

The model never needs pause-based cuts. Each emitted gloss also carries the time span where
the CTC path assigned it, which gives word boundaries as a by-product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

TEMPORAL_DOWNSAMPLE = 4


class CSLRModel(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Dropout(dropout), nn.Linear(input_dim, hidden))
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.sequence = nn.LSTM(
            hidden, hidden // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.conv_head = nn.Linear(hidden, vocab_size)
        self.head = nn.Linear(hidden, vocab_size)

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return lengths // TEMPORAL_DOWNSAMPLE

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (sequence log-probs, conv log-probs, output lengths); log-probs are (T, B, V)."""
        hidden = self.temporal(self.input(features).transpose(1, 2)).transpose(1, 2)
        out_lengths = self.output_lengths(lengths).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(hidden, out_lengths.cpu(), batch_first=True, enforce_sorted=False)
        sequence, _ = self.sequence(packed)
        sequence, _ = nn.utils.rnn.pad_packed_sequence(sequence, batch_first=True, total_length=hidden.shape[1])
        log_probs = self.head(sequence).log_softmax(-1).transpose(0, 1)
        conv_log_probs = self.conv_head(hidden).log_softmax(-1).transpose(0, 1)
        return log_probs, conv_log_probs, out_lengths


@dataclass(frozen=True)
class AlignedGloss:
    gloss: str
    start_step: int
    end_step: int
    confidence: float


def greedy_decode(log_probs: np.ndarray, id_to_gloss: dict[int, str]) -> list[AlignedGloss]:
    """Collapse the best CTC path; blank is index 0. log_probs is (T, V) for one clip."""
    best = log_probs.argmax(-1)
    probabilities = np.exp(log_probs.max(-1))
    glosses: list[AlignedGloss] = []
    step = 0
    while step < len(best):
        token = int(best[step])
        end = step
        while end + 1 < len(best) and int(best[end + 1]) == token:
            end += 1
        if token != 0:
            glosses.append(
                AlignedGloss(id_to_gloss.get(token, f"<{token}>"), step, end, float(probabilities[step : end + 1].mean()))
            )
        step = end + 1
    return glosses


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp in enumerate(hypothesis, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp))
        previous = current
    return previous[-1]
