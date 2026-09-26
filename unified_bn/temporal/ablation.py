"""Opt-in temporal controls; the production BiGRU is used unchanged."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from unified_bn.core.types import TemporalOutput
from unified_bn.data.collate import collate_temporal
from unified_bn.temporal.base import BaseTemporalAggregator
from unified_bn.temporal.bigru import BiGRUAggregator, BiGRUConfig
from unified_bn.temporal.pooling import lengths_to_mask


VARIANTS = ("mlp_mean", "gru", "shuffled_bigru", "bigru")


class MLPMeanAggregator(BaseTemporalAggregator):
    """One shared two-linear-layer ReLU MLP, then mean over VALID windows."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.mlp = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(),
                                 nn.Linear(output_dim, output_dim), nn.ReLU())

    def _aggregate_impl(self, sequence, lengths):
        mask = lengths_to_mask(lengths, max_steps=sequence.shape[1], device=sequence.device)
        # Do not feed padding to the MLP, including its learned biases.
        transformed = sequence.new_zeros((*sequence.shape[:2], self.output_dim))
        transformed[mask] = self.mlp(sequence[mask])
        embedding = transformed.sum(dim=1) / lengths.to(sequence.device, sequence.dtype)[:, None]
        return TemporalOutput(embedding=embedding)


class GRUFinalAggregator(BaseTemporalAggregator):
    """Unidirectional GRU, with the final VALID hidden state as the readout."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.gru = nn.GRU(input_dim, output_dim, batch_first=True, bidirectional=False)

    def _aggregate_impl(self, sequence, lengths):
        packed = pack_padded_sequence(sequence, lengths.detach().cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, hidden = self.gru(packed)
        return TemporalOutput(embedding=hidden[-1])


def make_aggregator(variant: str, input_dim: int, hidden_dim: int):
    """All heads receive 2h features. Only the two BiGRUs match parameter counts.

    Shuffling is a data control, implemented by TemporalAblationCollate. A
    shuffled BiGRU is NOT a mathematically permutation-invariant model.
    """
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError("hidden_dim must be a positive integer")
    if variant == "mlp_mean":
        return MLPMeanAggregator(input_dim, 2 * hidden_dim)
    if variant == "gru":
        return GRUFinalAggregator(input_dim, 2 * hidden_dim)
    if variant in ("bigru", "shuffled_bigru"):
        return BiGRUAggregator(BiGRUConfig(input_dim=input_dim, hidden_dim=hidden_dim,
                                         num_layers=1, pooling="final"))
    raise ValueError(f"Unknown temporal ablation: {variant}")


def subject_permutation(subject_id: str, length: int, seed: int) -> torch.Tensor:
    """Fixed pseudorandom permutation per subject, independent of batch/RNG state.

    Hash-ranked window indices make the protocol portable across NumPy/Torch
    versions. No label, split role, feature value, or outcome enters the key.
    Length-one scans and identity permutations are allowed, not discarded.
    """
    if length < 1 or seed < 0:
        raise ValueError("Positive length and nonnegative shuffle seed required")
    prefix = json.dumps(["templasso-shuffle-v1", seed, str(subject_id)],
                        ensure_ascii=True, separators=(",", ":")).encode()
    order = sorted(range(length), key=lambda i: (
        hashlib.sha256(prefix + b":" + str(i).encode()).digest(), i))
    return torch.tensor(order, dtype=torch.long)


@dataclass(frozen=True)
class TemporalAblationCollate:
    """Same frozen subject permutation for training, validation and final test."""

    variant: str
    shuffle_seed: int

    def __post_init__(self):
        if self.variant not in VARIANTS or self.shuffle_seed < 0:
            raise ValueError("Invalid ablation variant or shuffle seed")

    def __call__(self, samples):
        batch = collate_temporal(samples)
        if self.variant == "shuffled_bigru":
            for row, (sid, length) in enumerate(zip(batch.subject_ids, batch.lengths.tolist())):
                perm = subject_permutation(sid, length, self.shuffle_seed).to(batch.dynamic_fc.device)
                batch.dynamic_fc[row, :length] = batch.dynamic_fc[row, :length].index_select(0, perm)
        return batch
