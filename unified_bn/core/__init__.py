"""Core shared contracts and validation utilities."""

from .types import BackboneOutput, FusionOutput, TemporalOutput
from .validation import validate_fc_batch, validate_lengths, validate_sequence_embeddings

__all__ = [
    "BackboneOutput", "TemporalOutput", "FusionOutput",
    "validate_fc_batch", "validate_lengths", "validate_sequence_embeddings",
]
