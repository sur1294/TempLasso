from __future__ import annotations

import torch


def validate_fc_batch(fc: torch.Tensor, *, n_rois: int | None = None, name: str = "fc", require_finite: bool = True) -> None:
    """Validate dense FC matrices shaped [B,R,R]."""
    if not isinstance(fc, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(fc)!r}.")
    if fc.ndim != 3:
        raise ValueError(f"{name} must have shape [B,R,R], got {tuple(fc.shape)}.")
    if fc.shape[1] != fc.shape[2]:
        raise ValueError(f"{name} must contain square matrices, got {tuple(fc.shape)}.")
    if n_rois is not None and fc.shape[1:] != (n_rois, n_rois):
        raise ValueError(f"{name} expected R={n_rois}, got shape {tuple(fc.shape)}.")
    if not torch.is_floating_point(fc):
        raise TypeError(f"{name} must be floating point, got dtype={fc.dtype}.")
    if require_finite and not torch.isfinite(fc).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def validate_sequence_embeddings(sequence: torch.Tensor, *, embedding_dim: int | None = None, name: str = "sequence", require_finite: bool = True) -> None:
    """Validate temporal embeddings shaped [B,K,D]."""
    if not isinstance(sequence, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(sequence)!r}.")
    if sequence.ndim != 3:
        raise ValueError(f"{name} must have shape [B,K,D], got {tuple(sequence.shape)}.")
    if embedding_dim is not None and sequence.shape[-1] != embedding_dim:
        raise ValueError(f"{name} expected D={embedding_dim}, got {sequence.shape[-1]}.")
    if not torch.is_floating_point(sequence):
        raise TypeError(f"{name} must be floating point, got dtype={sequence.dtype}.")
    if require_finite and not torch.isfinite(sequence).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def validate_lengths(lengths: torch.Tensor, *, batch_size: int, max_steps: int, name: str = "lengths") -> None:
    """Validate variable-length sequence lengths shaped [B]."""
    if not isinstance(lengths, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(lengths)!r}.")
    if lengths.ndim != 1 or lengths.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [{batch_size}], got {tuple(lengths.shape)}.")
    if lengths.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise TypeError(f"{name} must use an integer dtype, got {lengths.dtype}.")
    if torch.any(lengths < 1):
        raise ValueError(f"{name} must contain only positive lengths.")
    if torch.any(lengths > max_steps):
        raise ValueError(f"{name} contains a value larger than K={max_steps}.")
