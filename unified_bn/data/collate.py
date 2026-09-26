from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from unified_bn.data.common import StaticSample, TemporalSample
from unified_bn.data.abide_fusion import FusionSample

@dataclass
class StaticBatch:
    subject_ids: tuple[str, ...]
    labels: torch.Tensor
    fc: torch.Tensor
    source_indices: torch.Tensor
    def to(self, device, *, non_blocking=False):
        return StaticBatch(
            self.subject_ids,
            self.labels.to(device, non_blocking=non_blocking),
            self.fc.to(device, non_blocking=non_blocking),
            self.source_indices.to(device, non_blocking=non_blocking),
        )
    def pin_memory(self):
        return StaticBatch(
            self.subject_ids, self.labels.pin_memory(), self.fc.pin_memory(),
            self.source_indices.pin_memory()
        )

@dataclass
class TemporalBatch:
    subject_ids: tuple[str, ...]
    labels: torch.Tensor
    dynamic_fc: torch.Tensor
    lengths: torch.Tensor
    source_indices: torch.Tensor
    def to(self, device, *, non_blocking=False):
        return TemporalBatch(
            self.subject_ids,
            self.labels.to(device, non_blocking=non_blocking),
            self.dynamic_fc.to(device, non_blocking=non_blocking),
            self.lengths.to(device, non_blocking=non_blocking),
            self.source_indices.to(device, non_blocking=non_blocking),
        )
    def pin_memory(self):
        return TemporalBatch(
            self.subject_ids, self.labels.pin_memory(), self.dynamic_fc.pin_memory(),
            self.lengths.pin_memory(), self.source_indices.pin_memory()
        )

@dataclass
class FusionBatch:
    subject_ids: tuple[str, ...]
    labels: torch.Tensor
    static_fc: torch.Tensor
    dynamic_fc: torch.Tensor
    lengths: torch.Tensor
    static_source_indices: torch.Tensor
    temporal_source_indices: torch.Tensor
    def to(self, device, *, non_blocking=False):
        return FusionBatch(
            self.subject_ids,
            self.labels.to(device, non_blocking=non_blocking),
            self.static_fc.to(device, non_blocking=non_blocking),
            self.dynamic_fc.to(device, non_blocking=non_blocking),
            self.lengths.to(device, non_blocking=non_blocking),
            self.static_source_indices.to(device, non_blocking=non_blocking),
            self.temporal_source_indices.to(device, non_blocking=non_blocking),
        )
    def pin_memory(self):
        return FusionBatch(
            self.subject_ids,
            self.labels.pin_memory(),
            self.static_fc.pin_memory(),
            self.dynamic_fc.pin_memory(),
            self.lengths.pin_memory(),
            self.static_source_indices.pin_memory(),
            self.temporal_source_indices.pin_memory(),
        )

def collate_static(samples: Sequence[StaticSample]) -> StaticBatch:
    if not samples:
        raise ValueError("Cannot collate an empty static batch.")
    shape = tuple(samples[0].fc.shape)
    if any(tuple(s.fc.shape) != shape for s in samples):
        raise ValueError("Static samples have inconsistent FC shapes.")
    return StaticBatch(
        tuple(s.subject_id for s in samples),
        torch.tensor([s.label for s in samples], dtype=torch.long),
        torch.stack([s.fc for s in samples]),
        torch.tensor([s.source_index for s in samples], dtype=torch.long),
    )

def collate_temporal(samples: Sequence[TemporalSample]) -> TemporalBatch:
    if not samples:
        raise ValueError("Cannot collate an empty temporal batch.")
    n_rois = int(samples[0].dynamic_fc.shape[-1])
    dtype, device = samples[0].dynamic_fc.dtype, samples[0].dynamic_fc.device
    max_k = max(int(s.length) for s in samples)
    batch = torch.zeros(
        len(samples), max_k, n_rois, n_rois, dtype=dtype, device=device
    )
    for i, sample in enumerate(samples):
        if sample.length <= 0:
            raise ValueError("Temporal sample length must be positive.")
        if sample.dynamic_fc.ndim != 3:
            raise ValueError("Each temporal sample must have shape [K,R,R].")
        if sample.dynamic_fc.shape[0] != sample.length:
            raise ValueError("Temporal sample length disagrees with dynamic_fc K.")
        if sample.dynamic_fc.shape[1:] != (n_rois, n_rois):
            raise ValueError("Temporal samples have inconsistent ROI shapes.")
        if sample.dynamic_fc.dtype != dtype:
            raise TypeError("Temporal samples have inconsistent dtypes.")
        batch[i, :sample.length] = sample.dynamic_fc
    return TemporalBatch(
        tuple(s.subject_id for s in samples),
        torch.tensor([s.label for s in samples], dtype=torch.long),
        batch,
        torch.tensor([s.length for s in samples], dtype=torch.long),
        torch.tensor([s.source_index for s in samples], dtype=torch.long),
    )

def collate_fusion(samples: Sequence[FusionSample]) -> FusionBatch:
    if not samples:
        raise ValueError("Cannot collate an empty fusion batch.")

    first = samples[0]
    if first.static_fc.ndim != 2 or first.static_fc.shape[0] != first.static_fc.shape[1]:
        raise ValueError("Each static fusion FC must have shape [R,R].")

    n_rois = int(first.static_fc.shape[0])
    static_dtype = first.static_fc.dtype
    static_device = first.static_fc.device
    temporal_dtype = first.dynamic_fc.dtype
    temporal_device = first.dynamic_fc.device
    max_k = max(int(sample.length) for sample in samples)

    static_batch = []
    dynamic_batch = torch.zeros(
        len(samples),
        max_k,
        n_rois,
        n_rois,
        dtype=temporal_dtype,
        device=temporal_device,
    )

    for i, sample in enumerate(samples):
        if sample.static_fc.ndim != 2 or tuple(sample.static_fc.shape) != (
            n_rois,
            n_rois,
        ):
            raise ValueError("Fusion samples have inconsistent static FC shapes.")
        if sample.static_fc.dtype != static_dtype:
            raise TypeError("Fusion static samples have inconsistent dtypes.")
        if sample.static_fc.device != static_device:
            raise ValueError("Fusion static samples must be on the same device.")
        if sample.length <= 0:
            raise ValueError("Fusion temporal sample length must be positive.")
        if sample.dynamic_fc.ndim != 3:
            raise ValueError("Each fusion temporal sample must have shape [K,R,R].")
        if sample.dynamic_fc.shape[0] != sample.length:
            raise ValueError("Fusion temporal length disagrees with dynamic_fc K.")
        if tuple(sample.dynamic_fc.shape[1:]) != (n_rois, n_rois):
            raise ValueError(
                "Fusion static and temporal ROI shapes must match across samples."
            )
        if sample.dynamic_fc.dtype != temporal_dtype:
            raise TypeError("Fusion temporal samples have inconsistent dtypes.")
        if sample.dynamic_fc.device != temporal_device:
            raise ValueError("Fusion temporal samples must be on the same device.")

        static_batch.append(sample.static_fc)
        dynamic_batch[i, : sample.length] = sample.dynamic_fc

    return FusionBatch(
        subject_ids=tuple(sample.subject_id for sample in samples),
        labels=torch.tensor([sample.label for sample in samples], dtype=torch.long),
        static_fc=torch.stack(static_batch),
        dynamic_fc=dynamic_batch,
        lengths=torch.tensor([sample.length for sample in samples], dtype=torch.long),
        static_source_indices=torch.tensor(
            [sample.static_source_index for sample in samples], dtype=torch.long
        ),
        temporal_source_indices=torch.tensor(
            [sample.temporal_source_index for sample in samples], dtype=torch.long
        ),
    )
