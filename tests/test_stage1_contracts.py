from __future__ import annotations

import pytest
import torch

from unified_bn.backbones.base import BaseBackbone
from unified_bn.core.types import BackboneOutput, FusionOutput, TemporalOutput
from unified_bn.fusion.base import BaseFusion
from unified_bn.selectors.base import BaseSelector, SelectorState
from unified_bn.temporal.base import BaseTemporalAggregator


class DummyBackbone(BaseBackbone):
    def __init__(self) -> None:
        super().__init__(n_rois=100, embedding_dim=8)
    def _encode_impl(self, fc: torch.Tensor) -> BackboneOutput:
        emb = fc.mean(dim=(1, 2)).unsqueeze(1).repeat(1, 8)
        return BackboneOutput(embedding=emb)


class DummyTemporal(BaseTemporalAggregator):
    def __init__(self) -> None:
        super().__init__(input_dim=8, output_dim=8)
    def _aggregate_impl(self, sequence: torch.Tensor, lengths: torch.Tensor) -> TemporalOutput:
        rows = [sequence[i, :int(length)].mean(dim=0) for i, length in enumerate(lengths)]
        return TemporalOutput(embedding=torch.stack(rows))


class DummyFusion(BaseFusion):
    def __init__(self) -> None:
        super().__init__(static_dim=8, temporal_dim=8, output_dim=16)
    def _fuse_impl(self, h_static: torch.Tensor, h_temporal: torch.Tensor) -> FusionOutput:
        return FusionOutput(embedding=torch.cat([h_static, h_temporal], dim=-1))


class DummySelector(BaseSelector):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = None
    def _fit_impl(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        del y_train
        self.threshold = float(x_train.abs().mean())
    def _transform_impl(self, x: torch.Tensor) -> torch.Tensor:
        assert self.threshold is not None
        return torch.where(x.abs() >= self.threshold, x, torch.zeros_like(x))
    def state_dict(self) -> dict:
        return {"state": self.state.name, "threshold": self.threshold}


def test_backbone_contract() -> None:
    model = DummyBackbone()
    fc = torch.randn(4, 100, 100)
    out = model(fc)
    assert out.embedding.shape == (4, 8)
    assert torch.equal(model.encode(fc), out.embedding)


def test_backbone_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        DummyBackbone()(torch.randn(4, 100, 99))


def test_temporal_contract_with_variable_lengths() -> None:
    out = DummyTemporal()(torch.randn(3, 5, 8), torch.tensor([5, 3, 1], dtype=torch.long))
    assert out.embedding.shape == (3, 8)


def test_temporal_rejects_invalid_lengths() -> None:
    with pytest.raises(ValueError):
        DummyTemporal()(torch.randn(2, 5, 8), torch.tensor([5, 6], dtype=torch.long))


def test_fusion_contract() -> None:
    out = DummyFusion()(torch.randn(4, 8), torch.randn(4, 8))
    assert out.embedding.shape == (4, 16)


def test_selector_lifecycle_prevents_unsafe_transform() -> None:
    selector = DummySelector()
    x_train = torch.randn(20, 10)
    y_train = torch.randint(0, 2, (20,))
    assert selector.state is SelectorState.UNFITTED
    with pytest.raises(RuntimeError):
        selector.transform(x_train)
    selector.fit(x_train, y_train)
    assert selector.state is SelectorState.FITTED
    with pytest.raises(RuntimeError):
        selector.transform(x_train)
    selector.freeze()
    assert selector.state is SelectorState.FROZEN
    assert selector.transform(x_train).shape == x_train.shape


def test_selector_cannot_be_refit() -> None:
    selector = DummySelector()
    x_train = torch.randn(20, 10)
    y_train = torch.randint(0, 2, (20,))
    selector.fit(x_train, y_train)
    with pytest.raises(RuntimeError):
        selector.fit(x_train, y_train)
