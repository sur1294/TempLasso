"""Adapter for the supplied GraphTransformer (registry name: bnt)."""
from dataclasses import asdict, dataclass

from models.bnt.bnt_model import BNTConfig, StaticBNT
from unified_bn.backbones.base import BaseBackbone
from unified_bn.core.types import BackboneOutput


@dataclass(frozen=True)
class BNTBackboneConfig:
    n_rois: int = 100
    n_layers: int = 2
    readout: str = "mean"
    nhead: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1

    def __post_init__(self):
        BNTConfig(**asdict(self))

    def to_reference_config(self, num_classes=2):
        return BNTConfig(**asdict(self), num_classes=num_classes)


class BNTBackbone(BaseBackbone):
    def __init__(self, config=None, *, model=None, num_classes=2):
        config = BNTBackboneConfig() if config is None else config
        if not isinstance(config, BNTBackboneConfig):
            raise TypeError("config must be BNTBackboneConfig.")
        expected = config.to_reference_config(num_classes)
        if model is None:
            model = StaticBNT(expected)
        elif not isinstance(model, StaticBNT):
            raise TypeError("model must be StaticBNT.")
        elif model.config != expected:
            raise ValueError("Supplied BNT model configuration differs from its adapter.")
        super().__init__(n_rois=config.n_rois, embedding_dim=model.embedding_dim)
        self.config = config
        self.reference_model = model

    def _encode_impl(self, fc):
        return BackboneOutput(embedding=self.reference_model(fc, return_embedding=True), aux={})

    def classify(self, fc):
        return self.reference_model(fc)

    def reference_state_dict(self):
        return self.reference_model.state_dict()
