"""GCN adapter for the existing static, temporal and fusion pipelines."""
from dataclasses import asdict, dataclass

from models.gcn.gcn_model import GCNConfig, StaticGCN
from unified_bn.backbones.base import BaseBackbone
from unified_bn.core.types import BackboneOutput


@dataclass(frozen=True)
class GCNBackboneConfig:
    n_rois: int = 100
    n_layers: int = 2
    hidden_dim: int = 64
    dropout: float = 0.1
    adjacency: str = "positive"
    node_features: str = "fc_rows"
    readout: str = "mean"

    def __post_init__(self):
        GCNConfig(**asdict(self))

    def to_reference_config(self, num_classes=2):
        return GCNConfig(**asdict(self), num_classes=num_classes)


class GCNBackbone(BaseBackbone):
    def __init__(self, config=None, *, model=None, num_classes=2):
        config = GCNBackboneConfig() if config is None else config
        if not isinstance(config, GCNBackboneConfig):
            raise TypeError("config must be GCNBackboneConfig.")
        expected = config.to_reference_config(num_classes)
        if model is None:
            model = StaticGCN(expected)
        elif not isinstance(model, StaticGCN):
            raise TypeError("model must be StaticGCN.")
        elif model.config != expected:
            raise ValueError("Supplied GCN model configuration differs from its adapter.")
        super().__init__(n_rois=config.n_rois, embedding_dim=model.embedding_dim)
        self.config = config
        self.reference_model = model

    def _encode_impl(self, fc):
        return BackboneOutput(embedding=self.reference_model(fc, return_embedding=True), aux={})

    def classify(self, fc):
        return self.reference_model(fc)

    def reference_state_dict(self):
        return self.reference_model.state_dict()
