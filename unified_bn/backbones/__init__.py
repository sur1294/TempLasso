"""Backbone interfaces and implementations."""

from .base import BaseBackbone
from .bnt import BNTBackbone, BNTBackboneConfig
from .gcn import GCNBackbone, GCNBackboneConfig
from .alter import ALTERBackbone, ALTERBackboneConfig
from .bqn import BQNBackbone, BQNBackboneConfig
from .brainnetcnn import BrainNetCNNBackbone, BrainNetCNNBackboneConfig
from .selected_edges import SelectedEdgeBackbone, SelectedEdgeBackboneConfig

__all__ = [
    "BaseBackbone",
    "BNTBackbone",
    "BNTBackboneConfig",
    "GCNBackbone",
    "GCNBackboneConfig",
    "ALTERBackbone",
    "ALTERBackboneConfig",
    "BQNBackbone",
    "BQNBackboneConfig",
    "BrainNetCNNBackbone",
    "BrainNetCNNBackboneConfig",
    "SelectedEdgeBackbone",
    "SelectedEdgeBackboneConfig",
]
