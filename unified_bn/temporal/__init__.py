"""Temporal aggregation components."""

from .base import BaseTemporalAggregator
from .bigru import BiGRUAggregator, BiGRUConfig
from .pooling import (
    BaseTemporalPooler,
    FinalStatePooling,
    MaskedAttentionPooling,
    MaskedMeanPooling,
    PoolingOutput,
)
from .windowing import (
    WindowPolicyConfig,
    apply_window_policy,
    resolve_global_min_windows,
)

__all__ = [
    "BaseTemporalAggregator",
    "BiGRUAggregator",
    "BiGRUConfig",
    "BaseTemporalPooler",
    "FinalStatePooling",
    "MaskedAttentionPooling",
    "MaskedMeanPooling",
    "PoolingOutput",
    "WindowPolicyConfig",
    "apply_window_policy",
    "resolve_global_min_windows",
]
