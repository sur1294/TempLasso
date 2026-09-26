from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto

import torch


class SelectorState(Enum):
    UNFITTED = auto()
    FITTED = auto()
    FROZEN = auto()


class BaseSelector(ABC):
    """Train-only feature selector with an explicit leakage-safe lifecycle."""

    def __init__(self) -> None:
        self._state = SelectorState.UNFITTED

    @property
    def state(self) -> SelectorState:
        return self._state

    @property
    def is_frozen(self) -> bool:
        return self._state is SelectorState.FROZEN

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor) -> "BaseSelector":
        if self._state is not SelectorState.UNFITTED:
            raise RuntimeError(f"Selector.fit() may be called exactly once. Current state: {self._state.name}.")
        self._fit_impl(x_train, y_train)
        self._state = SelectorState.FITTED
        return self

    def freeze(self) -> "BaseSelector":
        if self._state is SelectorState.UNFITTED:
            raise RuntimeError("Cannot freeze an unfitted selector.")
        self._freeze_impl()
        self._state = SelectorState.FROZEN
        return self

    def transform(self, x: torch.Tensor, *, require_frozen: bool = True) -> torch.Tensor:
        if self._state is SelectorState.UNFITTED:
            raise RuntimeError("Selector must be fitted before transform().")
        if require_frozen and self._state is not SelectorState.FROZEN:
            raise RuntimeError("Validation/test-safe transform requires a frozen selector.")
        out = self._transform_impl(x)
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"{type(self).__name__}._transform_impl() must return a torch.Tensor.")
        if not torch.isfinite(out).all():
            raise ValueError(f"{type(self).__name__} produced NaN or Inf.")
        return out

    @abstractmethod
    def _fit_impl(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        raise NotImplementedError

    def _freeze_impl(self) -> None:
        """Optional hook for finalizing fitted state."""

    @abstractmethod
    def _transform_impl(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict:
        """Return serializable fitted state and provenance."""
        raise NotImplementedError
