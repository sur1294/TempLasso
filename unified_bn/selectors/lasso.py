from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from unified_bn.selectors.base import BaseSelector

EdgeInputSpace = Literal["fisher_z", "correlation"]

# arctanh(1 - 1e-6) ~= 7.25; keeps Fisher-z finite in float32 when a
# correlation matrix arrives at exactly +-1.
_CORR_CLIP = 1.0 - 1e-6


@dataclass(frozen=True)
class LassoEdgeSelectorConfig:
    """
    Configuration for TRAIN-only L1 edge selection.

    C
        Inverse L1 strength (sklearn convention). Smaller C -> fewer edges.
    random_state
        Pinned into liblinear. Its coordinate-descent order is otherwise
        randomised, which was measured to move |S| by +-1-2 edges across
        identical fits and, through the validation argmax, flip which
        candidate wins.
    input_space
        ``fisher_z``: fit and transform on arctanh(correlation). This is the
        variance-stabilised space the temporal artifact is stored in and the
        space in which the standalone TempLasso result was validated.
        ``correlation``: operate on the reconstructed correlation values the
        loader hands to BQN.
    min_edges
        Fail closed if fewer edges survive; a near-empty S is a degenerate
        candidate, not a model.
    """

    C: float = 0.01
    random_state: int = 0
    max_iter: int = 3000
    input_space: EdgeInputSpace = "fisher_z"
    min_edges: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.C, (int, float)) or isinstance(self.C, bool):
            raise TypeError("C must be a number.")
        if self.C <= 0.0:
            raise ValueError("C must be positive.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer.")
        if self.random_state < 0:
            raise ValueError("random_state must be nonnegative.")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int):
            raise TypeError("max_iter must be an integer.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.input_space not in ("fisher_z", "correlation"):
            raise ValueError(
                "input_space must be 'fisher_z' or 'correlation', "
                f"got {self.input_space!r}."
            )
        if isinstance(self.min_edges, bool) or not isinstance(self.min_edges, int):
            raise TypeError("min_edges must be an integer.")
        if self.min_edges <= 0:
            raise ValueError("min_edges must be positive.")


class LassoEdgeSelector(BaseSelector):
    """
    L1-logistic edge selection with a leakage-safe fit/freeze lifecycle.

    Fitting
    -------
    ``fit(x_train, y_train)`` takes the pooled TRAIN windows as rows,
    ``[N_windows, E]``, each row labelled with its subject's label. One
    L1-penalised logistic regression is fit over that matrix; the indices of
    the non-zero coefficients become the edge set ``S``. The fitted weights
    are discarded -- only the column list, and the per-edge standardisation
    statistics computed on the same TRAIN rows, are retained.

    Pooling windows rather than fitting per window index is deliberate.
    Per-index fitting has only ~700 rows against ~5,000 edges and produced
    near-disjoint edge sets across indices (mean pairwise Jaccard 0.01-0.04,
    with 0-1 edges surviving a >=50% vote); pooling gives ~12k rows and a
    stable set that also scored higher (0.729 +/- 0.016 vs 0.708 +/- 0.037
    AUC on ABIDE/Schaefer-100).

    One shared ``S`` is applied to every subject and every window, so feature
    ``j`` denotes the same edge at every timestep -- the property a recurrent
    aggregator needs to track an edge through time.

    Transform
    ---------
    ``transform(x)`` accepts ``[..., E]`` and returns ``[..., |S|]``, selected
    and standardised with the frozen TRAIN statistics. It never re-estimates
    anything, so applying it to validation/test rows cannot leak.
    """

    def __init__(self, config: LassoEdgeSelectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or LassoEdgeSelectorConfig()
        if not isinstance(self.config, LassoEdgeSelectorConfig):
            raise TypeError(
                "config must be LassoEdgeSelectorConfig, got "
                f"{type(self.config)!r}."
            )
        self._n_edges: int | None = None
        self._selected: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._n_train_rows: int | None = None

    # ------------------------------------------------------------ properties
    @property
    def n_edges(self) -> int:
        self._require_fitted()
        return int(self._n_edges)  # type: ignore[arg-type]

    @property
    def n_selected(self) -> int:
        self._require_fitted()
        return int(self._selected.size)  # type: ignore[union-attr]

    @property
    def selected_indices(self) -> np.ndarray:
        self._require_fitted()
        return self._selected.copy()  # type: ignore[union-attr]

    @property
    def train_mean(self) -> np.ndarray:
        self._require_fitted()
        return self._mu.copy()  # type: ignore[union-attr]

    @property
    def train_std(self) -> np.ndarray:
        self._require_fitted()
        return self._sd.copy()  # type: ignore[union-attr]

    def _require_fitted(self) -> None:
        if self._selected is None:
            raise RuntimeError("LassoEdgeSelector has not been fitted.")

    # ------------------------------------------------------------ lifecycle
    def _fit_impl(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LassoEdgeSelector requires scikit-learn."
            ) from exc

        x = _as_2d_float64(x_train, name="x_train")
        y = _as_binary_labels(y_train, n_rows=x.shape[0])

        n_rows, n_edges = x.shape
        if n_rows < 2:
            raise ValueError("LassoEdgeSelector needs at least two training rows.")
        if np.unique(y).size < 2:
            raise ValueError(
                "LassoEdgeSelector requires both classes in the training rows."
            )

        mu = x.mean(axis=0)
        sd = x.std(axis=0) + 1e-8
        xz = (x - mu) / sd

        def _l1_model() -> "LogisticRegression":
            return LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=float(self.config.C),
                max_iter=int(self.config.max_iter),
                random_state=int(self.config.random_state),
            )

        classes = np.unique(y)
        if classes.size == 2:
            # Historical binary path, unchanged.
            coefs = _l1_model().fit(xz, y).coef_
        else:
            # K > 2: explicit one-vs-rest (recent scikit-learn refuses
            # multiclass liblinear); the selected set is the UNION of the
            # edges any class-vs-rest L1 model keeps.
            coefs = np.stack(
                [_l1_model().fit(xz, (y == c).astype(np.int64)).coef_[0] for c in classes]
            )

        selected = np.flatnonzero((coefs != 0.0).any(axis=0)).astype(np.int64)
        if selected.size < self.config.min_edges:
            raise ValueError(
                f"L1 selection kept {selected.size} edges at C={self.config.C}, "
                f"below min_edges={self.config.min_edges}. Increase C."
            )

        self._n_edges = int(n_edges)
        self._n_train_rows = int(n_rows)
        self._selected = selected
        self._mu = mu.astype(np.float64)
        self._sd = sd.astype(np.float64)

    def _freeze_impl(self) -> None:
        # Nothing to finalise: fitted state is already immutable numpy arrays.
        return None

    def _transform_impl(self, x: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)!r}.")
        if x.ndim < 1 or int(x.shape[-1]) != self._n_edges:
            raise ValueError(
                f"x must have trailing dimension E={self._n_edges}, "
                f"got shape {tuple(x.shape)}."
            )
        idx = torch.as_tensor(self._selected, device=x.device, dtype=torch.long)
        mu = torch.as_tensor(self._mu[self._selected], device=x.device, dtype=x.dtype)
        sd = torch.as_tensor(self._sd[self._selected], device=x.device, dtype=x.dtype)
        return (x.index_select(-1, idx) - mu) / sd

    # ------------------------------------------------------------ provenance
    def state_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "selector": "lasso_edges",
            "config": asdict(self.config),
            "n_edges": int(self._n_edges),  # type: ignore[arg-type]
            "n_selected": int(self._selected.size),  # type: ignore[union-attr]
            "n_train_rows": int(self._n_train_rows),  # type: ignore[arg-type]
            "selected_indices": self._selected.tolist(),  # type: ignore[union-attr]
            "train_mean": self._mu.tolist(),  # type: ignore[union-attr]
            "train_std": self._sd.tolist(),  # type: ignore[union-attr]
            "edge_set_fingerprint": self.edge_set_fingerprint(),
            "state": self.state.name,
        }

    def edge_set_fingerprint(self) -> str:
        """SHA-256 over the ordered selected indices; provenance for S."""
        self._require_fitted()
        digest = hashlib.sha256()
        digest.update(int(self._n_edges).to_bytes(8, "big"))  # type: ignore[arg-type]
        digest.update(self._selected.astype("<i8").tobytes())  # type: ignore[union-attr]
        return "sha256:" + digest.hexdigest()

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "LassoEdgeSelector":
        """Rebuild a FROZEN selector from ``state_dict()`` output."""
        if payload.get("selector") != "lasso_edges":
            raise ValueError("state_dict does not describe a lasso_edges selector.")
        selector = cls(LassoEdgeSelectorConfig(**payload["config"]))
        selector._n_edges = int(payload["n_edges"])
        selector._n_train_rows = int(payload["n_train_rows"])
        selector._selected = np.asarray(payload["selected_indices"], dtype=np.int64)
        selector._mu = np.asarray(payload["train_mean"], dtype=np.float64)
        selector._sd = np.asarray(payload["train_std"], dtype=np.float64)
        if selector._mu.size != selector._n_edges or selector._sd.size != selector._n_edges:
            raise ValueError("state_dict statistics do not match n_edges.")
        if selector._selected.size != int(payload["n_selected"]):
            raise ValueError("state_dict n_selected disagrees with selected_indices.")
        from unified_bn.selectors.base import SelectorState

        selector._state = SelectorState.FITTED
        selector.freeze()
        if selector.edge_set_fingerprint() != payload["edge_set_fingerprint"]:
            raise ValueError("Rebuilt edge set does not match stored fingerprint.")
        return selector


# ---------------------------------------------------------------------------
# Pooled TRAIN windows from a temporal dataset
# ---------------------------------------------------------------------------

def pooled_edge_rows_from_dataset(
    dataset: Dataset,
    *,
    n_rois: int,
    input_space: EdgeInputSpace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stack every window of every subject in ``dataset`` as one row.

    Returns ``(x, y)`` with ``x`` shaped ``[sum_i K_i, E]`` and ``y`` the
    subject label repeated ``K_i`` times. Rows are in dataset order, then
    window order, so the result is deterministic for a fixed dataset.

    Intended to be called on the TRAIN partition only. Nothing in this
    function enforces that -- the caller owns the leakage boundary.
    """

    if isinstance(n_rois, bool) or not isinstance(n_rois, int) or n_rois <= 1:
        raise ValueError("n_rois must be an integer > 1.")

    rows, cols = torch.triu_indices(n_rois, n_rois, offset=1)
    x_chunks: list[torch.Tensor] = []
    y_chunks: list[torch.Tensor] = []

    for i in range(len(dataset)):  # type: ignore[arg-type]
        sample = dataset[i]
        fc = sample.dynamic_fc
        if fc.ndim != 3 or fc.shape[1:] != (n_rois, n_rois):
            raise ValueError(
                f"Sample {i} dynamic_fc must be [K,{n_rois},{n_rois}], "
                f"got {tuple(fc.shape)}."
            )
        edges = fc[:, rows, cols]
        edges = _to_input_space(edges, input_space)
        x_chunks.append(edges.to(torch.float32))
        y_chunks.append(
            torch.full((int(fc.shape[0]),), int(sample.label), dtype=torch.long)
        )

    if not x_chunks:
        raise ValueError("Dataset yielded no windows.")
    return torch.cat(x_chunks, dim=0), torch.cat(y_chunks, dim=0)


def upper_triangle_edges(fc: torch.Tensor, *, input_space: EdgeInputSpace) -> torch.Tensor:
    """``[..., R, R]`` -> ``[..., E]`` in the requested input space."""
    if fc.ndim < 2 or fc.shape[-1] != fc.shape[-2]:
        raise ValueError(f"fc must be [..., R, R], got {tuple(fc.shape)}.")
    n_rois = int(fc.shape[-1])
    rows, cols = torch.triu_indices(n_rois, n_rois, offset=1, device=fc.device)
    return _to_input_space(fc[..., rows, cols], input_space)


def _to_input_space(corr_edges: torch.Tensor, input_space: EdgeInputSpace) -> torch.Tensor:
    if input_space == "correlation":
        return corr_edges
    if input_space == "fisher_z":
        return torch.atanh(corr_edges.clamp(-_CORR_CLIP, _CORR_CLIP))
    raise ValueError(f"Unknown input_space {input_space!r}.")


def _as_2d_float64(x: torch.Tensor, *, name: str) -> np.ndarray:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)!r}.")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D [N,E], got {tuple(x.shape)}.")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return x.detach().cpu().to(torch.float64).numpy()


def _as_binary_labels(y: torch.Tensor, *, n_rows: int) -> np.ndarray:
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"y_train must be a torch.Tensor, got {type(y)!r}.")
    arr = y.detach().cpu().reshape(-1).numpy()
    if arr.shape[0] != n_rows:
        raise ValueError(
            f"y_train has {arr.shape[0]} rows but x_train has {n_rows}."
        )
    if not np.all(np.equal(np.mod(arr, 1), 0)):
        raise ValueError("y_train must contain integer class labels.")
    if (arr < 0).any():
        raise ValueError("y_train must contain nonnegative class labels (0..K-1).")
    return arr.astype(np.int64)


__all__: Sequence[str] = (
    "EdgeInputSpace",
    "LassoEdgeSelector",
    "LassoEdgeSelectorConfig",
    "pooled_edge_rows_from_dataset",
    "upper_triangle_edges",
)
