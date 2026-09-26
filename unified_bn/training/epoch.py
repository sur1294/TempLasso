from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Iterable, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from unified_bn.data.collate import FusionBatch, StaticBatch, TemporalBatch
from unified_bn.training.metrics import (
    BinaryClassificationMetrics,
    BinaryMetricAccumulator,
)


PipelineBatch = Union[StaticBatch, TemporalBatch, FusionBatch]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class EpochRunnerError(RuntimeError):
    """Raised when the generic Stage 6 epoch-runner contract is violated."""


def _forward_pipeline_batch(
    model: nn.Module,
    batch: PipelineBatch,
) -> torch.Tensor:
    """
    Dispatch one validated unified_bn batch to a pipeline model.

    Scientific boundary
    -------------------
    This is the only mode-specific adapter required by the epoch runner:

    - ``StaticBatch`` forwards only ``fc``.
    - ``TemporalBatch`` forwards only ``dynamic_fc`` and ``lengths``.
    - ``FusionBatch`` forwards only ``static_fc``, ``dynamic_fc``, and
      ``lengths``.

    Subject IDs and source indices are provenance metadata. They are never
    forwarded to the predictive model.
    """

    if isinstance(batch, StaticBatch):
        output = model(batch.fc)
    elif isinstance(batch, TemporalBatch):
        output = model(batch.dynamic_fc, batch.lengths)
    elif isinstance(batch, FusionBatch):
        output = model(
            batch.static_fc,
            batch.dynamic_fc,
            batch.lengths,
        )
    else:
        raise TypeError(
            "Unsupported batch type for unified Stage 6 epoch runner: "
            f"{type(batch)!r}. Expected StaticBatch, TemporalBatch, "
            "or FusionBatch."
        )

    if not hasattr(output, "logits"):
        raise EpochRunnerError(
            "Pipeline output must expose a 'logits' attribute."
        )

    logits = output.logits
    if not isinstance(logits, torch.Tensor):
        raise EpochRunnerError(
            "Pipeline output.logits must be a torch.Tensor, got "
            f"{type(logits)!r}."
        )

    return logits


def run_binary_epoch(
    model: nn.Module,
    batches: Iterable[PipelineBatch],
    *,
    loss_fn: LossFunction,
    device: torch.device | str,
    training: bool,
    optimizer: Optimizer | None = None,
    non_blocking: bool = False,
) -> BinaryClassificationMetrics:
    """
    Run one binary-classification epoch for a unified pipeline.

    Parameters
    ----------
    model:
        A pipeline model whose forward output exposes ``.logits`` with shape
        ``[B, 2]``. The epoch runner does not know backbone mathematics.
    batches:
        Iterable yielding ``StaticBatch``, ``TemporalBatch``, or
        ``FusionBatch`` objects from the validated data layer. HDF5 access is
        intentionally outside this function.
    loss_fn:
        Callable returning a *scalar mean per-sample loss* for one batch.
        Stage 6D will instantiate the Atlas ABIDE cross-entropy criterion.
        The scalar-mean contract is explicit because epoch loss is accumulated
        sample-weighted across possibly unequal final batch sizes.
    device:
        CPU or CUDA device to which each validated batch is moved via its
        existing ``.to(...)`` contract.
    training:
        ``True`` performs optimization under ``model.train()``. ``False``
        performs evaluation under ``model.eval()`` and ``torch.no_grad()``.
    optimizer:
        Required for training and prohibited for evaluation. Scheduler stepping
        is intentionally outside the epoch runner and belongs to Stage 6D.
    non_blocking:
        Forwarded to batch ``.to(...)``. This is useful with pinned-memory CUDA
        loaders but does not alter scientific behavior.

    Returns
    -------
    BinaryClassificationMetrics
        Sample-weighted mean loss, ROC-AUC, accuracy, sensitivity,
        specificity, and sample count using the Stage 6A metric protocol.

    Notes
    -----
    This function intentionally contains no split selection, checkpoint
    selection, early stopping, scheduler logic, hyperparameter search, test-set
    special cases, HDF5 reads, BQN-specific logic, or temporal-window logic.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model)!r}.")
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable.")
    if not isinstance(training, bool):
        raise TypeError("training must be a bool.")

    if training:
        if optimizer is None:
            raise EpochRunnerError("Training requires an optimizer.")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                "optimizer must be torch.optim.Optimizer during training, got "
                f"{type(optimizer)!r}."
            )
        model.train()
    else:
        if optimizer is not None:
            raise EpochRunnerError(
                "Evaluation must not receive an optimizer; pass optimizer=None."
            )
        model.eval()

    resolved_device = torch.device(device)
    accumulator = BinaryMetricAccumulator()

    grad_context = nullcontext() if training else torch.no_grad()

    with grad_context:
        for raw_batch in batches:
            if not isinstance(
                raw_batch,
                (StaticBatch, TemporalBatch, FusionBatch),
            ):
                raise TypeError(
                    "Unsupported batch type for unified Stage 6 epoch runner: "
                    f"{type(raw_batch)!r}. Expected StaticBatch, TemporalBatch, "
                    "or FusionBatch."
                )

            batch = raw_batch.to(
                resolved_device,
                non_blocking=non_blocking,
            )

            if training:
                # Clear gradients immediately before the forward/backward pass
                # for this optimization step. set_to_none=True is deterministic
                # and avoids unnecessary zero-fill work.
                optimizer.zero_grad(set_to_none=True)

            logits = _forward_pipeline_batch(model, batch)
            labels = batch.labels

            loss = loss_fn(logits, labels)
            if not isinstance(loss, torch.Tensor):
                raise EpochRunnerError(
                    "loss_fn must return a torch.Tensor scalar mean loss."
                )
            if loss.numel() != 1:
                raise EpochRunnerError(
                    "loss_fn must return a scalar mean per-sample loss; got "
                    f"shape {tuple(loss.shape)}."
                )
            if not torch.isfinite(loss).all():
                raise EpochRunnerError("loss_fn produced NaN or Inf.")

            if training:
                loss.backward()
                optimizer.step()

            batch_size = int(labels.shape[0])
            accumulator.update(
                logits=logits,
                labels=labels,
                loss_sum=loss.detach() * batch_size,
            )

    return accumulator.compute()
