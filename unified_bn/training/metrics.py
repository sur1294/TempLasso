from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class BinaryMetricError(ValueError):
    """Raised when binary metric inputs violate the Stage 6 contract."""


class UndefinedBinaryMetricError(BinaryMetricError):
    """Raised when a requested binary metric is mathematically undefined."""


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Epoch-level binary metrics used by the Atlas ABIDE protocol."""

    loss: float
    roc_auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    n_samples: int

    # Short aliases match the validated historical BQN result naming.
    @property
    def auc(self) -> float:
        return self.roc_auc

    @property
    def acc(self) -> float:
        return self.accuracy

    @property
    def sens(self) -> float:
        return self.sensitivity

    @property
    def spec(self) -> float:
        return self.specificity

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BinaryMetricAccumulator:
    """
    Accumulate sample-weighted loss and binary classification metrics.

    Protocol
    --------
    - ROC-AUC is computed from ``softmax(logits)[:, 1]``.
    - Class predictions use ``argmax(logits, dim=1)``.
    - No probability threshold is fitted or selected.
    - ``loss_sum`` is the SUM of per-sample losses for the supplied batch;
      the reported loss is therefore the exact sample-weighted epoch mean.

    Ground-truth edge case
    ----------------------
    ROC-AUC requires both classes. Sensitivity or specificity is also
    undefined if its corresponding ground-truth class is absent. Stage 6
    therefore fails loudly when an accumulated evaluation set contains only
    one ground-truth class instead of fabricating a value or serializing NaN.
    """

    def __init__(self) -> None:
        self._loss_sum = 0.0
        self._n_samples = 0
        self._logits: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def update(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_sum: float | torch.Tensor,
    ) -> None:
        self._validate_batch(logits=logits, labels=labels)

        if isinstance(loss_sum, torch.Tensor):
            if loss_sum.numel() != 1:
                raise BinaryMetricError("loss_sum tensor must be scalar.")
            loss_value = float(loss_sum.detach().cpu().item())
        else:
            loss_value = float(loss_sum)

        if not np.isfinite(loss_value):
            raise BinaryMetricError("loss_sum must be finite.")
        if loss_value < 0.0:
            raise BinaryMetricError("loss_sum must be nonnegative.")

        labels_cpu = labels.detach().to(device="cpu", dtype=torch.long)

        self._loss_sum += loss_value
        self._n_samples += int(labels_cpu.numel())
        self._logits.append(
            logits.detach().to(device="cpu", dtype=torch.float64)
        )
        self._labels.append(labels_cpu)

    def compute(self) -> BinaryClassificationMetrics:
        if self._n_samples == 0:
            raise BinaryMetricError(
                "Cannot compute binary metrics from zero accumulated samples."
            )

        widths = {int(t.shape[1]) for t in self._logits}
        if len(widths) != 1:
            raise BinaryMetricError(f"Inconsistent logit widths across batches: {sorted(widths)}.")
        logits = torch.cat(self._logits, dim=0)
        labels = torch.cat(self._labels, dim=0)

        if logits.shape[0] != self._n_samples or labels.shape[0] != self._n_samples:
            raise RuntimeError("Internal binary-metric sample-count mismatch.")

        n_classes = int(logits.shape[1])
        if n_classes > 2:
            return self._compute_multiclass(logits, labels, n_classes)

        n_positive = int((labels == 1).sum().item())
        n_negative = int((labels == 0).sum().item())

        if n_positive == 0 or n_negative == 0:
            raise UndefinedBinaryMetricError(
                "ROC-AUC, sensitivity, and specificity require both "
                "ground-truth classes in the accumulated evaluation set."
            )

        probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
        predictions = torch.argmax(logits, dim=1)

        tp = int(((predictions == 1) & (labels == 1)).sum().item())
        tn = int(((predictions == 0) & (labels == 0)).sum().item())
        fp = int(((predictions == 1) & (labels == 0)).sum().item())
        fn = int(((predictions == 0) & (labels == 1)).sum().item())

        # Both denominators are non-zero because the ground-truth class check
        # above requires at least one positive and one negative sample.
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / self._n_samples

        roc_auc = float(
            roc_auc_score(
                labels.numpy(),
                probabilities,
            )
        )

        return BinaryClassificationMetrics(
            loss=float(self._loss_sum / self._n_samples),
            roc_auc=roc_auc,
            accuracy=float(accuracy),
            sensitivity=float(sensitivity),
            specificity=float(specificity),
            n_samples=self._n_samples,
        )

    def _compute_multiclass(
        self, logits: torch.Tensor, labels: torch.Tensor, n_classes: int
    ) -> BinaryClassificationMetrics:
        """
        K > 2 classes. The dataclass fields keep their names so every result,
        history and checkpoint reader is unchanged, with these meanings:
          roc_auc      macro-averaged one-vs-rest ROC-AUC (sklearn
                       ``multi_class='ovr', average='macro'``)
          accuracy     top-1 accuracy
          sensitivity  macro recall  (mean over classes of TP_c / (TP_c + FN_c))
          specificity  macro specificity (mean over classes of TN_c / (TN_c + FP_c))
        Every class must be present in the ground truth, otherwise the macro
        quantities are undefined and this fails loudly like the binary case.
        """
        present = set(labels.tolist())
        if present != set(range(n_classes)):
            raise UndefinedBinaryMetricError(
                f"Macro metrics need every class 0..{n_classes - 1} in the "
                f"accumulated evaluation set; found {sorted(present)}."
            )
        probabilities = torch.softmax(logits, dim=1).numpy()
        predictions = torch.argmax(logits, dim=1)
        recalls, specs = [], []
        for c in range(n_classes):
            is_c = labels == c
            pred_c = predictions == c
            tp = int((pred_c & is_c).sum()); fn = int((~pred_c & is_c).sum())
            fp = int((pred_c & ~is_c).sum()); tn = int((~pred_c & ~is_c).sum())
            recalls.append(tp / (tp + fn)); specs.append(tn / (tn + fp))
        roc_auc = float(roc_auc_score(labels.numpy(), probabilities, multi_class="ovr",
                                      average="macro", labels=list(range(n_classes))))
        return BinaryClassificationMetrics(
            loss=float(self._loss_sum / self._n_samples),
            roc_auc=roc_auc,
            accuracy=float((predictions == labels).double().mean().item()),
            sensitivity=float(np.mean(recalls)),
            specificity=float(np.mean(specs)),
            n_samples=self._n_samples,
        )

    @staticmethod
    def _validate_batch(*, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if not isinstance(logits, torch.Tensor):
            raise BinaryMetricError("logits must be a torch.Tensor.")
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise BinaryMetricError(
                "Binary logits must have shape [B,2] (or [B,K], K>=2, for the "
                f"macro protocol), got {tuple(logits.shape)}."
            )
        if logits.shape[0] == 0:
            raise BinaryMetricError("Cannot update metrics with an empty batch.")
        if not torch.is_floating_point(logits):
            raise BinaryMetricError("logits must be floating point.")
        if not torch.isfinite(logits).all():
            raise BinaryMetricError("logits contain NaN or Inf.")

        if not isinstance(labels, torch.Tensor):
            raise BinaryMetricError("labels must be a torch.Tensor.")
        if labels.ndim != 1:
            raise BinaryMetricError(
                "labels must have shape [B], got "
                f"{tuple(labels.shape)}."
            )
        if labels.shape[0] != logits.shape[0]:
            raise BinaryMetricError("logits and labels batch sizes do not match.")

        labels_cpu = labels.detach().to(device="cpu", dtype=torch.long)
        unique_labels = set(labels_cpu.tolist())
        n_classes = int(logits.shape[1])
        if not unique_labels.issubset(set(range(n_classes))):
            if n_classes == 2:
                raise BinaryMetricError(
                    "Binary labels must contain only 0/1, got "
                    f"{sorted(unique_labels)}."
                )
            raise BinaryMetricError(
                f"Labels must lie in 0..{n_classes - 1}, got {sorted(unique_labels)}."
            )
