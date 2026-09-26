"""Dataset-specific validation selection for the Atlas comparison.

Atlas Matters, Section 4.1 (p. 6) and Appendix B.4 (p. 17), explicitly
specify macro OvR AUC for four-class PPMI and loss for binary datasets.
This is a selection rule, not a replacement for cross-entropy training.
"""
from __future__ import annotations

from unified_bn.training.config import ValidationSelectionRule


def dataset_selection_rule(dataset: str, num_classes: int) -> ValidationSelectionRule:
    """Resolve the frozen rule before constructing candidates or reading data.

    The existing metric key ``val_auc`` denotes unweighted macro one-vs-rest
    AUC for K>2 in BinaryMetricAccumulator. No test metric enters this rule.
    Preserve the historical loss rule for every dataset other than PPMI.
    """
    if dataset.strip().upper() == "PPMI":
        if num_classes != 4:
            raise ValueError("The Atlas PPMI protocol requires --num-classes 4.")
        return ValidationSelectionRule(metric="val_auc", mode="max")
    return ValidationSelectionRule(metric="val_loss", mode="min")
