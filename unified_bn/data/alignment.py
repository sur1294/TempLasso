from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from unified_bn.data.common import ensure_unique_subject_ids

@dataclass(frozen=True)
class AlignmentReport:
    n_left: int
    n_right: int
    n_common: int
    left_only: tuple[str, ...]
    right_only: tuple[str, ...]
    label_mismatches: tuple[str, ...]
    same_order: bool
    @property
    def exact_subject_match(self) -> bool:
        return not self.left_only and not self.right_only
    @property
    def labels_match(self) -> bool:
        return not self.label_mismatches
    @property
    def is_compatible(self) -> bool:
        return self.exact_subject_match and self.labels_match

def audit_subject_alignment(
    left_ids: Sequence[str], left_labels: Sequence[int],
    right_ids: Sequence[str], right_labels: Sequence[int],
) -> AlignmentReport:
    ensure_unique_subject_ids(left_ids)
    ensure_unique_subject_ids(right_ids)
    if len(left_ids) != len(left_labels):
        raise ValueError("left_ids and left_labels length mismatch.")
    if len(right_ids) != len(right_labels):
        raise ValueError("right_ids and right_labels length mismatch.")
    left_map = {sid: int(y) for sid, y in zip(left_ids, left_labels)}
    right_map = {sid: int(y) for sid, y in zip(right_ids, right_labels)}
    left_set, right_set = set(left_map), set(right_map)
    common = left_set & right_set
    mismatches = tuple(sorted(
        sid for sid in common if left_map[sid] != right_map[sid]
    ))
    return AlignmentReport(
        len(left_ids), len(right_ids), len(common),
        tuple(sorted(left_set-right_set)), tuple(sorted(right_set-left_set)),
        mismatches, tuple(left_ids) == tuple(right_ids),
    )

def require_aligned_modalities(
    left_ids: Sequence[str], left_labels: Sequence[int],
    right_ids: Sequence[str], right_labels: Sequence[int],
) -> AlignmentReport:
    report = audit_subject_alignment(left_ids, left_labels, right_ids, right_labels)
    if not report.exact_subject_match:
        raise ValueError(
            "Static/temporal subject sets differ: "
            f"left_only={report.left_only[:10]}, right_only={report.right_only[:10]}."
        )
    if not report.labels_match:
        raise ValueError(
            "Static/temporal labels disagree for subject IDs: "
            f"{report.label_mismatches[:10]}."
        )
    return report
