import pytest
from unified_bn.data.alignment import (
    audit_subject_alignment, require_aligned_modalities
)

def test_alignment_allows_different_order():
    report = require_aligned_modalities(
        ["a","b","c"], [0,1,0],
        ["c","a","b"], [0,0,1],
    )
    assert report.is_compatible
    assert not report.same_order
    assert report.n_common == 3

def test_alignment_detects_label_mismatch():
    report = audit_subject_alignment(
        ["a","b"], [0,1], ["b","a"], [0,1]
    )
    assert report.label_mismatches == ("a","b")
    with pytest.raises(ValueError, match="labels disagree"):
        require_aligned_modalities(
            ["a","b"], [0,1], ["b","a"], [0,1]
        )

def test_alignment_detects_missing_subjects():
    report = audit_subject_alignment(
        ["a","b"], [0,1], ["a","c"], [0,1]
    )
    assert report.left_only == ("b",)
    assert report.right_only == ("c",)
    assert not report.is_compatible
