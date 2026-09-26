"""
Optional cluster-only regression test against the actual ABIDE artifacts.

This test is skipped by default so normal unit-test execution does not depend
on cluster filesystem state. Set UNIFIED_BN_RUN_REAL_ABIDE=1 to enable it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from unified_bn.data.audit import run_abide_audit


RUN_REAL = os.environ.get("UNIFIED_BN_RUN_REAL_ABIDE") == "1"

ROOT = Path("/groups/bcoskunuzer/sxk230046/fall2026/Brainnetwork")
STATIC = ROOT / "derivatives/abide_data_tr.h5"
TEMPORAL = ROOT / "derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5"
SPLITS = [
    ROOT / "splits/abide_split_seed42.npz",
    ROOT / "splits/abide_split_seed1042.npz",
    ROOT / "splits/abide_split_seed2042.npz",
    ROOT / "splits/abide_split_seed3042.npz",
    ROOT / "splits/abide_split_seed4042.npz",
]


@pytest.mark.skipif(
    not RUN_REAL,
    reason="Set UNIFIED_BN_RUN_REAL_ABIDE=1 to audit cluster ABIDE artifacts.",
)
def test_real_abide_artifacts_match_frozen_protocol():
    report = run_abide_audit(
        static_h5=STATIC,
        temporal_h5=TEMPORAL,
        split_paths=SPLITS,
        expected_n_rois=100,
        expected_n_subjects=1025,
        expected_n_edges=4950,
        expected_total_windows=17551,
        expected_min_windows=12,
        expected_max_windows=26,
        expected_class_counts={0: 537, 1: 488},
        expected_split_sizes=(717, 103, 205),
    )

    assert report.alignment.n_common == 1025
    assert report.alignment.static_only == 0
    assert report.alignment.temporal_only == 0
    assert report.alignment.label_mismatches == 0
    assert report.alignment.same_order is False
