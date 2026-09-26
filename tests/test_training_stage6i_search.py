from __future__ import annotations

import json
from dataclasses import replace

import pytest

from unified_bn.training.config import (
    TrainingConfig,
    ValidationSelectionRule,
)
from unified_bn.training.metrics import BinaryClassificationMetrics
from unified_bn.training.results import CandidateTrainingResult
from unified_bn.training.search import (
    ValidationSearchError,
    select_validation_winner,
    write_validation_selection_json,
)
from unified_bn.training.search_space import (
    BQNSearchSpace,
    generate_vanilla_bqn_candidates,
)


def _metrics(
    *,
    loss: float,
    auc: float = 0.70,
    accuracy: float = 0.65,
    sensitivity: float = 0.60,
    specificity: float = 0.70,
) -> BinaryClassificationMetrics:
    return BinaryClassificationMetrics(
        loss=loss,
        roc_auc=auc,
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        n_samples=20,
    )


def _result(
    candidate,
    *,
    best_value: float,
    auc: float = 0.70,
    test_metrics=None,
    suffix: str = "0",
) -> CandidateTrainingResult:
    metrics = _metrics(loss=best_value, auc=auc)
    return CandidateTrainingResult(
        run_config=candidate.to_dict(),
        requested_epochs=200,
        completed_epochs=200,
        stopped_early=False,
        best_epoch=17,
        selection_metric=candidate.training.checkpoint_selection.metric,
        selection_mode=candidate.training.checkpoint_selection.mode,
        best_selection_value=best_value,
        best_validation_metrics=metrics,
        restored_validation_metrics=metrics,
        best_checkpoint_path=f"/runs/{suffix}/best.pt",
        last_checkpoint_path=f"/runs/{suffix}/last.pt",
        history_path=f"/runs/{suffix}/history.json",
        result_path=f"/runs/{suffix}/result.json",
        loss_function="torch.nn.modules.loss.CrossEntropyLoss",
        runtime_provenance={"torch": "test"},
        user_provenance={},
        test_metrics=test_metrics,
    )


def _three_candidates():
    return generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=BQNSearchSpace(
            n_layers=(1, 2, 3),
            activations=("elu",),
            dropouts=(0.0,),
        ),
    )


def test_selects_minimum_validation_loss_not_highest_auc() -> None:
    candidates = _three_candidates()

    results = (
        _result(candidates[0], best_value=0.55, auc=0.99, suffix="0"),
        _result(candidates[1], best_value=0.42, auc=0.61, suffix="1"),
        _result(candidates[2], best_value=0.48, auc=0.95, suffix="2"),
    )

    selection = select_validation_winner(candidates, results)

    assert selection.selected_index == 1
    assert selection.selection_metric == "val_loss"
    assert selection.selection_mode == "min"
    assert selection.selection_value == pytest.approx(0.42)
    assert selection.selected_best_checkpoint_path == "/runs/1/best.pt"


def test_result_completion_order_does_not_change_selection() -> None:
    candidates = _three_candidates()
    results = (
        _result(candidates[0], best_value=0.55, suffix="0"),
        _result(candidates[1], best_value=0.42, suffix="1"),
        _result(candidates[2], best_value=0.48, suffix="2"),
    )

    forward = select_validation_winner(candidates, results)
    reverse = select_validation_winner(candidates, tuple(reversed(results)))

    assert forward == reverse


def test_exact_tie_keeps_first_frozen_candidate() -> None:
    candidates = _three_candidates()
    results = (
        _result(candidates[0], best_value=0.40, suffix="0"),
        _result(candidates[1], best_value=0.40, suffix="1"),
        _result(candidates[2], best_value=0.50, suffix="2"),
    )

    selection = select_validation_winner(candidates, results)

    assert selection.selected_index == 0
    assert selection.selected_best_checkpoint_path == "/runs/0/best.pt"


def test_partial_search_is_rejected() -> None:
    candidates = _three_candidates()
    results = (
        _result(candidates[0], best_value=0.55, suffix="0"),
        _result(candidates[2], best_value=0.48, suffix="2"),
    )

    with pytest.raises(ValidationSearchError, match="incomplete"):
        select_validation_winner(candidates, results)


def test_extra_unknown_candidate_result_is_rejected() -> None:
    candidates = _three_candidates()

    extra = generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=BQNSearchSpace(
            n_layers=(4,),
            activations=("elu",),
            dropouts=(0.0,),
        ),
    )[0]

    results = (
        _result(candidates[0], best_value=0.55, suffix="0"),
        _result(candidates[1], best_value=0.42, suffix="1"),
        _result(candidates[2], best_value=0.48, suffix="2"),
        _result(extra, best_value=0.20, suffix="extra"),
    )

    with pytest.raises(ValidationSearchError, match="does not belong"):
        select_validation_winner(candidates, results)


def test_duplicate_result_for_same_candidate_is_rejected() -> None:
    candidates = _three_candidates()

    results = (
        _result(candidates[0], best_value=0.55, suffix="0a"),
        _result(candidates[0], best_value=0.55, suffix="0b"),
        _result(candidates[1], best_value=0.42, suffix="1"),
        _result(candidates[2], best_value=0.48, suffix="2"),
    )

    with pytest.raises(ValidationSearchError, match="Multiple completed"):
        select_validation_winner(candidates, results)


def test_any_test_metrics_make_search_fail_closed() -> None:
    candidates = _three_candidates()

    results = (
        _result(
            candidates[0],
            best_value=0.55,
            suffix="0",
            test_metrics=_metrics(loss=0.1),
        ),
        _result(candidates[1], best_value=0.42, suffix="1"),
        _result(candidates[2], best_value=0.48, suffix="2"),
    )

    with pytest.raises(ValidationSearchError, match="held-out test"):
        select_validation_winner(candidates, results)


def test_result_selection_rule_mismatch_is_rejected() -> None:
    candidates = _three_candidates()
    bad = replace(
        _result(candidates[0], best_value=0.55, suffix="0"),
        selection_metric="val_auc",
    )

    results = (
        bad,
        _result(candidates[1], best_value=0.42, suffix="1"),
        _result(candidates[2], best_value=0.48, suffix="2"),
    )

    with pytest.raises(ValidationSearchError, match="selection_metric"):
        select_validation_winner(candidates, results)


def test_candidates_with_mixed_selection_rules_are_rejected() -> None:
    base = generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=BQNSearchSpace(
            n_layers=(1,),
            activations=("elu",),
            dropouts=(0.0,),
        ),
    )[0]

    max_training = TrainingConfig(
        checkpoint_selection=ValidationSelectionRule(
            metric="val_auc",
            mode="max",
        )
    )
    other = replace(base, training=max_training)

    results = (
        _result(base, best_value=0.50, suffix="0"),
        CandidateTrainingResult(
            run_config=other.to_dict(),
            requested_epochs=200,
            completed_epochs=200,
            stopped_early=False,
            best_epoch=10,
            selection_metric="val_auc",
            selection_mode="max",
            best_selection_value=0.80,
            best_validation_metrics=_metrics(loss=0.6, auc=0.80),
            restored_validation_metrics=_metrics(loss=0.6, auc=0.80),
            best_checkpoint_path="/runs/1/best.pt",
            last_checkpoint_path="/runs/1/last.pt",
            history_path="/runs/1/history.json",
            result_path="/runs/1/result.json",
            loss_function="torch.nn.modules.loss.CrossEntropyLoss",
            runtime_provenance={},
            user_provenance={},
            test_metrics=None,
        ),
    )

    with pytest.raises(ValidationSearchError, match="different"):
        select_validation_winner((base, other), results)


def test_restored_validation_metric_must_match_best_selection_value() -> None:
    candidates = _three_candidates()
    inconsistent = replace(
        _result(candidates[0], best_value=0.55, suffix="0"),
        restored_validation_metrics=_metrics(loss=0.56),
    )

    results = (
        inconsistent,
        _result(candidates[1], best_value=0.42, suffix="1"),
        _result(candidates[2], best_value=0.48, suffix="2"),
    )

    with pytest.raises(ValidationSearchError, match="disagrees"):
        select_validation_winner(candidates, results)


def test_selection_artifact_is_test_free_and_json_serializable(tmp_path) -> None:
    candidates = _three_candidates()
    results = (
        _result(candidates[0], best_value=0.55, suffix="0"),
        _result(candidates[1], best_value=0.42, suffix="1"),
        _result(candidates[2], best_value=0.48, suffix="2"),
    )

    selection = select_validation_winner(candidates, results)
    path = write_validation_selection_json(
        tmp_path / "selection.json",
        selection,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "validation_selection_complete"
    assert payload["n_candidates"] == 3
    assert payload["selected_index"] == 1
    assert payload["selection_value"] == pytest.approx(0.42)

    def check_no_test_key(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert "test" not in str(key).lower()
                check_no_test_key(child)
        elif isinstance(value, list):
            for child in value:
                check_no_test_key(child)

    check_no_test_key(payload)


def test_selection_fingerprint_matches_selected_candidate() -> None:
    candidates = _three_candidates()
    results = (
        _result(candidates[0], best_value=0.55, suffix="0"),
        _result(candidates[1], best_value=0.42, suffix="1"),
        _result(candidates[2], best_value=0.48, suffix="2"),
    )

    selection = select_validation_winner(candidates, results)

    from unified_bn.training.search_space import candidate_fingerprint

    assert selection.selected_fingerprint == candidate_fingerprint(
        candidates[1]
    )


def test_restored_validation_metric_tolerates_gpu_inference_drift() -> None:
    """Sub-1e-4 drift between the tracked best value and the reloaded
    checkpoint's re-evaluation (cuDNN/cuBLAS non-reproducibility) must not
    fail the search; a real checkpoint mix-up (>= 1e-3) still must."""
    candidates = _three_candidates()
    drifted = replace(
        _result(candidates[0], best_value=0.55, suffix="0"),
        restored_validation_metrics=_metrics(loss=0.55 + 1e-6),
    )
    others = (
        _result(candidates[1], best_value=0.62, suffix="1"),
        _result(candidates[2], best_value=0.68, suffix="2"),
    )
    selection = select_validation_winner(candidates, (drifted,) + others)
    assert selection.selected_index == 0

    mixed_up = replace(
        _result(candidates[0], best_value=0.55, suffix="0"),
        restored_validation_metrics=_metrics(loss=0.55 + 1e-3),
    )
    with pytest.raises(ValidationSearchError, match="restored"):
        select_validation_winner(candidates, (mixed_up,) + others)
