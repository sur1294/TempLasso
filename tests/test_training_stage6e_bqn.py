from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from models.bqn.bqn_model import BQNConfig, StaticBQN
from unified_bn.backbones.bqn import BQNBackbone, BQNBackboneConfig
from unified_bn.data.collate import StaticBatch
from unified_bn.pipelines.vanilla import VanillaPipeline
from unified_bn.training.checkpoint import read_training_checkpoint
from unified_bn.training.config import (
    CandidateRunConfig,
    LRScheduleConfig,
    OptimizerConfig,
    RunIdentity,
    TrainingConfig,
)
from unified_bn.training.reproducibility import seed_everything
from unified_bn.training.trainer import evaluate_final_test, train_candidate


def _make_reference_pipeline(
    *,
    seed: int,
    n_layers: int,
    activation: str,
    dropout: float,
):
    """Construct the exact-reference Vanilla BQN path after seeding."""

    seed_everything(seed)

    reference_config = BQNConfig(
        n_rois=100,
        n_layers=n_layers,
        dropout=dropout,
        activation=activation,
        cluster_num=4,
        preserve_unused_mlp_b=True,
        orthogonal_centers=True,
        freeze_centers=True,
        project_assignment=True,
    )
    reference = StaticBQN(reference_config)

    adapter_config = BQNBackboneConfig(
        n_rois=100,
        n_layers=n_layers,
        dropout=dropout,
        activation=activation,
        cluster_num=4,
        preserve_unused_mlp_b=True,
        orthogonal_centers=True,
        freeze_centers=True,
        project_assignment=True,
    )
    backbone = BQNBackbone(adapter_config, model=reference)

    # Exact Atlas/reference-oriented Vanilla path: reuse the classifier that
    # belongs to the same validated StaticBQN object rather than introducing a
    # new generic Linear head.
    pipeline = VanillaPipeline(
        backbone,
        num_classes=2,
        classifier=reference.classifier,
    )
    return reference, adapter_config, pipeline


def _correlation_like_batch(
    *,
    seed: int,
    subject_prefix: str,
) -> StaticBatch:
    """Small deterministic balanced batch with signed symmetric FC inputs."""

    generator = torch.Generator().manual_seed(seed)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    fc = torch.empty(4, 100, 100, dtype=torch.float32)

    for index, label in enumerate(labels.tolist()):
        noise = 0.04 * torch.randn(100, 100, generator=generator)
        matrix = 0.5 * (noise + noise.transpose(0, 1))
        matrix.fill_diagonal_(1.0)

        # Add a deterministic class-linked signed edge only to make the
        # synthetic optimization smoke non-degenerate. This is test data, not
        # a scientific preprocessing substitute.
        edge_value = 0.25 if label == 1 else -0.25
        matrix[0, 1] = edge_value
        matrix[1, 0] = edge_value
        fc[index] = matrix

    return StaticBatch(
        subject_ids=tuple(f"{subject_prefix}{i}" for i in range(4)),
        labels=labels,
        fc=fc,
        source_indices=torch.arange(4, dtype=torch.long),
    )


def _candidate_config(
    adapter_config: BQNBackboneConfig,
    *,
    seed: int,
    epochs: int,
) -> CandidateRunConfig:
    return CandidateRunConfig(
        identity=RunIdentity(
            mode="vanilla",
            seed=seed,
            split_identity="stage6e-synthetic-static-split-v1",
            backbone_config={
                "name": "bqn",
                **asdict(adapter_config),
                "classifier": "reference_static_bqn_classifier",
            },
        ),
        training=TrainingConfig(
            epochs=epochs,
            batch_size=4,
            optimizer=OptimizerConfig(
                name="adam",
                lr=1e-4,
                weight_decay=1e-4,
            ),
            lr_schedule=LRScheduleConfig(
                name="cosine",
                min_lr=1e-5,
            ),
        ),
    )


def test_stage6e_reference_vanilla_path_is_exact_before_training() -> None:
    reference, _, pipeline = _make_reference_pipeline(
        seed=42,
        n_layers=3,
        activation="elu",
        dropout=0.0,
    )
    batch = _correlation_like_batch(seed=101, subject_prefix="exact-")

    reference.eval()
    pipeline.eval()
    with torch.no_grad():
        expected_logits = reference(batch.fc)
        observed = pipeline(batch.fc)

    assert torch.equal(observed.logits, expected_logits)
    assert observed.embedding.shape == (4, 32)
    assert pipeline.backbone.reference_model is reference
    assert pipeline.classifier is reference.classifier


def test_stage6e_generic_trainer_optimizes_actual_reference_bqn(
    tmp_path: Path,
) -> None:
    reference, adapter_config, pipeline = _make_reference_pipeline(
        seed=1042,
        n_layers=1,
        activation="elu",
        dropout=0.0,
    )
    run_config = _candidate_config(
        adapter_config,
        seed=1042,
        epochs=2,
    )

    train_batches = [_correlation_like_batch(seed=201, subject_prefix="train-")]
    val_batches = [_correlation_like_batch(seed=301, subject_prefix="val-")]

    result = train_candidate(
        pipeline,
        train_batches,
        val_batches,
        run_config=run_config,
        run_dir=tmp_path / "candidate",
        device="cpu",
        user_provenance={"stage": "6E", "model": "actual_static_bqn"},
    )

    assert result.completed_epochs == 2
    assert result.stopped_early is False
    assert result.selection_metric == "val_loss"
    assert result.selection_mode == "min"
    assert result.best_epoch in {1, 2}
    assert result.test_metrics is None
    assert result.best_validation_metrics.n_samples == 4
    assert result.restored_validation_metrics.n_samples == 4

    # Active reference BQN maps and the exact reference classifier must have
    # participated in optimization. The intentionally inactive mlp_b remains
    # gradient-free, protecting the validated BQN mathematics.
    for layer in reference.qp_layers:
        assert layer.mlp_a.weight.grad is not None
        assert layer.mlp_h.weight.grad is not None
        if layer.mlp_b is not None:
            assert layer.mlp_b.weight.grad is None

    for parameter in reference.classifier.parameters():
        assert parameter.grad is not None

    result_payload = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
    assert result_payload["status"] == "validation_complete"
    assert result_payload["test_metrics"] is None
    assert result_payload["run_config"] == run_config.to_dict()


def test_stage6e_trainer_returns_model_restored_to_actual_bqn_best_checkpoint(
    tmp_path: Path,
) -> None:
    _, adapter_config, pipeline = _make_reference_pipeline(
        seed=2042,
        n_layers=1,
        activation="leaky_relu",
        dropout=0.0,
    )
    run_config = _candidate_config(
        adapter_config,
        seed=2042,
        epochs=2,
    )

    result = train_candidate(
        pipeline,
        [_correlation_like_batch(seed=401, subject_prefix="train-")],
        [_correlation_like_batch(seed=501, subject_prefix="val-")],
        run_config=run_config,
        run_dir=tmp_path / "candidate",
        device="cpu",
    )

    checkpoint = read_training_checkpoint(
        result.best_checkpoint_path,
        expected_run_config=run_config,
        map_location="cpu",
    )
    live_state = pipeline.state_dict()

    assert live_state.keys() == checkpoint.model_state.keys()
    for name, tensor in live_state.items():
        assert torch.equal(tensor.cpu(), checkpoint.model_state[name].cpu()), name

    assert checkpoint.progress.completed_epoch == result.best_epoch
    assert checkpoint.progress.best_epoch == result.best_epoch
    assert checkpoint.progress.best_selection_value == pytest.approx(
        result.best_selection_value,
        rel=0.0,
        abs=1e-15,
    )


def test_stage6e_actual_bqn_test_evaluation_is_explicit_and_separate(
    tmp_path: Path,
) -> None:
    _, adapter_config, pipeline = _make_reference_pipeline(
        seed=3042,
        n_layers=1,
        activation="gelu",
        dropout=0.0,
    )
    run_config = _candidate_config(
        adapter_config,
        seed=3042,
        epochs=1,
    )

    result = train_candidate(
        pipeline,
        [_correlation_like_batch(seed=601, subject_prefix="train-")],
        [_correlation_like_batch(seed=701, subject_prefix="val-")],
        run_config=run_config,
        run_dir=tmp_path / "candidate",
        device="cpu",
    )

    # Training/model selection completed without any test-loader argument.
    assert result.test_metrics is None

    test_metrics = evaluate_final_test(
        pipeline,
        [_correlation_like_batch(seed=801, subject_prefix="test-")],
        device="cpu",
    )

    assert test_metrics.n_samples == 4
    assert 0.0 <= test_metrics.roc_auc <= 1.0
    assert 0.0 <= test_metrics.accuracy <= 1.0
    assert 0.0 <= test_metrics.sensitivity <= 1.0
    assert 0.0 <= test_metrics.specificity <= 1.0

    # evaluate_final_test() evaluates only; it does not silently mutate the
    # tuning result artifact. Recording test metrics is a separate explicit
    # Stage 6D action.
    result_payload = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
    assert result_payload["test_metrics"] is None
