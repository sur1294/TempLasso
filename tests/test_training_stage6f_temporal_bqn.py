from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from models.bqn.bqn_model import BQNConfig, StaticBQN
from unified_bn.backbones.bqn import BQNBackbone, BQNBackboneConfig
from unified_bn.data.collate import TemporalBatch
from unified_bn.pipelines.temporal import TemporalPipeline
from unified_bn.temporal.bigru import BiGRUAggregator, BiGRUConfig
from unified_bn.temporal.windowing import WindowPolicyConfig
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


def _make_temporal_reference_pipeline(
    *,
    seed: int,
    n_layers: int = 1,
    activation: str = "elu",
    bqn_dropout: float = 0.0,
    hidden_dim: int = 4,
):
    """Construct the validated shared-BQN -> BiGRU temporal path."""

    seed_everything(seed)

    reference_config = BQNConfig(
        n_rois=100,
        n_layers=n_layers,
        dropout=bqn_dropout,
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
        dropout=bqn_dropout,
        activation=activation,
        cluster_num=4,
        preserve_unused_mlp_b=True,
        orthogonal_centers=True,
        freeze_centers=True,
        project_assignment=True,
    )
    backbone = BQNBackbone(adapter_config, model=reference)

    temporal_config = BiGRUConfig(
        input_dim=backbone.embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.0,
        pooling="final",
        projection_dim=None,
        layer_norm=False,
    )
    aggregator = BiGRUAggregator(temporal_config)
    window_policy = WindowPolicyConfig(policy="native")

    pipeline = TemporalPipeline(
        backbone=backbone,
        temporal_aggregator=aggregator,
        num_classes=2,
        window_policy=window_policy,
    )

    return (
        reference,
        adapter_config,
        temporal_config,
        window_policy,
        pipeline,
    )


def _temporal_batch(
    *,
    seed: int,
    subject_prefix: str,
    padded_scale: float = 0.0,
) -> TemporalBatch:
    """
    Deterministic balanced temporal batch with native lengths [2, 1].

    The second subject has one padded window. ``padded_scale`` can replace
    that fake window with an extreme finite matrix to test padding isolation.
    """

    generator = torch.Generator().manual_seed(seed)
    labels = torch.tensor([0, 1], dtype=torch.long)
    lengths = torch.tensor([2, 1], dtype=torch.long)
    dynamic_fc = torch.zeros(2, 2, 100, 100, dtype=torch.float32)

    for subject_index, label in enumerate(labels.tolist()):
        for window_index in range(int(lengths[subject_index].item())):
            noise = 0.03 * torch.randn(100, 100, generator=generator)
            matrix = 0.5 * (noise + noise.transpose(0, 1))
            matrix.fill_diagonal_(1.0)

            # Small deterministic class-linked signal so the optimization
            # smoke is non-degenerate. This is synthetic test input only.
            edge_value = 0.20 if label == 1 else -0.20
            edge_value += 0.01 * window_index
            matrix[0, 1] = edge_value
            matrix[1, 0] = edge_value
            dynamic_fc[subject_index, window_index] = matrix

    if padded_scale != 0.0:
        fake = padded_scale * torch.randn(100, 100, generator=generator)
        dynamic_fc[1, 1] = fake

    return TemporalBatch(
        subject_ids=(f"{subject_prefix}0", f"{subject_prefix}1"),
        labels=labels,
        dynamic_fc=dynamic_fc,
        lengths=lengths,
        source_indices=torch.arange(2, dtype=torch.long),
    )


def _candidate_config(
    adapter_config: BQNBackboneConfig,
    temporal_config: BiGRUConfig,
    window_policy: WindowPolicyConfig,
    *,
    seed: int,
    epochs: int,
) -> CandidateRunConfig:
    return CandidateRunConfig(
        identity=RunIdentity(
            mode="temporal",
            seed=seed,
            split_identity="stage6f-synthetic-temporal-split-v1",
            backbone_config={
                "name": "bqn",
                **asdict(adapter_config),
            },
            temporal_config={
                "name": "bigru",
                **asdict(temporal_config),
                "classifier": "linear",
            },
            window_policy=window_policy,
        ),
        training=TrainingConfig(
            epochs=epochs,
            batch_size=2,
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


def test_stage6f_temporal_path_has_one_shared_actual_bqn() -> None:
    reference, _, _, _, pipeline = _make_temporal_reference_pipeline(seed=42)

    assert pipeline.backbone.reference_model is reference

    bqn_objects = [
        module
        for module in pipeline.modules()
        if isinstance(module, StaticBQN)
    ]
    assert len(bqn_objects) == 1
    assert bqn_objects[0] is reference

    assert pipeline.backbone.embedding_dim == 32
    assert pipeline.temporal_aggregator.input_dim == 32
    assert pipeline.temporal_aggregator.output_dim == 8
    assert pipeline.window_policy == WindowPolicyConfig(policy="native")


def test_stage6f_generic_trainer_optimizes_actual_bqn_bigru_and_excludes_padding(
    tmp_path: Path,
) -> None:
    (
        reference,
        adapter_config,
        temporal_config,
        window_policy,
        pipeline,
    ) = _make_temporal_reference_pipeline(seed=1042)

    run_config = _candidate_config(
        adapter_config,
        temporal_config,
        window_policy,
        seed=1042,
        epochs=1,
    )

    train_batch = _temporal_batch(
        seed=201,
        subject_prefix="train-",
        padded_scale=1.0e6,
    )
    val_batch = _temporal_batch(
        seed=301,
        subject_prefix="val-",
        padded_scale=-1.0e6,
    )

    seen_bqn_batch_sizes = []

    def pre_hook(_module, args):
        seen_bqn_batch_sizes.append(int(args[0].shape[0]))

    handle = reference.register_forward_pre_hook(pre_hook)
    try:
        result = train_candidate(
            pipeline,
            [train_batch],
            [val_batch],
            run_config=run_config,
            run_dir=tmp_path / "candidate",
            device="cpu",
            user_provenance={
                "stage": "6F",
                "model": "actual_bqn_bigru",
            },
        )
    finally:
        handle.remove()

    # lengths=[2,1], so every pipeline call must present exactly three real
    # windows to BQN. The one padded matrix is never forwarded, even though it
    # contains extreme finite values. Calls are train, validation, and the
    # post-restore validation audit performed by Stage 6D.
    assert seen_bqn_batch_sizes == [3, 3, 3]

    assert result.completed_epochs == 1
    assert result.best_epoch == 1
    assert result.selection_metric == "val_loss"
    assert result.selection_mode == "min"
    assert result.test_metrics is None

    # Active BQN mathematics, BiGRU, and temporal classifier all participate
    # in the generic optimization step. The validated reference-inactive
    # mlp_b remains gradient-free.
    for layer in reference.qp_layers:
        assert layer.mlp_a.weight.grad is not None
        assert layer.mlp_h.weight.grad is not None
        if layer.mlp_b is not None:
            assert layer.mlp_b.weight.grad is None

    assert pipeline.temporal_aggregator.gru.weight_ih_l0.grad is not None
    assert pipeline.temporal_aggregator.gru.weight_hh_l0.grad is not None
    assert pipeline.classifier.weight.grad is not None

    assert torch.isfinite(
        pipeline.temporal_aggregator.gru.weight_ih_l0.grad
    ).all()
    assert torch.isfinite(pipeline.classifier.weight.grad).all()

    result_payload = json.loads(
        Path(result.result_path).read_text(encoding="utf-8")
    )
    assert result_payload["status"] == "validation_complete"
    assert result_payload["test_metrics"] is None
    assert result_payload["run_config"] == run_config.to_dict()
    assert result_payload["run_config"]["identity"]["mode"] == "temporal"
    assert result_payload["run_config"]["identity"]["window_policy"] == {
        "policy": "native",
        "crop": "start",
        "min_windows": None,
    }


def test_stage6f_best_checkpoint_is_restored_for_actual_temporal_pipeline(
    tmp_path: Path,
) -> None:
    (
        _,
        adapter_config,
        temporal_config,
        window_policy,
        pipeline,
    ) = _make_temporal_reference_pipeline(seed=2042)

    run_config = _candidate_config(
        adapter_config,
        temporal_config,
        window_policy,
        seed=2042,
        epochs=2,
    )

    result = train_candidate(
        pipeline,
        [_temporal_batch(seed=401, subject_prefix="train-")],
        [_temporal_batch(seed=501, subject_prefix="val-")],
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


def test_stage6f_explicit_test_evaluation_is_padding_invariant_and_separate(
    tmp_path: Path,
) -> None:
    (
        _,
        adapter_config,
        temporal_config,
        window_policy,
        pipeline,
    ) = _make_temporal_reference_pipeline(seed=3042)

    run_config = _candidate_config(
        adapter_config,
        temporal_config,
        window_policy,
        seed=3042,
        epochs=1,
    )

    result = train_candidate(
        pipeline,
        [_temporal_batch(seed=601, subject_prefix="train-")],
        [_temporal_batch(seed=701, subject_prefix="val-")],
        run_config=run_config,
        run_dir=tmp_path / "candidate",
        device="cpu",
    )

    assert result.test_metrics is None

    clean = _temporal_batch(
        seed=801,
        subject_prefix="test-",
        padded_scale=0.0,
    )
    corrupted_padding = _temporal_batch(
        seed=801,
        subject_prefix="test-",
        padded_scale=1.0e6,
    )

    clean_metrics = evaluate_final_test(
        pipeline,
        [clean],
        device="cpu",
    )
    corrupted_metrics = evaluate_final_test(
        pipeline,
        [corrupted_padding],
        device="cpu",
    )

    assert clean_metrics.n_samples == 2
    assert corrupted_metrics.n_samples == 2
    assert clean_metrics.loss == pytest.approx(
        corrupted_metrics.loss,
        rel=0.0,
        abs=0.0,
    )
    assert clean_metrics.roc_auc == corrupted_metrics.roc_auc
    assert clean_metrics.accuracy == corrupted_metrics.accuracy
    assert clean_metrics.sensitivity == corrupted_metrics.sensitivity
    assert clean_metrics.specificity == corrupted_metrics.specificity

    # Explicit evaluation alone does not silently mutate tuning artifacts.
    result_payload = json.loads(
        Path(result.result_path).read_text(encoding="utf-8")
    )
    assert result_payload["test_metrics"] is None
