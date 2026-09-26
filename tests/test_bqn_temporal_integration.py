from __future__ import annotations

import torch

from models.bqn.bqn_model import BQNConfig, StaticBQN
from unified_bn.backbones.bqn import BQNBackbone, BQNBackboneConfig
from unified_bn.pipelines.temporal import TemporalPipeline
from unified_bn.temporal.bigru import BiGRUAggregator, BiGRUConfig


def _make_real_bqn_temporal_pipeline(
    *,
    seed: int = 1234,
) -> tuple[StaticBQN, BQNBackbone, TemporalPipeline]:
    """
    Build the actual validated StaticBQN, wrap that exact object in the
    unified BQN adapter, and use it as the shared per-window backbone of the
    generic TemporalPipeline.

    Dropout is disabled so padding-invariance can be tested exactly.
    """
    torch.manual_seed(seed)

    reference_config = BQNConfig(
        n_rois=100,
        n_layers=3,
        dropout=0.0,
        activation="leaky_relu",
        cluster_num=4,
        preserve_unused_mlp_b=True,
        orthogonal_centers=True,
        freeze_centers=True,
        project_assignment=True,
    )
    reference = StaticBQN(reference_config)

    adapter_config = BQNBackboneConfig(
        n_rois=100,
        n_layers=3,
        dropout=0.0,
        activation="leaky_relu",
        cluster_num=4,
        preserve_unused_mlp_b=True,
        orthogonal_centers=True,
        freeze_centers=True,
        project_assignment=True,
    )
    backbone = BQNBackbone(adapter_config, model=reference)

    aggregator = BiGRUAggregator(
        BiGRUConfig(
            input_dim=backbone.embedding_dim,
            hidden_dim=5,
            num_layers=1,
            dropout=0.0,
            pooling="final",
        )
    )

    pipeline = TemporalPipeline(
        backbone=backbone,
        temporal_aggregator=aggregator,
        num_classes=2,
    )

    return reference, backbone, pipeline


def _symmetric_fc(
    batch: int,
    windows: int,
    rois: int = 100,
) -> torch.Tensor:
    """
    Generate finite symmetric correlation-like matrices with unit diagonal.

    These are synthetic integration-test inputs, not a replacement for the
    already-passed real ABIDE artifact audit.
    """
    x = torch.empty(batch, windows, rois, rois).uniform_(-0.5, 0.5)
    x = 0.5 * (x + x.transpose(-1, -2))

    diagonal = torch.arange(rois)
    x[..., diagonal, diagonal] = 1.0
    return x


def test_real_bqn_temporal_pipeline_shapes_and_shared_model_identity() -> None:
    reference, backbone, pipeline = _make_real_bqn_temporal_pipeline()

    # The adapter must still wrap the exact validated reference object.
    assert backbone.reference_model is reference
    assert pipeline.backbone is backbone
    assert pipeline.backbone.reference_model is reference

    # There is one BQN object shared across every temporal window, not K
    # independently instantiated BQNs.
    bqn_objects = [
        module
        for module in pipeline.modules()
        if isinstance(module, StaticBQN)
    ]
    assert len(bqn_objects) == 1
    assert bqn_objects[0] is reference

    x = _symmetric_fc(batch=2, windows=4)
    lengths = torch.tensor([4, 2], dtype=torch.long)

    pipeline.eval()
    with torch.no_grad():
        out = pipeline(x, lengths, return_aux=True)

    assert out.logits.shape == (2, 2)
    assert out.temporal_embedding.shape == (2, 10)
    assert out.aux["window_embeddings"].shape == (2, 4, 32)
    assert out.aux["n_valid_windows"] == 6
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.temporal_embedding).all()
    assert torch.isfinite(out.aux["window_embeddings"]).all()
    
def test_real_bqn_temporal_pipeline_encodes_exactly_valid_windows() -> None:
    reference, _, pipeline = _make_real_bqn_temporal_pipeline(seed=21)

    seen_batch_sizes: list[int] = []

    def pre_hook(_module, args):
        fc = args[0]
        seen_batch_sizes.append(int(fc.shape[0]))

    handle = reference.register_forward_pre_hook(pre_hook)
    try:
        x = _symmetric_fc(batch=3, windows=5)
        lengths = torch.tensor([5, 3, 1], dtype=torch.long)

        pipeline.eval()
        with torch.no_grad():
            out = pipeline(x, lengths, return_aux=True)
    finally:
        handle.remove()

    # 5 + 3 + 1 = 9 genuine windows. Padded windows must never enter BQN.
    assert seen_batch_sizes == [9]
    assert out.aux["n_valid_windows"] == 9


def test_real_bqn_temporal_pipeline_is_exactly_padding_invariant() -> None:
    _, _, pipeline = _make_real_bqn_temporal_pipeline(seed=31)
    pipeline.eval()

    x1 = _symmetric_fc(batch=3, windows=5)
    lengths = torch.tensor([5, 3, 1], dtype=torch.long)

    x2 = x1.clone()

    # Deliberately replace only padded matrices with extreme but finite values.
    # If padding ever reaches BQN or influences BiGRU aggregation, this test
    # should fail.
    x2[1, 3:] = 1.0e6 * torch.randn_like(x2[1, 3:])
    x2[2, 1:] = 1.0e6 * torch.randn_like(x2[2, 1:])

    with torch.no_grad():
        out1 = pipeline(x1, lengths, return_aux=True)
        out2 = pipeline(x2, lengths, return_aux=True)

    torch.testing.assert_close(
        out1.aux["window_embeddings"],
        out2.aux["window_embeddings"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        out1.temporal_embedding,
        out2.temporal_embedding,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        out1.logits,
        out2.logits,
        rtol=0.0,
        atol=0.0,
    )


def test_real_bqn_temporal_pipeline_backpropagates_through_bqn_gru_classifier() -> None:
    reference, _, pipeline = _make_real_bqn_temporal_pipeline(seed=41)
    pipeline.train()

    x = _symmetric_fc(batch=2, windows=3)
    lengths = torch.tensor([3, 2], dtype=torch.long)

    out = pipeline(x, lengths)
    loss = out.logits.square().mean()
    loss.backward()

    # Active BQN quadratic paths must receive gradients.
    for index, layer in enumerate(reference.qp_layers):
        assert layer.mlp_a.weight.grad is not None, (
            f"BQN layer {index} mlp_a did not receive gradient."
        )
        assert layer.mlp_h.weight.grad is not None, (
            f"BQN layer {index} mlp_h did not receive gradient."
        )

        # This parameter exists only for exact reference fidelity and is not
        # active in the validated BQN forward path.
        if layer.mlp_b is not None:
            assert layer.mlp_b.weight.grad is None, (
                "Reference-inactive mlp_b unexpectedly received gradient."
            )

    # Temporal and subject-level classification components must also train.
    assert pipeline.temporal_aggregator.gru.weight_ih_l0.grad is not None
    assert pipeline.temporal_aggregator.gru.weight_hh_l0.grad is not None
    assert pipeline.classifier.weight.grad is not None

    assert torch.isfinite(
        pipeline.temporal_aggregator.gru.weight_ih_l0.grad
    ).all()
    assert torch.isfinite(pipeline.classifier.weight.grad).all()
