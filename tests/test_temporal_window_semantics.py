"""
What "TempX" and "X + TempX / X + TempLasso" actually compute, checked
numerically rather than by reading the pipeline code.

These are the properties a temporalised backbone must have for the paper's
TempX row to mean what it says:

  1. Shared encoder, per window: the k-th window embedding of subject i is
     exactly what the static backbone produces for that window on its own.
     (One module, applied independently to every real window.)
  2. Padding never enters the encoder: garbage in padded slots cannot change
     any output.
  3. Length-aware aggregation: the BiGRU reads only the first ``length``
     steps, so appending padding does not change the subject embedding.
  4. Batch-composition invariance: a subject's output does not depend on who
     else is in the batch (no cross-subject leakage through the packing).
  5. Fusion trains both branches: after one backward, every trainable
     parameter in the static branch AND the temporal branch has a gradient,
     for ALTER+TempALTER and ALTER+TempLasso alike. The only parameters
     without one are the wrapped reference heads (``StaticALTER.classifier``),
     which the pipelines bypass by design because they consume ``encode()``
     and own their own head -- the same situation as BQN's internal
     ``classifier.*`` in the validated BQN+BQN fusion baseline.

Run for both registered backbones so the guarantees are about the pipeline,
not about one model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unified_bn.selectors.lasso import LassoEdgeSelector, LassoEdgeSelectorConfig
from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.model_factory import (
    build_model_from_candidate,
    build_templasso_model_from_candidate,
)
from unified_bn.training.search_space import (
    ALTERSearchSpace,
    BQNSearchSpace,
    BiGRUSearchSpace,
    LassoEdgeSearchSpace,
    generate_fusion_candidates,
    generate_fusion_templasso_candidates,
    generate_temporal_candidates,
)


R = 20
ALTER_SMALL = ALTERSearchSpace(
    n_layers=(1,), n_rois=R, pos_embed_dim=4, nhead=4, ffn_dim=16, cluster_num=3,
    dropouts=(0.0,),
)
BQN_SMALL = BQNSearchSpace(
    n_layers=(1,), activations=("elu",), dropouts=(0.0,), n_rois=R, cluster_num=2
)
BIGRU = BiGRUSearchSpace(hidden_dims=(8,), num_layers=(1,), poolings=("final",))
SPACES = {"bqn": BQN_SMALL, "alter": ALTER_SMALL}


def _dynamic_like(batch_size: int, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, k, R, R, generator=generator)
    x = 0.5 * (x + x.transpose(-1, -2))
    x = torch.tanh(x)
    diagonal = torch.arange(R)
    x[..., diagonal, diagonal] = 1.0
    return x


def _temporal_model(name: str, seed: int = 0):
    candidate = generate_temporal_candidates(
        seed=42,
        split_identity="sha256:split42",
        window_policy=WindowPolicyConfig(policy="native"),
        temporal_space=BIGRU,
        backbone_space=SPACES[name],
    )[0]
    torch.manual_seed(seed)
    return build_model_from_candidate(candidate).eval()


@pytest.mark.parametrize("name", ["bqn", "alter"])
def test_window_embeddings_are_the_static_backbone_applied_per_window(name) -> None:
    model = _temporal_model(name)
    lengths = torch.tensor([4, 2, 3], dtype=torch.long)
    dynamic_fc = _dynamic_like(3, 4, seed=1)

    with torch.no_grad():
        out = model(dynamic_fc, lengths, return_aux=True)
        for i, length in enumerate(lengths.tolist()):
            for k in range(length):
                single = model.backbone.encode(dynamic_fc[i, k : k + 1])[0]
                # Same module, same window; the only difference is that the
                # pipeline packs all real windows into one GEMM batch, which
                # moves the last float32 ulp of a reduction. Semantically it
                # is the static encoder applied per window.
                torch.testing.assert_close(
                    out.aux["window_embeddings"][i, k], single, rtol=0.0, atol=1e-5
                )
            # padded slots are exactly zero, never an encoding of padding
            assert torch.equal(
                out.aux["window_embeddings"][i, length:],
                torch.zeros_like(out.aux["window_embeddings"][i, length:]),
            )


@pytest.mark.parametrize("name", ["bqn", "alter"])
def test_padding_contents_cannot_change_any_output(name) -> None:
    model = _temporal_model(name)
    lengths = torch.tensor([3, 1], dtype=torch.long)
    clean = _dynamic_like(2, 3, seed=2)
    dirty = clean.clone()
    dirty[1, 1:] = 1e6  # beyond length for subject 1
    with torch.no_grad():
        a = model(clean, lengths)
        b = model(dirty, lengths)
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.temporal_embedding, b.temporal_embedding)


@pytest.mark.parametrize("name", ["bqn", "alter"])
def test_appending_padding_steps_does_not_change_the_subject(name) -> None:
    # Same subjects, same true lengths, longer padded K: the BiGRU must read
    # only ``length`` steps for the result to be identical.
    model = _temporal_model(name)
    lengths = torch.tensor([3, 2], dtype=torch.long)
    short = _dynamic_like(2, 3, seed=3)
    longer = torch.cat([short, torch.zeros(2, 4, R, R)], dim=1)
    with torch.no_grad():
        a = model(short, lengths)
        b = model(longer, lengths)
    assert torch.equal(a.logits, b.logits)


@pytest.mark.parametrize("name", ["bqn", "alter"])
def test_batch_composition_does_not_leak_between_subjects(name) -> None:
    model = _temporal_model(name)
    dynamic_fc = _dynamic_like(3, 3, seed=4)
    lengths = torch.tensor([3, 2, 3], dtype=torch.long)
    with torch.no_grad():
        together = model(dynamic_fc, lengths).logits
        alone = torch.cat(
            [model(dynamic_fc[i : i + 1], lengths[i : i + 1]).logits for i in range(3)]
        )
    # Different packings of the same windows through the encoder: allow the
    # last-ulp reduction-order noise of the attention/GEMM kernels, nothing more.
    torch.testing.assert_close(together, alone, rtol=0.0, atol=1e-5)
    # ...and permuting the batch permutes the outputs exactly.
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        permuted = model(dynamic_fc[perm], lengths[perm]).logits
    torch.testing.assert_close(permuted, together[perm], rtol=0.0, atol=1e-5)


def _grad_names(module) -> tuple[set[str], set[str]]:
    with_grad, without = set(), set()
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (with_grad if p.grad is not None else without).add(n)
    return with_grad, without


def test_alter_plus_temp_alter_trains_both_branches() -> None:
    candidate = generate_fusion_candidates(
        seed=42,
        split_identity="sha256:split42",
        window_policy=WindowPolicyConfig(policy="native"),
        temporal_space=BIGRU,
        backbone_space=ALTER_SMALL,
    )[0]
    torch.manual_seed(5)
    model = build_model_from_candidate(candidate)
    model.train()

    static_fc = _dynamic_like(3, 1, seed=6)[:, 0]
    dynamic_fc = _dynamic_like(3, 3, seed=7)
    lengths = torch.tensor([3, 2, 1], dtype=torch.long)
    out = model(static_fc, dynamic_fc, lengths)
    out.logits.sum().backward()

    with_grad, without = _grad_names(model)
    # The ONLY parameters without a gradient are the two wrapped reference
    # heads (StaticALTER.classifier of each branch). The pipeline consumes
    # encode() -- the pre-classifier embedding -- and owns its own head, so
    # those modules are bypassed by design; same situation as the internal
    # ``classifier.*`` of BQN in the validated BQN+BQN fusion baseline.
    expected_unused = {
        n for n, _ in model.named_parameters()
        if ".reference_model.classifier." in n
    }
    assert without == expected_unused, sorted(without ^ expected_unused)
    assert len(expected_unused) == 12
    # Everything else on BOTH branches trains: RRWP is parameter-free, so the
    # trainable ALTER parameters are the transformer blocks, DEC encoder and
    # dim_reduction of each branch, plus the BiGRU and the fusion head.
    for prefix in ("static_backbone.", "temporal_backbone.", "temporal_aggregator.", "classifier."):
        assert any(n.startswith(prefix) for n in with_grad), prefix
    assert out.fused_embedding.shape[-1] == model.static_backbone.embedding_dim + 16


def _fitted_selector(C: float = 0.05) -> LassoEdgeSelector:
    rng = np.random.default_rng(0)
    n_edges = R * (R - 1) // 2
    x = rng.normal(0, 0.3, size=(120, n_edges)).astype(np.float32)
    y = rng.integers(0, 2, size=120)
    x[y == 1, 1] += 0.8
    x[y == 1, 4] += 0.8
    config = LassoEdgeSelectorConfig(C=C, min_edges=1)
    return (
        LassoEdgeSelector(config)
        .fit(torch.from_numpy(x), torch.from_numpy(y))
        .freeze()
    )


def test_alter_plus_temp_lasso_trains_the_static_branch_and_bigru() -> None:
    selector = _fitted_selector()
    candidate = generate_fusion_templasso_candidates(
        seed=42,
        split_identity="sha256:split42",
        window_policy=WindowPolicyConfig(policy="native"),
        temporal_space=BIGRU,
        backbone_space=ALTER_SMALL,
        selector_space=LassoEdgeSearchSpace(Cs=(0.05,), n_rois=R, min_edges=1),
    )[0]
    torch.manual_seed(8)
    model = build_templasso_model_from_candidate(candidate, edge_selector=selector)
    model.train()

    static_fc = _dynamic_like(3, 1, seed=9)[:, 0]
    dynamic_fc = _dynamic_like(3, 3, seed=10)
    lengths = torch.tensor([3, 2, 1], dtype=torch.long)
    model(static_fc, dynamic_fc, lengths).logits.sum().backward()

    with_grad, without = _grad_names(model)
    expected_unused = {
        n for n, _ in model.named_parameters()
        if n.startswith("static_backbone.reference_model.classifier.")
    }
    assert without == expected_unused, sorted(without ^ expected_unused)
    # the lasso window encoder is parameter-free by design
    assert not any(n.startswith("temporal_backbone.") for n in with_grad)
    assert any(n.startswith("static_backbone.") for n in with_grad)
    assert any(n.startswith("temporal_aggregator.") for n in with_grad)
    assert model.temporal_backbone.embedding_dim == selector.n_selected
