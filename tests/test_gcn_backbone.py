"""Independent graph-math checks and GCN integration with existing pipelines."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from models.gcn.gcn_model import GCNConfig, DenseGCNLayer, StaticGCN, normalized_positive_adjacency
from unified_bn.backbones.gcn import GCNBackbone, GCNBackboneConfig
from unified_bn.training.abide_cli import ABIDECLIError, build_parser, build_single_candidate
from unified_bn.training.model_factory import ModelFactoryError, build_model_from_candidate
from unified_bn.training.search_space import (
    GCNSearchSpace, BiGRUSearchSpace, SearchSpaceError, generate_gcn_backbone_configs,
    generate_vanilla_candidates, generate_temporal_candidates, generate_fusion_candidates,
    candidate_fingerprint,
)
from unified_bn.temporal.windowing import WindowPolicyConfig


def fc(batch=3, rois=8):
    raw = torch.randn(batch, rois, rois, generator=torch.Generator().manual_seed(12))
    values = torch.tanh((raw + raw.transpose(-1, -2)) / 2)
    values.diagonal(dim1=-2, dim2=-1).fill_(1)
    return values


def test_hand_calculated_normalization_and_exactly_one_self_loop():
    r = torch.tensor([[[1., .5, -.7], [.5, 1., -.2], [-.7, -.2, 1.]]], dtype=torch.float64)
    original = r.clone()
    expected = torch.tensor([[[2/3, 1/3, 0.], [1/3, 2/3, 0.], [0., 0., 1.]]], dtype=r.dtype)
    torch.testing.assert_close(normalized_positive_adjacency(r), expected)
    assert torch.equal(r, original)
    assert torch.equal(normalized_positive_adjacency(torch.eye(3).unsqueeze(0)), torch.eye(3).unsqueeze(0))
    negative = torch.full((2, 3, 3), -.5)
    negative.diagonal(dim1=-2, dim2=-1).fill_(1)
    assert torch.equal(normalized_positive_adjacency(negative), torch.eye(3).expand(2, 3, 3))


def test_layer_logits_and_gradients_against_edgewise_reference():
    # Unequal degrees distinguish post-aggregation bias from A_hat(HW + b).
    adjacency = normalized_positive_adjacency(torch.tensor(
        [[[1., .6, .2], [.6, 1., -.3], [.2, -.3, 1.]]], dtype=torch.float64))
    layer = DenseGCNLayer(2, 2).double()
    with torch.no_grad():
        layer.linear.weight.copy_(torch.tensor([[1., 2.], [-3., .5]]))
        layer.bias.copy_(torch.tensor([.4, -.8]))
    features = torch.tensor([[[1., -2.], [3., .4], [-.2, 2.]]], dtype=torch.float64, requires_grad=True)
    observed = layer(features, adjacency)
    f = features.detach().clone().requires_grad_()
    w = layer.linear.weight.detach().clone().requires_grad_()
    b = layer.bias.detach().clone().requires_grad_()
    expected = torch.stack([
        sum(adjacency[0, i, j] * (w @ f[0, j]) for j in range(3)) + b
        for i in range(3)
    ]).unsqueeze(0)
    torch.testing.assert_close(observed, expected)
    observed.square().sum().backward()
    expected.square().sum().backward()
    for actual, reference in [(features.grad, f.grad), (layer.linear.weight.grad, w.grad), (layer.bias.grad, b.grad)]:
        torch.testing.assert_close(actual, reference)


def test_signed_features_per_graph_adjacency_and_batch_independence():
    torch.manual_seed(1)
    model = StaticGCN(GCNConfig(n_rois=3, hidden_dim=5, dropout=0)).eval()
    r = torch.tensor([[[1., -.2, .4], [-.2, 1., -.7], [.4, -.7, 1.]]])
    changed = r.clone()
    changed[:, 0, 1] = changed[:, 1, 0] = -.9
    assert torch.equal(normalized_positive_adjacency(r), normalized_positive_adjacency(changed))
    seen = []
    model.layers[0].register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
    first = model(r, return_embedding=True)
    second = model(changed, return_embedding=True)
    assert torch.equal(seen[0], r) and torch.equal(seen[1], changed)
    assert not torch.allclose(first, second)
    other = r.clone()
    other[:, 0, 1] = other[:, 1, 0] = .8
    assert not torch.equal(normalized_positive_adjacency(r), normalized_positive_adjacency(other))
    torch.testing.assert_close(model(torch.cat([r, other]))[:1], model(r))


@pytest.mark.parametrize("classes", [2, 4])
def test_schaefer100_logits_and_single_subject_batch(classes):
    model = StaticGCN(GCNConfig(num_classes=classes)).eval()
    x = fc(1, 100)
    assert model(x).shape == (1, classes)
    assert model(x, return_embedding=True).shape == (1, 64)


@pytest.mark.parametrize("kwargs", [
    {"n_rois": 0}, {"n_layers": 0}, {"hidden_dim": 0}, {"n_layers": True},
    {"hidden_dim": 32.5}, {"num_classes": 1}, {"dropout": 1.},
    {"dropout": float("nan")}, {"dropout": True}, {"adjacency": "signed"},
    {"node_features": "identity"}, {"readout": "sum"},
])
def test_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        GCNConfig(**kwargs)


def test_invalid_fc_fails_closed():
    x = fc(2)
    nonsymmetric = x.clone(); nonsymmetric[:, 0, 1] = .123
    invalid = [x[:, :3, :3] * 2, x[:, :3, :2], torch.eye(8),
               nonsymmetric, torch.full_like(x, float("nan")), x * .5]
    for value in invalid:
        with pytest.raises(ValueError):
            normalized_positive_adjacency(value)
    with pytest.raises(TypeError):
        normalized_positive_adjacency(torch.ones(1, 3, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="Expected"):
        StaticGCN()(x)


def candidate(mode, classes):
    common = dict(seed=42, split_identity="sha256:gcn_test", num_classes=classes,
                  backbone_space=GCNSearchSpace(n_rois=8, hidden_dims=(12,)))
    if mode == "vanilla":
        return generate_vanilla_candidates(**common)[0]
    common.update(window_policy=WindowPolicyConfig(policy="native"),
                  temporal_space=BiGRUSearchSpace(hidden_dims=(8,), num_layers=(1,), poolings=("final",)))
    return (generate_temporal_candidates if mode == "temporal" else generate_fusion_candidates)(**common)[0]


@pytest.mark.parametrize("classes", [2, 4])
@pytest.mark.parametrize("mode", ["vanilla", "temporal", "fusion"])
def test_sequences_gradients_separate_fusion_weights_and_checkpoint_roundtrip(mode, classes):
    torch.set_num_threads(1)
    cand = candidate(mode, classes)
    model = build_model_from_candidate(cand).eval()
    x = fc(3)
    dynamic = torch.stack([x, x * .5, x * .1], dim=1)
    dynamic.diagonal(dim1=-2, dim2=-1).fill_(1)
    lengths = torch.tensor([3, 1, 2])
    dynamic[1, 1:] = 12345  # Invalid padding must never reach the GCN.
    dynamic[2, 2:] = 12345
    seen = []
    if mode == "vanilla":
        out = model(x)
        assert isinstance(model.backbone, GCNBackbone)
        assert model.classifier is model.backbone.reference_model.classifier
        assert torch.equal(out.logits, model.backbone.classify(x))
        encoders = [model.backbone]
    else:
        backbone = model.backbone if mode == "temporal" else model.temporal_backbone
        backbone.reference_model.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
        out = model(dynamic, lengths, return_aux=True) if mode == "temporal" else model(x, dynamic, lengths, return_aux=True)
        assert torch.equal(seen[0], torch.cat([dynamic[i, :k] for i, k in enumerate(lengths)]))
        assert out.aux["n_valid_windows"] == 6
        assert torch.equal(out.aux["effective_lengths"], lengths)
        encoders = [backbone]
        if mode == "fusion":
            assert model.static_backbone is not backbone
            assert {id(p) for p in model.static_backbone.parameters()}.isdisjoint({id(p) for p in backbone.parameters()})
            encoders.append(model.static_backbone)
    assert out.logits.shape == (3, classes)
    torch.nn.functional.cross_entropy(out.logits, torch.arange(3) % classes).backward()
    for encoder in encoders:
        for layer in encoder.reference_model.layers:
            grad = layer.linear.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    restored = build_model_from_candidate(cand).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    restored_out = restored(x) if mode == "vanilla" else restored(dynamic, lengths) if mode == "temporal" else restored(x, dynamic, lengths)
    assert torch.equal(out.logits, restored_out.logits)


def test_search_config_identity_and_invalid_fields():
    space = GCNSearchSpace(hidden_dims=(32, 64), dropouts=(.1, .3))
    configs = generate_gcn_backbone_configs(space)
    assert len(configs) == 4 and configs == generate_gcn_backbone_configs(space)
    assert set(configs[0]) - {"name"} == set(GCNBackboneConfig.__dataclass_fields__)
    fields = json.loads(json.dumps(asdict(GCNBackboneConfig())))
    assert GCNBackboneConfig(**fields) == GCNBackboneConfig()
    cand = candidate("vanilla", 2)
    assert candidate_fingerprint(cand) == candidate_fingerprint(candidate("vanilla", 2))
    changed = dict(cand.identity.backbone_config, hidden_dim=24)
    assert candidate_fingerprint(cand) != candidate_fingerprint(replace(cand, identity=replace(cand.identity, backbone_config=changed)))
    for mutation in ("extra", "missing"):
        cfg = dict(cand.identity.backbone_config)
        if mutation == "extra": cfg["unexpected"] = 1
        else: cfg.pop("adjacency")
        with pytest.raises(ModelFactoryError):
            build_model_from_candidate(replace(cand, identity=replace(cand.identity, backbone_config=cfg)))
    with pytest.raises(ValueError, match="differs"):
        GCNBackbone(GCNBackboneConfig(n_rois=8), model=StaticGCN(GCNConfig(n_rois=8)), num_classes=4)
    for kwargs in ({"hidden_dims": ()}, {"dropouts": (.1, .1)}, {"hidden_dims": (True,)}):
        with pytest.raises((SearchSpaceError, TypeError)):
            GCNSearchSpace(**kwargs)


def cli_args(mode="vanilla", *extra):
    return build_parser().parse_args(["--mode", mode, "--seed", "42", "--split-path", "s.npz",
        "--static-h5", "s.h5", "--temporal-h5", "t.h5", "--search-dir", "out", "--device", "cpu",
        "--backbone", "gcn", "--expected-n-rois", "8", *extra])


@pytest.mark.parametrize("mode", ["vanilla", "temporal", "fusion", "fusion_templasso"])
def test_cli_materializes_gcn_for_all_four_variants(mode):
    flags = [] if mode == "vanilla" else ["--bigru-hidden-dim", "8", "--bigru-layers", "1", "--bigru-pooling", "final"]
    if mode == "fusion_templasso": flags += ["--lasso-C", "1"]
    cand = build_single_candidate(cli_args(mode, "--num-classes", "4", "--gcn-hidden-dim", "16", *flags), split_identity="s")
    assert cand.identity.backbone_config["name"] == "gcn"
    assert cand.identity.backbone_config["hidden_dim"] == 16
    assert cand.identity.num_classes == 4
    if mode == "fusion_templasso":
        assert cand.identity.temporal_backbone_config["name"] == "lasso_edges"


@pytest.mark.parametrize("flags", [("--bqn-layers", "1"), ("--alter-layers", "1"), ("--bnc-dropout", ".1"), ("--bnt-layers", "1")])
def test_gcn_rejects_other_backbone_flags(flags):
    with pytest.raises(ABIDECLIError):
        build_single_candidate(cli_args("vanilla", *flags), split_identity="s")


@pytest.mark.parametrize("backbone", ["bqn", "alter", "brainnetcnn", "bnt"])
def test_other_backbones_cannot_ignore_gcn_flags(backbone):
    with pytest.raises(ABIDECLIError, match="GCN arguments"):
        build_single_candidate(cli_args("vanilla", "--backbone", backbone, "--gcn-layers", "2"), split_identity="s")
