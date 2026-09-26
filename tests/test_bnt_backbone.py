"""Reference equality and registration of the supplied GraphTransformer."""
import ast
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.bnt.bnt_model import BNTConfig, StaticBNT
from unified_bn.backbones.bnt import BNTBackbone, BNTBackboneConfig
from unified_bn.training.abide_cli import ABIDECLIError, build_parser, build_single_candidate
from unified_bn.training.model_factory import ModelFactoryError, build_model_from_candidate
from unified_bn.training.search_space import (BNTSearchSpace, BQNSearchSpace, ALTERSearchSpace,
    BrainNetCNNSearchSpace, BiGRUSearchSpace, SearchSpaceError, generate_bnt_backbone_configs,
    generate_vanilla_candidates, generate_temporal_candidates, generate_fusion_candidates,
    candidate_fingerprint)
from unified_bn.temporal.windowing import WindowPolicyConfig

ROOT = Path(__file__).resolve().parents[1]


def source_model(n_rois, readout):
    raw = (ROOT / "models/bnt/GraphTransformer.source.txt").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "7477f8d6d4f96c135b7d0f6c80ad60ff688f846cee7312ec43c90495c5aa679c"
    # Only the original class is evaluated. Its configuration annotation is
    # unused; the original BaseModel is simply nn.Module with an abstract stub.
    # This avoids importing the ALTER runner, CUDA factory, or OmegaConf.
    tree = ast.parse(raw.decode())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    assert len(tree.body) == 1 and tree.body[0].name == "GraphTransformer"
    scope = dict(torch=torch, nn=nn, TransformerEncoderLayer=nn.TransformerEncoderLayer,
                 BaseModel=nn.Module, DictConfig=object)
    exec(compile(tree, "GraphTransformer.source.txt", "exec"), scope)
    config = SimpleNamespace(model=SimpleNamespace(readout=readout, self_attention_layer=2),
                             dataset=SimpleNamespace(node_sz=n_rois, node_feature_sz=n_rois))
    return scope["GraphTransformer"](config)


def fc(batch=3, rois=8):
    raw = torch.randn(batch, rois, rois, generator=torch.Generator().manual_seed(12))
    values = torch.tanh((raw + raw.transpose(-1, -2)) / 2)
    values.diagonal(dim1=-2, dim2=-1).fill_(1)
    return values


@pytest.mark.parametrize("readout", ["mean", "concat", "max", "sum"])
@pytest.mark.parametrize("training", [False, True])
def test_weights_logits_and_gradients_match_original_exactly(readout, training):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    original = source_model(8, readout)
    torch.manual_seed(17)
    port = StaticBNT(BNTConfig(n_rois=8, readout=readout))
    assert set(original.state_dict()) == set(port.state_dict())
    for key, value in original.state_dict().items():
        assert torch.equal(value, port.state_dict()[key]), key
    original.train(training); port.train(training)
    a, b = fc().requires_grad_(), fc().requires_grad_()
    torch.manual_seed(27)
    expected = original(None, a)
    torch.manual_seed(27)
    observed = port(b)
    assert torch.equal(expected, observed)
    expected.square().sum().backward(); observed.square().sum().backward()
    assert torch.equal(a.grad, b.grad)
    for (name, p), (other, q) in zip(original.named_parameters(), port.named_parameters()):
        assert name == other and torch.equal(p.grad, q.grad), name


def test_default_schaefer100_and_no_grad_evaluation_match():
    torch.manual_seed(42)
    original = source_model(100, "mean").eval()
    torch.manual_seed(42)
    model = StaticBNT().eval()
    assert sum(p.numel() for p in model.parameters()) == 527594
    assert model.embedding_dim == 100
    with torch.no_grad():
        assert torch.equal(original(None, fc(2, 100)), model(fc(2, 100)))
    model.load_state_dict(original.state_dict(), strict=True)
    assert model.classifier is model.fc


@pytest.mark.parametrize("kwargs", [{"n_rois": 10}, {"n_layers": 0}, {"dropout": 1.0},
    {"dropout": float("nan")}, {"readout": "invalid"}, {"nhead": 0}, {"num_classes": 1},
    {"n_rois": 100.0}, {"n_layers": True}])
def test_invalid_config_fails_before_model_build(kwargs):
    with pytest.raises((ValueError, TypeError)):
        BNTConfig(**kwargs)


def test_invalid_inputs_and_original_sum_batchnorm_constraint():
    model = StaticBNT(BNTConfig(n_rois=8))
    for x in (fc(2, 12), torch.eye(8), torch.full((2, 8, 8), float("nan"))):
        with pytest.raises(ValueError):
            model(x)
    with pytest.raises(TypeError):
        model(torch.ones(2, 8, 8, dtype=torch.int64))
    with pytest.raises(ValueError, match="at least two"):
        StaticBNT(BNTConfig(n_rois=8, readout="sum"))(fc(1))
    assert StaticBNT(BNTConfig(n_rois=8, readout="sum")).eval()(fc(1)).shape == (1, 2)


def candidate(mode="vanilla", classes=2, readout="mean"):
    common = dict(seed=42, split_identity="sha256:s", num_classes=classes,
                  backbone_space=BNTSearchSpace(n_rois=8, n_layers=(1,), readouts=(readout,)))
    if mode == "vanilla":
        return generate_vanilla_candidates(**common)[0]
    common.update(window_policy=WindowPolicyConfig(policy="native"),
                  temporal_space=BiGRUSearchSpace(hidden_dims=(8,), num_layers=(1,), poolings=("final",)))
    return (generate_temporal_candidates if mode == "temporal" else generate_fusion_candidates)(**common)[0]


@pytest.mark.parametrize("classes", [2, 4])
@pytest.mark.parametrize("mode", ["vanilla", "temporal", "fusion"])
def test_all_modes_classes_and_sequence_lengths(mode, classes):
    cand = candidate(mode, classes)
    model = build_model_from_candidate(cand).eval()
    x = fc(3)
    dynamic = torch.stack([x, x * .5, x * .1], dim=1)
    lengths = torch.tensor([3, 1, 2])
    dynamic[1, 1:] = 12345
    dynamic[2, 2:] = 12345
    seen = []
    if mode == "vanilla":
        assert isinstance(model.backbone, BNTBackbone)
        assert model.classifier is model.backbone.reference_model.fc
        out = model(x)
        assert torch.equal(out.logits, model.backbone.classify(x))
    else:
        backbone = model.backbone if mode == "temporal" else model.temporal_backbone
        backbone.reference_model.register_forward_pre_hook(lambda m, inputs: seen.append(inputs[0].detach().clone()))
        out = model(dynamic, lengths, return_aux=True) if mode == "temporal" else model(x, dynamic, lengths, return_aux=True)
        assert torch.equal(seen[0], torch.cat([dynamic[i, :k] for i, k in enumerate(lengths)]))
        assert out.aux["n_valid_windows"] == 6
        assert torch.equal(out.aux["effective_lengths"], lengths)
        if mode == "fusion":
            assert model.static_backbone is not model.temporal_backbone
            assert {id(p) for p in model.static_backbone.parameters()}.isdisjoint(
                {id(p) for p in model.temporal_backbone.parameters()})
    assert out.logits.shape == (3, classes)
    torch.nn.functional.cross_entropy(out.logits, torch.arange(3) % classes).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    restored = build_model_from_candidate(cand).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    restored_out = restored(x) if mode == "vanilla" else restored(dynamic, lengths) if mode == "temporal" else restored(x, dynamic, lengths)
    assert torch.equal(out.logits, restored_out.logits)


def test_config_roundtrip_unknown_keys_and_adapter_mismatch():
    payload = generate_bnt_backbone_configs(BNTSearchSpace())[0]
    assert set(payload) - {"name"} == set(BNTBackboneConfig.__dataclass_fields__)
    fields = json.loads(json.dumps(asdict(BNTBackboneConfig())))
    assert BNTBackboneConfig(**fields) == BNTBackboneConfig()
    for mutation in ("extra", "missing"):
        cand = candidate()
        config = dict(cand.identity.backbone_config)
        if mutation == "extra": config["unexpected"] = 1
        else: config.pop("readout")
        with pytest.raises(ModelFactoryError):
            build_model_from_candidate(replace(cand, identity=replace(cand.identity, backbone_config=config)))
    with pytest.raises(ValueError, match="differs"):
        BNTBackbone(BNTBackboneConfig(n_rois=8), model=StaticBNT(BNTConfig(n_rois=8)), num_classes=4)


def cli_args(mode="vanilla", *extra):
    return build_parser().parse_args(["--mode", mode, "--seed", "42", "--split-path", "s.npz",
        "--static-h5", "s.h5", "--temporal-h5", "t.h5", "--search-dir", "out", "--device", "cpu",
        "--backbone", "bnt", "--expected-n-rois", "8", *extra])


@pytest.mark.parametrize("mode", ["vanilla", "temporal", "fusion", "fusion_templasso"])
def test_cli_materializes_bnt_in_each_variation(mode):
    flags = [] if mode == "vanilla" else ["--bigru-hidden-dim", "8", "--bigru-layers", "1", "--bigru-pooling", "final"]
    if mode == "fusion_templasso": flags += ["--lasso-C", "1"]
    cand = build_single_candidate(cli_args(mode, "--num-classes", "4", *flags), split_identity="s")
    assert cand.identity.backbone_config["name"] == "bnt"
    assert cand.identity.num_classes == 4
    if mode == "fusion_templasso":
        assert cand.identity.temporal_backbone_config["name"] == "lasso_edges"


@pytest.mark.parametrize("flags", [("--bqn-layers", "1"), ("--alter-layers", "1"), ("--bnc-dropout", ".1")])
def test_cli_rejects_other_backbone_arguments(flags):
    with pytest.raises(ABIDECLIError):
        build_single_candidate(cli_args("vanilla", *flags), split_identity="s")


def test_bnt_arguments_cannot_be_silently_ignored():
    args = cli_args("vanilla", "--backbone", "brainnetcnn", "--bnt-layers", "2")
    with pytest.raises(ABIDECLIError, match="BNT arguments"):
        build_single_candidate(args, split_identity="s")
    with pytest.raises((SearchSpaceError, TypeError)):
        BNTSearchSpace(n_layers=(1, 1))
    with pytest.raises((SearchSpaceError, TypeError)):
        BNTSearchSpace(n_rois=10)


def test_existing_backbone_identities_are_unchanged():
    spaces = [BQNSearchSpace(n_layers=(1,), activations=("elu",), dropouts=(.1,), n_rois=8),
              ALTERSearchSpace(n_layers=(1,), n_rois=8, pos_embed_dim=8, cluster_num=2, ffn_dim=16),
              BrainNetCNNSearchSpace(n_rois=8)]
    expected = ["2229499366e02f1edb500530ebbd728b06ccb6d0682e023a16fc45a6d7db362a",
                "0027c6adb78c407bab5e58fdc075847188aaeb497bc4fe65aa20125805cb9e6f",
                "6bfaab1e2ab09dc49bfd98a213dc5fffb138287ea75c81bcd148c72f0139acd1"]
    for space, fingerprint in zip(spaces, expected):
        cand = generate_vanilla_candidates(seed=42, split_identity="sha256:bnt_baseline", backbone_space=space)[0]
        assert candidate_fingerprint(cand) == fingerprint
