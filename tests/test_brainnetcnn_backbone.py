"""
BrainNetCNN as a third static backbone.

Pins (1) that the transcription is the collaborator's harness model up to the
documented deviations -- bitwise-equal logits against a verbatim copy of her
forward once its evaluation-time dropout is switched off; (2) that ``eval()``
really disables dropout here (the bug being fixed); (3) that the backbone,
search space, factory and CLI plumbing mirror BQN/ALTER exactly.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.brainnetcnn.brainnetcnn_model import BrainNetCNNConfig, StaticBrainNetCNN
from unified_bn.backbones import BrainNetCNNBackbone, BrainNetCNNBackboneConfig
from unified_bn.pipelines.temporal import TemporalPipeline
from unified_bn.pipelines.vanilla import VanillaPipeline
from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.abide_cli import ABIDECLIError, build_parser, _backbone_space_from_args
from unified_bn.training.config import CandidateRunConfig, RunIdentity
from unified_bn.training.model_factory import ModelFactoryError, build_model_from_candidate
from unified_bn.training.search_space import (
    BQNSearchSpace,
    BiGRUSearchSpace,
    BrainNetCNNSearchSpace,
    SearchSpaceError,
    candidate_fingerprint,
    generate_brainnetcnn_backbone_configs,
    generate_temporal_candidates,
    generate_vanilla_bqn_candidates,
    generate_vanilla_candidates,
)


# --------------------------------------------------------------------------
# Verbatim copy of the collaborator's forward (ALTER/models/brainnetcnn/
# brainnetcnn.py) with two mechanical changes: no omegaconf config object,
# and ``training=False`` on every F.dropout so the comparison is against the
# deterministic function her code *intends* at evaluation time.
class _HarnessE2EBlock(nn.Module):
    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class _HarnessBrainNetCNN(nn.Module):
    def __init__(self, node_sz):
        super().__init__()
        self.d = node_sz
        self.e2econv1 = _HarnessE2EBlock(1, 32, node_sz, bias=True)
        self.e2econv2 = _HarnessE2EBlock(32, 64, node_sz, bias=True)
        self.E2N = nn.Conv2d(64, 1, (1, self.d))
        self.N2G = nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, 30)
        self.dense3 = nn.Linear(30, 2)

    def forward(self, node_feature):
        node_feature = node_feature.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(node_feature), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5, training=False)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5, training=False)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5, training=False)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)
        return out


def _fc(batch: int, n_rois: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, n_rois, n_rois, generator=g)
    x = torch.tanh(0.5 * (x + x.transpose(-1, -2)))
    idx = torch.arange(n_rois)
    x[:, idx, idx] = 1.0
    return x


R = 12


def test_transcription_matches_the_harness_forward_bitwise() -> None:
    torch.manual_seed(0)
    ours = StaticBrainNetCNN(BrainNetCNNConfig(n_rois=R)).eval()
    theirs = _HarnessBrainNetCNN(R).eval()
    # Same parameter names by construction -> load ours into theirs.
    # ``classifier`` is a Sequential over the same dense modules, so those
    # parameters appear twice in our state dict (dense*.* and classifier.*).
    state = {k: v for k, v in ours.state_dict().items() if not k.startswith("classifier.")}
    missing, unexpected = theirs.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    fc = _fc(4, R, 1)
    with torch.no_grad():
        torch.testing.assert_close(ours(fc), theirs(fc), rtol=0.0, atol=0.0)


def test_same_seed_gives_the_same_parameter_draw_as_the_harness() -> None:
    torch.manual_seed(7)
    ours = StaticBrainNetCNN(BrainNetCNNConfig(n_rois=R))
    torch.manual_seed(7)
    theirs = _HarnessBrainNetCNN(R)
    for (name, p), (name2, q) in zip(ours.named_parameters(), theirs.named_parameters()):
        assert name == name2
        torch.testing.assert_close(p, q, rtol=0.0, atol=0.0)


def test_eval_disables_dropout_unlike_the_harness() -> None:
    torch.manual_seed(1)
    model = StaticBrainNetCNN(BrainNetCNNConfig(n_rois=R))
    fc = _fc(3, R, 2)
    model.eval()
    with torch.no_grad():
        a, b = model(fc), model(fc)
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
    model.train()
    with torch.no_grad():
        c, d = model(fc), model(fc)
    assert not torch.equal(c, d)


def test_embedding_is_the_n2g_vector_and_head_is_the_reference_dense_stack() -> None:
    torch.manual_seed(2)
    model = StaticBrainNetCNN(BrainNetCNNConfig(n_rois=R)).eval()
    fc = _fc(2, R, 3)
    with torch.no_grad():
        emb = model(fc, return_embedding=True)
        logits, aux = model(fc, return_aux=True)
    assert emb.shape == (2, 256) and model.embedding_dim == 256
    torch.testing.assert_close(aux["embedding"], emb)
    torch.testing.assert_close(model.classifier(emb), logits)


def test_config_fails_closed() -> None:
    with pytest.raises(ValueError):
        BrainNetCNNConfig(dropout=1.0)
    with pytest.raises(ValueError):
        BrainNetCNNConfig(e2e_channels=(32,))
    with pytest.raises(TypeError):
        BrainNetCNNConfig(n_rois=100.0)


# ---------------------------------------------------------------- backbone
def test_backbone_wraps_reference_and_validates_input() -> None:
    torch.manual_seed(3)
    backbone = BrainNetCNNBackbone(BrainNetCNNBackboneConfig(n_rois=R)).eval()
    fc = _fc(2, R, 4)
    with torch.no_grad():
        out = backbone(fc)
        torch.testing.assert_close(out.embedding, backbone.reference_model(fc, return_embedding=True))
        torch.testing.assert_close(backbone.classify(fc), backbone.reference_model(fc))
    with pytest.raises(ValueError):
        backbone(_fc(2, R + 1, 5))


def test_backbone_config_round_trips_through_json_lists() -> None:
    cfg = BrainNetCNNBackboneConfig(e2e_channels=[32, 64], dense_dims=[128, 30])
    assert cfg == BrainNetCNNBackboneConfig()
    assert hash(cfg) == hash(BrainNetCNNBackboneConfig())


# ---------------------------------------------------------------- search space
def test_search_space_payloads_mirror_the_backbone_config() -> None:
    payloads = generate_brainnetcnn_backbone_configs(
        BrainNetCNNSearchSpace(dropouts=(0.1, 0.3, 0.5), n_rois=R)
    )
    assert len(payloads) == 3
    for payload in payloads:
        assert payload["name"] == "brainnetcnn"
        mapping = dict(payload); mapping.pop("name")
        assert set(mapping) == set(BrainNetCNNBackboneConfig.__dataclass_fields__)
        BrainNetCNNBackboneConfig(**mapping)
    assert [p["dropout"] for p in payloads] == [0.1, 0.3, 0.5]


def test_search_space_fails_closed() -> None:
    with pytest.raises(SearchSpaceError):
        BrainNetCNNSearchSpace(dropouts=(1.5,))
    with pytest.raises((SearchSpaceError, TypeError)):
        BrainNetCNNSearchSpace(dropouts=0.5)  # not a tuple axis


def test_generic_vanilla_generator_equals_the_bqn_generator_for_bqn_spaces() -> None:
    space = BQNSearchSpace(n_layers=(1, 2), activations=("elu",), dropouts=(0.1,))
    a = generate_vanilla_candidates(seed=42, split_identity="sha256:s", backbone_space=space)
    b = generate_vanilla_bqn_candidates(seed=42, split_identity="sha256:s", backbone_space=space)
    assert [candidate_fingerprint(c) for c in a] == [candidate_fingerprint(c) for c in b]


# ---------------------------------------------------------------- factory
def _vanilla_candidate(dropout: float = 0.5) -> CandidateRunConfig:
    return generate_vanilla_candidates(
        seed=42, split_identity="sha256:split42",
        backbone_space=BrainNetCNNSearchSpace(dropouts=(dropout,), n_rois=R),
    )[0]


def test_vanilla_candidate_builds_the_reference_path() -> None:
    torch.manual_seed(4)
    model = build_model_from_candidate(_vanilla_candidate())
    assert isinstance(model, VanillaPipeline)
    assert isinstance(model.backbone, BrainNetCNNBackbone)
    assert model.classifier is model.backbone.reference_model.classifier
    assert {id(p) for p in model.parameters()} == {
        id(p) for p in model.backbone.reference_model.parameters()
    }
    model.eval()
    fc = _fc(3, R, 6)
    with torch.no_grad():
        torch.testing.assert_close(model(fc).logits, model.backbone.reference_model(fc), rtol=0.0, atol=0.0)


def test_factory_rejects_unknown_and_missing_keys() -> None:
    candidate = _vanilla_candidate()
    bad = dict(candidate.identity.backbone_config); bad["extra"] = 1
    with pytest.raises(ModelFactoryError):
        build_model_from_candidate(replace(candidate, identity=replace(candidate.identity, backbone_config=bad)))
    bad = dict(candidate.identity.backbone_config); bad.pop("dropout")
    with pytest.raises(ModelFactoryError):
        build_model_from_candidate(replace(candidate, identity=replace(candidate.identity, backbone_config=bad)))


def test_temporal_candidate_builds_temp_brainnetcnn() -> None:
    torch.manual_seed(5)
    candidate = generate_temporal_candidates(
        seed=42, split_identity="sha256:split42",
        window_policy=WindowPolicyConfig(policy="native"),
        temporal_space=BiGRUSearchSpace(hidden_dims=(8,), num_layers=(1,), poolings=("final",)),
        backbone_space=BrainNetCNNSearchSpace(dropouts=(0.3,), n_rois=R),
    )[0]
    model = build_model_from_candidate(candidate)
    assert isinstance(model, TemporalPipeline)
    assert isinstance(model.backbone, BrainNetCNNBackbone)
    assert model.backbone.config.dropout == 0.3


# ---------------------------------------------------------------- CLI
def _args(*extra: str):
    return build_parser().parse_args([
        "--mode", "vanilla", "--split-path", "s.npz", "--static-h5", "s.h5",
        "--temporal-h5", "t.h5", "--seed", "42", "--search-dir", "d", "--device", "cpu",
        *extra,
    ])


def test_cli_backbone_brainnetcnn_resolves_to_a_singleton_space() -> None:
    space = _backbone_space_from_args(_args("--backbone", "brainnetcnn"))
    assert space == BrainNetCNNSearchSpace(dropouts=(0.5,), n_rois=100)
    space = _backbone_space_from_args(_args("--backbone", "brainnetcnn", "--bnc-dropout", "0.2"))
    assert space.dropouts == (0.2,)


def test_cli_flag_groups_are_exclusive() -> None:
    with pytest.raises(ABIDECLIError):
        _backbone_space_from_args(_args("--backbone", "brainnetcnn", "--bqn-layers", "3"))
    with pytest.raises(ABIDECLIError):
        _backbone_space_from_args(_args("--backbone", "brainnetcnn", "--alter-layers", "1"))
    with pytest.raises(ABIDECLIError):
        _backbone_space_from_args(_args("--backbone", "alter", "--bnc-dropout", "0.2"))
    with pytest.raises(ABIDECLIError):
        _backbone_space_from_args(_args("--backbone", "bqn", "--bqn-layers", "3",
                                        "--bqn-activation", "elu", "--bqn-dropout", "0.1",
                                        "--bnc-dropout", "0.2"))
