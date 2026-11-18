"""
X + TempLasso — late fusion of a BQN static branch with a lasso_edges
temporal branch.

The one property this file exists to prove is in
``TestGradientFlow``: a single backward through ``FusionPipeline`` delivers
non-trivial gradients to the static BQN *and* to the BiGRU/head, while the
lasso_edges window encoder contributes no trainable parameters at all.

Also pins that adding ``RunIdentity.temporal_backbone_config`` left every
historical fusion / temporal / vanilla fingerprint untouched.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from unified_bn.backbones import BQNBackbone, SelectedEdgeBackbone
from unified_bn.data import FusionBatch, load_frozen_split
from unified_bn.pipelines.fusion import FusionPipeline
from unified_bn.selectors import LassoEdgeSelector, LassoEdgeSelectorConfig
from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.abide_runtime import (
    ABIDERuntimeConfig,
    ABIDERuntimeFactory,
    frozen_split_identity,
)
from unified_bn.training.config import CandidateRunConfig, RunIdentity
from unified_bn.training.model_factory import (
    ModelFactoryError,
    build_model_from_candidate,
    build_templasso_model_from_candidate,
)
from unified_bn.training.search_space import (
    BQNSearchSpace,
    BiGRUSearchSpace,
    LassoEdgeSearchSpace,
    candidate_fingerprint,
    generate_bqn_backbone_configs,
    generate_fusion_templasso_candidates,
)

R = 6
E = R * (R - 1) // 2
SUBJECTS = ("s1", "s2", "s3", "s4", "s5", "s6")
LABELS = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)


# ---------------------------------------------------------------- fixtures
def _write_split(path: Path) -> None:
    np.savez(path, train_ids=np.asarray(["s2", "s1", "s5", "s6"]),
             val_ids=np.asarray(["s3"]), test_ids=np.asarray(["s4"]),
             seed=np.asarray(42, dtype=np.int64))


def _write_temporal_h5(path: Path) -> None:
    rng = np.random.default_rng(7)
    lengths = np.asarray([3, 2, 4, 2, 3, 5], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    rows, cols = np.triu_indices(R, k=1)
    edges = rng.normal(0, 0.3, size=(int(lengths.sum()), len(rows))).astype(np.float32)
    w = 0
    for i, L in enumerate(lengths):
        sign = 1.0 if LABELS[i] == 1 else -1.0
        edges[w:w + L, 1] += 0.6 * sign; edges[w:w + L, 4] += 0.6 * sign; w += L
    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(SUBJECTS, dtype="S"))
        h5.create_dataset("labels", data=LABELS)
        h5.create_dataset("K", data=lengths); h5.create_dataset("offsets", data=offsets)
        h5.create_dataset("fc_z_upper", data=edges)
        h5.create_dataset("triu_row", data=rows.astype(np.int64))
        h5.create_dataset("triu_col", data=cols.astype(np.int64))


def _write_static_h5(path: Path) -> None:
    rng = np.random.default_rng(3)
    fc = np.zeros((len(SUBJECTS), R, R), dtype=np.float32)
    for i in range(len(SUBJECTS)):
        a = rng.normal(0, 0.3, size=(R, R)); a = 0.5 * (a + a.T); np.fill_diagonal(a, 1.0)
        fc[i] = np.clip(a, -0.99, 0.99)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(SUBJECTS, dtype="S"))
        h5.create_dataset("labels", data=LABELS); h5.create_dataset("fc", data=fc)


def _bqn_space() -> BQNSearchSpace:
    return BQNSearchSpace(n_layers=(1,), activations=("elu",), dropouts=(0.0,),
                          n_rois=R, cluster_num=2)


def _bigru_space() -> BiGRUSearchSpace:
    return BiGRUSearchSpace(hidden_dims=(4,), num_layers=(1,), poolings=("final",))


def _fusion_templasso_candidate(split_identity: str, *, C: float = 1.0) -> CandidateRunConfig:
    return generate_fusion_templasso_candidates(
        seed=42, split_identity=split_identity,
        window_policy=WindowPolicyConfig(policy="native"),
        temporal_space=_bigru_space(), backbone_space=_bqn_space(),
        selector_space=LassoEdgeSearchSpace(Cs=(C,), n_rois=R, min_edges=1),
    )[0]


def _fitted_selector(C: float = 1.0) -> LassoEdgeSelector:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=240)
    x = rng.normal(0, 1, size=(240, E))
    for e in (1, 4, 9):
        x[:, e] += 1.5 * (2 * y - 1)
    return LassoEdgeSelector(LassoEdgeSelectorConfig(C=C, min_edges=1)).fit(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y)).freeze()


def _sym(batch, k=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (batch, R, R) if k is None else (batch, k, R, R)
    a = torch.rand(*shape, generator=g) * 1.6 - 0.8
    a = 0.5 * (a + a.transpose(-1, -2))
    idx = torch.arange(R); a[..., idx, idx] = 1.0
    return a


# ================================================================ schema
class TestRunIdentitySchema:
    def test_temporal_backbone_config_is_fusion_only(self):
        lasso = {"name": "lasso_edges", "n_rois": R, "C": 1.0, "random_state": 0,
                 "max_iter": 3000, "input_space": "fisher_z", "min_edges": 1}
        bqn = generate_bqn_backbone_configs(_bqn_space())[0]
        with pytest.raises(ValueError, match="only meaningful for fusion"):
            RunIdentity(mode="vanilla", seed=42, split_identity="x",
                        backbone_config=bqn, temporal_backbone_config=lasso)
        with pytest.raises(ValueError, match="only meaningful for fusion"):
            RunIdentity(mode="temporal", seed=42, split_identity="x",
                        backbone_config=bqn, temporal_config={"name": "bigru", "hidden_dim": 4},
                        window_policy=WindowPolicyConfig(), temporal_backbone_config=lasso)

    def test_historical_fusion_serialization_is_byte_identical(self):
        """Adding the optional field must not touch existing fingerprints."""
        bqn = generate_bqn_backbone_configs(_bqn_space())[0]
        ident = RunIdentity(mode="fusion", seed=42, split_identity="x",
                            backbone_config=bqn, temporal_config={"name": "bigru", "hidden_dim": 4},
                            window_policy=WindowPolicyConfig(), fusion_config={"name": "concat"})
        payload = ident.to_dict()
        assert "temporal_backbone_config" not in payload
        assert set(payload) == {"mode", "seed", "split_identity", "backbone_config",
                                "temporal_config", "window_policy", "fusion_config"}

    def test_x_templasso_serializes_and_fingerprints_distinctly(self):
        a = _fusion_templasso_candidate("x", C=0.5)
        b = _fusion_templasso_candidate("x", C=1.0)
        pa = a.identity.to_dict()
        assert pa["temporal_backbone_config"]["name"] == "lasso_edges"
        assert pa["backbone_config"]["name"] == "bqn"
        assert pa["fusion_config"] == {"name": "concat"}
        assert candidate_fingerprint(a) != candidate_fingerprint(b)
        assert candidate_fingerprint(a) == candidate_fingerprint(_fusion_templasso_candidate("x", C=0.5))


# ================================================================ factory
class TestFactory:
    def test_builds_bqn_static_plus_selected_edge_temporal(self):
        cand = _fusion_templasso_candidate("x", C=1.0)
        sel = _fitted_selector(C=1.0)
        model = build_templasso_model_from_candidate(cand, edge_selector=sel)
        assert isinstance(model, FusionPipeline)
        assert isinstance(model.static_backbone, BQNBackbone)
        assert isinstance(model.temporal_backbone, SelectedEdgeBackbone)
        assert model.temporal_backbone.embedding_dim == sel.n_selected
        # concat head width = D_static + D_temporal(BiGRU 2*hidden)
        assert model.fusion.output_dim == model.static_backbone.embedding_dim + 2 * 4

    def test_requires_frozen_selector(self):
        cand = _fusion_templasso_candidate("x", C=1.0)
        with pytest.raises(ModelFactoryError, match="requires a FROZEN"):
            build_templasso_model_from_candidate(cand, edge_selector=None)  # type: ignore[arg-type]
        with pytest.raises(ModelFactoryError, match="requires a FROZEN"):
            build_model_from_candidate(cand)

    def test_recipe_mismatch_rejected(self):
        cand = _fusion_templasso_candidate("x", C=1.0)
        with pytest.raises(ModelFactoryError, match="does not match"):
            build_templasso_model_from_candidate(cand, edge_selector=_fitted_selector(C=0.5))

    def test_lasso_edges_in_static_slot_is_rejected(self):
        lasso = _fusion_templasso_candidate("x").identity.temporal_backbone_config
        bad = RunIdentity(mode="fusion", seed=42, split_identity="x",
                          backbone_config=lasso, temporal_config={"name": "bigru", "hidden_dim": 4},
                          window_policy=WindowPolicyConfig(), fusion_config={"name": "concat"})
        with pytest.raises(ModelFactoryError, match="static branch must be a real FC encoder"):
            build_model_from_candidate(CandidateRunConfig(identity=bad))

    def test_historical_fusion_path_unchanged(self):
        """Same-config BQN fusion still builds two independent BQNs."""
        bqn = generate_bqn_backbone_configs(_bqn_space())[0]
        ident = RunIdentity(mode="fusion", seed=42, split_identity="x",
                            backbone_config=bqn, temporal_config={"name": "bigru", "hidden_dim": 4},
                            window_policy=WindowPolicyConfig(), fusion_config={"name": "concat"})
        model = build_model_from_candidate(CandidateRunConfig(identity=ident))
        assert isinstance(model.static_backbone, BQNBackbone)
        assert isinstance(model.temporal_backbone, BQNBackbone)
        assert model.static_backbone is not model.temporal_backbone
        assert model.static_backbone.config == model.temporal_backbone.config


# ============================================================ gradient flow
def _none_grad_names(mod) -> list[str]:
    return sorted(n for n, p in mod.named_parameters() if p.requires_grad and p.grad is None)


def _grad_norm(mod) -> float:
    gs = [p.grad for p in mod.parameters() if p.grad is not None]
    return float(torch.sqrt(sum((g ** 2).sum() for g in gs))) if gs else 0.0


class TestGradientFlow:
    """
    What "gradient reaches both branches" has to mean here.

    BQN keeps two groups of parameters that never enter *any* fusion forward
    graph, by the reference design rather than by our wiring:
      * ``qp_layers.*.mlp_b.*``  - preserved only for state_dict compatibility
        (``preserve_unused_mlp_b=True``); unused even in vanilla.
      * ``reference_model.classifier.*`` - BQN's own head. Fusion calls
        ``backbone.encode()`` (the pre-classifier embedding) and trains its own
        head, so this module is bypassed. Only vanilla uses it.
    Those legitimately have ``grad=None``. The invariant we pin is therefore:
    the static BQN in X+TempLasso has *exactly the same* None-grad set as the
    static BQN in the already-validated BQN+BQN fusion, every other static
    parameter and every BiGRU/head parameter receives a finite gradient, and
    the per-branch gradient norms over used parameters are strictly positive.
    """

    def _ours(self):
        cand = _fusion_templasso_candidate("x", C=1.0)
        return build_templasso_model_from_candidate(cand, edge_selector=_fitted_selector(C=1.0))

    def _reference_bqn_bqn_fusion(self):
        bqn = generate_bqn_backbone_configs(_bqn_space())[0]
        ident = RunIdentity(mode="fusion", seed=42, split_identity="x",
                            backbone_config=bqn, temporal_config={"name": "bigru", "hidden_dim": 4},
                            window_policy=WindowPolicyConfig(), fusion_config={"name": "concat"})
        return build_model_from_candidate(CandidateRunConfig(identity=ident))

    @staticmethod
    def _backward(model):
        torch.manual_seed(0)
        model.train()
        out = model(_sym(4, seed=1), _sym(4, 5, seed=2), torch.tensor([5, 3, 4, 2]))
        model.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(out.logits, torch.tensor([0, 1, 1, 0])).backward()

    def test_static_branch_matches_validated_bqn_fusion_exactly(self):
        ours, ref = self._ours(), self._reference_bqn_bqn_fusion()
        self._backward(ours); self._backward(ref)

        ours_none = _none_grad_names(ours.static_backbone)
        ref_none = _none_grad_names(ref.static_backbone)
        # identical None-grad set to the validated homogeneous fusion
        assert ours_none == ref_none, {"ours_only": set(ours_none) - set(ref_none),
                                        "ref_only": set(ref_none) - set(ours_none)}
        # and every one of them is a known-bypassed BQN module, nothing else
        assert all(("mlp_b" in n) or (".classifier." in n) for n in ours_none), ours_none

        # everything else on the static branch got a finite gradient
        used = [(n, p.grad) for n, p in ours.static_backbone.named_parameters()
                if p.requires_grad and n not in ours_none]
        assert used and all(g is not None and torch.isfinite(g).all() for _, g in used)
        assert _grad_norm(ours.static_backbone) > 0

    def test_temporal_branch_and_head_are_fully_gradded(self):
        ours = self._ours(); self._backward(ours)
        for name, mod in (("BiGRU", ours.temporal_aggregator), ("head", ours.classifier)):
            params = [(n, p) for n, p in mod.named_parameters() if p.requires_grad]
            assert params, f"{name} has no trainable parameters"
            assert _none_grad_names(mod) == [], f"{name} params without grad"
            assert all(torch.isfinite(p.grad).all() for _, p in params)
            assert _grad_norm(mod) > 0
        # the lasso_edges window encoder is a fixed, train-fitted projection
        assert list(ours.temporal_backbone.parameters()) == []

    def test_each_branch_gradient_depends_on_its_own_input(self):
        torch.manual_seed(0)
        model = self._ours(); model.train()
        y = torch.tensor([0, 1, 1, 0]); L = torch.tensor([5, 3, 4, 2])
        s0, d0 = _sym(4, seed=1), _sym(4, 5, seed=2)

        def gvec(mod, static_fc, dyn_fc):
            model.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(static_fc, dyn_fc, L).logits, y).backward()
            return torch.cat([p.grad.flatten() for p in mod.parameters() if p.grad is not None])

        # perturb temporal input -> BiGRU gradient changes
        assert not torch.allclose(gvec(model.temporal_aggregator, s0, d0),
                                  gvec(model.temporal_aggregator, s0, _sym(4, 5, seed=99)))
        # perturb static input -> static BQN gradient changes
        assert not torch.allclose(gvec(model.static_backbone, s0, d0),
                                  gvec(model.static_backbone, _sym(4, seed=77), d0))

    def test_optimizer_step_updates_both_branches_and_never_the_edge_set(self):
        torch.manual_seed(0)
        model = self._ours(); model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        before_static = [p.detach().clone() for p in model.static_backbone.parameters()]
        before_bigru = [p.detach().clone() for p in model.temporal_aggregator.parameters()]
        before_S = model.temporal_backbone.selected_indices.clone()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            out = model(_sym(4, seed=1), _sym(4, 5, seed=2), torch.tensor([5, 3, 4, 2]))
            nn.functional.cross_entropy(out.logits, torch.tensor([0, 1, 1, 0])).backward()
            opt.step()
        assert any(not torch.equal(a, b) for a, b in zip(before_static, model.static_backbone.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(before_bigru, model.temporal_aggregator.parameters()))
        assert torch.equal(before_S, model.temporal_backbone.selected_indices)


# ================================================================ runtime
class TestRuntime:
    def _factory(self, tmp_path):
        split, static, temporal = tmp_path / "split.npz", tmp_path / "static.h5", tmp_path / "temporal.h5"
        _write_split(split); _write_static_h5(static); _write_temporal_h5(temporal)
        f = ABIDERuntimeFactory(ABIDERuntimeConfig(split_path=split, static_h5=static, temporal_h5=temporal,
                                                   device="cpu", expected_n_rois=R, pin_memory=False))
        return f, frozen_split_identity(load_frozen_split(split))

    def test_fits_selector_on_train_and_builds_heterogeneous_fusion(self, tmp_path):
        factory, ident = self._factory(tmp_path)
        cand = _fusion_templasso_candidate(ident, C=1.0)
        rt = factory(cand, 0, tmp_path / "run")
        assert isinstance(rt.model, FusionPipeline)
        assert isinstance(rt.model.static_backbone, BQNBackbone)
        assert isinstance(rt.model.temporal_backbone, SelectedEdgeBackbone)
        prov = rt.user_provenance["edge_selector"]
        assert prov["fitted_on"] == "train" and prov["n_train_subjects"] == 4
        assert prov["n_train_windows"] == 13
        batch = next(iter(rt.train_batches))
        assert isinstance(batch, FusionBatch)

    def test_selector_fit_sees_exactly_train_windows(self, tmp_path, monkeypatch):
        factory, ident = self._factory(tmp_path)
        seen = {}
        orig = LassoEdgeSelector.fit
        def spy(self, x, y):
            seen["n"] = int(x.shape[0]); return orig(self, x, y)
        monkeypatch.setattr(LassoEdgeSelector, "fit", spy)
        factory(_fusion_templasso_candidate(ident, C=1.0), 0, tmp_path / "run")
        assert seen["n"] == 13      # train only: s2(2)+s1(3)+s5(3)+s6(5)

    def test_historical_bqn_fusion_gets_no_selector(self, tmp_path):
        factory, ident = self._factory(tmp_path)
        bqn = generate_bqn_backbone_configs(_bqn_space())[0]
        cand = CandidateRunConfig(identity=RunIdentity(
            mode="fusion", seed=42, split_identity=ident, backbone_config=bqn,
            temporal_config={"name": "bigru", "hidden_dim": 4},
            window_policy=WindowPolicyConfig(policy="native"), fusion_config={"name": "concat"}))
        rt = factory(cand, 0, tmp_path / "run")
        assert "edge_selector" not in rt.user_provenance
        assert isinstance(rt.model.temporal_backbone, BQNBackbone)
