"""
TempLasso — L1 edge selection + parameter-free window encoder + BiGRU.

Covers:
  * LassoEdgeSelector lifecycle, determinism, leakage boundary, state round-trip
  * SelectedEdgeBackbone contract, buffers-not-parameters, device movement
  * TemporalPipeline end-to-end with the new backbone (padding invariance)
  * model_factory dispatch and fail-closed behaviour
  * search-space candidate generation and fingerprint stability
  * ABIDERuntimeFactory fitting the selector on TRAIN only, with provenance
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from unified_bn.backbones import SelectedEdgeBackbone, SelectedEdgeBackboneConfig
from unified_bn.data import TemporalABIDEDataset, TemporalBatch, load_frozen_split
from unified_bn.pipelines.temporal import TemporalPipeline
from unified_bn.selectors import (
    LassoEdgeSelector,
    LassoEdgeSelectorConfig,
    SelectorState,
    pooled_edge_rows_from_dataset,
    upper_triangle_edges,
)
from unified_bn.temporal.bigru import BiGRUAggregator, BiGRUConfig
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
    BiGRUSearchSpace,
    LassoEdgeSearchSpace,
    candidate_fingerprint,
    generate_lasso_edge_backbone_configs,
    generate_templasso_candidates,
)


# ---------------------------------------------------------------- fixtures
R = 6                       # ROIs
E = R * (R - 1) // 2        # 15 edges


def _synthetic_rows(n: int = 240, *, seed: int = 0, signal_edges=(1, 4, 9)):
    """Rows where a few edges carry class signal; the rest are noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    x = rng.normal(0, 1, size=(n, E))
    for e in signal_edges:
        x[:, e] += 1.5 * (2 * y - 1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def _fitted_selector(C: float = 0.3, **kw) -> LassoEdgeSelector:
    # Synthetic rows carry only 3 signal edges, so relax the production
    # min_edges floor unless a test overrides it explicitly.
    kw.setdefault("min_edges", 1)
    x, y = _synthetic_rows()
    return LassoEdgeSelector(LassoEdgeSelectorConfig(C=C, **kw)).fit(x, y).freeze()


def _sym_fc(batch: int, k: int, *, seed: int = 0) -> torch.Tensor:
    """Random symmetric correlation-like [B,K,R,R] with unit diagonal."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(batch, k, R, R, generator=g) * 1.6 - 0.8
    a = 0.5 * (a + a.transpose(-1, -2))
    idx = torch.arange(R)
    a[..., idx, idx] = 1.0
    return a


SUBJECTS = ("s1", "s2", "s3", "s4", "s5", "s6")
LABELS = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)


def _write_split(path: Path, *, seed: int = 42) -> None:
    np.savez(
        path,
        train_ids=np.asarray(["s2", "s1", "s5", "s6"]),
        val_ids=np.asarray(["s3"]),
        test_ids=np.asarray(["s4"]),
        seed=np.asarray(seed, dtype=np.int64),
    )


def _write_temporal_h5(path: Path, *, n_rois: int = R) -> None:
    rng = np.random.default_rng(7)
    lengths = np.asarray([3, 2, 4, 2, 3, 5], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    rows, cols = np.triu_indices(n_rois, k=1)
    n_edges = len(rows)
    W = int(lengths.sum())
    # Fisher-z edges with mild label signal on edges 1 and 4
    edges = rng.normal(0, 0.3, size=(W, n_edges)).astype(np.float32)
    w = 0
    for i, L in enumerate(lengths):
        sign = 1.0 if LABELS[i] == 1 else -1.0
        edges[w:w + L, 1] += 0.6 * sign
        edges[w:w + L, 4] += 0.6 * sign
        w += L
    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(SUBJECTS, dtype="S"))
        h5.create_dataset("labels", data=LABELS)
        h5.create_dataset("K", data=lengths)
        h5.create_dataset("offsets", data=offsets)
        h5.create_dataset("fc_z_upper", data=edges)
        h5.create_dataset("triu_row", data=rows.astype(np.int64))
        h5.create_dataset("triu_col", data=cols.astype(np.int64))


def _write_static_h5(path: Path, *, n_rois: int = R) -> None:
    fc = np.zeros((len(SUBJECTS), n_rois, n_rois), dtype=np.float32)
    for i in range(len(SUBJECTS)):
        fc[i] = np.eye(n_rois, dtype=np.float32)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(SUBJECTS, dtype="S"))
        h5.create_dataset("labels", data=LABELS)
        h5.create_dataset("fc", data=fc)


def _templasso_candidate(split_path: Path, *, C: float = 1.0) -> CandidateRunConfig:
    return generate_templasso_candidates(
        seed=42,
        split_identity=frozen_split_identity(load_frozen_split(split_path)),
        window_policy=WindowPolicyConfig(policy="native"),
        temporal_space=BiGRUSearchSpace(
            hidden_dims=(4,), num_layers=(1,), poolings=("final",),
        ),
        selector_space=LassoEdgeSearchSpace(Cs=(C,), n_rois=R, min_edges=1),
    )[0]


# ================================================================ selector
class TestLassoEdgeSelector:
    def test_lifecycle_and_shapes(self):
        x, y = _synthetic_rows()
        sel = LassoEdgeSelector(LassoEdgeSelectorConfig(C=0.3, min_edges=1))
        assert sel.state is SelectorState.UNFITTED
        with pytest.raises(RuntimeError):
            sel.transform(x)
        sel.fit(x, y)
        assert sel.state is SelectorState.FITTED
        with pytest.raises(RuntimeError):
            sel.transform(x)                     # require_frozen default
        sel.freeze()
        assert sel.is_frozen
        out = sel.transform(x)
        assert out.shape == (x.shape[0], sel.n_selected)
        assert 1 <= sel.n_selected <= E
        assert torch.isfinite(out).all()

    def test_fit_is_single_shot(self):
        x, y = _synthetic_rows()
        sel = LassoEdgeSelector(LassoEdgeSelectorConfig(C=0.3, min_edges=1)).fit(x, y)
        with pytest.raises(RuntimeError):
            sel.fit(x, y)

    def test_recovers_signal_edges(self):
        sel = _fitted_selector(C=0.3)
        chosen = set(sel.selected_indices.tolist())
        assert {1, 4, 9} <= chosen, chosen

    def test_deterministic_across_repeats(self):
        a = _fitted_selector(C=0.3)
        b = _fitted_selector(C=0.3)
        assert np.array_equal(a.selected_indices, b.selected_indices)
        assert a.edge_set_fingerprint() == b.edge_set_fingerprint()
        x, _ = _synthetic_rows()
        assert torch.equal(a.transform(x), b.transform(x))

    def test_random_state_matters_only_through_liblinear(self):
        # Different random_state may or may not change S on easy synthetic
        # data; what must hold is that the same random_state is reproducible.
        a = _fitted_selector(C=0.3, random_state=3)
        b = _fitted_selector(C=0.3, random_state=3)
        assert a.edge_set_fingerprint() == b.edge_set_fingerprint()

    def test_transform_uses_frozen_train_statistics(self):
        sel = _fitted_selector(C=0.3)
        x, _ = _synthetic_rows()
        S = sel.selected_indices
        mu = torch.tensor(sel.train_mean[S], dtype=torch.float32)
        sd = torch.tensor(sel.train_std[S], dtype=torch.float32)
        expected = (x[:, S] - mu) / sd
        assert torch.allclose(sel.transform(x), expected, atol=1e-6)
        # A different (validation-like) matrix must be transformed with the
        # SAME statistics, not re-standardised.
        xv, _ = _synthetic_rows(n=50, seed=99)
        got = sel.transform(xv)
        assert torch.allclose(got, (xv[:, S] - mu) / sd, atol=1e-6)
        assert not torch.allclose(got.mean(0), torch.zeros_like(got.mean(0)), atol=1e-2)

    def test_transform_supports_leading_dims(self):
        sel = _fitted_selector(C=0.3)
        x = torch.randn(3, 7, E)
        out = sel.transform(x)
        assert out.shape == (3, 7, sel.n_selected)

    def test_rejects_wrong_edge_dim(self):
        sel = _fitted_selector(C=0.3)
        with pytest.raises(ValueError):
            sel.transform(torch.randn(4, E + 1))

    def test_min_edges_fails_closed(self):
        x, y = _synthetic_rows()
        with pytest.raises(ValueError, match="min_edges"):
            LassoEdgeSelector(LassoEdgeSelectorConfig(C=1e-4, min_edges=5)).fit(x, y)

    def test_requires_both_classes(self):
        x, _ = _synthetic_rows()
        y = torch.zeros(x.shape[0], dtype=torch.long)
        with pytest.raises(ValueError, match="both classes"):
            LassoEdgeSelector(LassoEdgeSelectorConfig(C=0.3, min_edges=1)).fit(x, y)

    def test_state_dict_roundtrip(self):
        sel = _fitted_selector(C=0.3)
        payload = sel.state_dict()
        rebuilt = LassoEdgeSelector.from_state_dict(payload)
        assert rebuilt.is_frozen
        assert rebuilt.edge_set_fingerprint() == sel.edge_set_fingerprint()
        x, _ = _synthetic_rows()
        assert torch.allclose(rebuilt.transform(x), sel.transform(x))

    def test_config_validation(self):
        with pytest.raises(ValueError):
            LassoEdgeSelectorConfig(C=0.0)
        with pytest.raises(ValueError):
            LassoEdgeSelectorConfig(input_space="raw")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            LassoEdgeSelectorConfig(min_edges=0)


class TestPooling:
    def test_pooled_rows_stack_every_window_with_subject_label(self, tmp_path):
        h5 = tmp_path / "t.h5"; _write_temporal_h5(h5)
        ds = TemporalABIDEDataset(h5, subject_ids=("s2", "s1"), expected_n_rois=R)
        x, y = pooled_edge_rows_from_dataset(ds, n_rois=R, input_space="fisher_z")
        ds.close()
        # s2 has K=2, s1 has K=3 -> 5 rows, in that order
        assert x.shape == (5, E)
        assert y.tolist() == [1, 1, 0, 0, 0]

    def test_fisher_z_roundtrip_matches_stored_edges(self, tmp_path):
        h5 = tmp_path / "t.h5"; _write_temporal_h5(h5)
        ds = TemporalABIDEDataset(h5, subject_ids=("s1",), expected_n_rois=R)
        x, _ = pooled_edge_rows_from_dataset(ds, n_rois=R, input_space="fisher_z")
        ds.close()
        with h5py.File(h5, "r") as f:
            stored = torch.tensor(f["fc_z_upper"][0:3])
        # dataset applies tanh; we apply atanh -> recover stored Fisher-z
        assert torch.allclose(x, stored, atol=1e-5)

    def test_upper_triangle_edges_order_is_canonical(self):
        fc = _sym_fc(2, 3)[0, 0]
        rows, cols = torch.triu_indices(R, R, offset=1)
        assert torch.equal(
            upper_triangle_edges(fc, input_space="correlation"), fc[rows, cols]
        )


# ================================================================ backbone
class TestSelectedEdgeBackbone:
    def test_contract_and_no_trainable_parameters(self):
        sel = _fitted_selector(C=0.3)
        bb = SelectedEdgeBackbone(sel, n_rois=R)
        assert bb.embedding_dim == sel.n_selected
        assert sum(p.numel() for p in bb.parameters()) == 0
        names = {n for n, _ in bb.named_buffers()}
        assert {"selected_indices", "train_mean", "train_std"} <= names
        fc = _sym_fc(4, 1)[:, 0]                   # [B,R,R]
        out = bb(fc)
        assert out.embedding.shape == (4, sel.n_selected)
        assert torch.isfinite(out.embedding).all()

    def test_matches_selector_transform(self):
        sel = _fitted_selector(C=0.3)
        bb = SelectedEdgeBackbone(sel, n_rois=R)
        fc = _sym_fc(3, 1)[:, 0]
        edges = upper_triangle_edges(fc, input_space="fisher_z")
        assert torch.allclose(bb.encode(fc), sel.transform(edges), atol=1e-6)

    def test_requires_frozen_selector(self):
        x, y = _synthetic_rows()
        unfrozen = LassoEdgeSelector(LassoEdgeSelectorConfig(C=0.3, min_edges=1)).fit(x, y)
        with pytest.raises(RuntimeError, match="FROZEN"):
            SelectedEdgeBackbone(unfrozen, n_rois=R)

    def test_rejects_roi_mismatch(self):
        sel = _fitted_selector(C=0.3)
        with pytest.raises(ValueError, match="implies E"):
            SelectedEdgeBackbone(sel, n_rois=R + 1)

    def test_state_dict_carries_buffers(self):
        sel = _fitted_selector(C=0.3)
        bb = SelectedEdgeBackbone(sel, n_rois=R)
        sd = bb.state_dict()
        assert torch.equal(sd["selected_indices"], torch.as_tensor(sel.selected_indices))

    def test_end_to_end_temporal_pipeline_is_padding_invariant(self):
        sel = _fitted_selector(C=0.3)
        bb = SelectedEdgeBackbone(sel, n_rois=R)
        agg = BiGRUAggregator(BiGRUConfig(input_dim=bb.embedding_dim, hidden_dim=4))
        model = TemporalPipeline(bb, agg, num_classes=2,
                                 window_policy=WindowPolicyConfig(policy="native"))
        model.eval()
        torch.manual_seed(0)
        fc = _sym_fc(3, 5, seed=1)
        lengths = torch.tensor([5, 2, 3])
        with torch.no_grad():
            base = model(fc, lengths).logits
            corrupted = fc.clone()
            for b, L in enumerate(lengths.tolist()):
                corrupted[b, L:] = 1e6           # garbage in padded windows
            after = model(corrupted, lengths).logits
        assert torch.allclose(base, after, atol=1e-6)
        assert base.shape == (3, 2)

    def test_gradients_flow_to_bigru_and_head_only(self):
        sel = _fitted_selector(C=0.3)
        bb = SelectedEdgeBackbone(sel, n_rois=R)
        agg = BiGRUAggregator(BiGRUConfig(input_dim=bb.embedding_dim, hidden_dim=4))
        model = TemporalPipeline(bb, agg, num_classes=2,
                                 window_policy=WindowPolicyConfig(policy="native"))
        fc = _sym_fc(2, 3); lengths = torch.tensor([3, 2])
        loss = nn.functional.cross_entropy(model(fc, lengths).logits, torch.tensor([0, 1]))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads and all(g is not None for g in grads)


# ============================================================ model factory
class TestModelFactoryLassoEdges:
    def _candidate(self, tmp_path, C=0.3, mode="temporal"):
        split = tmp_path / "s.npz"; _write_split(split)
        cand = _templasso_candidate(split, C=C)
        if mode == "temporal":
            return cand
        # coerce to another mode to test fail-closed paths
        ident = cand.identity
        if mode == "vanilla":
            new_ident = RunIdentity(mode="vanilla", seed=ident.seed,
                                    split_identity=ident.split_identity,
                                    backbone_config=ident.backbone_config)
        else:
            new_ident = RunIdentity(mode="fusion", seed=ident.seed,
                                    split_identity=ident.split_identity,
                                    backbone_config=ident.backbone_config,
                                    temporal_config=ident.temporal_config,
                                    window_policy=ident.window_policy,
                                    fusion_config={"name": "concat"})
        return CandidateRunConfig(identity=new_ident, training=cand.training)

    def test_temporal_requires_selector(self, tmp_path):
        with pytest.raises(ModelFactoryError, match="requires a FROZEN"):
            build_model_from_candidate(self._candidate(tmp_path))

    def test_temporal_builds_with_frozen_selector(self, tmp_path):
        cand = self._candidate(tmp_path, C=0.3)
        sel = _fitted_selector(C=0.3)
        model = build_templasso_model_from_candidate(cand, edge_selector=sel)
        assert isinstance(model, TemporalPipeline)
        assert isinstance(model.backbone, SelectedEdgeBackbone)
        assert model.backbone.embedding_dim == sel.n_selected

    def test_recipe_mismatch_is_rejected(self, tmp_path):
        cand = self._candidate(tmp_path, C=0.3)
        wrong = _fitted_selector(C=0.5)          # different C than candidate
        with pytest.raises(ModelFactoryError, match="does not match"):
            build_templasso_model_from_candidate(cand, edge_selector=wrong)

    def test_unfrozen_selector_is_rejected(self, tmp_path):
        cand = self._candidate(tmp_path, C=0.3)
        x, y = _synthetic_rows()
        unfrozen = LassoEdgeSelector(LassoEdgeSelectorConfig(C=0.3, min_edges=1)).fit(x, y)
        with pytest.raises(ModelFactoryError, match="frozen"):
            build_templasso_model_from_candidate(cand, edge_selector=unfrozen)

    def test_vanilla_and_fusion_reject_lasso_edges(self, tmp_path):
        with pytest.raises(ModelFactoryError, match="temporal"):
            build_model_from_candidate(self._candidate(tmp_path, mode="vanilla"))
        with pytest.raises(ModelFactoryError, match="static branch must be a real FC encoder"):
            build_model_from_candidate(self._candidate(tmp_path, mode="fusion"))

    def test_bqn_candidates_reject_stray_selector(self, tmp_path):
        from unified_bn.training.search_space import (
            BQNSearchSpace, generate_vanilla_bqn_candidates,
        )
        split = tmp_path / "s.npz"; _write_split(split)
        bqn = generate_vanilla_bqn_candidates(
            seed=42, split_identity=frozen_split_identity(load_frozen_split(split)),
            backbone_space=BQNSearchSpace(n_layers=(1,), activations=("elu",),
                                          dropouts=(0.0,), n_rois=R, cluster_num=2),
        )[0]
        with pytest.raises(ModelFactoryError, match="requires"):
            build_templasso_model_from_candidate(bqn, edge_selector=_fitted_selector())
        # and the historical factory's signature is untouched
        import inspect
        assert tuple(inspect.signature(build_model_from_candidate).parameters) == ("run_config",)


# ============================================================ search space
class TestSearchSpace:
    def test_recipe_generation_and_uniqueness(self):
        cfgs = generate_lasso_edge_backbone_configs(
            LassoEdgeSearchSpace(Cs=(0.005, 0.01, 0.03), n_rois=R)
        )
        assert len(cfgs) == 3
        assert all(c["name"] == "lasso_edges" for c in cfgs)
        assert [c["C"] for c in cfgs] == [0.005, 0.01, 0.03]
        expected_keys = {"name"} | set(SelectedEdgeBackboneConfig.__dataclass_fields__)
        assert all(set(c) == expected_keys for c in cfgs)

    def test_candidates_are_temporal_mode_and_fingerprint_stable(self, tmp_path):
        split = tmp_path / "s.npz"; _write_split(split)
        ident = frozen_split_identity(load_frozen_split(split))
        make = lambda: generate_templasso_candidates(
            seed=42, split_identity=ident,
            window_policy=WindowPolicyConfig(policy="native"),
            temporal_space=BiGRUSearchSpace(hidden_dims=(4, 8), num_layers=(1,),
                                            poolings=("final", "mean")),
            selector_space=LassoEdgeSearchSpace(Cs=(0.01, 0.03), n_rois=R),
        )
        a, b = make(), make()
        assert len(a) == 2 * 2 * 2 == 8
        assert all(c.identity.mode == "temporal" for c in a)
        assert [candidate_fingerprint(c) for c in a] == [candidate_fingerprint(c) for c in b]
        assert len({candidate_fingerprint(c) for c in a}) == 8

    def test_space_validation(self):
        with pytest.raises(Exception):
            LassoEdgeSearchSpace(Cs=(0.0,))
        with pytest.raises(Exception):
            LassoEdgeSearchSpace(input_space="nope")


# ================================================================ runtime
class TestRuntimeFitsSelectorOnTrainOnly:
    def _factory(self, tmp_path):
        split = tmp_path / "split.npz"; static = tmp_path / "static.h5"; temporal = tmp_path / "temporal.h5"
        _write_split(split); _write_static_h5(static); _write_temporal_h5(temporal)
        cfg = ABIDERuntimeConfig(split_path=split, static_h5=static, temporal_h5=temporal,
                                 device="cpu", expected_n_rois=R, pin_memory=False)
        return split, ABIDERuntimeFactory(cfg)

    def test_builds_templasso_runtime_and_records_provenance(self, tmp_path):
        split, factory = self._factory(tmp_path)
        cand = _templasso_candidate(split, C=1.0)
        rt = factory(cand, 0, tmp_path / "run")
        assert isinstance(rt.model, TemporalPipeline)
        assert isinstance(rt.model.backbone, SelectedEdgeBackbone)
        prov = rt.user_provenance["edge_selector"]
        assert prov["kind"] == "lasso_edges"
        assert prov["fitted_on"] == "train"
        assert prov["n_train_subjects"] == 4
        # train windows: s2(2)+s1(3)+s5(3)+s6(5) = 13
        assert prov["n_train_windows"] == 13
        assert prov["n_edges"] == E
        assert 1 <= prov["n_selected"] <= E
        assert prov["edge_set_fingerprint"].startswith("sha256:")
        # loaders exist for train/val only, and batches are TemporalBatch
        batch = next(iter(rt.train_batches))
        assert isinstance(batch, TemporalBatch)
        assert not hasattr(rt, "test_batches")

    def test_selector_never_sees_validation_rows(self, tmp_path, monkeypatch):
        """Spy on fit(): the pooled rows must equal exactly the TRAIN windows."""
        split, factory = self._factory(tmp_path)
        cand = _templasso_candidate(split, C=1.0)
        seen = {}
        orig_fit = LassoEdgeSelector.fit

        def spy(self, x, y):
            seen["x"] = x.clone(); seen["y"] = y.clone()
            return orig_fit(self, x, y)

        monkeypatch.setattr(LassoEdgeSelector, "fit", spy)
        factory(cand, 0, tmp_path / "run")

        ds = TemporalABIDEDataset(tmp_path / "temporal.h5",
                                  subject_ids=("s2", "s1", "s5", "s6"), expected_n_rois=R)
        x_train, y_train = pooled_edge_rows_from_dataset(ds, n_rois=R, input_space="fisher_z")
        ds.close()
        assert torch.equal(seen["x"], x_train)
        assert torch.equal(seen["y"], y_train)
        assert seen["x"].shape[0] == 13           # not 14 (val s3 has 4 windows) nor 19

    def test_same_candidate_gives_identical_edge_set(self, tmp_path):
        split, factory = self._factory(tmp_path)
        cand = _templasso_candidate(split, C=1.0)
        a = factory(cand, 0, tmp_path / "a").user_provenance["edge_selector"]
        b = factory(cand, 0, tmp_path / "b").user_provenance["edge_selector"]
        assert a["edge_set_fingerprint"] == b["edge_set_fingerprint"]

    def test_bqn_temporal_path_is_untouched(self, tmp_path):
        """Existing BQN temporal candidates must not gain a selector block."""
        from unified_bn.training.search_space import (
            BQNSearchSpace, generate_temporal_bqn_bigru_candidates,
        )
        split, factory = self._factory(tmp_path)
        cand = generate_temporal_bqn_bigru_candidates(
            seed=42, split_identity=frozen_split_identity(load_frozen_split(split)),
            window_policy=WindowPolicyConfig(policy="native"),
            backbone_space=BQNSearchSpace(n_layers=(1,), activations=("elu",),
                                          dropouts=(0.0,), n_rois=R, cluster_num=2),
            temporal_space=BiGRUSearchSpace(hidden_dims=(4,), num_layers=(1,), poolings=("final",)),
        )[0]
        rt = factory(cand, 0, tmp_path / "run")
        assert "edge_selector" not in rt.user_provenance
