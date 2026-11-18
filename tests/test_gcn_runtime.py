"""GCN on synthetic versions of the project's shared HDF5 schemas."""
import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from unified_bn.data import (StaticABIDEDataset, TemporalABIDEDataset, FusionABIDEDataset,
    collate_static, collate_temporal, collate_fusion, load_frozen_split)
from unified_bn.training.abide_runtime import ABIDERuntimeConfig, ABIDERuntimeFactory, frozen_split_identity
from unified_bn.training.checkpoint import read_training_checkpoint
from unified_bn.training.config import TrainingConfig
from unified_bn.training.dataset_protocol import dataset_selection_rule
from unified_bn.training.orchestrator import run_validation_search
from unified_bn.training.search_space import (GCNSearchSpace, BiGRUSearchSpace, LassoEdgeSearchSpace,
    generate_vanilla_candidates, generate_temporal_candidates, generate_fusion_candidates,
    generate_fusion_templasso_candidates)
from unified_bn.training.trainer import evaluate_final_test_and_record
from unified_bn.temporal.windowing import WindowPolicyConfig


@pytest.fixture
def artifacts(tmp_path):
    def create(dataset):
        classes = 4 if dataset == "PPMI" else 2
        ids = np.asarray([f"s{i:03d}" for i in range(8 * classes)])
        labels = np.arange(len(ids)) % classes
        rng = np.random.default_rng(32)
        r, c = np.triu_indices(8, 1)
        matrices = np.broadcast_to(np.eye(8), (len(ids), 8, 8)).copy().astype(np.float32)
        for i, label in enumerate(labels):
            edges = rng.normal(0, .1, len(r))
            edges[label] += .5
            matrices[i, r, c] = matrices[i, c, r] = edges
        static, temporal = tmp_path / "static.h5", tmp_path / "temporal.h5"
        with h5py.File(static, "w") as f:
            f.create_dataset("subject_ids", data=ids.astype("S"))
            f.create_dataset("labels", data=labels)
            f.create_dataset("fc", data=matrices)
        lengths = 2 + np.arange(len(ids)) % 3
        offsets = np.r_[0, np.cumsum(lengths)]
        values = []
        for i, k in enumerate(lengths):
            row = rng.normal(0, .2, (k, len(r)))
            row[:, labels[i]] += 1
            values.extend(row)
        with h5py.File(temporal, "w") as f:
            for key, data in dict(subject_ids=ids.astype("S"), labels=labels, K=lengths,
                offsets=offsets, fc_z_upper=np.asarray(values, dtype=np.float32), triu_row=r, triu_col=c).items():
                f.create_dataset(key, data=data)
        split = tmp_path / f"{dataset.lower()}_split_seed42.npz"
        np.savez(split, seed=np.asarray(42), train_ids=ids[:4*classes],
                 val_ids=ids[4*classes:6*classes], test_ids=ids[6*classes:])
        factory = ABIDERuntimeFactory(ABIDERuntimeConfig(split, static, temporal, "cpu",
            expected_n_rois=8, num_classes=classes, dataset_name=dataset, pin_memory=False))
        return static, temporal, split, factory, classes
    return create


def candidate(mode, factory, classes, *, epochs=1):
    common = dict(seed=42, split_identity=factory.split_identity, num_classes=classes,
        backbone_space=GCNSearchSpace(n_rois=8, n_layers=(1,), hidden_dims=(12,)),
        training_configs=(TrainingConfig(epochs=epochs, batch_size=4,
            checkpoint_selection=dataset_selection_rule(factory.config.dataset_name, classes)),))
    if mode == "vanilla": return generate_vanilla_candidates(**common)[0]
    common.update(window_policy=WindowPolicyConfig(policy="native"),
        temporal_space=BiGRUSearchSpace(hidden_dims=(8,), num_layers=(1,), poolings=("final",)))
    if mode == "temporal": return generate_temporal_candidates(**common)[0]
    if mode == "fusion": return generate_fusion_candidates(**common)[0]
    return generate_fusion_templasso_candidates(**common,
        selector_space=LassoEdgeSearchSpace(Cs=(1.,), n_rois=8, min_edges=1))[0]


@pytest.mark.parametrize("dataset", ["ABIDE", "ADNI", "PPMI"])
@pytest.mark.parametrize("mode", ["vanilla", "temporal", "fusion", "fusion_templasso"])
def test_train_select_restore_and_synthetic_final_test(dataset, mode, artifacts, tmp_path, monkeypatch):
    import unified_bn.training.abide_runtime as runtime
    static, temporal, split_path, factory, classes = artifacts(dataset)
    split = load_frozen_split(split_path)
    allowed = set(split.train_ids + split.val_ids)
    original_pool = runtime.pooled_edge_rows_from_dataset
    fitted_subjects = []
    def train_only_pool(dataset, **kwargs):
        assert tuple(dataset.subject_ids) == tuple(split.train_ids)
        fitted_subjects.append(tuple(dataset.subject_ids))
        return original_pool(dataset, **kwargs)
    monkeypatch.setattr(runtime, "pooled_edge_rows_from_dataset", train_only_pool)
    for cls in (StaticABIDEDataset, TemporalABIDEDataset):
        original = cls.__getitem__
        def guard(self, index, original=original):
            assert self.subject_ids[index] in allowed, "Search must not read TEST feature samples"
            return original(self, index)
        monkeypatch.setattr(cls, "__getitem__", guard)
    cand = candidate(mode, factory, classes)
    search = tmp_path / "search"
    selection = run_validation_search((cand,), search_dir=search, runtime_factory=factory)
    assert run_validation_search((cand,), search_dir=search, runtime_factory=factory) == selection
    result_path = Path(selection.selected_result_path)
    payload = json.loads(result_path.read_text())
    assert payload["test_metrics"] is None
    assert payload["run_config"]["identity"]["backbone_config"]["name"] == "gcn"
    checkpoint = read_training_checkpoint(selection.selected_best_checkpoint_path, expected_run_config=cand)
    rebuilt = factory(cand, 0, result_path.parent)
    rebuilt.model.load_state_dict(checkpoint.model_state, strict=True)
    if mode == "fusion_templasso":
        assert fitted_subjects
        assert rebuilt.model.static_backbone.__class__.__name__ == "GCNBackbone"
        assert rebuilt.model.temporal_backbone.__class__.__name__ == "SelectedEdgeBackbone"
    # Only now permit the synthetic held-out partition, using the existing API.
    allowed.update(split.test_ids)
    common = dict(subject_ids=split.test_ids, expected_n_rois=8, num_classes=classes)
    if mode == "vanilla": ds, collate = StaticABIDEDataset(static, **common), collate_static
    elif mode == "temporal": ds, collate = TemporalABIDEDataset(temporal, **common), collate_temporal
    else: ds, collate = FusionABIDEDataset(static, temporal, **common), collate_fusion
    try:
        loader = DataLoader(ds, batch_size=4, collate_fn=collate)
        metrics = evaluate_final_test_and_record(rebuilt.model, loader, result_path=result_path, device="cpu")
        assert metrics.n_samples == len(split.test_ids) and np.isfinite(metrics.roc_auc)
        with pytest.raises(RuntimeError):
            evaluate_final_test_and_record(rebuilt.model, loader, result_path=result_path, device="cpu")
    finally:
        ds.close()


def test_temporal_resume_matches_uninterrupted_training(artifacts, tmp_path, monkeypatch):
    import unified_bn.training.trainer as trainer
    _, _, _, factory, classes = artifacts("PPMI")
    cand = candidate("temporal", factory, classes, epochs=2)
    full = run_validation_search((cand,), search_dir=tmp_path / "full", runtime_factory=factory)
    original = trainer.run_binary_epoch
    count = []
    def interrupt(*args, **kwargs):
        count.append(1)
        if len(count) == 3: raise RuntimeError("interrupted after saved epoch")
        return original(*args, **kwargs)
    monkeypatch.setattr(trainer, "run_binary_epoch", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_validation_search((cand,), search_dir=tmp_path / "resume", runtime_factory=factory)
    monkeypatch.setattr(trainer, "run_binary_epoch", original)
    resumed = run_validation_search((cand,), search_dir=tmp_path / "resume", runtime_factory=factory)
    assert full.selection_value == resumed.selection_value
    a = read_training_checkpoint(full.selected_best_checkpoint_path, expected_run_config=cand)
    b = read_training_checkpoint(resumed.selected_best_checkpoint_path, expected_run_config=cand)
    assert set(a.model_state) == set(b.model_state)
    assert all(torch.equal(a.model_state[k], b.model_state[k]) for k in a.model_state)


@pytest.mark.parametrize("dataset", ["ABIDE", "ADNI", "PPMI"])
def test_existing_late_fusion_script_accepts_gcn_checkpoints(dataset, artifacts, tmp_path, monkeypatch):
    import experiments.late_fusion_from_checkpoints as script
    static, temporal, split, factory, classes = artifacts(dataset)
    for mode in ("vanilla", "temporal"):
        run_validation_search((candidate(mode, factory, classes),),
            search_dir=tmp_path / mode, runtime_factory=factory)
    out = tmp_path / "fused.csv"
    args = SimpleNamespace(repo=Path(__file__).resolve().parents[1], dataset=dataset,
        split_prefix=dataset.lower(), split_dir=split.parent, static_h5=static,
        temporal_h5=temporal, num_classes=classes, x_runs=str(tmp_path / "vanilla"),
        t_runs=str(tmp_path / "temporal"), label="gcn_test", out=out,
        seeds=[42], device="cpu", expected_n_rois=8)
    monkeypatch.setattr(script, "parse_args", lambda: args)
    assert script.main() == 0
    with out.open() as stream: rows = list(csv.DictReader(stream))
    assert {row["fusion"] for row in rows} >= {"X", "T", "avg", "alpha*", "logit-avg"}
    assert all(np.isfinite(float(row["test_auc"])) for row in rows)


@pytest.mark.parametrize("dataset,classes", [("ABIDE", 2), ("ADNI", 2), ("PPMI", 4)])
@pytest.mark.parametrize("row,expected", [("x", 8), ("tempx", 24)])
def test_grid_cli_lists_gcn_without_reading_data(dataset, classes, row, expected, tmp_path, monkeypatch, capsys):
    import experiments.run_grid as script
    seeds = [42, 1042, 2042, 3042, 4042]
    for seed in seeds:
        np.savez(tmp_path / f"{dataset.lower()}_split_seed{seed}.npz", seed=np.asarray(seed),
                 train_ids=np.asarray(["a", "b"]), val_ids=np.asarray(["c"]), test_ids=np.asarray(["d"]))
    args = SimpleNamespace(repo=Path(__file__).resolve().parents[1], command="list", row=row,
        backbone="gcn", dataset=dataset, split_prefix=dataset.lower(), split_dir=tmp_path,
        static_h5=tmp_path / "must_not_open.h5", temporal_h5=tmp_path / "also_absent.h5",
        grid_dir=tmp_path / "grid", epochs=60, batch_size=16, device="cpu",
        expected_n_rois=100, num_classes=classes, seeds=seeds, task_index=None, lasso_cs=None, out=None)
    monkeypatch.setattr(script, "parse_args", lambda: args)
    assert script.main() == 0
    assert int(capsys.readouterr().out.strip()) == 5 * expected
    assert not args.grid_dir.exists()


def test_ppmi_late_fusion_rejects_legacy_loss_checkpoint_before_data_access(artifacts, tmp_path, monkeypatch):
    import experiments.late_fusion_from_checkpoints as script
    import unified_bn.training.abide_runtime as runtime_module
    from unified_bn.training.config import ValidationSelectionRule
    static, temporal, split, factory, classes = artifacts("PPMI")
    legacy = candidate("vanilla", factory, classes)
    legacy = replace(legacy, training=replace(legacy.training, checkpoint_selection=ValidationSelectionRule()))
    run_validation_search((legacy,), search_dir=tmp_path/"legacy", runtime_factory=factory)
    args = SimpleNamespace(repo=Path(__file__).resolve().parents[1], dataset="PPMI",
        split_prefix="ppmi", split_dir=split.parent, static_h5=static, temporal_h5=temporal,
        num_classes=4, x_runs=str(tmp_path/"legacy"), t_runs=str(tmp_path/"must_not_read"),
        label="legacy", out=tmp_path/"must_not_write.csv", seeds=[42], device="cpu", expected_n_rois=8)
    monkeypatch.setattr(script, "parse_args", lambda: args)
    def reject(*args, **kwargs):
        raise AssertionError("Legacy run reached data/model reconstruction")
    monkeypatch.setattr(runtime_module, "ABIDERuntimeFactory", reject)
    with pytest.raises(ValueError, match="macro-AUC-selected"):
        script.main()
    assert not args.out.exists()
