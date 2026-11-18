"""Real tiny CPU runs through the updated generic grid and held-out path."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_gcn_runtime import artifacts


@pytest.mark.parametrize("row", ["x", "tempx", "templasso", "fusion", "fusion_templasso"])
def test_ppmi_grid_train_freeze_test_and_cached_rerun(row, artifacts, tmp_path, monkeypatch):
    from experiments import run_grid
    from unified_bn.data import StaticABIDEDataset, TemporalABIDEDataset, load_frozen_split
    import unified_bn.training.search_space as spaces
    import unified_bn.training.trainer as trainer
    static, temporal, split_path, factory, classes = artifacts("PPMI")
    args = SimpleNamespace(repo=Path(__file__).resolve().parents[1], command="train", row=row,
        backbone="gcn", dataset="PPMI", split_prefix="ppmi", split_dir=tmp_path,
        static_h5=static, temporal_h5=temporal, grid_dir=tmp_path/"grid", epochs=2,
        batch_size=4, device="cpu", expected_n_rois=8, num_classes=4, seeds=[42],
        task_index=0, lasso_cs=None, out=None)
    monkeypatch.setattr(run_grid, "parse_args", lambda: args)
    name = {"x":"generate_vanilla_candidates", "tempx":"generate_temporal_candidates",
            "templasso":"generate_templasso_candidates", "fusion":"generate_fusion_candidates",
            "fusion_templasso":"generate_fusion_templasso_candidates"}[row]
    generate = getattr(spaces, name)
    def small_grid(**kwargs):
        if "backbone_space" in kwargs:
            kwargs["backbone_space"] = spaces.GCNSearchSpace(n_rois=8, n_layers=(1,), hidden_dims=(6,))
        if "temporal_space" in kwargs:
            kwargs["temporal_space"] = spaces.BiGRUSearchSpace(hidden_dims=(4,), num_layers=(1,), poolings=("final",))
        if "selector_space" in kwargs:
            kwargs["selector_space"] = spaces.LassoEdgeSearchSpace(n_rois=8, Cs=(1.,), min_edges=1)
        result = generate(**kwargs)
        assert len(result) == 2
        return result
    monkeypatch.setattr(spaces, name, small_grid)
    split = load_frozen_split(split_path)
    allowed = set(split.train_ids + split.val_ids)
    for cls in (StaticABIDEDataset, TemporalABIDEDataset):
        original = cls.__getitem__
        def guard(self, index, original=original):
            assert self.subject_ids[index] in allowed, "TEST subject read before frozen selection"
            return original(self, index)
        monkeypatch.setattr(cls, "__getitem__", guard)
    assert run_grid.main() == 0
    # Candidate 1 is missing: selecting must not touch any test data.
    args.command = "select"
    with pytest.raises(ValueError, match="exactly one"):
        run_grid.main()
    assert not list(args.grid_dir.rglob("frozen_validation_winners.json"))
    args.command, args.task_index = "train", 1
    assert run_grid.main() == 0
    original_test = trainer.evaluate_final_test_and_record
    calls = []
    def checked_test(*positional, **kwargs):
        frozen = list(args.grid_dir.rglob("frozen_validation_winners.json"))
        assert len(frozen) == 1
        snapshot = json.loads(frozen[0].read_text())
        assert snapshot["winners"]["42"]["selection_mode"] == "max"
        assert len(snapshot["candidates"]) == 2
        allowed.update(split.test_ids)
        calls.append(1)
        return original_test(*positional, **kwargs)
    monkeypatch.setattr(trainer, "evaluate_final_test_and_record", checked_test)
    args.command = "select"
    assert run_grid.main() == 0
    assert len(calls) == 1
    assert run_grid.main() == 0
    assert len(calls) == 1  # reuse recorded test metrics; never evaluate again
    csv_path = next(args.grid_dir.rglob("all_hyperparameters.csv"))
    with csv_path.open() as stream: rows = list(csv.DictReader(stream))
    tested = [r for r in rows if r["test_auc"]]
    assert len(tested) == 1
    assert float(tested[0]["selection_value"]) == max(float(r["val_auc"]) for r in rows)
    assert all(r["selection_metric"] == "val_auc" for r in rows)
    assert all(r["test_auc"] == "" for r in rows if r is not tested[0])
    args.command = "train"
    with pytest.raises(SystemExit, match="frozen"):
        run_grid.main()
