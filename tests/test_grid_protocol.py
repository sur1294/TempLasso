from dataclasses import replace
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.grid_protocol import collect_grid, freeze_grid, grid_action_lock
from unified_bn.training.config import ValidationSelectionRule
from unified_bn.training.results import CandidateTrainingResult, write_candidate_result
from test_dataset_selection import cli_candidate, metrics


@pytest.fixture
def grid(tmp_path):
    args = SimpleNamespace(dataset="PPMI", row="x", backbone="gcn", num_classes=4)
    base = cli_candidate("PPMI", epochs=3)
    candidates = {}
    for seed in (42, 1042):
        first = replace(base, identity=replace(base.identity, seed=seed))
        second = replace(first, training=replace(first.training,
                         optimizer=replace(first.training.optimizer, lr=3e-4)))
        candidates[seed] = (first, second)
    def task_dir(seed, index, candidate):
        return tmp_path / str(seed) / str(index)
    paths = {}
    for seed, cands in candidates.items():
        for index, candidate in enumerate(cands):
            directory = task_dir(seed, index, candidate) / "candidate_0"
            directory.mkdir(parents=True)
            checkpoint = directory / "best.pt"; checkpoint.write_bytes(b"synthetic fixture")
            path = directory / "result.json"
            # Better loss deliberately belongs to the worse-AUC candidate.
            m = metrics(.2 if index == 0 else .8, .6 if index == 0 else .9)
            result = CandidateTrainingResult(run_config=candidate.to_dict(), requested_epochs=3,
                completed_epochs=3, stopped_early=False, best_epoch=2, selection_metric="val_auc",
                selection_mode="max", best_selection_value=m.roc_auc, best_validation_metrics=m,
                restored_validation_metrics=m, best_checkpoint_path=str(checkpoint),
                last_checkpoint_path=str(directory/"last.pt"), history_path=str(directory/"history.json"),
                result_path=str(path), loss_function="CrossEntropyLoss", runtime_provenance={},
                user_provenance={"dataset":"PPMI"})
            write_candidate_result(result)
            paths[seed, index] = path
    return args, candidates, task_dir, tmp_path, paths


def mutate(path, fn):
    p = json.loads(path.read_text()); fn(p); path.write_text(json.dumps(p))


def test_grid_selects_max_auc_then_allows_only_frozen_test_record(grid):
    args, candidates, task_dir, root, paths = grid
    with grid_action_lock(root):
        winners = freeze_grid(args, candidates, task_dir, root)
    assert [w["selected_index"] for w in winners.values()] == [1, 1]
    mutate(paths[42, 1], lambda p:p.update(status="final_test_complete", test_metrics=metrics(3., .1).to_dict()))
    assert freeze_grid(args, candidates, task_dir, root) == winners  # poor test AUC cannot reselect
    collect_grid(args, candidates, task_dir, root)
    with (root/"csv/all_hyperparameters.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and {r["selection_mode"] for r in rows} == {"max"}
    assert sum(bool(r["test_auc"]) for r in rows) == 1
    with (root/"csv/seed42.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "epochs", "wrong_rule", "nonfinite",
                                       "wrong_config", "wrong_metric", "empty_checkpoint", "test_before_freeze"])
def test_no_freeze_when_any_seed_has_invalid_candidate(grid, corruption):
    args, candidates, task_dir, root, paths = grid
    path = paths[1042, 1]  # the last seed must be checked BEFORE testing the first
    if corruption == "missing": path.unlink()
    elif corruption == "duplicate":
        other = path.parent.parent / "candidate_duplicate"; other.mkdir()
        (other/"result.json").write_text(path.read_text())
    elif corruption == "epochs": mutate(path, lambda p:p.update(completed_epochs=2))
    elif corruption == "wrong_rule": mutate(path, lambda p:p.update(selection_metric="val_loss", selection_mode="min"))
    elif corruption == "nonfinite": mutate(path, lambda p:p.update(best_selection_value=float("nan")))
    elif corruption == "wrong_config": mutate(path, lambda p:p["run_config"]["identity"].update(seed=999))
    elif corruption == "wrong_metric": mutate(path, lambda p:p.update(best_selection_value=.4))
    elif corruption == "empty_checkpoint": (path.parent/"best.pt").write_bytes(b"")
    elif corruption == "test_before_freeze":
        mutate(path, lambda p:p.update(status="final_test_complete", test_metrics=metrics(.1, .99).to_dict()))
    with pytest.raises(ValueError): freeze_grid(args, candidates, task_dir, root)
    assert not (root/"frozen_validation_winners.json").exists()


@pytest.mark.parametrize("corruption", ["validation", "checkpoint", "nonwinner_test"])
def test_frozen_artifacts_cannot_be_replaced_using_test_results(grid, corruption):
    args, candidates, task_dir, root, paths = grid
    freeze_grid(args, candidates, task_dir, root)
    frozen = (root/"frozen_validation_winners.json").read_bytes()
    if corruption == "validation":
        mutate(paths[42, 0], lambda p:p.update(best_epoch=1))
    elif corruption == "checkpoint": (paths[42, 1].parent/"best.pt").write_bytes(b"replaced checkpoint")
    else: mutate(paths[42, 0], lambda p:p.update(status="final_test_complete", test_metrics=metrics(.01, 1.).to_dict()))
    with pytest.raises(ValueError): freeze_grid(args, candidates, task_dir, root)
    assert (root/"frozen_validation_winners.json").read_bytes() == frozen


def test_exact_ties_keep_lowest_index(grid):
    args, candidates, task_dir, root, paths = grid
    for seed in candidates:
        def tie(p):
            p["best_selection_value"] = .9
            for name in ("best_validation_metrics", "restored_validation_metrics"):
                p[name]["roc_auc"] = .9
        mutate(paths[seed, 0], tie)
    winners = freeze_grid(args, candidates, task_dir, root)
    assert all(w["selected_index"] == 0 for w in winners.values())
