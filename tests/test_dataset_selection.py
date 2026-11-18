"""Atlas dataset rule across CLI, metrics, actual checkpointing and grids."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from unified_bn.training.abide_cli import build_parser, build_single_candidate
from unified_bn.training.config import TrainingConfig, ValidationSelectionRule
from unified_bn.training.dataset_protocol import dataset_selection_rule
from unified_bn.training.metrics import BinaryClassificationMetrics, BinaryMetricAccumulator, UndefinedBinaryMetricError
from unified_bn.training.search_space import candidate_fingerprint


CASES = [(backbone, mode) for backbone in ("bqn", "alter", "brainnetcnn", "bnt", "gcn")
         for mode in ("vanilla", "temporal", "fusion", "fusion_templasso")] + [("bqn", "templasso")]


def cli_candidate(dataset, backbone="gcn", mode="vanilla", epochs=3):
    args = ["--mode", mode, "--seed", "42", "--split-path", "split.npz", "--static-h5", "static.h5",
            "--temporal-h5", "temporal.h5", "--search-dir", "search", "--device", "cpu",
            "--dataset-name", dataset, "--num-classes", "4" if dataset.upper() == "PPMI" else "2",
            "--epochs", str(epochs)]
    if mode != "templasso":
        args += ["--backbone", backbone]
        if backbone == "bqn":
            args += ["--bqn-layers", "1", "--bqn-activation", "elu", "--bqn-dropout", "0.1"]
    if mode != "vanilla":
        args += ["--bigru-hidden-dim", "32", "--bigru-layers", "1", "--bigru-pooling", "final"]
    if mode in ("templasso", "fusion_templasso"):
        args += ["--lasso-C", "0.03"]
    return build_single_candidate(build_parser().parse_args(args), split_identity="fixture")


@pytest.mark.parametrize("dataset", ["ABIDE", "ADNI", "PPMI"])
@pytest.mark.parametrize("backbone,mode", CASES)
def test_every_cli_variant_uses_dataset_rule(dataset, backbone, mode):
    candidate = cli_candidate(dataset, backbone, mode)
    expected = ValidationSelectionRule("val_auc", "max") if dataset == "PPMI" else ValidationSelectionRule()
    assert candidate.training.checkpoint_selection == expected
    assert candidate.training.early_stopping.enabled is False
    if dataset != "PPMI":
        assert candidate.training == TrainingConfig(epochs=3)
    else:
        legacy = replace(candidate, training=replace(candidate.training, checkpoint_selection=ValidationSelectionRule()))
        assert candidate_fingerprint(legacy) != candidate_fingerprint(candidate)


def test_ppmi_case_and_class_count():
    assert dataset_selection_rule(" ppmi ", 4) == ValidationSelectionRule("val_auc", "max")
    for classes in (2, 3, 5):
        with pytest.raises(ValueError, match="num-classes 4"):
            dataset_selection_rule("PPMI", classes)
    assert dataset_selection_rule("other", 4) == ValidationSelectionRule()


def test_macro_auc_is_unweighted_ovr_over_entire_subject_set():
    labels = torch.tensor([0]*2 + [1]*9 + [2]*3 + [3]*6)
    logits = torch.tensor(np.random.default_rng(7).normal(size=(20, 4)), dtype=torch.float64)
    accumulator = BinaryMetricAccumulator()
    for chunk in range(0, 20, 3):  # individual batches may lack classes
        accumulator.update(logits=logits[chunk:chunk+3], labels=labels[chunk:chunk+3], loss_sum=1.)
    probabilities = torch.softmax(logits, dim=1).numpy()
    expected = roc_auc_score(labels.numpy(), probabilities, multi_class="ovr", average="macro")
    weighted = roc_auc_score(labels.numpy(), probabilities, multi_class="ovr", average="weighted")
    assert accumulator.compute().roc_auc == pytest.approx(expected)
    assert abs(expected - weighted) > .001
    missing = BinaryMetricAccumulator()
    missing.update(logits=logits[:11], labels=labels[:11], loss_sum=1.)
    with pytest.raises(UndefinedBinaryMetricError):
        missing.compute()  # no silent switch to loss or fewer classes


def metrics(loss, auc):
    return BinaryClassificationMetrics(loss, auc, .5, .5, .5, 8)


class MarkerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(0.))


@pytest.mark.parametrize("dataset,best_epoch", [("ABIDE", 3), ("ADNI", 3), ("PPMI", 2)])
def test_epoch_selection_restore_and_resume(dataset, best_epoch, tmp_path, monkeypatch):
    import unified_bn.training.trainer as trainer
    from unified_bn.training.checkpoint import read_training_checkpoint, CheckpointCompatibilityError
    candidate = cli_candidate(dataset, epochs=4)
    losses, aucs = [.4, .5, .1, .6], [.6, .9, .7, .9]
    interrupted = [False]
    def epoch(model, batches, *, training, **kwargs):
        if training:
            if interrupted[0] and model.marker.item() == 2:
                raise RuntimeError("simulated interruption")
            with torch.no_grad(): model.marker.add_(1)
        index = int(model.marker.item()) - 1
        return metrics(losses[index], aucs[index])
    monkeypatch.setattr(trainer, "run_binary_epoch", epoch)
    full = trainer.train_candidate(MarkerModel(), [], [], run_config=candidate,
                                   run_dir=tmp_path / "full", device="cpu")
    assert full.best_epoch == best_epoch
    state = read_training_checkpoint(full.best_checkpoint_path, expected_run_config=candidate)
    assert state.model_state["marker"].item() == best_epoch
    interrupted[0] = True
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.train_candidate(MarkerModel(), [], [], run_config=candidate,
                                run_dir=tmp_path / "resumed", device="cpu")
    interrupted[0] = False
    resumed = trainer.train_candidate(MarkerModel(), [], [], run_config=candidate,
        run_dir=tmp_path / "resumed", device="cpu", resume_from=tmp_path / "resumed/last.pt")
    assert resumed.best_epoch == full.best_epoch
    assert resumed.best_selection_value == full.best_selection_value
    if dataset == "PPMI":
        legacy = replace(candidate, training=replace(candidate.training, checkpoint_selection=ValidationSelectionRule()))
        with pytest.raises(CheckpointCompatibilityError, match="configuration|config"):
            read_training_checkpoint(full.best_checkpoint_path, expected_run_config=legacy)


@pytest.mark.parametrize("row,count", [("x", 8), ("tempx", 24), ("templasso", 18),
                                      ("fusion", 24), ("fusion_templasso", 72)])
@pytest.mark.parametrize("dataset", ["ABIDE", "ADNI", "PPMI"])
def test_grid_dispatch_uses_same_rule_for_all_rows(row, count, dataset, tmp_path, monkeypatch):
    from experiments import run_grid
    import unified_bn.training.orchestrator as orchestrator
    classes = 4 if dataset == "PPMI" else 2
    np.savez(tmp_path / "sample_split_seed42.npz", seed=42, train_ids=["a", "b"], val_ids=["c"], test_ids=["d"])
    args = SimpleNamespace(command="list", repo=tmp_path, row=row, backbone="gcn", dataset=dataset,
        num_classes=classes, split_prefix="sample", split_dir=tmp_path, static_h5=tmp_path/"absent.h5",
        temporal_h5=tmp_path/"also-absent.h5", grid_dir=tmp_path/"grid", epochs=200, batch_size=16,
        device="cpu", expected_n_rois=100, seeds=[42], task_index=count-1, lasso_cs=None, out=None)
    monkeypatch.setattr(run_grid, "parse_args", lambda: args)
    from unified_bn.training import search_space
    seen = []
    names = {"x":"generate_vanilla_candidates", "tempx":"generate_temporal_candidates",
             "templasso":"generate_templasso_candidates", "fusion":"generate_fusion_candidates",
             "fusion_templasso":"generate_fusion_templasso_candidates"}
    original = getattr(search_space, names[row])
    def capture(**kwargs):
        candidates = original(**kwargs); seen.extend(candidates); return candidates
    monkeypatch.setattr(search_space, names[row], capture)
    assert run_grid.main() == 0
    assert len(seen) == count
    assert all(c.training.checkpoint_selection == dataset_selection_rule(dataset, classes) for c in seen)
    assert not args.grid_dir.exists()


def test_ppmi_late_fusion_weight_uses_macro_auc():
    from experiments.late_fusion_from_checkpoints import select_multiclass_alpha
    y = np.tile(np.arange(4), 8)
    rng = np.random.default_rng(42)
    x, t = (rng.dirichlet(np.ones(4), len(y)) for _ in range(2))
    alphas = np.linspace(0, 1, 21)
    expected = alphas[np.argmax([roc_auc_score(y, a*x+(1-a)*t, average="macro", multi_class="ovr") for a in alphas])]
    assert select_multiclass_alpha(y, x, t, dataset="PPMI", num_classes=4) == expected
    assert select_multiclass_alpha(y, x, x, dataset="PPMI", num_classes=4) == 0.  # tie
