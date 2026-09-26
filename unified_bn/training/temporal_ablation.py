"""Frozen-input runtime for the standalone TempLasso aggregation ablation.

This module is opt-in. No production factory, selector, trainer, or backbone
definition is changed. All four variants share one serialized TRAIN selector.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from unified_bn.backbones.selected_edges import SelectedEdgeBackbone
from unified_bn.data import TemporalABIDEDataset, load_frozen_split
from unified_bn.pipelines.temporal import TemporalPipeline
from unified_bn.selectors.lasso import (LassoEdgeSelector, LassoEdgeSelectorConfig,
                                        pooled_edge_rows_from_dataset)
from unified_bn.temporal.ablation import VARIANTS, TemporalAblationCollate, make_aggregator
from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.abide_runtime import frozen_split_identity, read_stored_split_seed
from unified_bn.training.config import (CandidateRunConfig, RunIdentity, TrainingConfig,
    OptimizerConfig, LRScheduleConfig, DeterminismConfig, EarlyStoppingConfig,
    ValidationSelectionRule)
from unified_bn.training.dataset_protocol import dataset_selection_rule
from unified_bn.training.orchestrator import CandidateRuntime


SCHEMA = "templasso-temporal-ablation-v1"
SHUFFLE_POLICY = "fixed_per_subject_sha256_rank_v1"
# These bounds validate a TRAIN refit against the historical float32 buffers.
# They never change a mask, select a model, or loosen frozen-model checks.
REFERENCE_SCALING_ATOL = 1e-7
REFERENCE_SCALING_RTOL = 1e-6
REFERENCE_NORMALIZED_DRIFT = 1e-5


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software_versions():
    return {name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "scipy", "scikit-learn", "h5py")}


def source_hash():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "unified_bn").rglob("*.py"))
    paths += [root / "experiments" / "templasso_ablation.py",
              root / "experiments" / "grid_protocol.py"]
    return content_hash({p.relative_to(root).as_posix(): file_hash(p) for p in paths})


def training_from_dict(payload):
    p = dict(payload)
    early = dict(p["early_stopping"])
    early["rule"] = ValidationSelectionRule(**early["rule"])
    p["early_stopping"] = early
    for name, cls in (("optimizer", OptimizerConfig), ("lr_schedule", LRScheduleConfig),
                      ("checkpoint_selection", ValidationSelectionRule),
                      ("determinism", DeterminismConfig), ("early_stopping", EarlyStoppingConfig)):
        p[name] = cls(**p[name])
    return TrainingConfig(**p)


def candidate_from_dict(payload):
    identity = dict(payload["identity"])
    identity["window_policy"] = WindowPolicyConfig(**identity["window_policy"])
    return CandidateRunConfig(RunIdentity(**identity), training_from_dict(payload["training"]))


def selector_from_snapshot(record, n_rois):
    state = record["selector_state"]
    if content_hash(state) != record["selector_sha256"]:
        raise ValueError("Frozen selector content changed")
    n_edges = n_rois * (n_rois - 1) // 2
    indices = np.asarray(state["selected_indices"])
    if (state["state"] != "FROZEN" or state["n_edges"] != n_edges or indices.ndim != 1
            or indices.size < state["config"]["min_edges"] or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= n_edges)
            or np.any(np.diff(indices) <= 0)):
        raise ValueError("Invalid frozen edge support")
    for key in ("train_mean", "train_std"):
        values = np.asarray(state[key], dtype=float)
        if values.shape != (n_edges,) or not np.isfinite(values).all():
            raise ValueError("Invalid frozen scaling statistics")
        if key == "train_std" and np.any(values <= 0):
            raise ValueError("Nonpositive frozen standard deviation")
    return LassoEdgeSelector.from_state_dict(state)


def validate_anchor(candidate, dataset, num_classes, n_rois, seed, split_identity):
    """Only the existing one-layer, final-state, native-window baseline is an anchor."""
    i = candidate.identity
    t = dict(i.temporal_config or {})
    if (i.mode != "temporal" or i.seed != seed or i.split_identity != split_identity
            or i.num_classes != num_classes or i.backbone_config.get("name") != "lasso_edges"
            or i.backbone_config.get("n_rois") != n_rois
            or i.window_policy != WindowPolicyConfig(policy="native")
            or t.get("name") != "bigru" or t.get("num_layers") != 1
            or t.get("pooling") != "final" or t.get("dropout", 0.) != 0.
            or t.get("projection_dim") is not None or t.get("layer_norm", False)
            or candidate.training.early_stopping.enabled):
        raise ValueError("Reference is not the supported native-window standalone TempLasso baseline")
    if candidate.training.checkpoint_selection != dataset_selection_rule(dataset, num_classes):
        raise ValueError("Reference uses an obsolete dataset selection rule; PPMI requires macro AUC/max")


def reference_anchor(reference_grid, *, dataset, num_classes, n_rois, seed, split_identity,
                     temporal_h5):
    """Select from all 18 existing baseline candidates using validation only.

    Uses no test metric for choosing C, h, LR, or support. The original result
    files/checkpoints are read-only. Refuse partial grids or duplicate indices.
    """
    from experiments.grid_protocol import read_completed
    from unified_bn.training.search_space import candidate_fingerprint
    dirs = list((Path(reference_grid) / f"seed{seed}").glob("cand_*"))
    if len(dirs) != 18:
        raise ValueError(f"Reference seed {seed}: expected all 18 TempLasso candidates, found {len(dirs)}")
    records = []
    for directory in dirs:
        paths = list(directory.glob("candidate_*/result.json"))
        if len(paths) != 1:
            raise ValueError(f"Missing/duplicate reference result: {directory}")
        payload = json.loads(paths[0].read_text(encoding="utf-8"))
        candidate = candidate_from_dict(payload["run_config"])
        validate_anchor(candidate, dataset, num_classes, n_rois, seed, split_identity)
        path, payload = read_completed(candidate, directory, dataset)
        index = int(directory.name.split("_")[1])
        if directory.name != f"cand_{index:04d}_{candidate_fingerprint(candidate)[:12]}":
            raise ValueError(f"Unexpected reference directory identity: {directory}")
        provenance = payload["user_provenance"]
        if (Path(provenance["data_artifact"]).resolve() != Path(temporal_h5).resolve()
                or provenance.get("split_identity") != split_identity
                or provenance.get("edge_selector", {}).get("fitted_on") != "train"):
            raise ValueError("Reference data/split/TRAIN-selector provenance differs")
        validation = {k: v for k, v in payload.items() if k not in ("status", "test_metrics")}
        records.append((index, candidate, path, payload, content_hash(validation)))
    records.sort(key=lambda r: r[0])
    if [r[0] for r in records] != list(range(18)):
        raise ValueError("Incomplete/duplicate reference candidate indices")
    # Check the declared historical Cartesian grid, not just the file count.
    configs = {(r[1].identity.backbone_config["C"], r[1].identity.temporal_config["hidden_dim"],
                r[1].training.optimizer.lr) for r in records}
    cs = {r[0] for r in configs}; hs = {r[1] for r in configs}; lrs = {r[2] for r in configs}
    if len(cs) != 3 or hs != {32, 64, 128} or lrs != {1e-4, 3e-4} or len(configs) != 18:
        raise ValueError("Reference is not the complete 3 C x 3 hidden widths x 2 learning rates grid")
    common = []
    for _, candidate, *_ in records:
        cfg = candidate.to_dict()
        cfg["identity"]["backbone_config"].pop("C")
        cfg["identity"]["temporal_config"].pop("hidden_dim")
        cfg["training"]["optimizer"].pop("lr")
        common.append(canonical(cfg))
    if len(set(common)) != 1:
        raise ValueError("Reference candidates differ beyond C, hidden width and learning rate")
    rule = dataset_selection_rule(dataset, num_classes)
    best = records[0]
    for record in records[1:]:
        if rule.is_better(record[3]["best_selection_value"], best[3]["best_selection_value"]):
            best = record
    _, candidate, path, p, _ = best
    anchor = dict(kind="validation_selected_baseline", result_path=str(path.resolve()),
        checkpoint_path=p["checkpoints"]["best"], checkpoint_sha256=file_hash(p["checkpoints"]["best"]),
        candidate_index=best[0], selection_metric=rule.metric, selection_mode=rule.mode,
        validation_records=[dict(index=r[0], path=str(r[2].resolve()), validation_sha256=r[4]) for r in records])
    return candidate, anchor


def reference_selector_snapshot(selector, checkpoint, *, n_rois, n_train_subjects):
    """Validate TRAIN reproduction, then preserve exact historical model inputs.

    The checkpoint stores float32 statistics only for the selected coordinates.
    Small platform-dependent refit differences must not change those inputs.
    Unselected coordinates retain their TRAIN-refitted statistics and are never
    presented to an ablation model. All post-preparation checks remain exact.
    """
    state = selector.state_dict()
    expected_provenance = dict(kind="lasso_edges", fitted_on="train",
        n_train_subjects=n_train_subjects, n_train_windows=state["n_train_rows"],
        n_edges=state["n_edges"], n_selected=state["n_selected"],
        edge_set_fingerprint=state["edge_set_fingerprint"], config=state["config"])
    actual = checkpoint.user_provenance.get("edge_selector", {})
    if any(actual.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("Reference checkpoint TRAIN-selector provenance differs from refit")

    refitted = SelectedEdgeBackbone(selector, n_rois=n_rois).state_dict()
    historical = {}
    for key, value in refitted.items():
        saved = checkpoint.model_state.get("backbone." + key)
        if (not isinstance(saved, torch.Tensor) or saved.dtype != value.dtype
                or saved.shape != value.shape):
            raise ValueError(f"Reference checkpoint {key}: wrong tensor dtype/shape")
        historical[key] = saved.detach().cpu()
    indices = historical["selected_indices"]
    if not torch.equal(refitted["selected_indices"], indices):
        raise ValueError("Reference selected edge indices differ from TRAIN refit; refusing changed edges")
    for key in ("train_mean", "train_std"):
        if not torch.isfinite(historical[key]).all():
            raise ValueError(f"Reference checkpoint {key}: nonfinite scaling")
    saved_std = historical["train_std"].double()
    if not (saved_std > 0).all():
        raise ValueError("Reference checkpoint train_std must be positive")

    audit = dict(policy="exact_support_checkpoint_scaling_v1",
        selected_scaling_source="validation_selected_baseline_checkpoint_float32",
        unselected_scaling_source="current_TRAIN_refit_unused_by_model",
        atol=REFERENCE_SCALING_ATOL, rtol=REFERENCE_SCALING_RTOL,
        max_normalized_drift=REFERENCE_NORMALIZED_DRIFT,
        support_exact=True, statistics={})
    for key in ("train_mean", "train_std"):
        saved = historical[key].double()
        recomputed = refitted[key].double()
        difference = (recomputed - saved).abs()
        normalized = difference / saved_std
        limit = REFERENCE_SCALING_ATOL + REFERENCE_SCALING_RTOL * saved.abs()
        if (not torch.isfinite(recomputed).all() or torch.any(difference > limit)
                or torch.any(normalized > REFERENCE_NORMALIZED_DRIFT)):
            raise ValueError(f"Reference {key} materially differs from TRAIN refit: "
                f"max_abs={difference.max().item():.9g}, "
                f"max_normalized={normalized.max().item():.9g}")
        audit["statistics"][key] = dict(exact_refit=torch.equal(refitted[key], historical[key]),
            differing_values=int(torch.count_nonzero(difference).item()),
            max_absolute_difference=float(difference.max().item()),
            max_normalized_difference=float(normalized.max().item()))
        full_values = np.asarray(state[key], dtype=np.float64).copy()
        full_values[indices.numpy()] = saved.numpy()
        state[key] = full_values.tolist()

    # Round-trip the JSON-compatible snapshot through the actual production
    # encoder; every model-facing coordinate must equal the checkpoint exactly.
    restored = SelectedEdgeBackbone(LassoEdgeSelector.from_state_dict(state), n_rois=n_rois)
    for key, value in restored.state_dict().items():
        if not torch.equal(value, historical[key]):
            raise ValueError(f"Reference {key} was not preserved exactly in frozen snapshot")
    return state, audit


def prepare_protocol(*, dataset, num_classes, n_rois, seeds, split_dir, split_prefix,
                     temporal_h5, lasso_c, hidden_dim, training, reference_grid=None):
    """Fit once per seed on TRAIN only; never construct validation/test loaders."""
    dataset = dataset.upper()
    rule = dataset_selection_rule(dataset, num_classes)
    if dataset not in ("ABIDE", "ADNI", "PPMI") or num_classes != (4 if dataset == "PPMI" else 2):
        raise ValueError("Use ABIDE/ADNI with 2 classes or PPMI with 4 classes")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Unique nonnegative seeds required")
    if reference_grid is None and (lasso_c is None or training.checkpoint_selection != rule):
        raise ValueError("A prespecified C and the dataset selection rule are required")
    temporal_h5 = Path(temporal_h5).resolve()
    initial_data_hash = file_hash(temporal_h5)
    records = {}
    for seed in seeds:
        split_path = (Path(split_dir) / f"{split_prefix}_split_seed{seed}.npz").resolve()
        split = load_frozen_split(split_path)
        if read_stored_split_seed(split_path) != seed:
            raise ValueError("Stored split seed differs from requested seed")
        identity = frozen_split_identity(split)
        if reference_grid is not None:
            candidate, anchor = reference_anchor(reference_grid, dataset=dataset, num_classes=num_classes,
                n_rois=n_rois, seed=seed, split_identity=identity, temporal_h5=temporal_h5)
            train_config = candidate.training
            h = candidate.identity.temporal_config["hidden_dim"]
            recipe = dict(candidate.identity.backbone_config)
            recipe.pop("name"); recipe.pop("n_rois")
            selector_config = LassoEdgeSelectorConfig(**recipe)
        else:
            h, train_config = hidden_dim, training
            selector_config = LassoEdgeSelectorConfig(C=lasso_c)
            anchor = dict(kind="prespecified", C=lasso_c)
        if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        ds = TemporalABIDEDataset(temporal_h5, subject_ids=split.train_ids,
                                 expected_n_rois=n_rois, num_classes=num_classes)
        try:
            if tuple(ds.subject_ids) != tuple(split.train_ids):
                raise ValueError("Training subject order differs from frozen split")
            x, y = pooled_edge_rows_from_dataset(ds, n_rois=n_rois, input_space=selector_config.input_space)
        finally:
            ds.close()
        if set(y.tolist()) != set(range(num_classes)):
            raise ValueError("TRAIN must contain every target class")
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            selector = LassoEdgeSelector(selector_config).fit(x, y).freeze()
        state = selector.state_dict()
        if reference_grid is not None:
            from unified_bn.training.checkpoint import read_training_checkpoint
            if file_hash(anchor["checkpoint_path"]) != anchor["checkpoint_sha256"]:
                raise ValueError("Reference checkpoint changed during preparation")
            checkpoint = read_training_checkpoint(anchor["checkpoint_path"], expected_run_config=candidate)
            state, audit = reference_selector_snapshot(selector, checkpoint, n_rois=n_rois,
                                                       n_train_subjects=len(split.train_ids))
            if file_hash(anchor["checkpoint_path"]) != anchor["checkpoint_sha256"]:
                raise ValueError("Reference checkpoint changed during preparation")
            anchor["selector_reproduction"] = audit
            print(f"Seed {seed}: reference support exact ({state['n_selected']} edges); "
                  "checkpoint scaling preserved exactly; "
                  f"refit max differences: mean={audit['statistics']['train_mean']['max_absolute_difference']:.9g}, "
                  f"std={audit['statistics']['train_std']['max_absolute_difference']:.9g}", flush=True)
        records[str(seed)] = dict(split_path=str(split_path), split_sha256=file_hash(split_path),
            split_identity=identity, train_ids=list(split.train_ids),
            selector_state=state, selector_sha256=content_hash(state),
            hidden_dim=h, training=train_config.to_dict(), anchor=anchor)
    if file_hash(temporal_h5) != initial_data_hash:
        raise ValueError("Temporal artifact changed during selector preparation")
    payload = dict(schema=SCHEMA, dataset=dataset, num_classes=num_classes, n_rois=n_rois,
        seeds=list(seeds), variants=list(VARIANTS), temporal_h5=str(temporal_h5),
        temporal_sha256=initial_data_hash, selector_method="existing pooled TRAIN windows L1-logistic",
        shuffle_policy=SHUFFLE_POLICY, records=records,
        source_sha256=source_hash(), software=software_versions())
    return dict(payload, protocol_sha256=content_hash(payload))


def load_protocol(path):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_hash = content_hash({k: v for k, v in p.items() if k != "protocol_sha256"})
    if (p.get("schema") != SCHEMA or p.get("protocol_sha256") != expected_hash
            or p.get("variants") != list(VARIANTS) or p.get("shuffle_policy") != SHUFFLE_POLICY):
        raise ValueError("Invalid or modified ablation protocol")
    if p["source_sha256"] != source_hash() or p["software"] != software_versions():
        raise ValueError("Source code or scientific package versions changed since preparation")
    if file_hash(p["temporal_h5"]) != p["temporal_sha256"]:
        raise ValueError("Temporal HDF5 changed since preparation")
    for seed in p["seeds"]:
        r = p["records"][str(seed)]
        split = load_frozen_split(r["split_path"])
        if (file_hash(r["split_path"]) != r["split_sha256"]
                or frozen_split_identity(split) != r["split_identity"]
                or list(split.train_ids) != r["train_ids"]
                or read_stored_split_seed(r["split_path"]) != seed):
            raise ValueError("Frozen split changed")
        if training_from_dict(r["training"]).checkpoint_selection != dataset_selection_rule(p["dataset"], p["num_classes"]):
            raise ValueError("Wrong dataset selection rule")
        selector_from_snapshot(r, p["n_rois"])
    return p


def make_candidate(protocol, seed, variant):
    if variant not in protocol["variants"] or seed not in protocol["seeds"]:
        raise ValueError("Candidate is outside the frozen protocol")
    r = protocol["records"][str(seed)]
    return CandidateRunConfig(RunIdentity(mode="temporal", seed=seed,
        split_identity=r["split_identity"], num_classes=protocol["num_classes"],
        backbone_config=dict(name="lasso_edges", n_rois=protocol["n_rois"],
                             **r["selector_state"]["config"]),
        temporal_config=dict(name="templasso_ablation", variant=variant,
            hidden_dim=r["hidden_dim"], output_dim=2*r["hidden_dim"], num_layers=1,
            shuffle_policy=protocol["shuffle_policy"] if variant == "shuffled_bigru" else "none",
            shuffle_seed=seed, protocol_sha256=protocol["protocol_sha256"],
            selector_sha256=r["selector_sha256"]),
        window_policy=WindowPolicyConfig(policy="native")), training_from_dict(r["training"]))


def make_model(protocol, seed, variant):
    r = protocol["records"][str(seed)]
    selector = selector_from_snapshot(r, protocol["n_rois"])
    backbone = SelectedEdgeBackbone(selector, n_rois=protocol["n_rois"])
    aggregator = make_aggregator(variant, backbone.embedding_dim, r["hidden_dim"])
    return TemporalPipeline(backbone, aggregator, num_classes=protocol["num_classes"],
                            window_policy=WindowPolicyConfig(policy="native"))


def provenance(protocol, seed, variant, model):
    r = protocol["records"][str(seed)]
    return dict(dataset=protocol["dataset"], experiment=SCHEMA, variant=variant,
        protocol_sha256=protocol["protocol_sha256"], split_identity=r["split_identity"],
        temporal_sha256=protocol["temporal_sha256"], selector_sha256=r["selector_sha256"],
        edge_set_fingerprint=r["selector_state"]["edge_set_fingerprint"],
        n_selected=r["selector_state"]["n_selected"], fitted_on="train",
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        shuffle_policy=protocol["shuffle_policy"] if variant == "shuffled_bigru" else "none")


def make_loader(protocol, seed, variant, partition, *, generator=None, device="cpu"):
    r = protocol["records"][str(seed)]
    split = load_frozen_split(r["split_path"])
    ids = {"train": split.train_ids, "val": split.val_ids, "test": split.test_ids}[partition]
    ds = TemporalABIDEDataset(protocol["temporal_h5"], subject_ids=ids,
        expected_n_rois=protocol["n_rois"], num_classes=protocol["num_classes"])
    if tuple(ds.subject_ids) != tuple(ids):
        ds.close()
        raise ValueError("Dataset subject order differs from the frozen split")
    config = training_from_dict(r["training"])
    return DataLoader(ds, batch_size=config.batch_size, num_workers=config.num_workers,
        shuffle=partition == "train", generator=generator, drop_last=False,
        pin_memory=torch.device(device).type == "cuda",
        collate_fn=TemporalAblationCollate(variant, seed))


class AblationRuntimeFactory:
    """Rehydrate the frozen selector; fitting is impossible in this runtime."""

    def __init__(self, protocol, seed, variant, device):
        self.protocol, self.seed, self.variant, self.device = protocol, seed, variant, device
        self.loaders = []

    def __call__(self, candidate, index, run_dir):
        p, seed, variant = self.protocol, self.seed, self.variant
        if candidate.to_dict() != make_candidate(p, seed, variant).to_dict() or index != 0:
            raise ValueError("Ablation candidate differs from frozen protocol")
        model = make_model(p, seed, variant)
        generator = torch.Generator().manual_seed(seed)
        train = make_loader(p, seed, variant, "train", generator=generator, device=self.device)
        self.loaders.append(train)
        val = make_loader(p, seed, variant, "val", device=self.device)
        self.loaders.append(val)
        return CandidateRuntime(model=model, train_batches=train, val_batches=val,
            device=self.device, dataloader_generator=generator,
            user_provenance=provenance(p, seed, variant, model),
            non_blocking=torch.device(self.device).type == "cuda")

    def close(self):
        for loader in self.loaders:
            loader.dataset.close()
        self.loaders.clear()
