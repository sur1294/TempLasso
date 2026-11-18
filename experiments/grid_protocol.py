"""Validation-only grid audit, immutable winner freeze, and CSV collection.

This module never constructs a model or a test loader. Selection uses the
candidate's serialized rule at both levels; test fields cannot break ties.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(payload):
    return json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))


@contextmanager
def grid_action_lock(grid):
    """Prevent concurrent freeze/evaluation of the same grid (CPU or GPU)."""
    grid = Path(grid)
    grid.mkdir(parents=True, exist_ok=True)
    with (grid / ".selection.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def read_completed(candidate, directory, dataset):
    """Reject stale configurations, partial grids and ambiguous results."""
    paths = list(Path(directory).glob("candidate_*/result.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one completed result in {directory}; found {len(paths)}")
    path = paths[0]
    p = json.loads(path.read_text(encoding="utf-8"))
    from unified_bn.training.results import RESULT_SCHEMA_NAME, RESULT_SCHEMA_VERSION
    if (p.get("schema_name"), p.get("schema_version")) != (RESULT_SCHEMA_NAME, RESULT_SCHEMA_VERSION):
        raise ValueError(f"Unsupported result schema: {path}")
    if p.get("run_config") != candidate.to_dict():
        raise ValueError(f"Result configuration/selection protocol mismatch: {path}. Use a new grid directory.")
    if p.get("status") not in ("validation_complete", "final_test_complete"):
        raise ValueError(f"Incomplete result: {path}")
    if (p["status"] == "final_test_complete") != (p.get("test_metrics") is not None):
        raise ValueError(f"Inconsistent test status: {path}")
    if (p.get("requested_epochs") != candidate.training.epochs
            or p.get("completed_epochs") != candidate.training.epochs or p.get("stopped_early")):
        raise ValueError(f"Incomplete epoch budget: {path}")
    if not 1 <= p.get("best_epoch", 0) <= candidate.training.epochs:
        raise ValueError(f"Invalid best epoch: {path}")
    if p.get("user_provenance", {}).get("dataset", "").strip().upper() != dataset.strip().upper():
        raise ValueError(f"Wrong dataset provenance: {path}")
    if Path(p["result_path"]).resolve() != path.resolve():
        raise ValueError(f"Result path mismatch: {path}")
    checkpoint = Path(p["checkpoints"]["best"])
    if checkpoint.resolve() != (path.parent / "best.pt").resolve():
        raise ValueError(f"Unexpected best checkpoint path: {path}")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"Missing/empty best checkpoint: {checkpoint}")
    rule = candidate.training.checkpoint_selection
    from unified_bn.training.search import RESTORED_METRIC_ABS_TOL
    if (p.get("selection_metric"), p.get("selection_mode")) != (rule.metric, rule.mode):
        raise ValueError(f"Wrong selection rule: {path}")
    value = float(p["best_selection_value"])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite selection value: {path}")
    metric = {"val_loss": "loss", "val_auc": "roc_auc"}[rule.metric]
    for key in ("best_validation_metrics", "restored_validation_metrics"):
        metrics = p[key]
        if any(not math.isfinite(float(metrics[m])) for m in
               ("loss", "roc_auc", "accuracy", "sensitivity", "specificity")):
            raise ValueError(f"Nonfinite validation metrics: {path}")
        tolerance = RESTORED_METRIC_ABS_TOL if key == "restored_validation_metrics" else 1e-12
        if not math.isclose(value, float(metrics[metric]), abs_tol=tolerance, rel_tol=0):
            raise ValueError(f"Selection value disagrees with {metric}: {path}")
    return path, p


def freeze_grid(args, per_seed_candidates, task_dir, grid_root):
    """Audit ALL seeds before freezing or permitting ANY held-out evaluation.

    A later call verifies the frozen validation content and selected checkpoint
    hashes. Attaching test metrics does not change the validation digest.
    """
    from unified_bn.training.search_space import candidate_fingerprint
    grid_root = Path(grid_root)
    frozen_path = grid_root / "frozen_validation_winners.json"
    old = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    records, winners = [], {}
    for seed, candidates in per_seed_candidates.items():
        best = None
        rule = candidates[0].training.checkpoint_selection
        for index, candidate in enumerate(candidates):
            if candidate.training.checkpoint_selection != rule:
                raise ValueError("Candidates have inconsistent selection rules")
            path, payload = read_completed(candidate, task_dir(seed, index, candidate), args.dataset)
            if payload.get("test_metrics") is not None:
                if old is None or old["winners"][str(seed)]["selected_index"] != index:
                    raise ValueError(f"Test metrics exist before a valid winner freeze: {path}")
            validation = {k: v for k, v in payload.items() if k not in ("status", "test_metrics")}
            records.append(dict(seed=seed, candidate_index=index,
                fingerprint=candidate_fingerprint(candidate), result_path=str(path.resolve()),
                validation_sha256=hashlib.sha256(canonical(validation).encode()).hexdigest()))
            value = float(payload["best_selection_value"])
            if best is None or rule.is_better(value, best["selection_value"]):
                best = dict(selected_index=index, selection_value=value,
                    selection_metric=rule.metric, selection_mode=rule.mode,
                    fingerprint=candidate_fingerprint(candidate), result_path=str(path.resolve()),
                    checkpoint_path=str(Path(payload["checkpoints"]["best"]).resolve()))
        best["checkpoint_sha256"] = digest(best["checkpoint_path"])
        winners[str(seed)] = best
    snapshot = dict(schema="dataset-grid-winners-v1", dataset=args.dataset, row=args.row,
                    backbone="" if args.row == "templasso" else args.backbone,
                    num_classes=args.num_classes, seeds=list(per_seed_candidates),
                    tie_break="lowest candidate index; earliest epoch for an epoch tie",
                    candidates=records, winners=winners)
    if old is not None:
        if snapshot != old:
            raise ValueError("Frozen validation results or selected checkpoints changed; refusing test evaluation")
    else:
        grid_root.mkdir(parents=True, exist_ok=True)
        temp = frozen_path.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, frozen_path)  # caller holds grid_action_lock
    print(f"Frozen {len(winners)} validation winners across {len(records)} candidates: {frozen_path}")
    return winners


def collect_grid(args, per_seed_candidates, task_dir, grid_root):
    """Collect saved metrics only; never fit, select, or evaluate anything."""
    from unified_bn.training.search_space import candidate_fingerprint
    rows = []
    for seed, candidates in per_seed_candidates.items():
        for index, candidate in enumerate(candidates):
            rule = candidate.training.checkpoint_selection
            row = dict(dataset=args.dataset, row=args.row,
                       backbone="" if args.row == "templasso" else args.backbone,
                       seed=seed, candidate_index=index, fingerprint=candidate_fingerprint(candidate),
                       selection_metric=rule.metric, selection_mode=rule.mode,
                       selection_value="", config=canonical(candidate.to_dict()),
                       status="pending", best_epoch="", result_path="",
                       val_loss="", val_auc="", val_accuracy="", val_sensitivity="", val_specificity="",
                       test_loss="", test_auc="", test_accuracy="", test_sensitivity="", test_specificity="")
            directory = task_dir(seed, index, candidate)
            if list(Path(directory).glob("candidate_*/result.json")):
                path, p = read_completed(candidate, directory, args.dataset)
                row.update(status=p["status"], best_epoch=p["best_epoch"], result_path=str(path),
                           selection_value=p["best_selection_value"])
                for prefix, key in (("val", "best_validation_metrics"), ("test", "test_metrics")):
                    if p.get(key) is not None:
                        for dest, source in (("loss", "loss"), ("auc", "roc_auc"), ("accuracy", "accuracy"),
                                             ("sensitivity", "sensitivity"), ("specificity", "specificity")):
                            row[f"{prefix}_{dest}"] = p[key][source]
            rows.append(row)
    out = Path(grid_root) / "csv"
    out.mkdir(parents=True, exist_ok=True)
    tables = [("all_hyperparameters.csv", rows)] + [
        (f"seed{seed}.csv", [r for r in rows if r["seed"] == seed]) for seed in per_seed_candidates]
    for name, table in tables:
        temp = out / f".{name}.{os.getpid()}.tmp"
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(table)
        os.replace(temp, out / name)
    complete = sum(r["status"] != "pending" for r in rows)
    print(f"CSV: {out / 'all_hyperparameters.csv'} ({complete}/{len(rows)} validation runs complete)")
