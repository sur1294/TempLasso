from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from unified_bn.data import (
    FrozenSplit,
    FusionABIDEDataset,
    StaticABIDEDataset,
    TemporalABIDEDataset,
    collate_fusion,
    collate_static,
    collate_temporal,
    load_frozen_split,
)
from unified_bn.selectors.lasso import (
    LassoEdgeSelector,
    LassoEdgeSelectorConfig,
    pooled_edge_rows_from_dataset,
)
from unified_bn.training.config import CandidateRunConfig
from unified_bn.training.model_factory import (
    build_model_from_candidate,
    build_templasso_model_from_candidate,
)
from unified_bn.training.orchestrator import CandidateRuntime


class ABIDERuntimeError(RuntimeError):
    """Raised when a frozen ABIDE candidate cannot be executed safely."""


@dataclass(frozen=True)
class ABIDERuntimeConfig:
    """
    Non-scientific runtime inputs for one frozen ABIDE validation search.

    These values locate already-frozen artifacts and control data transport.
    Model/training hyperparameters remain inside CandidateRunConfig.
    """

    split_path: str | Path
    static_h5: str | Path
    temporal_h5: str | Path
    device: str | torch.device
    expected_n_rois: int = 100
    # Number of classes in the label column (2 = the historical binary protocol).
    num_classes: int = 2
    pin_memory: Optional[bool] = None
    require_stored_split_seed: bool = True
    # Recorded in run provenance only. The runtime itself is dataset-agnostic:
    # it consumes any dense static / ragged temporal HDF5 pair plus a frozen
    # ID split, so ADNI etc. reuse it unchanged with a different label here.
    dataset_name: str = "ABIDE"
    atlas_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise ValueError("dataset_name must be a non-empty string.")
        if isinstance(self.expected_n_rois, bool) or not isinstance(
            self.expected_n_rois, int
        ):
            raise TypeError("expected_n_rois must be an integer.")
        if self.expected_n_rois <= 0:
            raise ValueError("expected_n_rois must be positive.")
        if self.pin_memory is not None and not isinstance(
            self.pin_memory, bool
        ):
            raise TypeError("pin_memory must be bool or None.")
        if not isinstance(self.require_stored_split_seed, bool):
            raise TypeError("require_stored_split_seed must be a bool.")


class ABIDERuntimeFactory:
    """
    Construct one real ABIDE train/validation runtime per frozen candidate.

    Safety boundaries
    -----------------
    * Loads an existing frozen subject-ID split only.
    * Never generates or modifies a split.
    * Builds TRAIN and VALIDATION datasets/loaders only.
    * Does not construct a held-out test dataset or loader.
    * Verifies candidate seed and semantic split identity.
    * Uses the validated model factory; no model mathematics live here.
    * For temporal candidates whose backbone is ``lasso_edges`` (TempLasso),
      fits the L1 edge selector on the TRAIN partition only, freezes it, and
      hands it to the factory. Validation rows never reach the selector's
      fit(). The materialised edge set's fingerprint goes into provenance.
    * Uses the candidate TrainingConfig for batch_size and num_workers.
    * Returns the dedicated training DataLoader generator so Stage 6B
      checkpoint/resume can restore shuffle state exactly.
    """

    def __init__(self, config: ABIDERuntimeConfig) -> None:
        if not isinstance(config, ABIDERuntimeConfig):
            raise TypeError(
                "config must be ABIDERuntimeConfig, got "
                f"{type(config)!r}."
            )

        self.config = config
        self.split_path = Path(config.split_path)
        self.static_h5 = Path(config.static_h5)
        self.temporal_h5 = Path(config.temporal_h5)
        self.device = torch.device(config.device)

        self.split = load_frozen_split(self.split_path)
        self.split_identity = frozen_split_identity(self.split)
        self.stored_split_seed = read_stored_split_seed(self.split_path)

        if (
            config.require_stored_split_seed
            and self.stored_split_seed is None
        ):
            raise ABIDERuntimeError(
                "Frozen split does not contain a scalar 'seed' field."
            )

    def __call__(
        self,
        candidate: CandidateRunConfig,
        index: int,
        run_dir: Path,
    ) -> CandidateRuntime:
        if not isinstance(candidate, CandidateRunConfig):
            raise TypeError(
                "candidate must be CandidateRunConfig, got "
                f"{type(candidate)!r}."
            )
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer.")
        if index < 0:
            raise ValueError("index must be nonnegative.")
        if not isinstance(run_dir, Path):
            raise TypeError("run_dir must be pathlib.Path.")

        self._validate_candidate_identity(candidate)

        edge_selector: LassoEdgeSelector | None = None
        selector_provenance: dict[str, Any] | None = None
        if self._is_templasso(candidate):
            edge_selector, selector_provenance = self._fit_edge_selector(candidate)

        # Keep the exact historical call for every non-TempLasso candidate so
        # the BQN path is byte-for-byte unchanged (and remains patchable).
        if edge_selector is None:
            model = build_model_from_candidate(candidate)
        else:
            model = build_templasso_model_from_candidate(
                candidate, edge_selector=edge_selector
            )

        generator = torch.Generator()
        generator.manual_seed(candidate.identity.seed)

        pin_memory = self._resolved_pin_memory()
        loader_kwargs: dict[str, Any] = {
            "batch_size": candidate.training.batch_size,
            "num_workers": candidate.training.num_workers,
            "drop_last": False,
            "pin_memory": pin_memory,
        }

        if candidate.identity.mode == "vanilla":
            train_dataset = StaticABIDEDataset(
                self.static_h5,
                subject_ids=self.split.train_ids,
                expected_n_rois=self.config.expected_n_rois,
                num_classes=self.config.num_classes,
            )
            val_dataset = StaticABIDEDataset(
                self.static_h5,
                subject_ids=self.split.val_ids,
                expected_n_rois=self.config.expected_n_rois,
                num_classes=self.config.num_classes,
            )
            collate_fn = collate_static
            data_artifact = str(self.static_h5)

        elif candidate.identity.mode == "temporal":
            train_dataset = TemporalABIDEDataset(
                self.temporal_h5,
                subject_ids=self.split.train_ids,
                expected_n_rois=self.config.expected_n_rois,
                num_classes=self.config.num_classes,
            )
            val_dataset = TemporalABIDEDataset(
                self.temporal_h5,
                subject_ids=self.split.val_ids,
                expected_n_rois=self.config.expected_n_rois,
                num_classes=self.config.num_classes,
            )
            collate_fn = collate_temporal
            data_artifact: Any = str(self.temporal_h5)

        elif candidate.identity.mode == "fusion":
            train_dataset = FusionABIDEDataset(
                self.static_h5,
                self.temporal_h5,
                subject_ids=self.split.train_ids,
                expected_n_rois=self.config.expected_n_rois,
                num_classes=self.config.num_classes,
            )
            val_dataset = FusionABIDEDataset(
                self.static_h5,
                self.temporal_h5,
                subject_ids=self.split.val_ids,
                expected_n_rois=self.config.expected_n_rois,
                num_classes=self.config.num_classes,
            )
            collate_fn = collate_fusion
            data_artifact = {
                "static_h5": str(self.static_h5),
                "temporal_h5": str(self.temporal_h5),
            }

        else:
            raise ABIDERuntimeError(
                "ABIDE runtime supports only vanilla, temporal, or fusion mode."
            )

        _require_exact_partition_order(
            train_dataset.subject_ids,
            self.split.train_ids,
            partition="train",
        )
        _require_exact_partition_order(
            val_dataset.subject_ids,
            self.split.val_ids,
            partition="val",
        )

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            **loader_kwargs,
        )

        provenance = {
            "dataset": self.config.dataset_name,
            "atlas": self.config.atlas_name or f"Schaefer{self.config.expected_n_rois}",
            "mode": candidate.identity.mode,
            "data_artifact": data_artifact,
            "split_path": str(self.split_path),
            "split_identity": self.split_identity,
            "split_seed": self.stored_split_seed,
            "train_n": len(self.split.train_ids),
            "val_n": len(self.split.val_ids),
            "candidate_index": index,
        }
        if selector_provenance is not None:
            provenance["edge_selector"] = selector_provenance

        return CandidateRuntime(
            model=model,
            train_batches=train_loader,
            val_batches=val_loader,
            device=self.device,
            dataloader_generator=generator,
            user_provenance=provenance,
            non_blocking=bool(pin_memory and self.device.type == "cuda"),
        )

    @staticmethod
    def _templasso_recipe(candidate: CandidateRunConfig) -> dict[str, Any] | None:
        """Return the lasso_edges recipe mapping if this candidate has one."""
        ident = candidate.identity
        if ident.mode == "temporal":
            payload = ident.backbone_config
        elif ident.mode == "fusion":
            payload = ident.temporal_backbone_config
        else:
            return None
        if payload is None or payload.get("name") != "lasso_edges":
            return None
        return dict(payload)

    @classmethod
    def _is_templasso(cls, candidate: CandidateRunConfig) -> bool:
        return cls._templasso_recipe(candidate) is not None

    def _fit_edge_selector(
        self,
        candidate: CandidateRunConfig,
    ) -> tuple[LassoEdgeSelector, dict[str, Any]]:
        """
        Fit and freeze the TempLasso edge selector on TRAIN rows only.

        The TRAIN dataset is opened here solely to pool its windows; the
        DataLoader built later for optimisation re-reads the same frozen
        partition. Validation subjects are never passed to fit().
        """
        recipe = self._templasso_recipe(candidate)
        assert recipe is not None
        recipe.pop("name", None)
        n_rois = int(recipe.pop("n_rois"))
        config = LassoEdgeSelectorConfig(**recipe)

        train_dataset = TemporalABIDEDataset(
            self.temporal_h5,
            subject_ids=self.split.train_ids,
            expected_n_rois=self.config.expected_n_rois,
                num_classes=self.config.num_classes,
        )
        _require_exact_partition_order(
            train_dataset.subject_ids,
            self.split.train_ids,
            partition="train",
        )
        try:
            x_train, y_train = pooled_edge_rows_from_dataset(
                train_dataset,
                n_rois=n_rois,
                input_space=config.input_space,
            )
        finally:
            train_dataset.close()

        selector = LassoEdgeSelector(config).fit(x_train, y_train).freeze()

        provenance = {
            "kind": "lasso_edges",
            "fitted_on": "train",
            "n_train_subjects": len(self.split.train_ids),
            "n_train_windows": int(x_train.shape[0]),
            "n_edges": int(selector.n_edges),
            "n_selected": int(selector.n_selected),
            "edge_set_fingerprint": selector.edge_set_fingerprint(),
            "config": dict(recipe),
        }
        return selector, provenance

    def _validate_candidate_identity(
        self,
        candidate: CandidateRunConfig,
    ) -> None:
        if candidate.identity.split_identity != self.split_identity:
            raise ABIDERuntimeError(
                "Candidate split_identity does not match the semantic identity "
                "of the supplied frozen split."
            )

        if (
            self.stored_split_seed is not None
            and candidate.identity.seed != self.stored_split_seed
        ):
            raise ABIDERuntimeError(
                "Candidate seed does not match the seed stored in the frozen "
                f"split: candidate={candidate.identity.seed}, "
                f"split={self.stored_split_seed}."
            )

        backbone_n_rois = candidate.identity.backbone_config.get("n_rois")
        if backbone_n_rois != self.config.expected_n_rois:
            raise ABIDERuntimeError(
                "Candidate backbone ROI dimension does not match runtime data: "
                f"candidate={backbone_n_rois}, "
                f"runtime={self.config.expected_n_rois}."
            )

        if candidate.identity.mode == "vanilla":
            if not self.static_h5.is_file():
                raise FileNotFoundError(
                    f"Static ABIDE HDF5 not found: {self.static_h5}"
                )
        elif candidate.identity.mode == "temporal":
            if not self.temporal_h5.is_file():
                raise FileNotFoundError(
                    f"Temporal ABIDE HDF5 not found: {self.temporal_h5}"
                )
        elif candidate.identity.mode == "fusion":
            if not self.static_h5.is_file():
                raise FileNotFoundError(
                    f"Static ABIDE HDF5 not found: {self.static_h5}"
                )
            if not self.temporal_h5.is_file():
                raise FileNotFoundError(
                    f"Temporal ABIDE HDF5 not found: {self.temporal_h5}"
                )

    def _resolved_pin_memory(self) -> bool:
        if self.config.pin_memory is not None:
            return self.config.pin_memory
        return self.device.type == "cuda"


def frozen_split_identity(split: FrozenSplit) -> str:
    """
    Stable semantic SHA-256 identity of an ordered frozen split.

    Filesystem path, NPZ compression, timestamps, and unrelated metadata do
    not participate. Partition names, boundaries, counts, and ordered subject
    IDs do participate, so reordering subjects changes the identity.
    """

    if not isinstance(split, FrozenSplit):
        raise TypeError(
            f"split must be FrozenSplit, got {type(split)!r}."
        )

    digest = hashlib.sha256()

    for name, subject_ids in (
        ("train", split.train_ids),
        ("val", split.val_ids),
        ("test", split.test_ids),
    ):
        _hash_text(digest, name)
        digest.update(len(subject_ids).to_bytes(8, "big", signed=False))
        for subject_id in subject_ids:
            _hash_text(digest, subject_id)

    return "sha256:" + digest.hexdigest()


def read_stored_split_seed(path: str | Path) -> int | None:
    """
    Read only the optional scalar seed metadata from a frozen NPZ.

    Split labels and held-out outcomes are not loaded by this helper.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Frozen split not found: {source}")

    with np.load(source, allow_pickle=False) as npz:
        if "seed" not in npz.files:
            return None

        raw = np.asarray(npz["seed"])
        if raw.size != 1:
            raise ABIDERuntimeError(
                "Frozen split 'seed' field must contain exactly one value."
            )

        value = raw.reshape(-1)[0]
        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, bool) or not isinstance(
            value, (int, np.integer)
        ):
            raise ABIDERuntimeError(
                "Frozen split 'seed' field must be integer-valued."
            )

        seed = int(value)
        if seed < 0:
            raise ABIDERuntimeError(
                "Frozen split 'seed' field must be nonnegative."
            )
        return seed


def _hash_text(digest: Any, value: str) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big", signed=False))
    digest.update(encoded)


def _require_exact_partition_order(
    observed: tuple[str, ...],
    expected: tuple[str, ...],
    *,
    partition: str,
) -> None:
    if tuple(observed) != tuple(expected):
        raise ABIDERuntimeError(
            f"{partition} dataset subject order differs from the frozen split."
        )
