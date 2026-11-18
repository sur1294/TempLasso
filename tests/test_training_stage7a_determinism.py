from __future__ import annotations

import pytest
import torch

from unified_bn.training.config import DeterminismConfig, TrainingConfig
from unified_bn.training.reproducibility import seed_everything
from unified_bn.training.search_space import (
    BQNSearchSpace,
    candidate_fingerprint,
    generate_vanilla_bqn_candidates,
)


def test_production_determinism_defaults_are_strict() -> None:
    cfg = DeterminismConfig()
    assert cfg.deterministic_algorithms is True
    assert cfg.warn_only is False
    assert cfg.num_threads == 1
    assert TrainingConfig().determinism == cfg


def test_determinism_config_validates_thread_count() -> None:
    with pytest.raises(TypeError, match="num_threads"):
        DeterminismConfig(num_threads=True)
    with pytest.raises(TypeError, match="num_threads"):
        DeterminismConfig(num_threads=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        DeterminismConfig(num_threads=0)


def test_seed_everything_pins_threads_and_uses_strict_mode(monkeypatch) -> None:
    seen = {}

    monkeypatch.setattr(
        torch,
        "set_num_threads",
        lambda value: seen.__setitem__("num_threads", value),
    )

    def fake_use_deterministic_algorithms(enabled, *, warn_only=False):
        seen["enabled"] = bool(enabled)
        seen["warn_only"] = bool(warn_only)

    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        fake_use_deterministic_algorithms,
    )

    seed_everything(
        42,
        deterministic_algorithms=True,
        warn_only=False,
        num_threads=1,
    )

    assert seen == {
        "num_threads": 1,
        "enabled": True,
        "warn_only": False,
    }
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_seed_everything_rejects_invalid_thread_count() -> None:
    with pytest.raises(TypeError, match="num_threads"):
        seed_everything(42, num_threads=True)
    with pytest.raises(ValueError, match="positive"):
        seed_everything(42, num_threads=0)


def test_thread_policy_is_part_of_candidate_fingerprint() -> None:
    backbone = BQNSearchSpace(
        n_layers=(3,),
        activations=("elu",),
        dropouts=(0.3,),
    )

    c1 = generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=backbone,
        training_configs=(
            TrainingConfig(
                determinism=DeterminismConfig(num_threads=1)
            ),
        ),
    )[0]
    c2 = generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=backbone,
        training_configs=(
            TrainingConfig(
                determinism=DeterminismConfig(num_threads=2)
            ),
        ),
    )[0]

    assert candidate_fingerprint(c1) != candidate_fingerprint(c2)
