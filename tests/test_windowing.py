from __future__ import annotations

import pytest
import torch

from unified_bn.temporal.windowing import (
    WindowPolicyConfig,
    apply_window_policy,
    resolve_global_min_windows,
)


def test_resolve_global_min_windows() -> None:
    lengths = torch.tensor([15, 26, 17, 12], dtype=torch.long)
    assert resolve_global_min_windows(lengths) == 12


def test_native_policy_is_identity() -> None:
    x = torch.randn(3, 6, 4, 4)
    lengths = torch.tensor([6, 4, 2], dtype=torch.long)

    out, new_lengths = apply_window_policy(
        x,
        lengths,
        WindowPolicyConfig(policy="native"),
    )

    assert out is x
    assert new_lengths is lengths


def test_min_start_crop() -> None:
    x = torch.arange(3 * 6 * 2 * 2, dtype=torch.float32).reshape(3, 6, 2, 2)
    lengths = torch.tensor([6, 5, 4], dtype=torch.long)

    out, new_lengths = apply_window_policy(
        x,
        lengths,
        WindowPolicyConfig(
            policy="min",
            min_windows=4,
            crop="start",
        ),
    )

    assert out.shape == (3, 4, 2, 2)
    assert torch.equal(new_lengths, torch.tensor([4, 4, 4]))
    assert torch.equal(out, x[:, :4])


def test_min_center_crop_uses_each_subject_native_length() -> None:
    x = torch.zeros(2, 7, 1, 1)
    x[0, :, 0, 0] = torch.arange(7)
    x[1, :, 0, 0] = torch.arange(7) + 10
    lengths = torch.tensor([7, 5], dtype=torch.long)

    out, new_lengths = apply_window_policy(
        x,
        lengths,
        WindowPolicyConfig(
            policy="min",
            min_windows=3,
            crop="center",
        ),
    )

    assert torch.equal(out[0, :, 0, 0], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(out[1, :, 0, 0], torch.tensor([11.0, 12.0, 13.0]))
    assert torch.equal(new_lengths, torch.tensor([3, 3]))


def test_min_policy_requires_precomputed_value() -> None:
    x = torch.randn(2, 5, 3, 3)
    lengths = torch.tensor([5, 4], dtype=torch.long)

    with pytest.raises(ValueError, match="precomputed"):
        apply_window_policy(
            x,
            lengths,
            WindowPolicyConfig(policy="min"),
        )
