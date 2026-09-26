import argparse
import pytest
from unified_bn.data.cli import add_window_policy_args, window_policy_from_args

def _parser():
    p = argparse.ArgumentParser()
    add_window_policy_args(p)
    return p

def test_cli_native_default():
    args = _parser().parse_args([])
    cfg = window_policy_from_args(args)
    assert cfg.policy == "native"
    assert cfg.min_windows is None

def test_cli_min_uses_resolved_global_value():
    args = _parser().parse_args(["--window-policy","min"])
    cfg = window_policy_from_args(args, resolved_min_windows=12)
    assert cfg.policy == "min"
    assert cfg.min_windows == 12

def test_cli_min_requires_value():
    args = _parser().parse_args(["--window-policy","min"])
    with pytest.raises(ValueError, match="requires"):
        window_policy_from_args(args)
