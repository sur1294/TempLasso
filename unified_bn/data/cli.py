from __future__ import annotations
import argparse
from unified_bn.temporal.windowing import WindowPolicyConfig

def add_window_policy_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("temporal window policy")
    group.add_argument(
        "--window-policy", choices=("native", "min"), default="native",
        help="native preserves K_i; min truncates every subject to one frozen common K.",
    )
    group.add_argument(
        "--window-crop", choices=("start", "center"), default="start",
        help="Deterministic crop rule used only with --window-policy=min.",
    )
    group.add_argument(
        "--min-windows", type=int, default=None,
        help=(
            "Frozen common K for min mode. If omitted, resolve once from the "
            "intended cohort before training; never per minibatch."
        ),
    )

def window_policy_from_args(args, *, resolved_min_windows=None):
    if args.window_policy == "native":
        return WindowPolicyConfig(
            policy="native", crop=args.window_crop, min_windows=None
        )
    min_windows = (
        args.min_windows if args.min_windows is not None else resolved_min_windows
    )
    if min_windows is None:
        raise ValueError(
            "--window-policy=min requires --min-windows or resolved_min_windows."
        )
    return WindowPolicyConfig(
        policy="min", crop=args.window_crop, min_windows=int(min_windows)
    )
