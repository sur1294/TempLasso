from __future__ import annotations

import os
import random

import numpy as np
import torch


_UINT32_MODULUS = 2**32


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    return seed


def seed_everything(
    seed: int,
    *,
    deterministic_algorithms: bool = True,
    warn_only: bool = False,
    num_threads: int = 1,
) -> None:
    """
    Seed Python, NumPy, PyTorch CPU, and PyTorch CUDA RNGs.

    This follows the validated historical BQN reproducibility policy.

    Notes
    -----
    Setting ``PYTHONHASHSEED`` inside an already-running Python interpreter
    cannot retroactively change that interpreter's hash randomization. The
    assignment is still useful because subsequently spawned child processes
    inherit the requested value. For exact command-level reproducibility it
    should also be exported before Python starts.

    With ``deterministic_algorithms=True``, cuDNN benchmarking is disabled,
    cuDNN deterministic mode is requested, CUBLAS_WORKSPACE_CONFIG is set if
    absent, and PyTorch deterministic algorithms are requested.

    Production defaults use ``warn_only=False`` so an operation without a
    deterministic implementation fails loudly. ``num_threads`` pins PyTorch
    CPU intra-op parallelism; the production default is one thread to avoid
    thread-order-dependent floating-point reductions.
    """

    seed = _validate_seed(seed)

    if not isinstance(deterministic_algorithms, bool):
        raise TypeError("deterministic_algorithms must be a bool.")
    if not isinstance(warn_only, bool):
        raise TypeError("warn_only must be a bool.")
    if isinstance(num_threads, bool) or not isinstance(num_threads, int):
        raise TypeError("num_threads must be an integer.")
    if num_threads <= 0:
        raise ValueError("num_threads must be positive.")

    # Set process-level determinism environment before CUDA RNG calls.
    # PYTHONHASHSEED is still best exported before Python starts; setting it
    # here ensures child processes inherit the requested value.
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Pin PyTorch CPU intra-op reduction order. We intentionally do not call
    # torch.set_num_interop_threads() here because this function is called
    # repeatedly across candidates in one process.
    torch.set_num_threads(num_threads)

    random.seed(seed)
    np.random.seed(seed % _UINT32_MODULUS)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
    torch.use_deterministic_algorithms(
        bool(deterministic_algorithms),
        warn_only=bool(warn_only),
    )


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Create a dedicated CPU RNG for reproducible DataLoader shuffling."""

    seed = _validate_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def seed_dataloader_worker(worker_id: int) -> None:
    """
    Seed Python and NumPy in a DataLoader worker from PyTorch's worker seed.

    Use this as ``worker_init_fn`` together with a dedicated generator from
    :func:`make_dataloader_generator` when ``num_workers > 0``.
    """

    if isinstance(worker_id, bool) or not isinstance(worker_id, int):
        raise TypeError("worker_id must be an integer.")
    if worker_id < 0:
        raise ValueError("worker_id must be nonnegative.")

    worker_seed = torch.initial_seed() % _UINT32_MODULUS
    np.random.seed(worker_seed)
    random.seed(worker_seed)
