"""Multi-process cache warming for parallel JIT compilation.

Spawns subprocesses that each rebuild the Model from cloudpickled Regimes,
solve it, and let JAX's persistent compilation cache store the results.
When the main process later compiles the same functions, it gets cache hits.

This parallelizes both lowering (JAX tracing) and compilation (XLA), which
are otherwise sequential in the main process. Lowering is GIL-bound and
cannot be parallelized with threads — only separate processes help.
"""

import logging
import multiprocessing as mp
import os
import pickle
import time

from lcm.utils.logging import format_duration


def warm_cache(
    *,
    pickled_model_args: bytes,
    n_workers: int,
    logger: logging.Logger,
) -> None:
    """Spawn subprocesses to warm the JAX compilation cache.

    Each subprocess rebuilds the Model from the pickled constructor arguments
    and calls `solve()`. JAX's persistent cache automatically stores the
    compiled XLA programs. When the main process later compiles the same
    functions, it hits the cache.

    Args:
        pickled_model_args: cloudpickled tuple of (regimes, regime_id_class,
            ages, fixed_params, enable_jit, params).
        n_workers: Number of subprocesses to spawn.
        logger: Logger for progress reporting.

    """
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    if not cache_dir:
        logger.warning(
            "JAX_COMPILATION_CACHE_DIR not set — skipping multi-process "
            "cache warming (no persistent cache to share across processes)"
        )
        return

    logger.info("Cache warming: spawning %d workers", n_workers)
    start = time.monotonic()

    ctx = mp.get_context("spawn")
    processes = []
    for _ in range(n_workers):
        p = ctx.Process(
            target=_cache_warming_worker,
            args=(pickled_model_args, cache_dir),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        if p.exitcode != 0:
            logger.warning(
                "Cache warming worker (pid %d) exited with code %d",
                p.pid,
                p.exitcode,
            )

    elapsed = time.monotonic() - start
    logger.info("Cache warming complete  (%s)", format_duration(seconds=elapsed))


def _cache_warming_worker(pickled_args: bytes, cache_dir: str) -> None:
    """Subprocess entry point: rebuild Model and solve to populate cache.

    Must set JAX_COMPILATION_CACHE_DIR before importing JAX so the
    persistent cache is initialized with the correct directory.
    """
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir

    from lcm.model import Model  # noqa: PLC0415

    regimes, regime_id_class, ages, fixed_params, enable_jit, params = pickle.loads(  # noqa: S301
        pickled_args
    )
    model = Model(
        regimes=regimes,
        regime_id_class=regime_id_class,
        ages=ages,
        fixed_params=fixed_params,
        enable_jit=enable_jit,
    )
    model.solve(params=params, max_compilation_workers=1, log_level="off")
