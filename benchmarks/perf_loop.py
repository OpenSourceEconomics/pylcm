"""Warm-host GridSearch benchmark: one process measures one arm.

The arm is the pylcm source this process imports (recorded as `lcm.__file__`
and its git SHA) plus a named pylcm-internal patch (`--arm`). Everything runs in
one process on one model instance, in this order:

1. cold public solve at `log_level="progress"`: phases and counters;
2. warm-up solve at `log_level="off"`, untimed;
3. counted pass: `--count-reps` warm solves at `log_level="progress"` with every
   counter installed; phases per call;
4. timed pass: `--reps` warm solves at `log_level="off"` with no wrapper
   installed, `block_until_ready` inside the timer;
5. changed-params call (`discount_factor` 0.95 -> 0.94) on the warm model,
   timed and counted, compared bitwise with a freshly built model's cold solve
   of the same parameters;
6. contract digest: sha256 of the optimized HLO of every distinct dispatched
   executable, read after all timing.

The JSON written to `--out` starts with the environment record (jax / jaxlib,
x64, published dtypes, `lcm.__file__`, git SHA). The run refuses a jax below
the floor in pylcm's `pyproject.toml` before building anything.

Usage (one process per arm, then compare the JSON files):
    pixi run -e benchmarks-cuda12 python benchmarks/perf_loop.py \\
        --model precautionary_savings --arm shipped --reps 20 --count-reps 3 \\
        --label base --out base.json
"""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

if not __package__:
    # Run as a path, the repository root is not on the path, so the sibling
    # benchmark modules are unreachable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

import _lcm
import lcm
import lcm.model
from _lcm.solution import backward_induction, grid_search
from benchmarks.asv._compile_counters import count_compile_requests
from benchmarks.warm_solve_phases import (
    CallPhases,
    _collecting_lcm_records,
    parse_phase_records,
)

MODEL_NAMES = ("precautionary_savings", "iskhakov")

_PYLCM_ROOT = Path(lcm.__file__).resolve().parents[2]


@contextlib.contextmanager
def _halving_off() -> Iterator[None]:
    """Resolve every execution config with gather halving switched off."""
    resolve = lcm.model.resolve_execution_config

    def patched(*, config: Any, **kwargs: Any) -> Any:
        return resolve(
            config=dataclasses.replace(config, halve_on_materialised_gather=False),
            **kwargs,
        )

    lcm.model.resolve_execution_config = patched  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        lcm.model.resolve_execution_config = resolve


@contextlib.contextmanager
def _broadcast_off() -> Iterator[None]:
    """Make GridSearch's continuation-broadcast selector select no state."""
    select = grid_search._continuation_unread_state_names  # noqa: SLF001
    grid_search._continuation_unread_state_names = lambda **_: ()  # noqa: SLF001  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        grid_search._continuation_unread_state_names = select  # noqa: SLF001


# Arm name -> context manager held around every model build and solve.
ARM_PATCHES: Mapping[str, Callable[[], contextlib.AbstractContextManager[None]]] = {
    "shipped": contextlib.nullcontext,
    "halving_off": _halving_off,
    "broadcast_off": _broadcast_off,
}


def assert_jax_floor(*, version: str | None = None, spec: str | None = None) -> str:
    """Refuse a jax that misses the `jax` floor in pylcm's `pyproject.toml`.

    Args:
        version: The jax version to check; the imported one when `None`.
        spec: The requirement specifier; read from `pyproject.toml` when `None`.

    Returns:
        The specifier that was checked.

    Raises:
        RuntimeError: `version` does not satisfy `spec`.

    """
    version = jax.__version__ if version is None else version
    if spec is None:
        manifest = tomllib.loads((_PYLCM_ROOT / "pyproject.toml").read_text())
        (spec,) = [
            str(req.specifier)
            for req in map(Requirement, manifest["project"]["dependencies"])
            if req.name == "jax"
        ]
    if not SpecifierSet(spec).contains(version, prereleases=True):
        msg = f"jax {version} does not satisfy pylcm's declared floor jax{spec}."
        raise RuntimeError(msg)
    return spec


def environment_record() -> dict[str, Any]:
    """Versions, precision, source location and git SHA of this process."""
    sha = subprocess.run(
        ["git", "-C", str(_PYLCM_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "env_x64": os.environ.get("JAX_ENABLE_X64"),
        "published_dtypes": {
            "float": str(jnp.asarray(1.0).dtype),
            "int": str(jnp.asarray(1).dtype),
        },
        "jax_floor": assert_jax_floor(),
        "lcm_file": lcm.__file__,
        "_lcm_file": _lcm.__file__,
        "git": {"pylcm": (sha.stdout or sha.stderr).strip()},
        "devices": [str(device) for device in jax.devices()],
        "host": socket.gethostname(),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "argv": sys.argv,
    }


def _builder(model_name: str) -> Callable[[float], tuple[Any, dict[str, Any]]]:
    """Return `discount -> (model, params)` for the named benchmark model."""
    if model_name == "precautionary_savings":
        from benchmarks.asv.bench_precautionary_savings import (
            _make_model,
        )

        def make(discount: float) -> tuple[Any, dict[str, Any]]:
            model, params = _make_model(wealth_n_points=500, consumption_n_points=500)
            return model, {**params, "discount_factor": discount}

        return make

    from benchmarks.asv.bench_iskhakov_et_al_2017 import (
        _SOLVE_CONSUMPTION_N_POINTS,
        _SOLVE_WEALTH_N_POINTS,
        _make_model_and_params,
    )

    def make(discount: float) -> tuple[Any, dict[str, Any]]:
        model, params = _make_model_and_params(
            wealth_n_points=_SOLVE_WEALTH_N_POINTS,
            consumption_n_points=_SOLVE_CONSUMPTION_N_POINTS,
        )
        return model, {**params, "discount_factor": discount}

    return make


def _values(result: Any) -> dict[str, np.ndarray]:
    values = getattr(result, "values", result)
    return {
        f"{period}/{regime}": np.asarray(values[period][regime])
        for period in sorted(values)
        for regime in sorted(values[period])
    }


def _fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(arrays[key].tobytes())
    return digest.hexdigest()


def _block(result: Any) -> None:
    jax.block_until_ready(getattr(result, "values", result))


def _phases(calls: tuple[CallPhases, ...]) -> list[dict[str, Any]]:
    return [
        {
            "call_id": call.call_id,
            "residual": call.residual_seconds(),
            "phases": [dataclasses.asdict(phase) for phase in call.phases],
        }
        for call in calls
    ]


class _Counters:
    """Count the HLO-inspection path and record every dispatched executable."""

    def __init__(self) -> None:
        self.counts: dict[str, Any] = {}
        self.dispatch: dict[tuple[str, ...], Any] = {}
        self._originals: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self.counts = {
            "as_text": 0,
            "as_text_bytes": 0,
            "classify": 0,
            "gather_checks": 0,
            "inspected_ids": set(),
            "dispatches": 0,
            "inspect_seconds": 0.0,
        }

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        compiled_cls = jax.stages.Compiled
        as_text = compiled_cls.as_text

        def counted_as_text(obj: Any, *args: Any, **kwargs: Any) -> Any:
            text = as_text(obj, *args, **kwargs)
            self.counts["as_text"] += 1
            self.counts["as_text_bytes"] += len(text or "")
            return text

        originals = {
            name: getattr(backward_induction, name)
            for name in (
                "classify_reduce_fusions",
                "_checked_gather_fusion",
                "_run_period_kernel",
            )
            if hasattr(backward_induction, name)
        }

        def wrap_check(*, name: str, key: str) -> Callable[..., Any]:
            original = originals[name]

            def wrapped(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    self.counts[key] += 1
                    if key == "gather_checks":
                        self.counts["inspect_seconds"] += time.perf_counter() - start
                        compiled = kwargs.get("compiled")
                        executable = getattr(
                            compiled, "runtime_executable", lambda: None
                        )()
                        self.counts["inspected_ids"].add(id(executable or compiled))

            return wrapped

        def run(*args: Any, **kwargs: Any) -> Any:
            self.counts["dispatches"] += 1
            for core, planned in dict(kwargs["compiled_cores"]).items():
                signature = (
                    kwargs["regime_name"],
                    str(int(kwargs["period"])),
                    core,
                    getattr(planned, "name", "?"),
                    str(sorted(dict(getattr(planned, "tile_widths", {})).items())),
                    str(tuple(getattr(planned, "donated_arguments", ()))),
                )
                self.dispatch[signature] = planned
            return originals["_run_period_kernel"](*args, **kwargs)

        compiled_cls.as_text = counted_as_text  # ty: ignore[invalid-assignment]
        for name, key in (
            ("classify_reduce_fusions", "classify"),
            ("_checked_gather_fusion", "gather_checks"),
        ):
            if name in originals:
                setattr(backward_induction, name, wrap_check(name=name, key=key))
        backward_induction._run_period_kernel = run  # noqa: SLF001  # ty: ignore[invalid-assignment]
        try:
            yield
        finally:
            compiled_cls.as_text = as_text  # ty: ignore[invalid-assignment]
            for name, original in originals.items():
                setattr(backward_induction, name, original)

    def snapshot(self) -> dict[str, Any]:
        return {**self.counts, "inspected_ids": len(self.counts["inspected_ids"])}


def _counted_solve(
    *, model: Any, params: Any, counters: _Counters, log_level: str
) -> tuple[Any, dict[str, Any]]:
    """Solve once with every counter installed; return result and its record."""
    counters.reset()
    with (
        counters.installed(),
        _collecting_lcm_records() as lines,
        count_compile_requests() as requests,
    ):
        start = time.perf_counter()
        result = model.solve(params=params, log_level=log_level)
        _block(result)
        seconds = time.perf_counter() - start
    return result, {
        "seconds": seconds,
        "counts": counters.snapshot(),
        "requests": dataclasses.asdict(requests),
        "phases": _phases(parse_phase_records(lines=lines)),
    }


def _timed_warm_calls(
    *, model: Any, params: Any, reps: int, perturb: Callable[[], None]
) -> tuple[list[float], str]:
    """Time `reps` warm solves with no wrapper installed; return seconds and digest."""
    times = []
    result = None
    for _ in range(reps):
        perturb()
        start = time.perf_counter()
        result = model.solve(params=params, log_level="off")
        _block(result)
        times.append(time.perf_counter() - start)
    return times, _fingerprint(_values(result))


def _dispatch_contracts(*, counters: _Counters) -> dict[str, Any]:
    """Digest the optimized HLO of every distinct dispatched executable."""
    contracts = []
    for signature, planned in sorted(counters.dispatch.items()):
        compiled = planned.compiled
        text = compiled.as_text() if hasattr(compiled, "as_text") else None
        contracts.append(
            {
                "sig": list(signature),
                "hlo_sha": hashlib.sha256((text or "").encode()).hexdigest()[:16],
            }
        )
    return {
        "dispatch_contracts": contracts,
        "contract_digest": hashlib.sha256(
            json.dumps(contracts, sort_keys=True).encode()
        ).hexdigest()[:16],
        "sig_digest": hashlib.sha256(
            json.dumps([c["sig"] for c in contracts], sort_keys=True).encode()
        ).hexdigest()[:16],
    }


def main(*, argv: list[str] | None = None) -> None:
    """Measure one arm and write its JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_PATCHES), default="shipped")
    parser.add_argument("--perturb", choices=("none", "reread"), default="none")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--count-reps", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    record: dict[str, Any] = {
        "environment": environment_record(),
        "label": args.label,
        "model": args.model,
        "arm": args.arm,
        "perturb": args.perturb,
    }
    make = _builder(args.model)
    counters = _Counters()
    with ARM_PATCHES[args.arm]():
        model, params = make(0.95)
        _, changed = make(0.94)

        def perturb() -> None:
            if args.perturb == "reread" and hasattr(model, "_gather_checks"):
                model._gather_checks.clear()  # noqa: SLF001

        result, record["cold"] = _counted_solve(
            model=model, params=params, counters=counters, log_level="progress"
        )
        cold_fp = _fingerprint(_values(result))
        del result

        perturb()
        _block(model.solve(params=params, log_level="off"))

        counters.dispatch.clear()
        record["count_calls"] = []
        for _ in range(args.count_reps):
            perturb()
            result, call = _counted_solve(
                model=model, params=params, counters=counters, log_level="progress"
            )
            record["count_calls"].append(call)
        record["warm_fp"] = _fingerprint(_values(result))
        record["cold_fp"] = cold_fp
        del result

        times, record["time_fp"] = _timed_warm_calls(
            model=model, params=params, reps=args.reps, perturb=perturb
        )
        record["warm_seconds"] = times
        record["warm_median_ms"] = statistics.median(times) * 1e3

        perturb()
        result, record["changed"] = _counted_solve(
            model=model, params=changed, counters=counters, log_level="off"
        )
        changed_fp = _fingerprint(_values(result))
        del result
        fresh, _ = make(0.94)
        fresh_result = fresh.solve(params=changed, log_level="off")
        _block(fresh_result)
        fresh_fp = _fingerprint(_values(fresh_result))
    record["changed"].update(
        fp=changed_fp,
        fresh_fp=fresh_fp,
        bitwise_equal_fresh=changed_fp == fresh_fp,
        differs_from_base=changed_fp != cold_fp,
    )

    record.update(_dispatch_contracts(counters=counters))
    args.out.write_text(json.dumps(record, indent=1, default=str))
    summary_keys = ("label", "model", "arm", "perturb", "warm_median_ms", "cold_fp")
    summary = {key: record[key] for key in summary_keys}
    print(json.dumps({**summary, "contract_digest": record["contract_digest"]}))
    print("count_calls", [call["counts"] for call in record["count_calls"]])


if __name__ == "__main__":
    main()
