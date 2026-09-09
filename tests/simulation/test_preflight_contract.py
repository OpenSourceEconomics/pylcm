"""Host-read and warm-time contracts of the real public simulation preflight.

The private JAX property and NumPy conversion functions count the Python host
materialization routes used by the complete preflight, including cached requests.
The instrument does not measure physical DMA or prove a blocking wait occurred;
unknown native conversion routes remain outside it. Timing is measured separately.
"""

import functools
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from jax._src.array import ArrayImpl

import _lcm
import _lcm.simulation.initial_conditions as initial_module
import lcm.model
from benchmarks.asv._compile_counters import count_compile_requests
from benchmarks.asv._simulation_witnesses import WITNESSES
from lcm.exceptions import InvalidInitialConditionsError
from tests.simulation.test_compile_requests import (
    HOST_TIME_BAR,
    HOST_TIME_REPEATS,
    _lcm_log_output_held_fixed,
    _median_host_times,
)


@dataclass
class _Requests:
    """Host request records scoped to the actual Model-bound preflight."""

    active: str = "control"
    conversion_depth: int = 0
    calls: dict[str, int] = field(default_factory=dict)
    records: list[tuple[str, tuple[int, ...], bool]] = field(default_factory=list)
    sites: list[tuple[str, ...]] = field(default_factory=list)

    def record(self, array: jax.Array) -> None:
        """Keep attributable read sites separate from uninstrumented timing."""
        self.records.append(
            (
                self.active,
                tuple(array.shape),
                array._npy_value is not None,  # ty: ignore[unresolved-attribute]
            )
        )
        self.sites.append(
            tuple(
                f"{frame.filename}:{frame.lineno}:{frame.name}"
                for frame in traceback.extract_stack(limit=24)
                if "/src/_lcm/" in frame.filename
                or "/src/lcm/" in frame.filename
                or "/jax/" in frame.filename
                or "/tests/test_models/" in frame.filename
            )
        )


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> _Requests:
    """Wrap the real host-array property and live preflight entry point."""
    source = Path(_lcm.__file__).resolve()
    assert source.is_relative_to(Path(__file__).resolve().parents[2] / "src")
    capture = _Requests()
    original = ArrayImpl._value

    def read(array: jax.Array) -> np.ndarray:
        if capture.conversion_depth == 0:
            capture.record(array)
        return original.fget(array)

    monkeypatch.setattr(ArrayImpl, "_value", property(read))

    def wrap_converter(converter: Callable[..., Any]) -> Callable[..., Any]:
        def convert(value: Any, *args: Any, **kwargs: Any) -> Any:
            if not isinstance(value, jax.Array):
                return converter(value, *args, **kwargs)
            if capture.conversion_depth == 0:
                capture.record(value)
            capture.conversion_depth += 1
            try:
                return converter(value, *args, **kwargs)
            finally:
                capture.conversion_depth -= 1

        return convert

    for name in ("asarray", "array"):
        monkeypatch.setattr(np, name, wrap_converter(getattr(np, name)))
    for name in ("validate_simulation_inputs",):
        original_validator = getattr(lcm.model, name)

        def observe(
            *, name: str, validator: Callable[..., None], **kwargs: Any
        ) -> None:
            previous = capture.active
            capture.active = name
            capture.calls[name] = capture.calls.get(name, 0) + 1
            try:
                validator(**kwargs)
            finally:
                capture.active = previous

        monkeypatch.setattr(
            lcm.model,
            name,
            functools.partial(observe, name=name, validator=original_validator),
        )
    return capture


@pytest.mark.parametrize(
    "route", ["bool", "int", "float", "item", "tolist", "asarray", "array"]
)
def test_materialization_detector_observes_each_used_route(
    *, requests: _Requests, route: str
) -> None:
    """Both Python scalar conversions and NumPy's native fast path are observed."""
    value = jax.numpy.asarray(1)
    converters = {
        "bool": bool,
        "int": int,
        "float": float,
        "item": lambda array: array.item(),
        "tolist": lambda array: array.tolist(),
        "asarray": np.asarray,
        "array": np.array,
    }
    requests.records.clear()
    actual = converters[route](value)
    assert len(requests.records) == 1
    assert requests.records[0][:2] == ("control", ())
    assert actual == 1


@pytest.mark.parametrize("witness", sorted(WITNESSES))
def test_valid_whole_call_preflight_has_two_host_summaries(
    *,
    witness: str,
    requests: _Requests,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Preserving early semantic guards permits two summaries per valid call."""
    for name in (
        "_read_initial_cohorts",
        "_collect_feasibility_errors",
        "validate_transitions",
    ):
        operation = getattr(initial_module, name)

        def require_valid_fast_path(
            *, operation: Callable[..., Any], **kwargs: Any
        ) -> Any:
            try:
                return operation(**kwargs)
            except Exception as error:  # noqa: BLE001 - expose every fallback cause
                pytest.fail(
                    f"Valid preflight unexpectedly requested fallback: {error!r}"
                )

        monkeypatch.setattr(
            initial_module,
            name,
            functools.partial(require_valid_fast_path, operation=operation),
        )
    model, params, initial = WITNESSES[witness]()
    with _lcm_log_output_held_fixed():
        solution = model.solve(params=params, log_level="off")
        warm = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            seed=0,
            log_level="progress",
        )
        jax.block_until_ready(warm.raw_results)
        requests.records.clear()
        requests.sites.clear()
        requests.calls.clear()
        with count_compile_requests() as compilation:
            result = model.simulate(
                params=params,
                initial_conditions=initial,
                solution=solution,
                seed=0,
                log_level="progress",
            )
            jax.block_until_ready(result.raw_results)
    counts = {
        name: sum(record[0] == name for record in requests.records)
        for name in ("validate_simulation_inputs",)
    }
    record_property("requests", counts)
    record_property(
        "read_sites",
        [
            site
            for record, site in zip(requests.records, requests.sites, strict=True)
            if record[0] == "validate_simulation_inputs"
        ],
    )
    record_property(
        "uncached_requests",
        {
            name: sum(
                record[0] == name and not record[2] for record in requests.records
            )
            for name in counts
        },
    )
    stages = (
        compilation.trace_requests,
        compilation.lowering_requests,
        compilation.compile_requests,
    )
    record_property("compilation_requests", stages)
    assert requests.calls == dict.fromkeys(counts, 1)
    assert stages == (0, 0, 0)
    assert result.n_subjects == len(initial["regime_id"])
    assert sum(counts.values()) <= 2, counts


@pytest.mark.parametrize("witness", sorted(WITNESSES))
def test_unstubbed_warm_full_call_progress_meets_existing_time_bar(
    *, witness: str, record_property: Callable[[str, object], None]
) -> None:
    """Keep the original paired nine-repeat whole-call 1.5x acceptance bar."""
    start = time.perf_counter()
    measurement = _median_host_times(
        witness=witness,
        log_level="progress",
        repeats=HOST_TIME_REPEATS,
        stub_preflight=False,
    )
    off, progress = measurement.off_seconds, measurement.progress_seconds
    record_property("off_ms", off * 1000)
    record_property("progress_ms", progress * 1000)
    record_property("progress_over_off", progress / off)
    record_property("measurement_seconds", time.perf_counter() - start)
    assert progress / off <= HOST_TIME_BAR


def test_invalid_regime_wins_before_missing_state_or_transition_work(
    requests: _Requests,
) -> None:
    """A known invalid specimen activates validation and preserves error order."""
    model, params, initial = WITNESSES["multi_regime"]()
    invalid = dict(initial)
    invalid["regime_id"] = jax.numpy.full_like(initial["regime_id"], 999)
    del invalid["wealth"]
    assert np.asarray(invalid["regime_id"]).tolist() == [999] * 7
    requests.records.clear()
    requests.calls.clear()
    with (
        _lcm_log_output_held_fixed(),
        pytest.raises(
            InvalidInitialConditionsError, match="Invalid regime IDs \\[999\\]"
        ),
    ):
        model.simulate(
            params=params, initial_conditions=invalid, seed=0, log_level="debug"
        )
    assert requests.calls == {"validate_simulation_inputs": 1}
    assert any(row[0] == "validate_simulation_inputs" for row in requests.records)
