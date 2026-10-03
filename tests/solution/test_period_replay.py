"""One regime-period can be captured during a solve and re-run on its own.

Diagnosing a kernel that is slow, or whose allocation is refused, otherwise costs a
full backward induction: every period above the one in question has to be solved
before the interesting one is reached. Capturing the inputs of a single regime-period
turns that into one kernel invocation.

The capture is written from the funnel every regime-period passes through, so what is
replayed is what ran — not a reconstruction that might differ from it.
"""

import json
import logging
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cloudpickle
import jax
import numpy as np
import pytest

from _lcm.execution.output_layout import PlannedCore
from _lcm.execution.workspace_planning import _tiled_bootstrap_cap, bootstrap_width
from _lcm.solution import backward_induction, period_replay, public_period_capture
from lcm import AgeGrid, ExecutionConfig, LinSpacedGrid, Model
from lcm.persistence import PeriodCapture, load_period_capture, replay_period
from lcm.solver_api import ResultRetention
from tests.regime_building.test_gated_edges_collective_solve import (
    EKLRegimeId,
    _make_full_topology_regimes,
)
from tests.test_models.deterministic import base as retirement_model
from tests.test_models.deterministic.discrete import (
    RegimeId,
    get_model,
    get_params,
)
from tests.test_models.initial_regimes import initial_regimes_of

_N_PERIODS = 3


def _solve_capturing(*, monkeypatch, tmp_path, target: str | None):
    """Solve the discrete toy, optionally capturing one regime-period."""
    if target is None:
        monkeypatch.delenv("LCM_CAPTURE_PERIOD", raising=False)
    else:
        monkeypatch.setenv("LCM_CAPTURE_PERIOD", target)
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    base = get_model(n_periods=_N_PERIODS)
    model = Model(
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        initial_regimes=initial_regimes_of(model=base),
    )
    params = get_params(n_periods=_N_PERIODS)
    solution = model.solve(params=params, log_level="off").values
    return model, solution


def test_no_capture_is_written_without_the_target(*, monkeypatch, tmp_path):
    """The instrument costs nothing until a regime-period is named."""
    _solve_capturing(monkeypatch=monkeypatch, tmp_path=tmp_path, target=None)
    assert list(tmp_path.iterdir()) == []


def test_exactly_the_named_regime_period_is_captured(*, monkeypatch, tmp_path):
    """Naming one regime-period writes one capture, not one per period."""
    _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    captures = sorted(path.name for path in tmp_path.iterdir())
    assert captures == ["working_life@1"]


def test_a_capture_names_the_regime_period_and_age_it_holds(*, monkeypatch, tmp_path):
    """The capture is self-describing, so a directory of them can be read."""
    model, _ = _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    replay = replay_period(directory=tmp_path / "working_life@1")
    assert (replay.regime_name, replay.period, replay.age) == (
        "working_life",
        1,
        float(model.ages.values[1]),
    )


def test_replay_reproduces_the_value_function_of_the_full_solve(
    *, monkeypatch, tmp_path
):
    """Re-running the captured kernel gives back the array the solve produced.

    This is what makes the harness usable as a stand-in for the full run: the
    replayed kernel is the same computation on the same inputs, so a measurement
    taken on it transfers.
    """
    _, solution = _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    replay = replay_period(directory=tmp_path / "working_life@1")

    np.testing.assert_array_equal(
        np.asarray(replay.output.value),
        np.asarray(solution[1]["working_life"]),
    )


def test_replay_does_not_recapture_when_the_selector_remains_set(
    *, monkeypatch, tmp_path
):
    """Replay reads a selected capture without writing into its directory."""
    _, solution = _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    capture = tmp_path / "working_life@1"
    capture.chmod(0o555)
    try:
        replay = replay_period(directory=capture)
    finally:
        capture.chmod(0o755)

    np.testing.assert_array_equal(
        np.asarray(replay.output.value),
        np.asarray(solution[1]["working_life"]),
    )


def test_replay_runs_one_kernel_and_no_others(*, monkeypatch, tmp_path, caplog):
    """Replay does not re-solve the periods above the captured one."""
    _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    monkeypatch.setenv("LCM_LOG_KERNEL_ATTRIBUTION", "1")
    with caplog.at_level(logging.NOTSET, logger="lcm"):
        replay_period(directory=tmp_path / "working_life@1")

    executed = [
        record.getMessage()
        for record in caplog.records
        if re.search(r"\[attr\] \S+ age", record.getMessage())
    ]
    assert len(executed) == 1


def test_an_unknown_regime_period_is_refused(*, monkeypatch, tmp_path):
    """A target that never runs produces no capture rather than an empty one."""
    _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@99"
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("target", ["working_life", "working_life@", "@1", "x@y"])
def test_a_malformed_target_is_rejected_loudly(*, monkeypatch, tmp_path, target):
    """A typo in the target must not read as "nothing to capture"."""
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", target)
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model = get_model(n_periods=_N_PERIODS)
    params = get_params(n_periods=_N_PERIODS)
    with pytest.raises(ValueError, match="LCM_CAPTURE_PERIOD"):
        model.solve(params=params, log_level="off")


def test_a_malformed_target_is_rejected_before_kernel_compilation(
    *, monkeypatch, tmp_path
):
    """Capture selection is validated before solve kernels are compiled."""
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", "working_life")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))

    def fail_if_compilation_starts(*_args: object, **_kwargs: object) -> None:
        msg = "kernel compilation started"
        raise AssertionError(msg)

    monkeypatch.setattr(
        backward_induction, "_compile_all_functions", fail_if_compilation_starts
    )
    model = get_model(n_periods=_N_PERIODS)
    params = get_params(n_periods=_N_PERIODS)

    with pytest.raises(ValueError, match="LCM_CAPTURE_PERIOD"):
        model.solve(params=params, log_level="off")


def test_a_gated_edge_source_replays_to_the_value_the_solve_published(
    *, monkeypatch, tmp_path
):
    """A regime whose continuation reads a gated edge replays to the solved `V_arr`.

    The source's kernel does not read its target's raw value function: it reads the
    gated continuation folded onto the target's grid. Replay must hand the kernel
    that same object, so a captured gated-edge period returns the array the solve
    published rather than refusing at lowering.
    """
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", "single_f@0")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model = Model(
        regimes=_make_full_topology_regimes(),
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=EKLRegimeId,
        initial_regimes={0: ("single_f", "single_m")},
    )
    params = {"discount_factor": 0.95, "delta_f": 0.5, "delta_m": 0.2}
    solution = model.solve(params=params, log_level="off").values

    replay = replay_period(directory=tmp_path / "single_f@0")

    np.testing.assert_array_equal(
        np.asarray(replay.output.value),
        np.asarray(solution[0]["single_f"]),
    )


@pytest.mark.parametrize(
    ("retention", "retain_replay"),
    [
        pytest.param(ResultRetention.VALUES, False, id="values"),
        pytest.param(ResultRetention.VALUES_AND_REPLAY, True, id="values-and-replay"),
    ],
)
def test_replay_lowers_the_scope_the_solve_dispatched(
    *, monkeypatch, tmp_path, retention, retain_replay
):
    """A replay selects the programs the captured solve dispatched for the regime.

    The solve dispatches a regime's replay-scoped programs only when the retention
    keeps replay artifacts and the regime declares a route that consumes them; the
    capture records that decision and the replay lowers exactly that scope.
    """
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", "working_life@1")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model = get_model(n_periods=_N_PERIODS)
    params = get_params(n_periods=_N_PERIODS)
    model.solve(params=params, log_level="off", retention=retention)
    regime_retains_replay = (
        retain_replay
        and model._regimes["working_life"].simulation.egm_policy_read is not None
    )

    observed: list[tuple[bool, frozenset[object]]] = []
    real_select = period_replay.select_programs

    def record_select(**kwargs):
        observed.append((kwargs["retain_replay"], kwargs["selected_artifact_keys"]))
        return real_select(**kwargs)

    monkeypatch.setattr(period_replay, "select_programs", record_select)
    replay_period(directory=tmp_path / "working_life@1")

    assert observed == [(regime_retains_replay, frozenset())]


def _rewrite_capture(*, directory, mutate):
    """Load, mutate, and rewrite one capture payload."""
    path = directory / period_replay._PAYLOAD_NAME
    with path.open("rb") as stream:
        payload = cloudpickle.load(stream)
    mutate(payload)
    with path.open("wb") as stream:
        cloudpickle.dump(payload, stream)


def test_a_capture_records_the_tile_widths_the_solve_dispatched(
    *, monkeypatch, tmp_path
):
    """The captured widths are the bootstrap widths of the unbudgeted solve."""
    model, _ = _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    path = tmp_path / "working_life@1" / period_replay._PAYLOAD_NAME
    with path.open("rb") as stream:
        payload = cloudpickle.load(stream)

    regime = model._regimes["working_life"]
    space = regime.solution.state_action_space(
        regime_params=model._process_params(get_params(n_periods=_N_PERIODS))[
            "working_life"
        ]
    )
    action_extent = math.prod(space.actions_grid_shapes)
    cell_extent = math.prod(nodes.size for nodes in space.states.values())
    action_width = bootstrap_width(extent=action_extent)
    assert payload["core_tile_widths"] == {
        "main": {
            "action_product": action_width,
            "cell": bootstrap_width(
                extent=cell_extent, cap=_tiled_bootstrap_cap(block=action_width)
            ),
        }
    }


def test_a_capture_without_tile_widths_is_refused(*, monkeypatch, tmp_path):
    """Replay never plans widths itself, so a capture must state them."""
    _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    _rewrite_capture(
        directory=tmp_path / "working_life@1",
        mutate=lambda payload: payload.pop("core_tile_widths"),
    )

    with pytest.raises(ValueError, match="core_tile_widths"):
        replay_period(directory=tmp_path / "working_life@1")


@pytest.mark.parametrize(
    ("widths", "error"),
    [
        ({"main": {"action_product": 0}}, ValueError),
        ({"main": {"action_product": 2.0}}, TypeError),
        ({"other": {"action_product": 2}}, ValueError),
    ],
    ids=["nonpositive-width", "noninteger-width", "wrong-core-names"],
)
def test_malformed_captured_tile_widths_are_refused(
    *, monkeypatch, tmp_path, widths, error
):
    _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    _rewrite_capture(
        directory=tmp_path / "working_life@1",
        mutate=lambda payload: payload.__setitem__("core_tile_widths", widths),
    )

    with pytest.raises(error):
        replay_period(directory=tmp_path / "working_life@1")


@pytest.mark.parametrize("budget", [None, 32 * 1024**2])
@pytest.mark.parametrize("persistent_compilation_cache", [False], indirect=True)
@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_public_solve_captures_adjacent_periods_for_fresh_model_replay(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: int | None,
    caplog: pytest.LogCaptureFixture,
    persistent_compilation_cache: bool,
) -> None:
    """A fresh public model reproduces completed captured values bit for bit."""
    assert jax.config.jax_enable_compilation_cache is persistent_compilation_cache
    monkeypatch.setenv("LCM_LOG_KERNEL_ATTRIBUTION", "1")
    capture = PeriodCapture(
        directory=tmp_path,
        periods=(("retirement", 1), ("working_life", 0)),
        source_identity={"model": "tiny-public-model-v1"},
    )
    params = retirement_model.get_params(n_periods=_N_PERIODS)
    model = _make_public_capture_model(budget=budget)
    result = model.solve(
        params=params,
        log_level="off",
        period_capture=capture,
    )
    fresh = _make_public_capture_model(budget=budget)
    for regime_name, period in capture.periods:
        caplog.clear()
        replay = fresh.replay_period(
            directory=tmp_path / f"{regime_name}@{period}",
            params=params,
            source_identity={"model": "tiny-public-model-v1"},
        )
        np.testing.assert_array_equal(
            np.asarray(replay.value).view(np.uint8),
            np.asarray(result.values[period][regime_name]).view(np.uint8),
        )
        assert replay.reference_matches is True
        assert replay.optimized_hlo_matches is True
        assert replay.in_context_seconds is not None
        assert replay.in_context_seconds > 0
        assert replay.replay_seconds > 0
        executed = [
            record.getMessage()
            for record in caplog.records
            if re.search(r"\[attr\] \S+ age", record.getMessage())
        ]
        assert len(executed) == 1
        assert f"[attr] {regime_name} age" in executed[0]
        assert f"period {period}:" in executed[0]
        for mask in (np.isnan, np.isposinf, np.isneginf):
            np.testing.assert_array_equal(
                mask(np.asarray(replay.value)),
                mask(np.asarray(result.values[period][regime_name])),
            )
    assert not tuple(tmp_path.rglob("*.pkl"))

    def reject_compilation(**_kwargs: object) -> None:
        raise AssertionError("An incompatible capture reached compilation")

    monkeypatch.setattr(
        period_replay, "_compile_cores_for_one_period", reject_compilation
    )
    for incompatible_model, incompatible_params, source in (
        (fresh, {**params, "discount_factor": 0.9}, capture.source_identity),
        (
            _make_public_capture_model(wealth_stop=5, budget=budget),
            params,
            capture.source_identity,
        ),
        (
            _make_public_capture_model(axis_widths={"cell": 1}, budget=budget),
            params,
            capture.source_identity,
        ),
        (fresh, params, {"model": "different-source"}),
    ):
        with pytest.raises(ValueError, match=r"identity|incompatible"):
            incompatible_model.replay_period(
                directory=tmp_path / "working_life@0",
                params=incompatible_params,
                source_identity=source,
            )


@pytest.mark.parametrize("persistent_compilation_cache", [False], indirect=True)
@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_public_interrupted_capture_has_inputs_without_a_reference(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persistent_compilation_cache: bool,
) -> None:
    """An interruption after entry leaves readable inputs and no parity claim."""
    assert persistent_compilation_cache is False
    capture = PeriodCapture(
        directory=tmp_path,
        periods=(("working_life", 0),),
        source_identity={"model": "tiny-public-model-v1"},
    )
    monkeypatch.setenv("LCM_LOG_KERNEL_ATTRIBUTION", "1")
    interrupt = _InterruptCapturedPeriod()
    logger = logging.getLogger("lcm")
    logger.addFilter(interrupt)
    params = retirement_model.get_params(n_periods=_N_PERIODS)
    try:
        with pytest.raises(RuntimeError, match="Interrupted captured entry"):
            _make_public_capture_model().solve(
                params=params,
                log_level="off",
                period_capture=capture,
            )
    finally:
        logger.removeFilter(interrupt)
    directory = tmp_path / "working_life@0"
    record = load_period_capture(directory=directory)
    assert record.completed is False
    assert record.reference is None
    fresh = _make_public_capture_model()
    with pytest.raises(ValueError, match="completed reference"):
        fresh.replay_period(
            directory=directory,
            params=params,
            source_identity=capture.source_identity,
        )
    replay = fresh.replay_period(
        directory=directory,
        params=params,
        source_identity=capture.source_identity,
        require_reference=False,
    )
    assert replay.reference_matches is None
    assert replay.in_context_seconds is None


@pytest.mark.requires(device="gpu")
@pytest.mark.coverage(backends=("gpu-small", "gpu-large"), precisions="both")
def test_public_capture_from_separately_warmed_gpu_cache_requires_native_metadata(
    *, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A selected native cache hit requires actual buffer-assignment evidence."""
    stage = os.environ.get("PUBLIC_REPLAY_CACHE_STAGE")
    if stage is None:
        pytest.skip("Requires the serial warm/capture CI process pair")
    assert stage in {"warm", "capture"}
    root = Path(os.environ["PUBLIC_REPLAY_CACHE_EVIDENCE"])
    root.mkdir(parents=True, exist_ok=True)
    assert os.environ["JAX_ENABLE_COMPILATION_CACHE"] == "true"
    assert jax.config.jax_enable_compilation_cache is True
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == -1
    caplog.set_level(logging.DEBUG, logger="jax._src.compilation_cache")
    caplog.set_level(logging.DEBUG, logger="jax._src.compiler")
    params = retirement_model.get_params(n_periods=_N_PERIODS)
    if stage == "warm":
        _make_public_capture_model().solve(params=params, log_level="off")
        writes = [
            match.groups()
            for record in caplog.records
            if record.name == "jax._src.compilation_cache"
            and (
                match := re.fullmatch(
                    r"Writing (\S+) to persistent compilation cache "
                    r"with key '([^']+)'",
                    record.getMessage(),
                )
            )
        ]
        assert writes
        (root / "warm.json").write_text(
            json.dumps({"pid": os.getpid(), "writes": writes})
        )
        return

    warmed = json.loads((root / "warm.json").read_text())
    assert warmed["pid"] != os.getpid()
    observed: dict[str, Any] = {}
    original = public_period_capture.optimized_hlo_records

    def observe(*, compiled_cores: Mapping[str, PlannedCore]) -> dict[str, Any]:
        for name, core in compiled_cores.items():
            assert isinstance(core.compiled, jax.stages.Compiled)
            executable = core.compiled.runtime_executable()
            assert executable is not None
            memory = core.compiled.memory_analysis()
            proto = getattr(memory, "serialized_buffer_assignment_proto", None)
            observed[name] = {
                "modules": [module.name for module in executable.hlo_modules()],
                "proto_type": type(proto).__name__,
                "proto_bytes": len(proto) if isinstance(proto, bytes) else None,
                "memory": str(memory),
            }
        return original(compiled_cores=compiled_cores)

    monkeypatch.setattr(public_period_capture, "optimized_hlo_records", observe)
    monkeypatch.setenv("LCM_LOG_KERNEL_ATTRIBUTION", "1")
    interrupt = _InterruptCapturedPeriod()
    logger = logging.getLogger("lcm")
    logger.addFilter(interrupt)
    capture = PeriodCapture(
        directory=root / "capture",
        periods=(("working_life", 0),),
        source_identity={"model": "tiny-public-model-v1"},
    )
    try:
        with pytest.raises(ValueError, match="serialized_buffer_assignment_proto"):
            _make_public_capture_model().solve(
                params=params, log_level="off", period_capture=capture
            )
    finally:
        logger.removeFilter(interrupt)
        (root / "capture.json").write_text(
            json.dumps({"pid": os.getpid(), "observed": observed})
        )
        (root / "capture.log").write_text(caplog.text)
    assert observed
    for record in observed.values():
        assert record["proto_bytes"] in {None, 0}
        assert record["modules"]
        for name in record["modules"]:
            assert any(
                module == name
                and any(
                    log.getMessage()
                    == f"Persistent compilation cache hit for '{name}' with key {key!r}"
                    for log in caplog.records
                    if log.name == "jax._src.compiler"
                )
                for module, key in warmed["writes"]
            )
    assert not (capture.directory / "working_life@0" / "entry.h5").exists()


@pytest.mark.parametrize("operation", ["capture", "replay"])
@pytest.mark.parametrize(
    ("persistent_compilation_cache", "cache_environment"),
    [(True, "false"), (False, None)],
    indirect=["persistent_compilation_cache"],
)
@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_public_period_refuses_unsupported_cache_mode(
    *,
    operation: str,
    persistent_compilation_cache: bool,
    cache_environment: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unsupported modes before capture preparation or archive loading."""
    assert jax.config.jax_enable_compilation_cache is persistent_compilation_cache
    if cache_environment is None:
        monkeypatch.delenv("JAX_ENABLE_COMPILATION_CACHE", raising=False)
    else:
        monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", cache_environment)
    model = _make_public_capture_model()
    params = retirement_model.get_params(n_periods=_N_PERIODS)
    capture = PeriodCapture(
        directory=tmp_path / "absent",
        periods=(("working_life", 0),),
        source_identity={"model": "tiny-public-model-v1"},
    )

    def reject_preparation(**_kwargs: object) -> None:
        raise AssertionError("Unsupported cache mode reached capture preparation")

    monkeypatch.setattr(public_period_capture, "period_identity", reject_preparation)
    if operation == "capture":
        with pytest.raises(ValueError, match="JAX_ENABLE_COMPILATION_CACHE=false"):
            model.solve(params=params, log_level="off", period_capture=capture)
    else:
        with pytest.raises(ValueError, match="JAX_ENABLE_COMPILATION_CACHE=false"):
            model.replay_period(
                directory=capture.directory,
                params=params,
                source_identity=capture.source_identity,
            )
    assert not capture.directory.exists()


@pytest.mark.parametrize(("nested_value", "equal"), [(3, True), (4, False)])
def test_optimized_hlo_identity_preserves_backend_configuration_values(
    *, nested_value: int, equal: bool
) -> None:
    """JSON member ordering is irrelevant while backend configuration values bind."""
    canonicalize = getattr(public_period_capture, "_canonicalize_optimized_hlo", None)
    assert canonicalize is not None
    original = (
        'ROOT x = f32[] constant(1), backend_config={"z":1,"nested":{"b":2,"a":3}}\n'
    )
    reordered = (
        'ROOT x = f32[] constant(1), backend_config={"nested":{"a":'
        f'{nested_value},"b":2}},"z":1}}  \n\n'
    )
    assert (canonicalize(original) == canonicalize(reordered)) is equal


def test_optimized_hlo_identity_preserves_arrays_and_following_attributes() -> None:
    """JSON normalization retains arrays, literal strings and other HLO fields."""
    canonicalize = getattr(public_period_capture, "_canonicalize_optimized_hlo", None)
    assert canonicalize is not None
    prefix = 'ROOT x = f32[] constant(1), metadata={op_name="backend_config=01"}, '
    suffix = ', frontend_attributes={"keep":"yes"}'
    original = (
        prefix
        + 'backend_config={"z":[true, null, "01", {"b":2,"a":1}], "a":0.1}'
        + suffix
    )
    expected = (
        prefix + 'backend_config={"a":0.1,"z":[true,null,"01",{"a":1,"b":2}]}' + suffix
    )
    assert canonicalize(original) == expected


@pytest.mark.parametrize(
    ("configuration", "message"),
    [
        ('{"missing":', "Expecting value"),
        ('{"x":1,"x":2}', "Duplicate"),
        ('{"x":1,"\\u0078":2}', "Duplicate"),
        ('{"x":NaN}', "Nonstandard"),
        ('{"x":Infinity}', "Nonstandard"),
        ('{"x":-Infinity}', "Nonstandard"),
        ("01", "Malformed backend JSON boundary"),
        ("1e", "Malformed backend JSON boundary"),
        ('{"x":1}garbage', "Malformed backend JSON boundary"),
    ],
)
def test_optimized_hlo_identity_refuses_malformed_backend_configuration(
    *, configuration: str, message: str
) -> None:
    """Malformed backend JSON must not disappear during canonicalization."""
    canonicalize = getattr(public_period_capture, "_canonicalize_optimized_hlo", None)
    assert canonicalize is not None
    with pytest.raises(ValueError, match=message):
        canonicalize(f"ROOT x = f32[] constant(1), backend_config={configuration}")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("0.123456789012345678901", "0.123456789012345678902"),
        ("1e10000", "2e10000"),
        ("-0", "0"),
    ],
)
def test_optimized_hlo_identity_preserves_numeric_tokens(
    *, left: str, right: str
) -> None:
    """Ordering normalization cannot erase numeric distinctions or spelling."""
    canonicalize = getattr(public_period_capture, "_canonicalize_optimized_hlo", None)
    assert canonicalize is not None
    prefix = 'ROOT x = f32[] constant(1), backend_config={"x":'
    assert canonicalize(prefix + left + "}") != canonicalize(prefix + right + "}")


class _InterruptCapturedPeriod(logging.Filter):
    """Interrupt the real public solve immediately before the selected dispatch."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "[attr] working_life age" in message and "period 0:" in message:
            raise RuntimeError("Interrupted captured entry")
        return True


def _make_public_capture_model(
    *,
    wealth_stop: float = 4,
    axis_widths: dict[str, int] | None = None,
    budget: int | None = None,
) -> Model:
    """Build the public retirement fixture on a small complete candidate grid."""
    base = retirement_model.get_model(n_periods=_N_PERIODS)
    regimes = {
        name: regime.replace(
            states={"wealth": LinSpacedGrid(start=1, stop=wealth_stop, n_points=4)},
            actions={
                **regime.actions,
                "consumption": LinSpacedGrid(start=1, stop=4, n_points=5),
            },
        )
        if name != "dead"
        else regime
        for name, regime in base.user_regimes.items()
    }
    return Model(
        regimes=regimes,
        ages=base.ages,
        regime_id_class=retirement_model.RegimeId,
        execution_config=ExecutionConfig(
            device_memory_bytes=budget, axis_widths=axis_widths or {}
        ),
        initial_regimes=initial_regimes_of(model=base),
    )
