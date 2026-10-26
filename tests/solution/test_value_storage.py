"""`ExecutionConfig.value_storage="host"` keeps solved values in host memory.

The values, the reads and the simulated panel are those of the default device
storage, bit for bit. What changes is where a solved period's values live once no
period still to be solved reads them: in pinned host memory instead of on the
device. During the solve the device holds the resident window only, the values the
running period reads, while under device storage every solved period stays.

The models are the grid-search models of `test_period_target_value_reads`, whose
value arrays have two axes and so cannot be mistaken for a one-dimensional grid.
CPU devices have a pinned host memory kind too, so the copy path runs here as on
an accelerator.
"""

from collections.abc import Callable
from typing import Unpack

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.solution import backward_induction
from lcm import ExecutionConfig, InvariantBlockSchedule, Model
from tests.solution._callback_types import (
    RunPeriodKernelKwargs,
    RunPeriodKernelResult,
)
from tests.solution.test_period_target_value_reads import (
    _gated_model,
    _GatedRegimeId,
    _RegimeId,
    _three_regime_model,
)

_PARAMS = {"discount_factor": 0.95}

_MODELS = pytest.mark.parametrize(
    "build_model",
    [_three_regime_model, _gated_model],
    ids=["three-regime", "gated-edge"],
)


def _host() -> ExecutionConfig:
    return ExecutionConfig(value_storage="host")


def test_value_storage_defaults_to_the_device() -> None:
    """Without a choice, solved values stay on the device."""
    assert ExecutionConfig().value_storage == "device"


def test_value_storage_refuses_an_unknown_target() -> None:
    """Only the device and the host are storage targets."""
    with pytest.raises(BeartypeCallHintParamViolation, match="value_storage"):
        ExecutionConfig(value_storage="disk")  # ty: ignore[invalid-argument-type]


def test_host_value_storage_refuses_the_block_major_schedule() -> None:
    """Block-major already moves its values to the host on its own route."""
    with pytest.raises(ValueError, match="value_storage"):
        ExecutionConfig(
            value_storage="host",
            invariant_block_schedule=InvariantBlockSchedule.BLOCK_MAJOR,
        )


@_MODELS
def test_host_storage_publishes_the_device_storage_values(
    *, build_model: Callable[[ExecutionConfig], Model]
) -> None:
    """Every (period, regime) value is bitwise the one device storage publishes."""
    device = build_model(ExecutionConfig()).solve(params=_PARAMS, log_level="off")
    host = build_model(_host()).solve(params=_PARAMS, log_level="off")

    assert {p: set(v) for p, v in host.values.items()} == {
        p: set(v) for p, v in device.values.items()
    }
    for period, regime_to_value in device.values.items():
        for regime_name, value in regime_to_value.items():
            np.testing.assert_array_equal(
                np.asarray(host.value(period=period, regime=regime_name)),
                np.asarray(value),
                strict=True,
            )


@_MODELS
def test_host_storage_reads_a_value_on_the_device(
    *, build_model: Callable[[ExecutionConfig], Model]
) -> None:
    """A read hands out a device array, as under device storage."""
    solution = build_model(_host()).solve(params=_PARAMS, log_level="off")
    for period, regime_to_value in solution.values.items():
        for regime_name in regime_to_value:
            value = solution.value(period=period, regime=regime_name)
            assert value.sharding.memory_kind == "device"


def _three_regime_initial_conditions() -> dict[str, jax.Array]:
    return {
        "regime_id": jnp.array([_RegimeId.a, _RegimeId.a, _RegimeId.b]),
        "age": jnp.array([0.0, 0.0, 1.0]),
        "wealth": jnp.array([1.0, 3.5, 2.0]),
        "health": jnp.array([0, 1, 1]),
    }


def _gated_initial_conditions() -> dict[str, jax.Array]:
    return {
        "regime_id": jnp.array([_GatedRegimeId.src, _GatedRegimeId.src]),
        "age": jnp.array([0.0, 0.0]),
        "wealth": jnp.array([1.0, 4.5]),
        "health": jnp.array([1, 0]),
    }


@pytest.mark.parametrize(
    ("build_model", "initial_conditions"),
    [
        (_three_regime_model, _three_regime_initial_conditions),
        (_gated_model, _gated_initial_conditions),
    ],
    ids=["three-regime", "gated-edge"],
)
def test_host_storage_simulates_the_device_storage_panel(
    *,
    build_model: Callable[[ExecutionConfig], Model],
    initial_conditions: Callable[[], dict[str, jax.Array]],
) -> None:
    """A panel simulated from host-stored values is the device-storage panel."""
    panels = [
        build_model(config)
        .simulate(
            params=_PARAMS,
            initial_conditions=initial_conditions(),
            seed=3,
            log_level="off",
        )
        .to_dataframe()
        for config in (ExecutionConfig(), _host())
    ]

    assert panels[0].equals(panels[1])


def _value_periods_on_the_device_at_each_kernel(
    *, monkeypatch: pytest.MonkeyPatch, model: Model
) -> list[tuple[int, frozenset[int]]]:
    """Return, per kernel start, its period and the periods of live device values."""
    period_of: dict[int, int] = {}
    samples: list[tuple[int, frozenset[int]]] = []
    run_period_kernel = backward_induction._run_period_kernel

    def _run(**kwargs: Unpack[RunPeriodKernelKwargs]) -> RunPeriodKernelResult:
        samples.append(
            (
                kwargs["period"],
                frozenset(
                    period_of[pointer]
                    for array in jax.live_arrays()
                    if array.sharding.memory_kind == "device"
                    and (pointer := array.unsafe_buffer_pointer()) in period_of
                ),
            )
        )
        output = run_period_kernel(**kwargs)
        assert isinstance(output.value, jax.Array)
        period_of[output.value.unsafe_buffer_pointer()] = kwargs["period"]
        return output

    with monkeypatch.context() as patch:
        patch.setattr(backward_induction, "_run_period_kernel", _run)
        model.solve(params=_PARAMS, log_level="off")
    return samples


def test_host_storage_keeps_only_the_resident_window_on_the_device(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kernel finds on the device only values of its period and the two after.

    Its own period holds the values its peers already solved, the next period the
    values it reads, and the one after may still be moving to the host. Under
    device storage the check fires: every solved period is resident.
    """
    host = _value_periods_on_the_device_at_each_kernel(
        monkeypatch=monkeypatch, model=_three_regime_model(_host())
    )
    device = _value_periods_on_the_device_at_each_kernel(
        monkeypatch=monkeypatch, model=_three_regime_model(ExecutionConfig())
    )

    def _within_window(*, resident: frozenset[int], period: int) -> bool:
        return resident <= {period, period + 1, period + 2}

    assert all(
        _within_window(resident=resident, period=period) for period, resident in host
    ), host
    assert not all(
        _within_window(resident=resident, period=period) for period, resident in device
    ), device
