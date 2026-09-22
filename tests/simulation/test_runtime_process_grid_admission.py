"""Composite runtime process supports use exact admitted eager stages."""

import dataclasses
import os
import subprocess
import sys
from collections.abc import Callable
from functools import partialmethod
from pathlib import Path
from typing import Any

import jax
import jax.core
import numpy as np
import pytest

from _lcm.simulation import host_operations, process_grids
from _lcm.simulation.residency import (
    measure_buffer_footprint,
    resident_bytes_by_device,
)
from lcm import AgeGrid, LinSpacedGrid, Model, Regime, RouwenhorstAR1Process
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.execution.test_compiler_allocation_reservation import synthetic_memory
from tests.simulation.test_budget_lifecycle import _LifecycleRegimeId
from tests.simulation.test_process_grid_entry_admission import (
    _COMPOSITE_CASES,
    _inputs,
)

_RUNTIME_CASES = _COMPOSITE_CASES[1:]

_SELECTED_DEVICE_SCRIPT = r"""
import jax
import numpy as np

from _lcm.simulation import host_operations, process_grids
from _lcm.simulation.residency import measure_buffer_footprint
from tests.simulation.test_process_grid_entry_admission import _COMPOSITE_CASES

target = jax.devices()[2]
required = jax.sharding.SingleDeviceSharding(target)
for spec, parameters in _COMPOSITE_CASES[1:]:
    runtime = {
        name: jax.numpy.asarray(value) for name, value in parameters.items()
    }
    expected = np.asarray(spec.compute_gridpoints(**runtime))
    owner = process_grids.SimulationProcessGrids(
        live_footprint=lambda runtime=runtime: measure_buffer_footprint(tree=runtime),
        devices=(target,),
        budget_bytes=2**28,
        operations=host_operations.ProfiledSimulationOperations(),
    )
    actual = owner(spec=spec, parameters=runtime, required=required)
    assert set(actual.devices()) == {target}
    assert (np.asarray(actual).dtype, np.asarray(actual).tobytes()) == (
        expected.dtype,
        expected.tobytes(),
    )
print("RUNTIME-PROCESS-PLACEMENT-OK")
"""


def _controlled_post_validation_refusal(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise ExecutionPlanningError("controlled refusal after process validation")


def _owner(*, parameters: object) -> process_grids.SimulationProcessGrids:
    return process_grids.SimulationProcessGrids(
        live_footprint=lambda: measure_buffer_footprint(tree=parameters),
        devices=(jax.devices()[0],),
        budget_bytes=2**28,
        operations=host_operations.ProfiledSimulationOperations(),
    )


def test_runtime_process_outputs_use_selected_nondefault_device() -> None:
    """Every composite support is ready on the selected CPU device."""
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    }
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SELECTED_DEVICE_SCRIPT],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        timeout=600,
    )

    assert (result.returncode, "RUNTIME-PROCESS-PLACEMENT-OK" in result.stdout) == (
        0,
        True,
    ), result.stderr[-4000:]


@pytest.mark.parametrize(("spec", "parameters"), _RUNTIME_CASES)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("strong_runtime", [False, True])
def test_runtime_process_support_matches_eager_bytes_with_fixed_or_runtime_params(
    *,
    spec: Any,
    parameters: dict[str, float],
    fixed: bool,
    strong_runtime: bool,
) -> None:
    """Every composite family preserves bytes across scalar binding strengths."""
    fixed_names = tuple(parameters)[::2] if fixed else ()
    resolved_spec = dataclasses.replace(
        spec, **{name: parameters[name] for name in fixed_names}
    )
    runtime = {
        name: jax.numpy.asarray(
            value,
            dtype=(process_grids.canonical_float_dtype() if strong_runtime else None),
        )
        for name, value in parameters.items()
        if name not in fixed_names
    }
    eager = np.asarray(
        resolved_spec.compute_gridpoints(**resolved_spec.params, **runtime)
    )
    actual = np.asarray(
        _owner(parameters=runtime)(
            spec=resolved_spec,
            parameters=runtime,
            required=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        )
    )

    assert (actual.dtype, actual.tobytes()) == (eager.dtype, eager.tobytes())


_INTEGER_CASES = (
    (
        dataclasses.replace(_RUNTIME_CASES[0][0], mu=1, n_std=3),
        {"sigma": 2},
    ),
    (
        dataclasses.replace(_RUNTIME_CASES[1][0], n_std=3, p1=0, mu1=1),
        {
            "sigma1": 2,
            "mu2": 4,
            "sigma2": 1,
        },
    ),
    (
        dataclasses.replace(_RUNTIME_CASES[2][0], rho=0, sigma=2),
        {
            "mu": 1,
            "n_std": 3,
        },
    ),
    (
        dataclasses.replace(_RUNTIME_CASES[3][0], rho=0, sigma=2),
        {"mu": 1},
    ),
)


@pytest.mark.parametrize(("spec", "runtime"), _INTEGER_CASES)
def test_integer_capable_process_support_preserves_eager_bytes(
    *, spec: Any, runtime: dict[str, int]
) -> None:
    """Valid strong integer bindings keep eager promotion and support bytes."""
    placed_runtime = {
        name: jax.numpy.asarray(value, dtype=jax.numpy.int32)
        for name, value in runtime.items()
    }
    eager = np.asarray(spec.compute_gridpoints(**spec.params, **placed_runtime))
    actual = np.asarray(
        _owner(parameters=placed_runtime)(
            spec=spec,
            parameters=placed_runtime,
            required=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        )
    )

    assert (actual.dtype, actual.tobytes()) == (eager.dtype, eager.tobytes())


@pytest.mark.parametrize(("spec", "parameters"), _RUNTIME_CASES)
def test_automatic_simulation_consumes_each_admitted_runtime_process(
    *, spec: Any, parameters: dict[str, float]
) -> None:
    """Automatic solve and forward consumers reuse every admitted support."""
    model, params, initial = _inputs(
        budget=2**28,
        companion=spec,
        companion_params=parameters,
    )

    result = model.simulate(
        params=params,
        initial_conditions=initial,
        log_level="debug",
    )
    first = result.to_dataframe(use_labels=False).iloc[0]

    np.testing.assert_array_equal(
        first[["income", "saving", "value"]].to_numpy(dtype=float),
        [2.0, 1.0, 3.0],
    )


def test_runtime_process_support_changes_public_value_and_saving() -> None:
    """A changed two-point support changes the hand-evaluated saving decision.

    With rho=0 and sigma=1, the two support points are mu-1 and mu+1,
    equally weighted. Current assets and companion are flow endowments;
    saving costs 1.5 and pays next period's process value. At initial
    assets=0 and companion=mu, Q(saving)=mu+saving*(mu-1.5). The optimal
    saving/value pairs are (0, 0) at mu=0 and (1, 2.5) at mu=2. Returning to mu=0 must
    recover the first decision even though the model's executors are warm.
    """
    states = {
        "companion": RouwenhorstAR1Process(n_points=2, rho=0.0, sigma=1.0),
        "assets": LinSpacedGrid(start=0, stop=1, n_points=2),
    }
    model = Model(
        regimes={
            "alive": Regime(
                transition=_support_next_regime,
                active=lambda age: age == 0,
                states=states,
                actions={"saving": LinSpacedGrid(start=0, stop=1, n_points=2)},
                state_transitions={"assets": _support_next_assets},
                functions={"utility": _support_current_payoff},
            ),
            "done": Regime(
                transition=None,
                states=states,
                functions={"utility": _support_terminal_payoff},
            ),
        },
        regime_id_class=_LifecycleRegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        execution_config=ExecutionConfig(device_memory_bytes=2**28),
    )
    for mu, expected in ((0.0, [0.0, 0.0]), (2.0, [1.0, 2.5]), (0.0, [0.0, 0.0])):
        result = model.simulate(
            params={"mu": mu, "discount_factor": 1.0},
            initial_conditions={
                "companion": jax.numpy.asarray([mu]),
                "assets": jax.numpy.asarray([0.0]),
                "age": jax.numpy.asarray([0.0]),
                "regime_id": jax.numpy.asarray([_LifecycleRegimeId.alive]),
            },
            seed=0,
            log_level="debug",
        )
        frame = result.to_dataframe(use_labels=False)
        initial_row = frame.loc[frame["period"] == 0]
        assert len(initial_row) == 1
        np.testing.assert_array_equal(
            initial_row[["saving", "value"]].to_numpy(dtype=float)[0], expected
        )


def _support_current_payoff(
    *, saving: ContinuousAction, companion: ContinuousState, assets: ContinuousState
) -> FloatND:
    return assets + companion - 1.5 * saving


def _support_next_assets(*, saving: ContinuousAction) -> FloatND:
    return saving


def _support_terminal_payoff(
    *, companion: ContinuousState, assets: ContinuousState
) -> FloatND:
    return companion * assets


def _support_next_regime() -> ScalarInt:
    return _LifecycleRegimeId.done


# keyword-only-exempt: library-callback=functools.partialmethod
def _inject_process_refusal(
    self: host_operations.ProfiledSimulationOperations,
    *,
    original: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    result = original(self, **kwargs)
    if getattr(kwargs["function"], "__name__", "") == "_compute_process_stage":
        return dataclasses.replace(result, memory=synthetic_memory(2**30))
    return result


@pytest.mark.parametrize(("spec", "parameters"), _RUNTIME_CASES)
def test_runtime_process_refuses_before_completed_fallback_support(
    *,
    spec: Any,
    parameters: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declined first stage prevents the formerly eager fallback producer."""
    completed: list[object] = []
    process_type = type(spec)
    original_gridpoints = process_type.compute_gridpoints

    def observe_gridpoints(self: Any, **kwargs: Any) -> Any:
        result = original_gridpoints(self, **kwargs)
        if not any(
            isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(result)
        ):
            completed.append(jax.block_until_ready(result))
        return result

    monkeypatch.setattr(process_type, "compute_gridpoints", observe_gridpoints)
    monkeypatch.setattr(
        host_operations.ProfiledSimulationOperations,
        "compile_candidate",
        partialmethod(
            _inject_process_refusal,
            original=host_operations.ProfiledSimulationOperations.compile_candidate,
        ),
    )
    monkeypatch.setattr(
        Model, "_solve_from_flat_params", _controlled_post_validation_refusal
    )
    model, params, initial = _inputs(
        budget=2**20,
        companion=spec,
        companion_params=parameters,
    )

    with pytest.raises(ExecutionPlanningError) as error:
        model.simulate(
            params=params,
            initial_conditions=initial,
            log_level="warning",
        )

    assert (completed, "controlled refusal" in str(error.value)) == ([], False)


def test_staged_process_graph_is_preflighted_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown late primitive refuses before an earlier valid stage runs."""
    spec, parameters = _RUNTIME_CASES[0]
    owner = _owner(parameters=parameters)
    dispatched: list[str] = []
    original = process_grids.SimulationProcessGrids._produce

    def produce_and_record(self: Any, **kwargs: Any) -> Any:
        dispatched.append(kwargs["stage"])
        return original(self, **kwargs)

    original_trace = process_grids._trace_process_jaxpr
    unsupported = (
        jax.make_jaxpr(jax.numpy.sin)(
            jax.ShapeDtypeStruct(
                (spec.n_points,), process_grids.canonical_float_dtype()
            )
        )
        .eqns[0]
        .primitive
    )

    def inject_late_unknown(**kwargs: Any) -> Any:
        closed = original_trace(**kwargs)
        equations = list(closed.eqns)
        equations[-1] = equations[-1].replace(primitive=unsupported)
        return closed.replace(eqns=equations)

    monkeypatch.setattr(process_grids, "_trace_process_jaxpr", inject_late_unknown)
    monkeypatch.setattr(
        process_grids.SimulationProcessGrids, "_produce", produce_and_record
    )

    with pytest.raises(ExecutionPlanningError, match="unsupported stage 'sin'"):
        owner(
            spec=spec,
            parameters={
                name: jax.numpy.asarray(value) for name, value in parameters.items()
            },
            required=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        )

    assert dispatched == []


def test_nested_process_graph_rewiring_is_refused_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-aval nested intermediate rewiring cannot pass as alpha-renaming."""
    spec, parameters = _RUNTIME_CASES[0]
    owner = _owner(parameters=parameters)
    dispatched: list[str] = []
    original = process_grids.SimulationProcessGrids._produce
    original_trace = process_grids._trace_process_jaxpr

    def produce_and_record(self: Any, **kwargs: Any) -> Any:
        dispatched.append(kwargs["stage"])
        return original(self, **kwargs)

    def rewire_nested_operand(**kwargs: Any) -> Any:
        closed = original_trace(**kwargs)
        outer_equations = list(closed.eqns)
        outer_index = next(
            index
            for index, equation in enumerate(outer_equations)
            if equation.primitive.name == "jit"
        )
        outer = outer_equations[outer_index]
        nested = outer.params["jaxpr"]
        nested_equations = list(nested.eqns)
        nested_index = next(
            index
            for index, equation in enumerate(nested_equations)
            if equation.primitive.name == "sub"
            and len(equation.invars) == 2
            and equation.invars[0] != equation.invars[1]
            and equation.invars[0] not in nested.invars
            and equation.invars[1] not in nested.invars
            and equation.invars[0].aval == equation.invars[1].aval
        )
        equation = nested_equations[nested_index]
        nested_equations[nested_index] = equation.replace(
            invars=(equation.invars[0], equation.invars[0])
        )
        outer_params = dict(outer.params)
        outer_params["jaxpr"] = nested.replace(eqns=nested_equations)
        outer_equations[outer_index] = outer.replace(params=outer_params)
        return closed.replace(eqns=outer_equations)

    monkeypatch.setattr(process_grids, "_trace_process_jaxpr", rewire_nested_operand)
    monkeypatch.setattr(
        process_grids.SimulationProcessGrids, "_produce", produce_and_record
    )

    error: ExecutionPlanningError | None = None
    try:
        owner(
            spec=spec,
            parameters={
                name: jax.numpy.asarray(value) for name, value in parameters.items()
            },
            required=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        )
    except ExecutionPlanningError as caught:
        error = caught

    assert (
        error is not None and "nested linspace graph" in str(error),
        dispatched,
    ) == (True, [])


# keyword-only-exempt: library-callback=functools.partialmethod
def _observe_process_stages(
    self: process_grids.SimulationProcessGrids,
    *,
    original: Callable[..., Any],
    produced: list[jax.Array],
    host_constants: list[np.ndarray],
    **kwargs: Any,
) -> Any:
    for previous in produced:
        missing = resident_bytes_by_device(
            live=measure_buffer_footprint(tree=previous),
            arguments=self.snapshot(),
            devices=self.devices,
        )
        assert not any(missing.values())
    host_constants.extend(
        leaf
        for leaf in jax.tree.leaves(kwargs["parameters"])
        if isinstance(leaf, np.ndarray) and leaf.ndim == 1
    )
    result = original(self, **kwargs)
    assert set(result.devices()) == kwargs["required"].device_set
    produced.append(result)
    return result


@pytest.mark.parametrize("case_index", [4, 5])
def test_process_stages_retain_intermediates_and_place_host_constants(
    *, case_index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each stage sees prior outputs, and GH constants enter placed execution."""
    spec, parameters = _RUNTIME_CASES[case_index]
    runtime = {name: jax.numpy.asarray(value) for name, value in parameters.items()}
    owner = _owner(parameters=runtime)
    produced: list[jax.Array] = []
    host_constants: list[np.ndarray] = []
    monkeypatch.setattr(
        process_grids.SimulationProcessGrids,
        "_produce",
        partialmethod(
            _observe_process_stages,
            original=process_grids.SimulationProcessGrids._produce,
            produced=produced,
            host_constants=host_constants,
        ),
    )

    actual = owner(
        spec=spec,
        parameters=runtime,
        required=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )
    eager = np.asarray(spec.compute_gridpoints(**runtime))

    assert (
        bool(produced),
        bool(host_constants),
        np.asarray(actual).tobytes() == eager.tobytes(),
        owner.temporary_roots,
    ) == (True, case_index == 5, True, [])


def test_refused_process_stage_releases_all_intermediate_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later stage failure releases every completed scalar and vector root."""
    spec, parameters = _RUNTIME_CASES[4]
    runtime = {name: jax.numpy.asarray(value) for name, value in parameters.items()}
    owner = _owner(parameters=runtime)
    original = process_grids.SimulationProcessGrids._produce
    attempts = 0
    observed_prior_ownership: list[bool] = []

    def produce_then_refuse(self: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 5:
            observed_prior_ownership.append(
                all(
                    not any(
                        resident_bytes_by_device(
                            live=measure_buffer_footprint(tree=value),
                            arguments=self.snapshot(),
                            devices=self.devices,
                        ).values()
                    )
                    for value in self.temporary_roots
                )
            )
            raise RuntimeError("controlled process stage failure")
        return original(self, **kwargs)

    monkeypatch.setattr(
        process_grids.SimulationProcessGrids, "_produce", produce_then_refuse
    )

    with pytest.raises(RuntimeError, match="controlled process stage failure"):
        owner(
            spec=spec,
            parameters=runtime,
            required=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        )

    assert (attempts, observed_prior_ownership, owner.temporary_roots) == (
        5,
        [True],
        [],
    )
