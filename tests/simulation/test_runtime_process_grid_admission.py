"""Composite runtime process supports use exact admitted eager stages."""

import dataclasses
import math
import os
import subprocess
import sys
from collections.abc import Callable, Hashable, Mapping
from functools import partialmethod
from pathlib import Path
from typing import Literal, Never, NotRequired, TypedDict, Unpack

import jax
import jax.core
import numpy as np
import pytest
from jax.extend.core import Jaxpr
from numpy.typing import ArrayLike

from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.simulation import host_operations, process_grids
from _lcm.simulation.residency import (
    measure_buffer_footprint,
    resident_bytes_by_device,
)
from _lcm.typing import FootprintTree, PytreeValue, ShapeDtypePytree
from lcm import (
    AgeGrid,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Regime,
    RouwenhorstAR1Process,
    TauchenAR1Process,
    categorical,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    Float1D,
    FloatND,
    ParameterName,
    ReferenceName,
    ScalarFloat,
    ScalarInt,
    ValueND,
)
from tests.execution.test_compiler_allocation_reservation import synthetic_memory
from tests.simulation.test_budget_lifecycle import _LifecycleRegimeId
from tests.simulation.test_process_grid_entry_admission import (
    _COMPOSITE_CASES,
    _inputs,
)


class _ProduceArguments(TypedDict):
    parameters: Mapping[ParameterName, process_grids.ProcessValue]
    n_points: int
    required: jax.sharding.Sharding
    stage: NotRequired[process_grids._GridStage]
    exponent: NotRequired[int]
    dtype: NotRequired[str]
    weak_type: NotRequired[bool]
    process_stage: NotRequired[bool]


class _TraceArguments(TypedDict):
    spec: _ContinuousStochasticProcess
    parameter_names: tuple[ParameterName, ...]
    parameter_values: tuple[process_grids.ProcessValue, ...]


class _CompileCandidateArguments(TypedDict):
    key: Hashable
    function: Callable[..., PytreeValue]
    arguments: Mapping[ReferenceName, ShapeDtypePytree]
    static_arguments: Mapping[ReferenceName, host_operations.StaticArgument]
    output_sharding: NotRequired[jax.sharding.Sharding | None]


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


def _controlled_post_validation_refusal[Ignored](
    *args: Ignored, **kwargs: Ignored
) -> Never:
    del args, kwargs
    raise ExecutionPlanningError("controlled refusal after process validation")


def _owner(*, parameters: FootprintTree) -> process_grids.SimulationProcessGrids:
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
    spec: _ContinuousStochasticProcess,
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
    *, spec: _ContinuousStochasticProcess, runtime: dict[str, int]
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
    *, spec: _ContinuousStochasticProcess, parameters: dict[str, float]
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
    first = result.to_dataframe(use_categorical_codes=True).iloc[0]

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
                states=states,
                actions={"saving": LinSpacedGrid(start=0, stop=1, n_points=2)},
                state_transitions={"assets": _support_next_assets},
                functions={"utility": _support_current_payoff},
            ),
            "done": Regime(
                states=states,
                functions={"utility": _support_terminal_payoff},
            ),
        },
        regime_id_class=_LifecycleRegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        execution_config=ExecutionConfig(device_memory_bytes=2**28),
        initial_nodes={0: "alive"},
        edges={"alive": {"done": 0}},
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
        frame = result.to_dataframe(use_categorical_codes=True)
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


# keyword-only-exempt: library-callback=functools.partialmethod
def _inject_process_refusal[Result: host_operations._ProfiledOperation](
    self: host_operations.ProfiledSimulationOperations,
    *,
    original: Callable[..., Result],
    **kwargs: Unpack[_CompileCandidateArguments],
) -> Result:
    result = original(self, **kwargs)
    if getattr(kwargs["function"], "__name__", "") == "_compute_process_stage":
        return dataclasses.replace(result, memory=synthetic_memory(2**30))
    return result


@pytest.mark.parametrize(("spec", "parameters"), _RUNTIME_CASES)
def test_runtime_process_refuses_before_completed_fallback_support(
    *,
    spec: _ContinuousStochasticProcess,
    parameters: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declined first stage prevents the formerly eager fallback producer."""
    completed: list[Float1D] = []
    process_type = type(spec)
    original_gridpoints = process_type.compute_gridpoints

    def observe_gridpoints(
        self: _ContinuousStochasticProcess, **kwargs: ScalarFloat | ScalarInt
    ) -> Float1D:
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

    def produce_and_record(
        self: process_grids.SimulationProcessGrids, **kwargs: Unpack[_ProduceArguments]
    ) -> ValueND:
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

    def inject_late_unknown(**kwargs: Unpack[_TraceArguments]) -> Jaxpr:
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

    def produce_and_record(
        self: process_grids.SimulationProcessGrids, **kwargs: Unpack[_ProduceArguments]
    ) -> ValueND:
        dispatched.append(kwargs["stage"])
        return original(self, **kwargs)

    def rewire_nested_operand(**kwargs: Unpack[_TraceArguments]) -> Jaxpr:
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
def _observe_process_stages[Result: ValueND](
    self: process_grids.SimulationProcessGrids,
    *,
    original: Callable[..., Result],
    produced: list[jax.Array],
    host_constants: list[np.ndarray],
    **kwargs: Unpack[_ProduceArguments],
) -> Result:
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

    def produce_then_refuse(
        self: process_grids.SimulationProcessGrids, **kwargs: Unpack[_ProduceArguments]
    ) -> ValueND:
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


@pytest.mark.parametrize("runtime", [False, True])
def test_given_node_supports_preserve_distinct_axes(*, runtime: bool) -> None:
    """Identical law parameters cannot alias different supplied physical axes."""
    parameters = (
        {"rho": jax.numpy.asarray(1.0), "sigma": jax.numpy.asarray(0.1)}
        if runtime
        else {}
    )
    owner = _owner(parameters=parameters)
    required = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    for nodes in ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0)):
        process = TauchenAR1Process.on_nodes(
            nodes=nodes, rho=None if runtime else 1.0, sigma=None if runtime else 0.1
        )
        actual = owner(spec=process, parameters=parameters, required=required)
        np.testing.assert_array_equal(actual, jax.numpy.asarray(nodes))


def test_given_node_support_refuses_insufficient_placement_budget() -> None:
    """A constant physical axis cannot bypass device allocation admission."""
    owner = _owner(parameters={})
    owner.budget_bytes = 0
    process = TauchenAR1Process.on_nodes(nodes=(-1.0, 0.0, 1.0), rho=1.0, sigma=0.1)
    required = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    with pytest.raises(ExecutionPlanningError):
        owner(spec=process, parameters={}, required=required)
    assert not owner.grids


def _array_identity(value: ArrayLike) -> tuple[str, tuple[int, ...], bytes]:
    """Describe an array by its exact dtype, shape and represented bytes."""
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes()


@pytest.mark.parametrize("negative_first", [True, False])
@pytest.mark.parametrize("n_points", [3, 4])
@pytest.mark.parametrize("cache_boundary", ["binding", "content", "trace"])
def test_given_node_supports_keep_signed_zero_at_every_reuse_layer(
    *,
    cache_boundary: Literal["binding", "content", "trace"],
    negative_first: bool,
    n_points: int,
) -> None:
    """Equal-valued supplied axes with different zero signs never share a support.

    `binding` reuses the same runtime operands on one owner, `content` copies them,
    and `trace` uses a fresh owner, so each case reaches a different cache first.
    """
    lower, upper = {
        "binding": (-11.0, 13.0),
        "content": (-17.0, 19.0),
        "trace": (-23.0, 29.0),
    }[cache_boundary]
    # Distinct endpoints per case keep the process-wide trace cache from linking cases.
    displacement = 100.0 * (n_points - 3) + (1000.0 if negative_first else 0.0)
    lower, upper = lower - displacement, upper + displacement
    zeros = (-0.0, 0.0) if negative_first else (0.0, -0.0)
    first_params = {"rho": jax.numpy.asarray(1.0)}
    second_params = (
        first_params
        if cache_boundary == "binding"
        else {"rho": jax.numpy.array(first_params["rho"], copy=True)}
    )
    required = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    owners = [_owner(parameters=(first_params, second_params))]
    if cache_boundary == "trace":
        owners.append(_owner(parameters=(first_params, second_params)))
    observed = []
    try:
        for index, (zero, params) in enumerate(
            ((zeros[0], first_params), (zeros[1], second_params))
        ):
            nodes = (
                (lower, zero, upper) if n_points == 3 else (lower, -1.0, zero, upper)
            )
            process = TauchenAR1Process.on_nodes(nodes=nodes, sigma=0.1)
            actual = owners[min(index, len(owners) - 1)](
                spec=process, parameters=params, required=required
            )
            observed.append(
                _array_identity(actual) == _array_identity(jax.numpy.asarray(nodes))
            )
    finally:
        for owner in owners:
            owner.close()
    assert observed == [True, True]


def test_given_node_trace_identity_includes_active_float_dtype() -> None:
    """Integer law operands cannot hide a change in the supplied axis's dtype."""
    original = jax.config.read("jax_enable_x64")
    nodes = (-31.0, 0.0, 37.0)
    observed = []
    try:
        for x64 in (False, True):
            jax.config.update("jax_enable_x64", val=x64)
            params = {"rho": jax.numpy.asarray(1, dtype=jax.numpy.int32)}
            process = TauchenAR1Process.on_nodes(nodes=nodes, sigma=1, mu=0)
            owner = _owner(parameters=params)
            try:
                actual = owner(
                    spec=process,
                    parameters=params,
                    required=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
                )
                observed.append(
                    _array_identity(actual) == _array_identity(jax.numpy.asarray(nodes))
                )
            finally:
                owner.close()
    finally:
        jax.config.update("jax_enable_x64", val=original)
    assert observed == [True, True]


@categorical(ordered=False)
class _AxisIdentityRegimeId:
    done: ScalarInt


def _sign_sensitive_utility(*, x: ContinuousState, y: ContinuousState) -> FloatND:
    """Make a lost zero sign on either axis numerically observable."""
    return jax.numpy.where(jax.numpy.signbit(x), 1.0, 0.0) + jax.numpy.where(
        jax.numpy.signbit(y), 2.0, 0.0
    )


def _sign_sensitive_table(
    *, x_nodes: tuple[float, ...], y_nodes: tuple[float, ...]
) -> np.ndarray:
    """Tabulate the sign-sensitive payoff by scalar enumeration."""
    return np.asarray(
        [
            [
                (1.0 if math.copysign(1.0, x) < 0 else 0.0)
                + (2.0 if math.copysign(1.0, y) < 0 else 0.0)
                for y in y_nodes
            ]
            for x in x_nodes
        ]
    )


@pytest.mark.slow
@pytest.mark.parametrize("budget", [None, 2**28])
def test_automatic_solution_keeps_supplied_axis_zero_signs(
    budget: int | None,
) -> None:
    """A simulate-owned solve tabulates both prescribed signed-zero axes."""
    x_nodes = (-4.0, -0.0, 5.0)
    y_nodes = (-4.0, 0.0, 5.0)
    model = Model(
        regimes={
            "done": Regime(
                states={
                    "x": TauchenAR1Process.on_nodes(nodes=x_nodes, sigma=0.1),
                    "y": TauchenAR1Process.on_nodes(nodes=y_nodes, sigma=0.1),
                },
                functions={"utility": _sign_sensitive_utility},
            )
        },
        n_periods=1,
        initial_nodes=InitialNodes(by_period={0: "done"}),
        edges={},
        regime_id_class=_AxisIdentityRegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=budget),
    )
    params = {"rho": 1.0}
    expected = jax.numpy.asarray(
        _sign_sensitive_table(x_nodes=x_nodes, y_nodes=y_nodes)
    )
    result = model.simulate(
        params=params,
        initial_conditions={
            "x": jax.numpy.asarray([-0.0]),
            "y": jax.numpy.asarray([0.0]),
            "period": jax.numpy.asarray([0], dtype=jax.numpy.int32),
            "regime_id": jax.numpy.asarray([_AxisIdentityRegimeId.done]),
        },
        seed=0,
        log_level="off",
        runtime_checks=True,
    )
    assert result.solution is not None
    axes = result.solution.metadata.value_schemas[(0, "done")].named_axes
    observed = (
        _array_identity(result.solution.values[0]["done"]),
        *(
            _array_identity(np.asarray(axis.coordinates, dtype=expected.dtype))
            for axis in axes
        ),
    )
    assert observed == (
        _array_identity(expected),
        _array_identity(jax.numpy.asarray(x_nodes)),
        _array_identity(jax.numpy.asarray(y_nodes)),
    )
