"""Apply tagged current-shock paths only where a subject consumes their cells."""

import gc
import json
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Unpack

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.current_iid_node_adapter import _addressed_cell_keys, _cell_program
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory, run_simulation_producer
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.random_address import RandomAddress
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.typing import PytreeValue
from lcm import AgeGrid, ExecutionConfig, InitialNodes, IrregSpacedGrid, Model, Regime
from lcm.exceptions import ExecutionPlanningError, InvalidSimulationInputError
from lcm.result import SimulationResult
from lcm.shocks import BaseInnovations, PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import SolutionResult
from lcm.typing import ContinuousState, FloatND, ValueND
from tests.simulation._callback_types import ProducerAdmission
from tests.simulation._node_axis_models import (
    _child_regimes,
    _heterogeneous_entry_model,
    _income,
    _MixedIds,
    _path_model,
    _replay,
    _transition_model,
)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("seed", [0, 29])
def test_consumed_cell_keys_preserve_original_row_addresses(*, seed: int) -> None:
    """Each cell folds its named address, both row words, then its own period."""
    key = jax.random.key(seed, impl="threefry2x32")
    address_words = np.asarray(
        RandomAddress(kind="shock", name="eps").words(), dtype=np.uint32
    )
    row_high = np.asarray([0, 1, 0xFFFFFFFF], dtype=np.uint32)
    row_low = np.asarray([0xFFFFFFFF, 0, 7], dtype=np.uint32)
    periods = np.asarray([0, 1, 0xFFFFFFFF], dtype=np.uint32)
    expected = []
    for high, low, period in zip(row_high, row_low, periods, strict=True):
        cell = key
        for word in (*address_words, high, low, period):
            cell = jax.random.fold_in(cell, word)
        expected.append(np.asarray(jax.random.key_data(cell)))
    actual = _addressed_cell_keys(
        key=key,
        address_words=address_words,
        row_high=row_high,
        row_low=row_low,
        periods=periods,
    )
    np.testing.assert_array_equal(
        jax.random.key_data(actual), np.stack(expected)[:, None]
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_unread_adapter_path_cells_do_not_condition_the_draw() -> None:
    """Only column one is consumed; both outside-support columns are unread."""
    model = _path_model()
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    path = np.asarray([[2.0, 0.5, 2.0]], dtype=jnp.asarray(0.0).dtype)
    simulated = _replay(model=model, solution=solution, path=path, seed=29)
    child = simulated.raw_results["child"][1]
    np.testing.assert_array_equal(child.shocks["eps"], [0.5])
    np.testing.assert_allclose(
        child.actions["consumption"], [0.75], rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(
        simulated.raw_results["done"][2].states["wealth"], [0.75], rtol=2e-4, atol=2e-4
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    ("column", "common", "width", "budgeted", "seed"),
    [
        (0, False, 1, False, 1),
        (2, False, 2, True, 29),
        (0, True, 2, False, 71),
        (2, True, 1, True, 91),
    ],
)
def test_unread_cell_mutations_leave_replay_unchanged(
    *, column: int, common: bool, width: int, budgeted: bool, seed: int
) -> None:
    model = _path_model(width=width, budgeted=budgeted)
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    dtype = jnp.asarray(0.0).dtype
    good = np.asarray([0.5, 0.5, 0.5], dtype=dtype)
    if not common:
        good = np.tile(good, (3, 1))
        good[:, 1] = [0.0, 0.5, 1.0]  # consumed endpoints remain supported
    changed = good.copy()
    if common:
        changed[column] = 2.0
    else:
        changed[:, column] = 2.0
    reference = _replay(model=model, solution=solution, path=good, seed=seed)
    actual = _replay(model=model, solution=solution, path=changed, seed=seed)
    for regime, period in (("child", 1), ("done", 2)):
        left, right = (
            reference.raw_results[regime][period],
            actual.raw_results[regime][period],
        )
        for name, expected in left.states.items():
            np.testing.assert_array_equal(right.states[name], expected)
        for name, expected in left.actions.items():
            np.testing.assert_array_equal(right.actions[name], expected)
        for name, expected in left.shocks.items():
            np.testing.assert_array_equal(right.shocks[name], expected)
        np.testing.assert_array_equal(right.V_arr, left.V_arr)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("invalid", [-1.0, 2.0])
def test_consumed_adapter_path_cell_still_requires_support(*, invalid: float) -> None:
    model = _path_model()
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    path = np.asarray([[0.5, invalid, 0.5]], dtype=jnp.asarray(0.0).dtype)
    with pytest.raises(InvalidSimulationInputError):
        _replay(model=model, solution=solution, path=path, seed=29)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(("base", "reverse"), [(False, False), (True, True)])
def test_heterogeneous_nonreader_entry_ignores_all_unconsumed_cells(
    *, base: bool, reverse: bool
) -> None:
    """One subject never reads eps; another reads it only in column one.

    Uniform(0,1) makes the valid base and physical coordinates equal. The value
    two is outside both supports, but is placed only in unconsumed cells. A row
    permutation checks that consumption belongs to original subjects.
    """
    model = _heterogeneous_entry_model()
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off", max_compilation_workers=1)
    dtype = jnp.asarray(0.0).dtype
    order = np.asarray([1, 0] if reverse else [0, 1])
    initial = {
        "wealth": np.ones(2, dtype=dtype),
        "age": np.asarray([2, 1], dtype=np.int32)[order],
        "regime_id": np.asarray(
            [model.regime_names_to_ids["done"], model.regime_names_to_ids["child"]],
            dtype=np.int32,
        )[order],
    }
    reference = np.full((2, 3), 0.5, dtype=dtype)
    changed = np.asarray([[2.0, 2.0, 2.0], [2.0, 0.5, 2.0]], dtype=dtype)[order]
    tag = BaseInnovations if base else PhysicalValues
    frames = []
    for values in (reference, changed):
        simulated = model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            shock_paths={"eps": tag(values=values)},
            seed=29,
            log_level="off",
        )
        frames.append(simulated.to_dataframe())
    pd.testing.assert_frame_equal(frames[1], frames[0], check_exact=True)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("budgeted", [False, True])
@pytest.mark.parametrize("base", [False, True])
@pytest.mark.parametrize("dependent", [False, True])
def test_only_realized_destination_reads_its_supplied_shock(
    *, budgeted: bool, base: bool, dependent: bool
) -> None:
    """A terminal destination ignores the same-date reader's invalid path cell."""

    model = _transition_model(budgeted=budgeted, dependent=dependent)
    dtype = jnp.asarray(0.0).dtype
    values = np.asarray([[2.0, 0.5, 2.0], [2.0, 2.0, 2.0]], dtype=dtype)
    tag = BaseInnovations if base else PhysicalValues
    result = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": np.asarray([1.0, 2.0], dtype=dtype),
            "age": np.zeros(2, dtype=np.int32),
            "regime_id": np.full(
                2, model.regime_names_to_ids["parent"], dtype=np.int32
            ),
        },
        shock_paths={"eps": tag(values=values)},
        seed=29,
        log_level="off",
    )
    frame = result.to_dataframe(use_categorical_codes=True).reset_index()
    actual = frame.query("period == 1 and regime_name == 'child'")["consumption"]
    np.testing.assert_allclose(actual, [0.75], rtol=2e-4, atol=2e-4)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("budgeted", [False, True])
def test_other_current_group_keeps_its_tagged_path_and_entry_precedence(
    *, budgeted: bool
) -> None:
    """A node-axis group's unread cells do not intercept another group's tags."""

    def utility(*, eta: FloatND) -> FloatND:
        return eta

    regimes = _child_regimes()
    regimes["probe"] = Regime(functions={"utility": utility}, shocks=("other",))
    model = Model(
        regimes=regimes,
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={0: "probe", 1: "child"}),
        edges={"child": {"done": (1,)}},
        regime_id_class=_MixedIds,
        shocks={
            "income": _income(nodes=(0.0, 1.0), weights=(0.5, 0.5)),
            "other": ShockGroup(
                shocks={"eta": Uniform(start=0.0, stop=1.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[0.0], [1.0]]), weights=np.asarray([0.5, 0.5])
                ),
            ),
        },
        execution_config=ExecutionConfig(
            axis_widths={"subject": 1},
            device_memory_bytes=2**27 if budgeted else None,
        ),
    )
    dtype = jnp.asarray(0.0).dtype
    result = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": np.ones(1, dtype=dtype),
            "age": np.zeros(1, dtype=np.int32),
            "regime_id": np.full(1, model.regime_names_to_ids["probe"], dtype=np.int32),
            "eta": PhysicalValues(values=np.asarray([0.75], dtype=dtype)),
        },
        shock_paths={
            "eps": PhysicalValues(values=np.full((1, 3), 2.0, dtype=dtype)),
            "eta": PhysicalValues(values=np.asarray([0.25, 0.25, 0.25], dtype=dtype)),
        },
        seed=29,
        log_level="off",
    )
    np.testing.assert_array_equal(result.raw_results["probe"][0].shocks["eta"], [0.25])


def _add_destination_shock(
    *, wealth: ContinuousState, next_eps: ContinuousState
) -> ContinuousState:
    return wealth + next_eps


def _read_current_shock(*, eps: FloatND) -> FloatND:
    return eps


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("budgeted", [False, True])
def test_transition_local_shock_is_consumed_before_the_terminal_drops_it(
    *, budgeted: bool
) -> None:
    """A terminal wealth law reads its destination-date supplied innovation."""
    regimes = _child_regimes()
    regimes["probe"] = Regime(
        states={"wealth": IrregSpacedGrid(points=np.asarray([0.375, 1.0, 2.0]))},
        functions={"utility": _read_current_shock},
        state_transitions={"wealth": {"done": _add_destination_shock}},
        shocks=("income",),
    )
    model = Model(
        regimes=regimes,
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={0: "probe", 1: "child"}),
        edges={"probe": {"done": (0,)}, "child": {"done": (1,)}},
        regime_id_class=_MixedIds,
        shocks={"income": _income(nodes=(0.0, 1.0), weights=(0.5, 0.5))},
        execution_config=ExecutionConfig(
            axis_widths={"subject": 1},
            device_memory_bytes=2**27 if budgeted else None,
        ),
    )
    dtype = jnp.asarray(0.0).dtype
    result = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": np.ones(1, dtype=dtype),
            "age": np.zeros(1, dtype=np.int32),
            "regime_id": np.full(1, model.regime_names_to_ids["probe"], dtype=np.int32),
        },
        shock_paths={
            "eps": PhysicalValues(values=np.asarray([0.5, 0.25, 0.5], dtype=dtype))
        },
        seed=29,
        log_level="off",
    )
    np.testing.assert_array_equal(
        result.raw_results["done"][1].states["wealth"], [1.25]
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("persistent_compilation_cache", [False], indirect=True)
def test_budgeted_transition_reuses_compiled_shock_producers(
    *, persistent_compilation_cache: bool, tmp_path: Path
) -> None:
    """Identical public replays reuse code while preserving destination draws."""
    assert not persistent_compilation_cache
    model = _transition_model(budgeted=True, dependent=True)
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off")
    dtype = jnp.asarray(0.0).dtype
    initial_conditions = {
        "wealth": np.asarray([1.0, 2.0], dtype=dtype),
        "age": np.zeros(2, dtype=np.int32),
        "regime_id": np.full(2, model.regime_names_to_ids["parent"], dtype=np.int32),
    }
    shock_paths = {"eps": PhysicalValues(values=np.full((2, 3), 0.5, dtype=dtype))}
    events: list[dict[str, str | int | float]] = []

    # keyword-only-exempt: library-callback=jax.monitoring
    def record_compile(
        event: str, start_time: float, end_time: float, **metadata: str | int
    ) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            events.append({"start": start_time, "end": end_time, **metadata})

    def replay() -> SimulationResult:
        result = model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial_conditions,
            shock_paths=shock_paths,
            seed=0,
            log_level="off",
        )
        for periods in result.raw_results.values():
            for record in periods.values():
                jax.block_until_ready(
                    (
                        tuple(record.states.values()),
                        tuple(record.actions.values()),
                        tuple(record.shocks.values()),
                        record.V_arr,
                        record.in_regime,
                        record.own_stakeholder,
                        record.nested_policy_fallback,
                    )
                )
        jax.effects_barrier()
        return result

    jax.monitoring.register_event_time_span_listener(record_compile)
    try:
        replay()
        cold = events.copy()
        events.clear()
        result = replay()
        warm = events.copy()
    finally:
        jax.monitoring.unregister_event_time_span_listener(record_compile)
    (tmp_path / "compilation-events.json").write_text(
        json.dumps({"cold": cold, "warm": warm}, indent=2)
    )
    child = result.raw_results["child"][1]
    live = np.asarray(child.in_regime)
    assert live.sum() == 1
    np.testing.assert_array_equal(np.asarray(child.shocks["eps"])[live], [0.5])
    np.testing.assert_array_equal(np.asarray(child.states["wealth"])[live], [1.0])
    np.testing.assert_allclose(
        np.asarray(child.actions["consumption"])[live], [0.75], rtol=2e-4, atol=2e-4
    )
    assert cold, "The first public replay must exercise the compilation observer."
    assert not warm, f"Identical replay compiled again: {warm}"


def _raw_simulation_arrays(*, result: SimulationResult) -> tuple[ValueND, ...]:
    return tuple(
        leaf
        for periods in result.raw_results.values()
        for record in periods.values()
        for leaf in jax.tree.leaves(
            (
                record.states,
                record.actions,
                record.shocks,
                record.V_arr,
                record.in_regime,
                record.own_stakeholder,
                record.nested_policy_fallback,
            )
        )
    )


def _destination_replay(
    *,
    model: Model,
    solution: SolutionResult,
    discount_factor: float = 1.0,
    path_value: float = 0.5,
    n_subjects: int = 2,
) -> SimulationResult:
    dtype = jnp.asarray(0.0).dtype
    result = model.simulate(
        params={"discount_factor": discount_factor},
        solution=solution,
        initial_conditions={
            "wealth": np.tile(np.asarray([1.0, 2.0], dtype=dtype), n_subjects // 2),
            "age": np.zeros(n_subjects, dtype=np.int32),
            "regime_id": np.full(
                n_subjects, model.regime_names_to_ids["parent"], dtype=np.int32
            ),
        },
        shock_paths={
            "eps": PhysicalValues(
                values=np.full((n_subjects, 3), path_value, dtype=dtype)
            )
        },
        seed=0,
        log_level="off",
    )
    jax.block_until_ready(_raw_simulation_arrays(result=result))
    jax.effects_barrier()
    return result


def _assert_identical_replays(
    *, actual: SimulationResult, expected: SimulationResult
) -> None:
    assert actual.raw_results.keys() == expected.raw_results.keys()
    for name in actual.raw_results:
        assert actual.raw_results[name].keys() == expected.raw_results[name].keys()
    for got, wanted in zip(
        _raw_simulation_arrays(result=actual),
        _raw_simulation_arrays(result=expected),
        strict=True,
    ):
        a, b = np.asarray(got), np.asarray(wanted)
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert a.tobytes() == b.tobytes()


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("changed", ["params", "path"])
def test_cached_shock_producers_read_fresh_operands(
    *, changed: Literal["params", "path"]
) -> None:
    """A warmed replay agrees bitwise with a fresh model at the new inputs."""
    model = _transition_model(budgeted=True, dependent=True)
    original_solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    original = _destination_replay(model=model, solution=original_solution)
    beta = 1.0 / 3.0 if changed == "params" else 1.0
    path_value = 0.75 if changed == "path" else 0.5
    solution = model.solve(params={"discount_factor": beta}, log_level="off")
    actual = _destination_replay(
        model=model, solution=solution, discount_factor=beta, path_value=path_value
    )
    fresh = _transition_model(budgeted=True, dependent=True)
    fresh_solution = fresh.solve(params={"discount_factor": beta}, log_level="off")
    expected = _destination_replay(
        model=fresh,
        solution=fresh_solution,
        discount_factor=beta,
        path_value=path_value,
    )
    _assert_identical_replays(actual=actual, expected=expected)
    child = actual.raw_results["child"][1]
    live = np.asarray(child.in_regime)
    assert live.sum() == 1
    np.testing.assert_array_equal(np.asarray(child.shocks["eps"])[live], [path_value])
    np.testing.assert_array_equal(
        np.asarray(child.states["wealth"])[live],
        [1.25 if changed == "path" else 1.0],
    )
    baseline = original.raw_results["child"][1]
    assert not np.array_equal(
        np.asarray(child.actions["consumption"])[live],
        np.asarray(baseline.actions["consumption"])[np.asarray(baseline.in_regime)],
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("persistent_compilation_cache", [False], indirect=True)
def test_shock_producer_shape_changes_preserve_prior_replay_code(
    *, persistent_compilation_cache: bool, tmp_path: Path
) -> None:
    """A new cohort compiles; returning to a prior cohort reuses its code."""
    assert not persistent_compilation_cache
    model = _transition_model(budgeted=True, dependent=True)
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    events: list[dict[str, str | int | float]] = []

    # keyword-only-exempt: library-callback=jax.monitoring
    def record_compile(
        event: str, start_time: float, end_time: float, **metadata: str | int
    ) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            events.append({"start": start_time, "end": end_time, **metadata})

    jax.monitoring.register_event_time_span_listener(record_compile)
    try:
        original = _destination_replay(model=model, solution=solution)
        cold = events.copy()
        events.clear()
        larger = _destination_replay(model=model, solution=solution, n_subjects=4)
        changed = events.copy()
        events.clear()
        repeated = _destination_replay(model=model, solution=solution)
        returned = events.copy()
    finally:
        jax.monitoring.unregister_event_time_span_listener(record_compile)
    (tmp_path / "shape-compilation-events.json").write_text(
        json.dumps(
            {"first_two": cold, "four": changed, "return_two": returned}, indent=2
        )
    )
    assert cold, "The original shape must exercise compilation."
    assert changed, "The changed shape must exercise compilation."
    assert not returned, f"Returning to the prior shape compiled again: {returned}"
    _assert_identical_replays(actual=repeated, expected=original)
    fresh = _transition_model(budgeted=True, dependent=True)
    fresh_solution = fresh.solve(params={"discount_factor": 1.0}, log_level="off")
    _assert_identical_replays(
        actual=larger,
        expected=_destination_replay(
            model=fresh, solution=fresh_solution, n_subjects=4
        ),
    )
    for result, expected_count in ((original, 1), (larger, 2), (repeated, 1)):
        child = result.raw_results["child"][1]
        live = np.asarray(child.in_regime)
        assert live.sum() == expected_count
        np.testing.assert_array_equal(np.asarray(child.shocks["eps"])[live], 0.5)
        np.testing.assert_array_equal(np.asarray(child.states["wealth"])[live], 1.0)


def _shock_producer_case(
    *, name: Literal["keys", "law"]
) -> tuple[Callable[..., PytreeValue], dict[str, PytreeValue]]:
    key_arguments = {
        "key": jax.random.key(0),
        "address_words": np.asarray(
            RandomAddress(kind="shock", name="eps").words(), dtype=np.uint32
        ),
        "row_high": np.zeros(2, dtype=np.uint32),
        "row_low": np.arange(2, dtype=np.uint32),
        "periods": np.ones(2, dtype=np.uint32),
    }
    if name == "keys":
        return _addressed_cell_keys, key_arguments
    return (
        _cell_program(
            group_name="income",
            group=_income(nodes=(0.0, 1.0), weights=(0.5, 0.5)),
            kinds=(("eps", PhysicalValues),),
            flattened=True,
        ),
        {
            "cell_keys": {"eps": _addressed_cell_keys(**key_arguments)},
            "periods": np.ones((2, 1), dtype=np.int32),
            "ages": np.ones((2, 1), dtype=np.int32),
            "supplied": {"eps": np.full((2, 1), 0.5, dtype=jnp.asarray(0.0).dtype)},
            "enabled": {"eps": np.ones((2, 1), dtype=np.bool_)},
        },
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("producer", ["keys", "law"])
def test_cached_shock_producer_rechecks_budget_and_live_residency(
    *, producer: Literal["keys", "law"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached real key/law code cannot dispatch after either budget becomes tight."""
    function, raw_arguments = _shock_producer_case(name=producer)
    devices = (jax.devices()[0],)
    arguments = place_simulation_arguments(
        arguments=raw_arguments, subject_arg_names=(), value_reads=(), devices=devices
    )
    owner = ProfiledSimulationOperations()

    def memory(*, budget: int, retained: PytreeValue = ()) -> SimulationMemory:
        return SimulationMemory(
            budget_bytes=budget,
            devices=devices,
            subject_devices=devices,
            operations=owner,
            producers=owner,
            inputs=measure_buffer_footprint(tree=arguments),
            derived=retained,
        )

    first = run_simulation_producer(
        memory=memory(budget=2**27), function=function, arguments=arguments
    )
    assert len(owner.cache) == 1
    profiled = next(iter(owner.cache.values()))
    reservation = profiled.reservation_bytes
    footprint = measure_buffer_footprint(tree=arguments)
    assert reservation > sum(
        stop - start for start, stop in footprint.spans[devices[0]]
    )
    admissions: list[tuple[int, int]] = []
    dispatches: list[int] = []
    admit = ProfiledSimulationOperations.admit_producer
    dispatch = jax.stages.Compiled.__call__

    def observe_admission(
        self: ProfiledSimulationOperations, **kwargs: Unpack[ProducerAdmission]
    ) -> jax.stages.Compiled:
        if self is owner:
            admissions.append((kwargs["budget_bytes"], kwargs["resident_bytes"]))
        return admit(self, **kwargs)

    def observe_dispatch(
        self: jax.stages.Compiled, *args: PytreeValue, **kwargs: PytreeValue
    ) -> PytreeValue:
        if self is profiled.executable:
            dispatches.append(1)
        return dispatch(self, *args, **kwargs)

    monkeypatch.setattr(
        ProfiledSimulationOperations, "admit_producer", observe_admission
    )
    monkeypatch.setattr(jax.stages.Compiled, "__call__", observe_dispatch)
    retained = jnp.ones(64, dtype=jnp.uint8)
    repeated = run_simulation_producer(
        memory=memory(budget=reservation + retained.nbytes + 64, retained=retained),
        function=function,
        arguments=arguments,
    )
    for actual, expected in zip(
        jax.tree.leaves(repeated), jax.tree.leaves(first), strict=True
    ):
        if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(
                jax.random.key_data(actual), jax.random.key_data(expected)
            )
        else:
            np.testing.assert_array_equal(actual, expected)
    assert len(dispatches) == 1
    assert len(admissions) == 1
    for budget, live in (
        (reservation - 1, ()),
        (reservation + retained.nbytes - 1, retained),
    ):
        with pytest.raises(
            ExecutionPlanningError, match="No workspace-width candidate"
        ):
            run_simulation_producer(
                memory=memory(budget=budget, retained=live),
                function=function,
                arguments=arguments,
            )
    assert len(admissions) == 3, "Both refusals must reach fresh producer admission."
    assert admissions[1][1] == 0
    assert admissions[2][1] == retained.nbytes
    assert len(dispatches) == 1, (
        "Neither refused call may dispatch the cached executable."
    )
    assert len(owner.cache) == 1
    assert next(iter(owner.cache.values())) is profiled


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_cached_shock_producers_release_caller_arrays_and_results() -> None:
    """Retaining model/solution/code does not keep a previous caller's arrays alive."""
    model = _transition_model(budgeted=True, dependent=True)
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    dtype = jnp.asarray(0.0).dtype
    initial = {
        "wealth": np.asarray([1.0, 2.0], dtype=dtype),
        "age": np.zeros(2, dtype=np.int32),
        "regime_id": np.full(2, model.regime_names_to_ids["parent"], dtype=np.int32),
    }
    bank = np.full((2, 3), 0.5, dtype=dtype)
    paths = {"eps": PhysicalValues(values=bank)}
    references = tuple(weakref.ref(value) for value in (*initial.values(), bank))
    result = model.simulate(
        params={"discount_factor": 1.0},
        solution=solution,
        initial_conditions=initial,
        shock_paths=paths,
        seed=0,
        log_level="off",
    )
    jax.block_until_ready(_raw_simulation_arrays(result=result))
    jax.effects_barrier()
    result_ref = weakref.ref(result)
    assert all(reference() is not None for reference in references)
    del result, initial, bank, paths
    gc.collect()
    assert result_ref() is None
    assert all(reference() is None for reference in references)
    # The next call still uses the retained model and its solution.
    again = _destination_replay(model=model, solution=solution)
    assert np.asarray(again.raw_results["child"][1].in_regime).sum() == 1
