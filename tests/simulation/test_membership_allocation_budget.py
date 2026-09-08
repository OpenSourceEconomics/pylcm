"""Subject membership allocations must enter the simulation workspace guard."""

import dataclasses
import functools
import weakref
from collections.abc import Callable
from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation import membership as membership_module
from _lcm.simulation import simulate as simulation_module
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.membership import (
    activate_subject_membership,
    initialize_subject_membership,
)
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resident_bytes_by_device,
)
from lcm import DiscreteGrid, Model, categorical
from lcm.execution import ExecutionConfig
from lcm.typing import (
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)


def _run_with_membership_guard(
    *,
    original: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    observations: list[bool],
    **kwargs: Any,
) -> Any:
    """Inspect membership setup after public entry validation has finished."""
    function_name = "full_like" if operation == "empty_carriers" else "where"
    numerical_function = getattr(jnp, function_name)
    with monkeypatch.context() as guard:
        guard.setattr(
            jnp,
            function_name,
            functools.partial(
                _guard_first_subject_allocation,
                original=numerical_function,
                operation=operation,
                observations=observations,
            ),
        )
        return original(**kwargs)


def _guard_first_subject_allocation(
    *args: Any,
    original: Callable[..., Any],
    operation: str,
    observations: list[bool],
    **kwargs: Any,
) -> Any:
    """The first subject-sized operation must be traced before it executes."""
    operand = (
        args[0]
        if args
        else kwargs["a" if operation == "empty_carriers" else "condition"]
    )
    if not observations and getattr(operand, "shape", None) == (1,):
        traced = isinstance(operand, jax.core.Tracer)
        observations.append(traced)
        if not traced:
            raise AssertionError(
                f"Unprofiled simulation membership allocation: {operation}"
            )
    return original(*args, **kwargs)


@pytest.mark.parametrize("operation", ["empty_carriers", "activation"])
def test_budgeted_membership_setup_allocates_inside_profiled_code(
    *, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Workspace admission covers empty membership and each subject's entry."""
    model = _stateful_target_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    initial = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([0.0]),
        "regime_id": jnp.asarray([_LifecycleRegimeId.alive], dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="off")
    observations: list[bool] = []
    monkeypatch.setattr(
        simulation_module,
        "_simulate_subject_chunk",
        functools.partial(
            _run_with_membership_guard,
            original=simulation_module._simulate_subject_chunk,
            monkeypatch=monkeypatch,
            operation=operation,
            observations=observations,
        ),
    )
    result = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="off",
        seed=42,
    )
    assert observations == [True]
    np.testing.assert_array_equal(result.raw_results["alive"][0].in_regime, [True])


@pytest.mark.parametrize("budgeted", [False, True])
def test_membership_activation_preserves_rows_outside_their_entry_period(
    *,
    budgeted: bool,
) -> None:
    """A row-wise entry rule preserves prior regimes and roles at every other age."""
    inputs = {
        "starting_periods": jnp.asarray([0, 2, 1, 2], dtype=jnp.int32),
        "initial_regime_ids": jnp.asarray([1, 2, 3, 4], dtype=jnp.int32),
        "initial_own_stakeholder": jnp.asarray([5, 6, 7, 8], dtype=jnp.int32),
        "regime_ids": jnp.asarray([11, 12, 13, 14], dtype=jnp.int32),
        "own_stakeholder": jnp.asarray([21, 22, 23, 24], dtype=jnp.int32),
    }
    originals = {name: np.asarray(value).copy() for name, value in inputs.items()}
    devices = (jax.devices()[0],)
    memory = (
        SimulationMemory(
            budget_bytes=2**28,
            devices=devices,
            subject_devices=devices,
            operations=ProfiledSimulationOperations(),
            inputs=measure_buffer_footprint(tree=inputs),
        )
        if budgeted
        else None
    )
    empty_regimes, empty_roles = initialize_subject_membership(
        initial_regime_ids=inputs["initial_regime_ids"],
        initial_own_stakeholder=inputs["initial_own_stakeholder"],
        memory=memory,
    )
    np.testing.assert_array_equal(empty_regimes, [-(2**31)] * 4)
    np.testing.assert_array_equal(empty_roles, [-1, -1, -1, -1])
    for period in range(4):
        regimes, roles = activate_subject_membership(
            period=period, memory=memory, **inputs
        )
        for actual, old_name, initial_name in (
            (regimes, "regime_ids", "initial_regime_ids"),
            (roles, "own_stakeholder", "initial_own_stakeholder"),
        ):
            expected = [
                initial if entry == period else old
                for entry, initial, old in zip(
                    originals["starting_periods"],
                    originals[initial_name],
                    originals[old_name],
                    strict=True,
                )
            ]
            assert actual.dtype == np.dtype(np.int32)
            np.testing.assert_array_equal(actual, expected)
    for name, original in originals.items():
        np.testing.assert_array_equal(inputs[name], original)
    if memory is not None:
        # Period is a dynamic input: one executable for setup and one for entry.
        assert len(memory.operations.cache) == 2


@categorical(ordered=False)
class _MembershipKind:
    first: ScalarInt
    second: ScalarInt


def _membership_next_kind() -> ScalarInt:
    return _MembershipKind.second


def _membership_terminal_utility(
    *, wealth: ContinuousState, kind: DiscreteState
) -> FloatND:
    return wealth + kind


@dataclasses.dataclass(kw_only=True)
class _ObservedMembershipRoots:
    channel: str
    initial: DeviceBufferFootprint | None = None
    states: tuple[weakref.ReferenceType[jax.Array], ...] = ()
    age: weakref.ReferenceType[jax.Array] | None = None
    original_ages: DeviceBufferFootprint | None = None
    expected_age: object = None
    reached: int = 0


def _capture_initial_state_roots(
    *, original: Callable[..., Any], observed: _ObservedMembershipRoots, **kwargs: Any
) -> Any:
    observed.initial = measure_buffer_footprint(tree=kwargs["initial_states"])
    result = original(**kwargs)
    observed.states = tuple(weakref.ref(array) for array in jax.tree.leaves(result))
    return result


def _capture_placed_age(
    *, original: Callable[..., Any], observed: _ObservedMembershipRoots, **kwargs: Any
) -> Any:
    result = original(**kwargs)
    if set(kwargs["arguments"]) == {"age"}:
        observed.age = weakref.ref(result["age"])
        observed.expected_age = np.asarray(kwargs["arguments"]["age"]).copy()
    return result


# keyword-only-exempt: library-callback=functools.partialmethod
def _inspect_membership_roots(
    self: ProfiledSimulationOperations,
    *,
    original: Callable[..., Any],
    observed: _ObservedMembershipRoots,
    **kwargs: Any,
) -> Any:
    target = (
        membership_module._empty_subject_membership
        if observed.channel == "states"
        else membership_module._activate_subject_membership
    )
    if kwargs["function"] is target:
        devices = kwargs["budget_devices"]
        if observed.channel == "states":
            arrays = tuple(reference() for reference in observed.states)
            originals = observed.initial
        else:
            assert observed.age is not None
            arrays = (observed.age(),)
            originals = observed.original_ages
            np.testing.assert_array_equal(arrays[0], observed.expected_age)
        assert arrays
        assert all(array is not None for array in arrays)
        assert originals is not None
        actual = measure_buffer_footprint(tree=arrays)
        fresh = resident_bytes_by_device(
            live=actual, arguments=originals, devices=devices
        )
        assert sum(fresh.values()) > 0, "Inventory fixture must expose fresh payload"
        missing = resident_bytes_by_device(
            live=actual, arguments=kwargs["live_footprint"](), devices=devices
        )
        assert all(size == 0 for size in missing.values()), (
            f"Membership {observed.channel} missing from live inventory: "
            f"{dict(missing)}"
        )
        assert all(
            array is not None and array.sharding.device_set == set(kwargs["devices"])
            for array in arrays
        )
        observed.reached += 1
    return original(self, **kwargs)


# keyword-only-exempt: library-callback=functools.partialmethod
def _omit_observed_membership_hold(
    self: SimulationMemory,
    tree: object,
    *,
    original: Callable[..., Any],
    observed: _ObservedMembershipRoots,
) -> None:
    if (
        observed.channel == "age"
        and observed.age is not None
        and tree is observed.age()
    ):
        return
    if observed.channel == "states" and isinstance(tree, tuple) and len(tree) == 2:
        supplied = jax.tree.leaves(tree[0])
        if observed.states and all(
            any(array is reference() for array in supplied)
            for reference in observed.states
        ):
            return
    original(self, tree)


def _assert_public_membership_inventory(
    *, monkeypatch: pytest.MonkeyPatch, channel: str, omit_hold: bool
) -> None:
    base = _stateful_target_model()
    model = Model(
        regimes={
            name: dataclasses.replace(
                regime,
                functions={
                    **regime.functions,
                    "utility": (
                        _membership_terminal_utility
                        if regime.terminal
                        else regime.functions["utility"]
                    ),
                },
                states=(
                    {**regime.states, "kind": DiscreteGrid(_MembershipKind)}
                    if regime.terminal
                    else regime.states
                ),
                state_transitions=(
                    {}
                    if regime.terminal
                    else {**regime.state_transitions, "kind": _membership_next_kind}
                ),
            )
            for name, regime in base.user_regimes.items()
        },
        ages=base.ages,
        regime_id_class=_LifecycleRegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=2**32),
    )
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    initial = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([0.0]),
        "regime_id": jnp.asarray([_LifecycleRegimeId.alive], dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="off")
    observed = _ObservedMembershipRoots(
        channel=channel, original_ages=measure_buffer_footprint(tree=model.ages.values)
    )
    monkeypatch.setattr(
        simulation_module,
        "build_initial_states",
        functools.partial(
            _capture_initial_state_roots,
            original=simulation_module.build_initial_states,
            observed=observed,
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "place_simulation_arguments",
        functools.partial(
            _capture_placed_age,
            original=simulation_module.place_simulation_arguments,
            observed=observed,
        ),
    )
    monkeypatch.setattr(
        ProfiledSimulationOperations,
        "dispatch",
        functools.partialmethod(
            _inspect_membership_roots,
            original=ProfiledSimulationOperations.dispatch,
            observed=observed,
        ),
    )
    if omit_hold:
        monkeypatch.setattr(
            SimulationMemory,
            "hold",
            functools.partialmethod(
                _omit_observed_membership_hold,
                original=SimulationMemory.hold,
                observed=observed,
            ),
        )
    result = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="off",
        seed=42,
    )
    assert observed.reached == (1 if channel == "states" else model.n_periods)
    np.testing.assert_array_equal(result.raw_results["done"][1].states["kind"], [1])
    np.testing.assert_array_equal(initial["wealth"], [2.0])


@pytest.mark.parametrize("channel", ["states", "age"])
def test_membership_admission_counts_new_public_path_roots(
    *, monkeypatch: pytest.MonkeyPatch, channel: str
) -> None:
    """Actual filled/cast state and placed age arrays enter the next admission."""
    _assert_public_membership_inventory(
        monkeypatch=monkeypatch, channel=channel, omit_hold=False
    )


@pytest.mark.parametrize("channel", ["states", "age"])
def test_missing_membership_root_handoff_is_detected(
    *, monkeypatch: pytest.MonkeyPatch, channel: str
) -> None:
    """The public-path observer rejects removal of each exact lifetime handoff."""
    with pytest.raises(AssertionError, match=f"Membership {channel} missing"):
        _assert_public_membership_inventory(
            monkeypatch=monkeypatch, channel=channel, omit_hold=True
        )
