"""Full-population setup must admit its concrete operations before dispatch."""

from types import MappingProxyType
from typing import Any

import jax
import jax._src.core
import jax.core
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.simulate import _compute_starting_periods, _initial_own_stakeholder
from lcm import AgeGrid
from lcm.exceptions import ExecutionPlanningError
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)
from tests.test_models.deterministic.regression import get_model


class _UnadmittedAllocationError(AssertionError):
    """A concrete numerical dispatch occurred before a budget refusal."""


def _forbid_concrete(*_args: object, **_kwargs: object) -> object:
    raise _UnadmittedAllocationError("Population setup allocated before admission")


def _memory(*, inputs: object, budget: int) -> SimulationMemory:
    devices = (jax.devices()[0],)
    return SimulationMemory(
        budget_bytes=budget,
        devices=devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations(),
        inputs=measure_buffer_footprint(tree=inputs),
    )


def test_starting_periods_are_admitted_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ages = AgeGrid(start=18, stop=20, step="Y")
    initial = jnp.asarray([18.0, 20.0, 19.0])
    memory = _memory(inputs=(initial, ages.values), budget=1)
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_concrete)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_concrete)
        with pytest.raises(_UnadmittedAllocationError):
            jnp.zeros(3)
        with pytest.raises(ExecutionPlanningError):
            _compute_starting_periods(initial_ages=initial, ages=ages, memory=memory)
    generous = _memory(inputs=(initial, ages.values), budget=1_000_000)
    actual = _compute_starting_periods(initial_ages=initial, ages=ages, memory=generous)
    np.testing.assert_array_equal(actual, [0, 2, 1])
    assert actual.dtype == jnp.int32
    np.testing.assert_array_equal(initial, [18.0, 20.0, 19.0])


def test_default_roles_are_admitted_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = get_model(n_periods=2)
    initial = MappingProxyType({"regime_id": jnp.zeros(3, dtype=jnp.int32)})
    memory = _memory(inputs=initial, budget=1)
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_concrete)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_concrete)
        with pytest.raises(_UnadmittedAllocationError):
            jnp.zeros(3)
        with pytest.raises(ExecutionPlanningError):
            _initial_own_stakeholder(
                initial_conditions=initial,
                regimes=model._regimes,
                regime_names_to_ids=model.regime_names_to_ids,
                memory=memory,
            )
    generous = _memory(inputs=initial, budget=1_000_000)
    actual = _initial_own_stakeholder(
        initial_conditions=initial,
        regimes=model._regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        memory=generous,
    )
    np.testing.assert_array_equal(actual, [-1, -1, -1])
    assert actual.dtype == jnp.int32
    np.testing.assert_array_equal(initial["regime_id"], [0, 0, 0])


@pytest.mark.parametrize("operation", ["full_like", "searchsorted"])
def test_public_population_setup_uses_its_admitted_pure_bodies(
    *, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """The real public route must reach the profile before either constructor."""
    model = _stateful_target_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    initial = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([0.0]),
        "regime_id": jnp.asarray([_LifecycleRegimeId.alive], dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="off")
    original = getattr(jnp, operation)
    traced: list[str] = []

    def guard(first: object, *args: Any, **kwargs: Any) -> object:
        if not isinstance(first, jax.core.Tracer):
            raise _UnadmittedAllocationError(f"Unprofiled {operation} population setup")
        traced.append(operation)
        return original(first, *args, **kwargs)

    monkeypatch.setattr(jnp, operation, guard)
    sentinel_args = (
        (initial["regime_id"], -1)
        if operation == "full_like"
        else (model.ages.values, initial["age"])
    )
    with pytest.raises(_UnadmittedAllocationError):
        getattr(jnp, operation)(*sentinel_args)
    result = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=17,
        log_level="off",
    )
    assert result.n_subjects == 1
    assert traced
