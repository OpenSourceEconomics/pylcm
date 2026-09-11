"""Actual carrier construction refuses before its first new device allocation."""

from types import MappingProxyType

import jax
import jax._src.core
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.initial_conditions import build_initial_states
from lcm.exceptions import ExecutionPlanningError
from tests.simulation.test_budget_lifecycle import _stateful_target_model
from tests.simulation.test_population_allocation_budget import (
    _forbid_concrete,
    _memory,
    _UnadmittedAllocationError,
)


@pytest.mark.parametrize("provided", [False, True])
def test_chunk_carriers_are_admitted_before_fill_cast_or_placement(
    *, monkeypatch: pytest.MonkeyPatch, provided: bool
) -> None:
    model = _stateful_target_model()
    initial = MappingProxyType(
        {
            "age": jnp.zeros(3),
            **({"wealth": jnp.asarray([1, 2, 3], dtype=jnp.int32)} if provided else {}),
        }
    )
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_concrete)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_concrete)
        with pytest.raises(_UnadmittedAllocationError):
            jnp.zeros(3)
        with pytest.raises(ExecutionPlanningError):
            build_initial_states(
                initial_states=initial,
                regimes=model._regimes,
                memory=_memory(inputs=initial, budget=1),
            )
    actual = build_initial_states(
        initial_states=initial,
        regimes=model._regimes,
        memory=_memory(inputs=initial, budget=1_000_000),
    )
    expected = np.asarray([1, 2, 3]) if provided else np.full(3, np.nan)
    for states in actual.values():
        np.testing.assert_array_equal(states["wealth"], expected)
        assert states["wealth"].dtype == jnp.asarray(0.0).dtype
        assert states["wealth"].devices() == {jax.devices()[0]}
    np.testing.assert_array_equal(initial["age"], np.zeros(3))
    if provided:
        np.testing.assert_array_equal(initial["wealth"], [1, 2, 3])
