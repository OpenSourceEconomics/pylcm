"""Automatic solving must charge the simulation inputs that coexist with it."""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm.model as model_module
from _lcm.execution.scheduler import shares_a_buffer
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.residency import (
    measure_buffer_footprint,
    resident_bytes_by_device,
)
from _lcm.solution import backward_induction
from lcm import ExecutionConfig
from tests.solution.test_solution_result import _small_grid_search_inputs


def _record_initial(
    *,
    recorded: list[Mapping[str, jax.Array]],
    original: Callable[..., object],
    initial_conditions: Mapping[str, jax.Array],
    **arguments: Any,
) -> object:
    recorded.append(initial_conditions)
    return original(initial_conditions=initial_conditions, **arguments)


def _inspect_fixed_inventory(
    *,
    original_inputs: object,
    normalized: list[Mapping[str, jax.Array]],
    calls: list[bool],
    original: Callable[..., Mapping[int, int]],
    tree: object,
) -> Mapping[int, int]:
    assert len(normalized) == 1
    expected = measure_buffer_footprint(tree=(original_inputs, normalized[0]))
    actual = measure_buffer_footprint(tree=tree)
    missing = resident_bytes_by_device(
        live=expected, arguments=actual, devices=tuple(expected.spans)
    )
    assert not any(missing.values()), "Automatic solve omitted live entry inputs"
    calls.append(True)
    return original(tree=tree)


@pytest.mark.parametrize("omit_owners", [False, True])
def test_automatic_solve_charges_original_and_canonical_inputs(
    *, omit_owners: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspect the actual compiler inventory, with an owner-dropping control."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(
            device_memory_bytes=2**28, axis_widths={"subject": 2}
        )
    )
    initial = {name: jnp.repeat(value, 3) for name, value in initial.items()}
    # Accepted integer wealth owns different bytes from its canonical float copy.
    original_wealth = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    initial["wealth"] = original_wealth
    normalized: list[Mapping[str, jax.Array]] = []
    calls: list[bool] = []
    monkeypatch.setattr(
        model_module,
        "validate_simulation_inputs",
        partial(
            _record_initial,
            recorded=normalized,
            original=model_module.validate_simulation_inputs,
        ),
    )
    monkeypatch.setattr(
        backward_induction,
        "concrete_device_bytes",
        partial(
            _inspect_fixed_inventory,
            original_inputs=initial,
            normalized=normalized,
            calls=calls,
            original=backward_induction.concrete_device_bytes,
        ),
    )
    if omit_owners:
        monkeypatch.setattr(
            SimulationEntryAllocations, "solve_input_roots", lambda _: ()
        )
        with pytest.raises(AssertionError, match="omitted live entry inputs"):
            model.simulate(
                params=params,
                initial_conditions=initial,
                log_level="off",
            )
        assert not calls
    else:
        result = model.simulate(
            params=params,
            initial_conditions=initial,
            log_level="off",
        )
        assert result.n_subjects == 3
        assert calls == [True]
    # Automatic solve precedes outer-chunk padding; canonical entry has three rows.
    assert len(normalized[0]["wealth"]) == 3
    assert normalized[0]["wealth"].dtype.kind == "f"
    assert not shares_a_buffer(first=original_wealth, second=normalized[0]["wealth"])
    np.testing.assert_array_equal(original_wealth, [1, 2, 3])
