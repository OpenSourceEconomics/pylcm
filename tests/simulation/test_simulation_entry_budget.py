"""Budget admission sees original caller buffers before conversion or snapshotting."""

import gc
import weakref
from collections import UserDict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.scheduler import shares_a_buffer
from _lcm.simulation.residency import measure_buffer_footprint, resident_bytes_by_device
from _lcm.simulation.runtime import SimulationRuntime
from lcm import ExecutionConfig, Model
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import UserInitialConditions, UserParams
from tests.solution.test_solution_result import _small_grid_search_inputs


@pytest.mark.parametrize("source", ["params", "initial_conditions"])
@pytest.mark.parametrize("unregistered_mapping", [False, True])
def test_existing_caller_buffers_are_checked_before_parameter_conversion(
    *,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    unregistered_mapping: bool,
) -> None:
    """A one-byte budget is refused before any parameter normalization can run."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=1)
    )
    host_initial: dict[str, jax.Array | np.ndarray] = {
        name: np.asarray(value) for name, value in initial.items()
    }
    if source == "params":
        params = {**params, "discount_factor": jnp.asarray(0.95)}
    else:
        host_initial["wealth"] = jnp.asarray([2.0])
    call_params: UserParams = UserDict(params) if unregistered_mapping else params
    call_initial: UserInitialConditions = (
        UserDict(host_initial) if unregistered_mapping else host_initial
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Parameter conversion started before entry admission")

    monkeypatch.setattr(Model, "_process_params", forbidden)
    with pytest.raises(ExecutionPlanningError, match="budget"):
        model.simulate(
            params=call_params, initial_conditions=call_initial, log_level="off"
        )


def test_original_inputs_survive_and_remain_charged_after_conversion_and_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Physical source buffers remain in admission while distinct copies execute."""
    model, base_params, _ = _small_grid_search_inputs(
        execution_config=ExecutionConfig(
            device_memory_bytes=2**32, axis_widths={"subject": 2}
        )
    )
    original_refs: list[weakref.ReferenceType[jax.Array]] = []
    observed: list[int] = []

    def remember(array: jax.Array) -> jax.Array:
        original_refs.append(weakref.ref(array))
        return array

    def initial_inputs() -> dict[str, jax.Array]:
        return {
            "wealth": remember(jnp.asarray([2.0, 2.0, 2.0], dtype=jnp.float16)),
            "age": remember(jnp.asarray([18.0, 18.0, 18.0], dtype=jnp.float16)),
            "regime_id": remember(jnp.zeros(3, dtype=jnp.int16)),
        }

    def user_params() -> UserParams:
        return {
            **base_params,
            "discount_factor": remember(jnp.asarray(0.95, dtype=jnp.float16)),
        }

    dispatch = SimulationRuntime.dispatch

    def observed_dispatch(self: SimulationRuntime, **kwargs: Any) -> object:
        residency = kwargs["residency"]
        assert residency is not None
        originals = tuple(reference() for reference in original_refs)
        assert all(array is not None for array in originals), (
            "An original caller buffer died before forward dispatch"
        )
        argument_arrays = tuple(
            leaf
            for leaf in jax.tree.leaves(kwargs["arguments"])
            if isinstance(leaf, jax.Array)
        )
        assert argument_arrays
        for original in originals:
            assert isinstance(original, jax.Array)
            assert all(
                not shares_a_buffer(first=original, second=argument)
                for argument in argument_arrays
            ), "The source-versus-converted-copy premise must be physical"
        missing = resident_bytes_by_device(
            live=measure_buffer_footprint(tree=originals),
            arguments=residency.live_footprint(),
            devices=residency.budget_devices,
        )
        assert all(size == 0 for size in missing.values()), (
            f"Original caller buffers are absent from residency: {dict(missing)}"
        )
        observed.append(len(originals))
        return dispatch(self, **kwargs)

    monkeypatch.setattr(SimulationRuntime, "dispatch", observed_dispatch)
    result = model.simulate(
        params=user_params(),
        initial_conditions=initial_inputs(),
        log_level="off",
    )
    assert result.n_subjects == 3
    assert observed
    assert all(count == 4 for count in observed)
    gc.collect()
    assert all(reference() is None for reference in original_refs)
