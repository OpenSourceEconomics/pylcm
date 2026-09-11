"""Exact chunk-entry and publication operations with explicit admission.

Population slicing stays on its actual selected source devices. Only the narrow
result later moves to the subject layout, so a slice never first replicates the
whole population onto the subject mesh. Dynamic positions share code by width.
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from lcm.exceptions import ExecutionPlanningError


def slice_population(
    *, array: jax.Array, start: int, width: int, memory: SimulationMemory | None
) -> jax.Array:
    """Admit one exact valid subject window before its output is allocated."""
    if (
        type(start) is not int
        or type(width) is not int
        or start < 0
        or width <= 0
        or start + width > array.shape[0]
    ):
        raise ExecutionPlanningError("A population window must lie inside its source.")
    if start == 0 and width == array.shape[0]:
        return array
    if memory is None:
        return array[start : start + width]
    devices = tuple(
        device for device in memory.devices if device in array.sharding.device_set
    )
    if set(devices) != array.sharding.device_set:
        raise ExecutionPlanningError(
            "A population slice has an unbudgeted source device."
        )
    result = cast(
        "jax.Array",
        memory.operations.dispatch(
            function=_slice_population,
            arguments={"array": array, "start": np.int32(start)},
            static_arguments={"width": width},
            subject_arg_names=("array",) if len(devices) > 1 else (),
            subject_outputs=len(devices) > 1,
            devices=devices,
            live_footprint=memory.snapshot,
            budget_devices=memory.devices,
            budget_bytes=memory.budget_bytes,
        ),
    )
    memory.hold(tree=result)
    return result


def _slice_population(*, array: jax.Array, start: jax.Array, width: int) -> jax.Array:
    """Use a dynamic start without changing the caller's validated slice range."""
    return jax.lax.dynamic_slice_in_dim(array, start, width, axis=0)


def period_age(
    *, values: jax.Array, period: int, memory: SimulationMemory | None
) -> jax.Array:
    """Extract an exact grid age inside the profiled body, before scalar upload."""
    if type(period) is not int or not 0 <= period < values.shape[0]:
        raise ExecutionPlanningError("A simulation period must index its age grid.")
    if memory is None:
        return values[period]
    return memory.run(
        function=_period_age,
        arguments={"values": values, "period": np.int32(period)},
    )


def _period_age(*, values: jax.Array, period: jax.Array) -> jax.Array:
    """Read the original age grid, with one compilation shared by all periods."""
    return jax.lax.dynamic_index_in_dim(values, period, axis=0, keepdims=False)


def regime_mask(
    *, regime_ids: jax.Array, regime_id: jax.Array, memory: SimulationMemory | None
) -> jax.Array:
    """Publish the original exact categorical membership predicate."""
    return run_simulation_operation(
        memory=memory,
        function=_regime_mask,
        arguments={"regime_ids": regime_ids, "regime_id": regime_id},
        subject_arg_names=("regime_ids",),
        subject_outputs=True,
    )


def _regime_mask(*, regime_ids: jax.Array, regime_id: jax.Array) -> jax.Array:
    """Preserve membership equality without a concrete host-side temporary."""
    return jnp.asarray(regime_id == regime_ids)


def broadcast_collective(
    *,
    indices: jax.Array,
    value: jax.Array,
    n_subjects: int,
    memory: SimulationMemory | None,
) -> tuple[jax.Array, jax.Array]:
    """Add the subject axis to the original stateless collective outputs."""
    return run_simulation_operation(
        memory=memory,
        function=_broadcast_collective,
        arguments={"indices": indices, "value": value},
        static_arguments={"n_subjects": n_subjects},
        subject_outputs=True,
    )


def _broadcast_collective(
    *, indices: jax.Array, value: jax.Array, n_subjects: int
) -> tuple[jax.Array, jax.Array]:
    """Keep the original index and stakeholder-axis broadcast expressions."""
    return (
        jnp.broadcast_to(jnp.asarray(indices), (n_subjects,)),
        jnp.broadcast_to(value[None, ...], (n_subjects, *value.shape)),
    )


def broadcast_value(
    *, value: jax.Array, n_subjects: int, memory: SimulationMemory | None
) -> jax.Array:
    """Give a scalar terminal value the published subject extent."""
    return run_simulation_operation(
        memory=memory,
        function=_broadcast_value,
        arguments={"value": value},
        static_arguments={"n_subjects": n_subjects},
        subject_outputs=True,
    )


def _broadcast_value(*, value: jax.Array, n_subjects: int) -> jax.Array:
    """Preserve the scalar value exactly, including its signed zero and NaN."""
    return jnp.broadcast_to(value, (n_subjects,))


def empty_fallback(*, mask: jax.Array, memory: SimulationMemory | None) -> jax.Array:
    """Allocate the original all-false fallback publication inside admission."""
    return run_simulation_operation(
        memory=memory,
        function=_empty_fallback,
        arguments={"mask": mask},
        subject_arg_names=("mask",),
        subject_outputs=True,
    )


def _empty_fallback(*, mask: jax.Array) -> jax.Array:
    """Use the same shape and Boolean dtype as the existing publication."""
    return jnp.zeros_like(mask, dtype=bool)
