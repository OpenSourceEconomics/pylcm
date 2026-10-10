"""Engine simulation dataclasses take their mapping fields as read-only views.

Their callers are the engine itself, so a plain dict at construction is a bug the
runtime type check rejects rather than a value the class copies.
"""

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.execution.workspace_planning import compiler_memory_reservation
from _lcm.simulation.chunk_planning import (
    SimulationChunkPlan,
    SimulationChunkProfile,
    SimulationStageProfile,
)
from _lcm.simulation.diagnostic_operations import DiagnosticBinding
from _lcm.simulation.forward_program_profiles import (
    AbstractSimulationProfile,
    ForwardProgramProfile,
)
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.program_types import SimulationBuildContext
from _lcm.simulation.residency import DeviceBufferFootprint
from _lcm.simulation.runtime import SimulationDispatchContext
from _lcm.simulation.subject_groups import SubjectGroupingRoute


def _increment(*, source: jax.Array) -> jax.Array:
    return source + 1


def _compiled() -> jax.stages.Compiled:
    source = jax.device_put(np.arange(4, dtype=np.int32), jax.devices()[0])
    return jax.jit(_increment).lower(source=source).compile()


def _profile(**overrides: Any) -> SimulationChunkProfile:
    device = jax.devices()[0]
    compiled = _compiled()
    stage = SimulationStageProfile(
        name="increment",
        executable=compiled,
        devices=(device,),
        memory=compiler_memory_reservation(
            compiled=compiled, widths=MappingProxyType({})
        ),
    )
    fields: dict[str, Any] = {
        "n_subjects": 4,
        "padded_population": 4,
        "stages": (stage,),
        "fixed_reservation": MappingProxyType({device: 16}),
        "output_reservation": MappingProxyType({}),
        "setup_reservation": MappingProxyType({}),
        "axis_widths": MappingProxyType({}),
    }
    return SimulationChunkProfile(**fields | overrides)


def _abstract_profile(
    cls: type[AbstractSimulationProfile], **overrides: Any
) -> AbstractSimulationProfile:
    compiled = _compiled()
    fields: dict[str, Any] = {
        "executable": compiled,
        "arguments": MappingProxyType(
            {"source": jax.ShapeDtypeStruct((4,), jnp.int32)}
        ),
        "memory": compiler_memory_reservation(
            compiled=compiled, widths=MappingProxyType({})
        ),
    }
    return cls(**fields | overrides)


def _footprint() -> DeviceBufferFootprint:
    return DeviceBufferFootprint(spans=MappingProxyType({}))


def _build_context(**overrides: Any) -> SimulationBuildContext:
    fields: dict[str, Any] = {
        "state_action_space": None,
        "next_regime_to_V_arr": MappingProxyType({}),
        "next_regime_to_continuation": MappingProxyType({}),
        "flat_params": MappingProxyType({}),
        "period": 0,
        "ages": None,
        "call_arguments": MappingProxyType({"state": jnp.zeros(4)}),
    }
    return SimulationBuildContext(**fields | overrides)


_PLAIN_DICT_CONSTRUCTIONS: dict[str, Callable[[], object]] = {
    "footprint.spans": lambda: DeviceBufferFootprint(spans={}),  # ty: ignore[invalid-argument-type]
    "memory.axis_widths": lambda: SimulationMemory(
        budget_bytes=1,
        devices=(jax.devices()[0],),
        subject_devices=(jax.devices()[0],),
        operations=ProfiledSimulationOperations(),
        inputs=_footprint(),
        axis_widths={},  # ty: ignore[invalid-argument-type]
    ),
    "dispatch_context.axis_widths": lambda: SimulationDispatchContext(
        live_footprint=_footprint,
        budget_devices=(jax.devices()[0],),
        axis_widths={},  # ty: ignore[invalid-argument-type]
    ),
    "chunk_profile.fixed_reservation": lambda: _profile(
        fixed_reservation={jax.devices()[0]: 16}
    ),
    "chunk_profile.output_reservation": lambda: _profile(output_reservation={}),
    "chunk_profile.setup_reservation": lambda: _profile(setup_reservation={}),
    "chunk_profile.axis_widths": lambda: _profile(axis_widths={}),
    "chunk_plan.required_bytes": lambda: SimulationChunkPlan(
        profile=_profile(),
        required_bytes={jax.devices()[0]: 16},  # ty: ignore[invalid-argument-type]
    ),
    "diagnostic.arguments": lambda: DiagnosticBinding(
        function=_increment,
        arguments={},  # ty: ignore[invalid-argument-type]
        subject_arg_names=(),
    ),
    "diagnostic.static_arguments": lambda: DiagnosticBinding(
        function=_increment,
        arguments=MappingProxyType({}),
        subject_arg_names=(),
        static_arguments={},  # ty: ignore[invalid-argument-type]
    ),
    "abstract_profile.arguments": lambda: _abstract_profile(
        AbstractSimulationProfile, arguments={}
    ),
    "forward_profile.arguments": lambda: _abstract_profile(
        ForwardProgramProfile, arguments={}
    ),
    "build_context.call_arguments": lambda: _build_context(call_arguments={}),
    "grouping_route.value_axis_names": lambda: SubjectGroupingRoute(
        state_name="health",
        codes=(0, 1),
        value_axis_names={},  # ty: ignore[invalid-argument-type]
    ),
}


@pytest.mark.parametrize(
    "construct",
    list(_PLAIN_DICT_CONSTRUCTIONS.values()),
    ids=list(_PLAIN_DICT_CONSTRUCTIONS),
)
def test_engine_mapping_field_rejects_a_plain_dict(
    construct: Callable[[], object],
) -> None:
    """Constructing an engine dataclass with a plain dict mapping field fails."""
    with pytest.raises(BeartypeCallHintParamViolation):
        construct()


_TREE_FIELDS: dict[str, Callable[[], MappingProxyType]] = {
    "diagnostic.arguments": lambda: (
        DiagnosticBinding(
            function=_increment,
            arguments=MappingProxyType(
                {"source": jax.ShapeDtypeStruct((4,), jnp.int32)}
            ),
            subject_arg_names=("source",),
        ).arguments
    ),
    "abstract_profile.arguments": lambda: (
        _abstract_profile(AbstractSimulationProfile).arguments
    ),
    "build_context.call_arguments": lambda: _build_context().call_arguments,
}


@pytest.mark.parametrize("read", list(_TREE_FIELDS.values()), ids=list(_TREE_FIELDS))
def test_engine_tree_field_round_trips_through_jax_tree_utilities(
    read: Callable[[], MappingProxyType],
) -> None:
    """A mapping field that reaches JAX tree utilities rebuilds as the same view."""
    field = read()
    leaves, treedef = jax.tree_util.tree_flatten(field)

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert (type(field), type(rebuilt), tuple(rebuilt)) == (
        MappingProxyType,
        MappingProxyType,
        tuple(field),
    )
