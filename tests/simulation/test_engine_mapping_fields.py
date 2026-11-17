"""Engine simulation dataclasses keep their mapping fields as read-only views."""

from collections.abc import Callable
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.workspace_planning import compiler_memory_reservation
from _lcm.simulation.diagnostic_operations import DiagnosticBinding
from _lcm.simulation.forward_program_profiles import AbstractSimulationProfile
from _lcm.simulation.program_types import SimulationBuildContext
from _lcm.typing import PytreeValue, ShapeDtypePytree

type _EngineTree = MappingProxyType[str, PytreeValue | ShapeDtypePytree]


def _increment(*, source: jax.Array) -> jax.Array:
    return source + 1


def _compiled() -> jax.stages.Compiled:
    source = jax.device_put(np.arange(4, dtype=np.int32), jax.devices()[0])
    return jax.jit(_increment).lower(source=source).compile()


def _abstract_profile() -> AbstractSimulationProfile:
    compiled = _compiled()
    return AbstractSimulationProfile(
        executable=compiled,
        arguments=MappingProxyType({"source": jax.ShapeDtypeStruct((4,), jnp.int32)}),
        memory=compiler_memory_reservation(
            compiled=compiled, widths=MappingProxyType({})
        ),
    )


def _build_context() -> SimulationBuildContext:
    return SimulationBuildContext(
        state_action_space=None,
        next_regime_to_V_arr=MappingProxyType({}),
        next_regime_to_continuation=MappingProxyType({}),
        flat_params=MappingProxyType({}),
        period=0,
        ages=None,
        call_arguments=MappingProxyType({"state": jnp.zeros(4)}),
    )


_TREE_FIELDS: dict[str, Callable[[], _EngineTree]] = {
    "diagnostic.arguments": lambda: (
        DiagnosticBinding(
            function=_increment,
            arguments=MappingProxyType(
                {"source": jax.ShapeDtypeStruct((4,), jnp.int32)}
            ),
            subject_arg_names=("source",),
        ).arguments
    ),
    "abstract_profile.arguments": lambda: _abstract_profile().arguments,
    "build_context.call_arguments": lambda: _build_context().call_arguments,
}


@pytest.mark.parametrize("read", list(_TREE_FIELDS.values()), ids=list(_TREE_FIELDS))
def test_engine_tree_field_round_trips_through_jax_tree_utilities(
    read: Callable[[], _EngineTree],
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
