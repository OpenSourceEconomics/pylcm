"""Explicit device-local execution of independent leading-axis agent programs.

This module depends only on JAX and the standard library. It chooses no memory
budget, allocates no population, and performs no device placement. Its wrapper is
lowered by the normal simulation compiler, whose actual memory profile remains
the input to both chunk selection and live dispatch admission.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from _lcm.execution.output_layout import OutputRoleTree, StateAxesLeading
from _lcm.typing import ArgumentTree, PytreeValue, ShapeDtypePytree
from lcm.typing import ReferenceName


@runtime_checkable
class AgentShardable(Protocol):
    """Explicit capability: independent agent inputs and leading-axis outputs.

    Merely having an array of the population's length is not this declaration.
    Every output leaf must retain the input agent axis, without cross-agent
    operations. Parameter arrays and solution reads are shared even when their
    leading extent happens to equal the population.
    """

    @property
    def agent_shard_arg_names(self) -> tuple[str, ...]:
        """Name all argument subtrees whose leaves carry the agent axis."""
        ...


def declared_agent_shard_arg_names(
    *, function: Callable[..., PytreeValue]
) -> tuple[str, ...] | None:
    """Return the agent arguments a callable declares, seen through partials.

    The capability belongs to the body that was built, not to the keyword
    bindings the model layer later wraps around it: a regime with fixed params
    reaches the runtime as ``functools.partial(body, **fixed)``, and a partial
    proxies no attribute of the callable it wraps. Read the declaration off the
    innermost callable and drop every name a binding has already consumed, so
    only arguments still supplied per dispatch are offered for partitioning.
    A positional binding is refused: it renames nothing and would shift the
    keyword contract this declaration is written in.

    Returns ``None`` when no callable in the chain declares the capability.
    """
    bound: set[str] = set()
    inner: Callable[..., PytreeValue] = function
    while isinstance(inner, partial):
        if inner.args:
            return None
        bound.update(inner.keywords)
        inner = inner.func
    if not isinstance(inner, AgentShardable):
        return None
    return tuple(name for name in inner.agent_shard_arg_names if name not in bound)


def shard_agent_function(
    *,
    function: Callable[..., PytreeValue],
    agent_arg_names: tuple[ReferenceName, ...],
    arguments: Mapping[str, ArgumentTree],
    static_kwargs: Mapping[str, int],
    devices: tuple[jax.Device, ...],
    agent_width_keyword: str,
    output_roles: OutputRoleTree | None = None,
) -> Callable[..., PytreeValue]:
    """Wrap the complete tile loop, not each tile, in one manual device map.

    Only names, shapes, static widths, and the function survive construction.
    Neither the wrapper nor its JAX map captures arrays from ``arguments``.
    The caller must already have admitted/placed agent and shared operands.
    Global rows must divide evenly over ``devices``; Model supplies padding and
    later trims it using its existing original-agent identities.
    """
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Agent sharding needs distinct, nonempty devices.")
    if not agent_arg_names or len(set(agent_arg_names)) != len(agent_arg_names):
        raise ValueError("Agent sharding needs distinct, nonempty agent names.")
    if set(static_kwargs).intersection(arguments):
        raise ValueError("Static widths must not also be dynamic agent arguments.")
    extent = _agent_extent(arguments=arguments, agent_arg_names=agent_arg_names)
    if extent % len(devices):
        raise ValueError("The padded agent extent must divide over agent devices.")
    local_extent = extent // len(devices)
    local_widths = dict(static_kwargs)
    width = local_widths.get(agent_width_keyword)
    if type(width) is not int or width <= 0:
        raise ValueError("Agent sharding requires a positive static agent width.")
    # A configured agent width is a per-device upper bound in this opt-in mode.
    local_widths[agent_width_keyword] = min(width, local_extent)
    mesh = jax.make_mesh(
        (len(devices),), ("X",), (jax.sharding.AxisType.Auto,), devices=devices
    )
    names = frozenset(agent_arg_names)
    in_specs = {name: jax.P("X") if name in names else jax.P() for name in arguments}
    scalar_positions = frozenset(
        index
        for index, role in enumerate(jax.tree.leaves(output_roles))
        if isinstance(role, StateAxesLeading)
        and role.state_names == ()
        and role.n_free_leading_axes == 0
        and role.dtype == jnp.bool_
        and role.shape == ()
    )
    out_specs = (
        jax.tree.map(
            lambda role: (
                jax.P()
                if isinstance(role, StateAxesLeading)
                and role.state_names == ()
                and role.n_free_leading_axes == 0
                and role.dtype == jnp.bool_
                and role.shape == ()
                else jax.P("X")
            ),
            output_roles,
        )
        if scalar_positions
        else jax.P("X")
    )
    mapped = jax.shard_map(
        partial(
            _evaluate_local_agents,
            function=function,
            static_kwargs=MappingProxyType(local_widths),
            local_extent=local_extent,
            scalar_positions=scalar_positions,
        ),
        mesh=mesh,
        in_specs=(in_specs,),
        out_specs=out_specs,
        check_vma=True,
    )
    return _KeywordAgentMap(mapped=mapped)


def _agent_extent(
    *,
    arguments: Mapping[str, ArgumentTree],
    agent_arg_names: tuple[ReferenceName, ...],
) -> int:
    """Validate leading dimensions; empty subtrees contribute no agent extent."""
    extents: set[int] = set()
    for name in agent_arg_names:
        if name not in arguments:
            raise ValueError(f"Declared agent argument {name!r} is missing.")
        leaves = jax.tree.leaves(arguments[name])
        for leaf in leaves:
            shape = getattr(leaf, "shape", ())
            if not shape or type(shape[0]) is not int or shape[0] <= 0:
                raise ValueError(
                    f"Agent argument {name!r} needs a positive leading axis."
                )
            extents.add(shape[0])
    if len(extents) != 1:
        raise ValueError("Every agent operand must have the same leading extent.")
    return next(iter(extents))


# keyword-only-exempt: library-callback=jax.shard_map
def _evaluate_local_agents(
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    *,
    function: Callable[..., PytreeValue],
    static_kwargs: Mapping[str, int],
    local_extent: int,
    scalar_positions: frozenset[int] = frozenset(),
) -> PytreeValue:
    """Run one partition's existing tiling body, retaining its output tree."""
    result = function(**arguments, **static_kwargs)
    leaves = jax.tree.leaves(result)
    for index, leaf in enumerate(leaves):
        shape = getattr(leaf, "shape", ())
        if index in scalar_positions:
            if shape != () or leaf.dtype != jnp.bool_:
                raise TypeError(
                    "Agent integrity outputs must be scalar Boolean arrays."
                )
            leaves[index] = jax.lax.pmax(leaf.astype(jnp.int32), "X").astype(jnp.bool_)
            continue
        if not shape or shape[0] != local_extent:
            raise ValueError("An agent-local output must preserve its leading extent.")
    return (
        jax.tree.unflatten(jax.tree.structure(result), leaves)
        if scalar_positions
        else result
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _KeywordAgentMap:
    """Preserve the existing keyword-only dynamic-argument compiler interface."""

    mapped: Callable[
        [Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]], PytreeValue
    ]

    def __call__(self, **arguments: PytreeValue | ShapeDtypePytree) -> PytreeValue:
        """Dispatch all device partitions together, with no Python device loop."""
        return self.mapped(arguments)
