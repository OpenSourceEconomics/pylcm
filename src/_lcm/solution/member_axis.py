"""Publish a one-member regime's value with the regime's member axis.

Every regime publishes its value array with a trailing axis over its members.
An endogenous-grid solver solves a regime of one member, so its value gains a
trailing axis of length one. Each program of its period kernels appends that
axis inside the compiled body, and a program that reads another program's value
as an internal input reads the member's slice of it.
"""

import functools
from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp

from _lcm.execution.core_program import CoreProgram, CoreProgramGraphAware
from _lcm.execution.output_layout import VALUE
from _lcm.regime_building.collective import landing_member_index
from _lcm.solution.contract import PeriodKernel, SolutionKernels
from _lcm.typing import DataclassInstance, PytreeValue
from lcm.typing import ReferenceName

# The wrapper built for each core, keyed by the core's identity and value input.
type _Wrappers = dict[tuple[int, ReferenceName | None], Callable[..., PytreeValue]]


def publish_member_axis(
    *, kernels: SolutionKernels, members: tuple[str, ...]
) -> SolutionKernels:
    """Return `kernels` whose programs publish the value with the member axis.

    Every period kernel keeps its type and its program graph's names; each
    program publishing a value appends the trailing member axis to it.
    """
    # Periods sharing a core share its one wrapper, so they keep one callable.
    wrappers: _Wrappers = {}
    return replace(
        kernels,
        period_kernels=MappingProxyType(
            {
                period: _kernel_with_member_axis(
                    kernel=kernel, members=members, wrappers=wrappers
                )
                for period, kernel in kernels.period_kernels.items()
            }
        ),
    )


def with_member_axis(
    *,
    program: CoreProgram,
    members: tuple[str, ...],
    value_input: ReferenceName | None = None,
    wrappers: _Wrappers | None = None,
) -> CoreProgram:
    """Return `program` publishing its value with the member axis of `members`.

    `value_input` names an internal input holding another program's output whose
    first leaf is that program's value; the wrapped program reads the member's
    slice of it. `wrappers` holds the wrapper already built for a core, which is
    reused. A program publishing no value is returned unchanged.
    """
    if not publishes_value(program=program):
        return program
    member_index = _member_index(members=members)
    wrappers = {} if wrappers is None else wrappers

    def wrap(function: Callable[..., PytreeValue]) -> Callable[..., PytreeValue]:
        key = (id(function), value_input)
        if key not in wrappers:
            wrappers[key] = _wrapped(
                function=function, member_index=member_index, value_input=value_input
            )
        return wrappers[key]

    variant = program.checked_variant
    return replace(
        program,
        function=wrap(program.function),
        checked_variant=None
        if variant is None
        else replace(variant, function=wrap(variant.function)),
    )


def drop_member_axis(*, output: PytreeValue, members: tuple[str, ...]) -> PytreeValue:
    """Return a program output whose first leaf, its value, is the member's slice."""
    return _map_value_leaf(
        output=output,
        function=functools.partial(
            _member_slice, member_index=_member_index(members=members)
        ),
    )


def _kernel_with_member_axis(
    *, kernel: PeriodKernel, members: tuple[str, ...], wrappers: _Wrappers
) -> PeriodKernel:
    """Return the dataclass `kernel` with each program publishing the member axis."""
    return cast(
        "PeriodKernel",
        replace(
            cast("DataclassInstance", kernel),
            _core_programs=MappingProxyType(
                {
                    name: with_member_axis(
                        program=program, members=members, wrappers=wrappers
                    )
                    for name, program in cast("CoreProgramGraphAware", kernel)
                    .core_programs()
                    .items()
                }
            ),
        ),
    )


def _member_index(*, members: tuple[str, ...]) -> int:
    """Index of the regime's one member, the member a route into it lands on."""
    return landing_member_index(members=members, role=members[0])


def publishes_value(*, program: CoreProgram) -> bool:
    """Whether `program` publishes the regime's value, its first output leaf."""
    leaves = jax.tree.leaves(program.output_roles)
    return bool(leaves) and leaves[0] is VALUE


def _wrapped(
    *,
    function: Callable[..., PytreeValue],
    member_index: int,
    value_input: ReferenceName | None,
) -> Callable[..., PytreeValue]:
    """Wrap `function` so that equal cores keep one compilation identity.

    The engine groups programs by the callable and the identities of a
    partial's bound values, so the wrapper binds the core's own function and
    bound values rather than one fresh object per period.
    """
    if isinstance(function, functools.partial):
        return functools.partial(
            _publish_value_with_member_axis,
            *function.args,
            _member_axis_core=function.func,
            _member_axis_index=member_index,
            _member_axis_value_input=value_input,
            **function.keywords,
        )
    return functools.partial(
        _publish_value_with_member_axis,
        _member_axis_core=function,
        _member_axis_index=member_index,
        _member_axis_value_input=value_input,
    )


def _publish_value_with_member_axis(
    *args: object,  # noqa: PAN001 - forwards the core's own arguments unchanged
    _member_axis_core: Callable[..., PytreeValue],
    _member_axis_index: int,
    _member_axis_value_input: ReferenceName | None,
    **kwargs: object,  # noqa: PAN001 - forwards the core's own arguments unchanged
) -> PytreeValue:
    """Run the core, its value input sliced to the member, and append the axis."""
    if _member_axis_value_input is not None:
        kwargs[_member_axis_value_input] = _map_value_leaf(
            output=cast("PytreeValue", kwargs[_member_axis_value_input]),
            function=functools.partial(_member_slice, member_index=_member_axis_index),
        )
    return _map_value_leaf(
        output=_member_axis_core(*args, **kwargs), function=_with_trailing_axis
    )


def _map_value_leaf(
    *, output: PytreeValue, function: Callable[..., jax.Array]
) -> PytreeValue:
    """Apply `function` to the first leaf of `output`, the program's value."""
    leaves, tree = jax.tree.flatten(output)
    return jax.tree.unflatten(tree, [function(value=leaves[0]), *leaves[1:]])


def _member_slice(*, value: jax.Array, member_index: int) -> jax.Array:
    return value[..., member_index]


def _with_trailing_axis(*, value: jax.Array) -> jax.Array:
    return jnp.expand_dims(value, -1)
