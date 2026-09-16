"""Explicit device-local execution of independent leading-axis subject programs.

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


@runtime_checkable
class SubjectShardable(Protocol):
    """Explicit capability: independent subject inputs and leading-axis outputs.

    Merely having an array of the population's length is not this declaration.
    Every output leaf must retain the input subject axis, without cross-subject
    operations. Parameter arrays and solution reads are shared even when their
    leading extent happens to equal the population.
    """

    @property
    def subject_shard_arg_names(self) -> tuple[str, ...]:
        """Name all argument subtrees whose leaves carry the subject axis."""
        ...


def declared_subject_shard_arg_names(
    *, function: Callable[..., object]
) -> tuple[str, ...] | None:
    """Return the subject arguments a callable declares, seen through partials.

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
    inner: object = function
    while isinstance(inner, partial):
        if inner.args:
            return None
        bound.update(inner.keywords)
        inner = inner.func
    if not isinstance(inner, SubjectShardable):
        return None
    return tuple(name for name in inner.subject_shard_arg_names if name not in bound)


def shard_subject_function(
    *,
    function: Callable[..., object],
    subject_arg_names: tuple[str, ...],
    arguments: Mapping[str, object],
    static_kwargs: Mapping[str, int],
    devices: tuple[jax.Device, ...],
    subject_width_keyword: str,
) -> Callable[..., object]:
    """Wrap the complete tile loop, not each tile, in one manual device map.

    Only names, shapes, static widths, and the function survive construction.
    Neither the wrapper nor its JAX map captures arrays from ``arguments``.
    The caller must already have admitted/placed subject and shared operands.
    Global rows must divide evenly over ``devices``; Model supplies padding and
    later trims it using its existing original-subject identities.
    """
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Subject sharding needs distinct, nonempty devices.")
    if not subject_arg_names or len(set(subject_arg_names)) != len(subject_arg_names):
        raise ValueError("Subject sharding needs distinct, nonempty subject names.")
    if set(static_kwargs).intersection(arguments):
        raise ValueError("Static widths must not also be dynamic subject arguments.")
    extent = _subject_extent(arguments=arguments, subject_arg_names=subject_arg_names)
    if extent % len(devices):
        raise ValueError("The padded subject extent must divide over subject devices.")
    local_extent = extent // len(devices)
    local_widths = dict(static_kwargs)
    width = local_widths.get(subject_width_keyword)
    if type(width) is not int or width <= 0:
        raise ValueError("Subject sharding requires a positive static subject width.")
    # A configured subject width is a per-device upper bound in this opt-in mode.
    local_widths[subject_width_keyword] = min(width, local_extent)
    mesh = jax.make_mesh(
        (len(devices),), ("X",), (jax.sharding.AxisType.Auto,), devices=devices
    )
    names = frozenset(subject_arg_names)
    in_specs = {name: jax.P("X") if name in names else jax.P() for name in arguments}
    mapped = jax.shard_map(
        partial(
            _evaluate_local_subjects,
            function=function,
            static_kwargs=MappingProxyType(local_widths),
            local_extent=local_extent,
        ),
        mesh=mesh,
        in_specs=(in_specs,),
        out_specs=jax.P("X"),
        check_vma=True,
    )
    return _KeywordSubjectMap(mapped=mapped)


def _subject_extent(
    *, arguments: Mapping[str, object], subject_arg_names: tuple[str, ...]
) -> int:
    """Validate declared leading dimensions from metadata, never device values."""
    extents: set[int] = set()
    for name in subject_arg_names:
        if name not in arguments:
            raise ValueError(f"Declared subject argument {name!r} is missing.")
        leaves = jax.tree.leaves(arguments[name])
        if not leaves:
            raise ValueError(f"Declared subject argument {name!r} has no array leaves.")
        for leaf in leaves:
            shape = getattr(leaf, "shape", ())
            if not shape or type(shape[0]) is not int or shape[0] <= 0:
                raise ValueError(
                    f"Subject argument {name!r} needs a positive leading axis."
                )
            extents.add(shape[0])
    if len(extents) != 1:
        raise ValueError("Every subject operand must have the same leading extent.")
    return next(iter(extents))


# keyword-only-exempt: library-callback=jax.shard_map
def _evaluate_local_subjects(
    arguments: Mapping[str, object],
    *,
    function: Callable[..., object],
    static_kwargs: Mapping[str, int],
    local_extent: int,
) -> object:
    """Run one partition's existing tiling body, retaining its output tree."""
    result = function(**arguments, **static_kwargs)
    for leaf in jax.tree.leaves(result):
        shape = getattr(leaf, "shape", ())
        if not shape or shape[0] != local_extent:
            raise ValueError("A subject-local output must preserve its leading extent.")
    return result


@dataclass(frozen=True, kw_only=True, eq=False)
class _KeywordSubjectMap:
    """Preserve the existing keyword-only dynamic-argument compiler interface."""

    mapped: Callable[..., object]

    def __call__(self, **arguments: object) -> object:
        """Dispatch all device partitions together, with no Python device loop."""
        return self.mapped(arguments)
