"""Certified monotone affine structure and actual floating-map pullbacks."""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax.extend.core import Jaxpr, Literal, Var

from _lcm.egm.outer_affine_structure import (
    OuterAffineCertificate,
    certify_outer_coefficient,
)
from _lcm.typing import PytreeChild
from lcm.typing import (
    FloatND,
    LoopIndex,
    ReferenceName,
    ScalarBool,
    ScalarFloat,
    StateName,
    ValueND,
)


@dataclass(frozen=True, kw_only=True)
class ResourcesMapCertificate:
    """An affine coefficient and monotonicity of the rounded primitive chain."""

    coefficient: Fraction | None
    """Exact coefficient from the canonical affine structure authority."""
    violation: str | None
    """Unsupported structure, including tainted cancellation or dynamic scale."""


def certify_resources_map(
    *,
    func: Callable[..., FloatND],
    state_name: StateName,
    abstract_args: Iterable[jax.ShapeDtypeStruct | ValueND],
    arg_names: tuple[ReferenceName, ...],
) -> ResourcesMapCertificate:
    """Require positive affine structure and a monotone floating implementation.

    The extra walk refuses cancellation between tainted paths. An exact real
    affine slope alone does not establish monotonicity after intermediate
    floating rounding. No derivatives or sampled secants certify this map.
    """
    args = tuple(abstract_args)
    affine: OuterAffineCertificate = certify_outer_coefficient(
        func=func,
        outer_action_name=state_name,
        abstract_args=args,
        arg_names=arg_names,
    )
    if affine.violation is not None or affine.coefficient is None:
        return ResourcesMapCertificate(
            coefficient=None,
            violation=affine.violation or "Missing affine coefficient.",
        )
    if affine.coefficient <= 0:
        return ResourcesMapCertificate(
            coefficient=None, violation="Resources require a positive liquid slope."
        )
    closed = jax.make_jaxpr(func)(*args)
    outputs, violation = _monotone_outputs(
        jaxpr=closed.jaxpr,
        inputs=tuple(
            index == arg_names.index(state_name) for index in range(len(args))
        ),
    )
    if violation is not None or outputs != (True,):
        return ResourcesMapCertificate(
            coefficient=None,
            violation=violation or "Resources must return one monotone scalar path.",
        )
    return ResourcesMapCertificate(coefficient=affine.coefficient, violation=None)


def _monotone_outputs(
    *, jaxpr: Jaxpr, inputs: tuple[bool, ...]
) -> tuple[tuple[bool, ...], str | None]:
    tainted: set[Var] = {
        variable
        for variable, active in zip(jaxpr.invars, inputs, strict=True)
        if active
    }
    for equation in jaxpr.eqns:
        marks = tuple(
            not isinstance(variable, Literal) and variable in tainted
            for variable in equation.invars
        )
        if not any(marks):
            continue
        name = equation.primitive.name
        nested = [
            equation.params[key]
            for key in ("jaxpr", "call_jaxpr", "fun_jaxpr")
            if key in equation.params
        ]
        if len(nested) == 1:
            body = getattr(nested[0], "jaxpr", nested[0])
            outputs, violation = _monotone_outputs(jaxpr=body, inputs=marks)
            if violation is not None:
                return (), violation
            tainted.update(
                variable
                for variable, active in zip(equation.outvars, outputs, strict=True)
                if active
            )
            continue
        supported = name in {
            "add",
            "add_any",
            "reshape",
            "broadcast_in_dim",
            "squeeze",
            "copy",
            "stop_gradient",
        }
        if name == "sub":
            supported = marks == (True, False)
        if name in {"mul", "div"}:
            fixed_index = 1 if marks == (True, False) else 0
            supported = sum(marks) == 1 and (name != "div" or marks[0])
            literal = equation.invars[fixed_index]
            if supported and isinstance(literal, Literal):
                supported = math.isfinite(float(literal.val)) and literal.val > 0
            else:
                supported = False
        if not supported:
            return (), f"Resources primitive {name!r} lacks a rounded-monotone proof."
        tainted.update(equation.outvars)
    return tuple(
        not isinstance(variable, Literal) and variable in tainted
        for variable in jaxpr.outvars
    ), None


class BoundaryPullback(NamedTuple):
    """First represented pre-state reaching one native resources boundary."""

    point: ScalarFloat
    interior: ScalarBool
    valid: ScalarBool

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[PytreeChild]], BoundaryPullback],
        tuple[None, tuple[PytreeChild, ...]],
    ]:
        """Retain the fixed keyed schema through original-buffer reconstruction."""
        return _unflatten_boundary_pullback, (None, tuple(self))


def pullback_resource_boundary(
    *,
    resources: Callable[[ScalarFloat], ScalarFloat],
    boundary: ScalarFloat,
    lower: ScalarFloat,
    upper: ScalarFloat,
    strict: bool = False,
) -> BoundaryPullback:
    """Invert a structurally certified monotone map on ordered floating keys.

    The model-owned caller must bind the exact function its resources
    certificate traced. This search evaluates that actual rounded function,
    including a plateau, instead of inverting a different endpoint secant.
    Every key in the finite bracket is covered by the integer bisection.
    """
    dtype = lower.dtype
    unsigned = jnp.uint64 if dtype == jnp.float64 else jnp.uint32
    bits = jnp.finfo(dtype).bits
    sign = jnp.asarray(1 << (bits - 1), dtype=unsigned)

    def key(value: ScalarFloat) -> jax.Array:
        raw = jax.lax.bitcast_convert_type(value, unsigned)
        return jnp.where((raw & sign) != 0, ~raw, raw ^ sign)

    def coordinate(ordered: jax.Array) -> ScalarFloat:
        raw = jnp.where((ordered & sign) != 0, ordered ^ sign, ~ordered)
        return jax.lax.bitcast_convert_type(raw, dtype)

    left, right = key(lower), key(upper)
    lower_resources, upper_resources = resources(lower), resources(upper)
    valid = (
        jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & jnp.isfinite(boundary)
        & jnp.isfinite(lower_resources)
        & jnp.isfinite(upper_resources)
        & (lower < upper)
        & (lower_resources <= upper_resources)
    )

    def reached(value: ScalarFloat) -> ScalarBool:
        return value > boundary if strict else value >= boundary

    interior = ~reached(lower_resources) & reached(upper_resources)

    # keyword-only-exempt: library-callback=jax.lax.fori_loop
    def bisect(
        index: LoopIndex, interval: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        del index
        lo, hi = interval
        middle = lo + (hi - lo) // jnp.asarray(2, dtype=unsigned)
        at_boundary = reached(resources(coordinate(middle)))
        return (
            jnp.where((lo < hi) & ~at_boundary, middle + 1, lo),
            jnp.where((lo < hi) & at_boundary, middle, hi),
        )

    first, last = jax.lax.fori_loop(0, bits, bisect, (left, right))
    point = coordinate(first)
    predecessor = coordinate(jnp.maximum(first, left + 1) - 1)
    resolved = (
        (first == last)
        & reached(resources(point))
        & ((first == left) | ~reached(resources(predecessor)))
    )
    return BoundaryPullback(
        point=point,
        interior=interior,
        valid=valid & (~interior | resolved),
    )


def _flatten_boundary_pullback(
    record: BoundaryPullback,
) -> tuple[tuple[PytreeChild, ...], None]:
    return tuple(record), None


def _flatten_boundary_pullback_with_keys(
    record: BoundaryPullback,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(BoundaryPullback._fields, record, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_boundary_pullback(
    metadata: None,
    children: Iterable[PytreeChild],
) -> BoundaryPullback:
    """Restore native numeric, abstract or role leaves under the fixed schema."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(BoundaryPullback._fields):
        raise TypeError(
            "Native BoundaryPullback reconstruction requires its exact fields."
        )
    return cast("BoundaryPullback", tuple.__new__(BoundaryPullback, leaves))


jax.tree_util.register_pytree_with_keys(
    BoundaryPullback,
    _flatten_boundary_pullback_with_keys,
    _unflatten_boundary_pullback,
    _flatten_boundary_pullback,
)
