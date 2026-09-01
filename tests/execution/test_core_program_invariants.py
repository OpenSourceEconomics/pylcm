"""Fail-closed contracts for solver-declared core programs."""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.core_program import (
    CoreExecutionRequirements,
    CoreProgram,
    ReductionSemantics,
    StreamableProductAxis,
    resolve_core_program,
)
from _lcm.execution.output_layout import VALUE
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION

_WIDTH_KEYWORD = "_test_action_tile_width"


@dataclass(frozen=True, kw_only=True)
class _FakeReductionSemantics:
    """Test adapter proving execution depends on semantics, not a solver class."""

    semantic_key: Hashable


def _core(*, choice: jax.Array, _test_action_tile_width: int) -> jax.Array:
    """Return one scalar while accepting the planner's static width binding."""
    return choice[0] + jnp.asarray(_test_action_tile_width, dtype=choice.dtype)


def _alternate_core(*, choice: jax.Array, _test_action_tile_width: int) -> jax.Array:
    """Provide a route-distinct core with the same dynamic/static signature."""
    return choice[-1] + jnp.asarray(_test_action_tile_width, dtype=choice.dtype)


@dataclass
class _UnhashableCore:
    """Weak-referenceable callable JAX cannot use as a cache key."""

    offset: float = 0.0

    def __call__(self, *, choice: jax.Array, _test_action_tile_width: int) -> jax.Array:
        return (
            choice[0]
            + jnp.asarray(_test_action_tile_width, dtype=choice.dtype)
            + self.offset
        )


@dataclass(frozen=True)
class _EqualHashableCore:
    """Callable whose value equality aliases route-distinct JAX cache keys."""

    offset: float

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EqualHashableCore)

    def __call__(self, *, choice: jax.Array, _test_action_tile_width: int) -> jax.Array:
        return (
            choice[0]
            + jnp.asarray(_test_action_tile_width, dtype=choice.dtype)
            + self.offset
        )


class _NonWeakrefableCore:
    """Callable that JAX cannot retain as a raw trace-cache key."""

    __slots__ = ()

    def __call__(self, *, choice: jax.Array, _test_action_tile_width: int) -> jax.Array:
        return choice[0] + jnp.asarray(_test_action_tile_width, dtype=choice.dtype)


def _program(
    *,
    arguments: Mapping[str, object] | None = None,
    coordinate_extent: int = 2,
    reduction: ReductionSemantics = HARD_MAX_REDUCTION,
    function: Callable[..., object] = _core,
) -> CoreProgram:
    """Build one canonical action-product declaration for resolver tests."""
    if arguments is None:
        arguments = {"choice": jnp.asarray([1.0, 2.0])}
    return CoreProgram(
        function=function,
        arguments=arguments,
        requirements=CoreExecutionRequirements(
            streamable_axes=(
                StreamableProductAxis(
                    name="action",
                    coordinate_names=("choice",),
                    coordinate_extents=(coordinate_extent,),
                    canonical_order="c",
                    reduction=reduction,
                    width_keyword=_WIDTH_KEYWORD,
                ),
            )
        ),
        output_roles=VALUE,
    )


def test_reduction_semantics_supply_stable_specialization_identity() -> None:
    first = resolve_core_program(
        program=_program(
            reduction=_FakeReductionSemantics(semantic_key=("fake-reduction", 1))
        ),
        tile_widths={"action": 1},
    )
    equivalent = resolve_core_program(
        program=_program(
            reduction=_FakeReductionSemantics(semantic_key=("fake-reduction", 1))
        ),
        tile_widths={"action": 1},
    )
    changed = resolve_core_program(
        program=_program(
            reduction=_FakeReductionSemantics(semantic_key=("fake-reduction", 2))
        ),
        tile_widths={"action": 1},
    )

    assert first.specialization_key == equivalent.specialization_key
    assert first.specialization_key != changed.specialization_key


def test_equivalent_static_resolution_reuses_callable_identity() -> None:
    """Equivalent programs retain JAX's trace-cache key across solve calls."""
    first = resolve_core_program(
        program=_program(),
        tile_widths={"action": 1},
    )
    equivalent = resolve_core_program(
        program=_program(arguments={"choice": jnp.asarray([3.0, 4.0])}),
        tile_widths={"action": 1},
    )
    changed_width = resolve_core_program(
        program=_program(),
        tile_widths={"action": 2},
    )
    changed_route = resolve_core_program(
        program=_program(function=_alternate_core),
        tile_widths={"action": 1},
    )

    assert first.function is equivalent.function
    assert first.function is changed_width.function
    assert first.static_kwargs != changed_width.static_kwargs
    assert first.specialization_key != changed_width.specialization_key
    assert first.function is not changed_route.function


def test_core_program_requires_identity_based_callable_semantics() -> None:
    """Reject value-equal routes that JAX could alias in its trace cache."""
    first = _EqualHashableCore(offset=1.0)
    second = _EqualHashableCore(offset=2.0)

    assert first is not second
    assert first == second
    assert hash(first) == hash(second)

    for function in (first, second):
        with pytest.raises(TypeError, match="identity-based equality and hashing"):
            resolve_core_program(
                program=_program(function=function),
                tile_widths={"action": 1},
            )


def test_core_program_rejects_unhashable_raw_callable() -> None:
    """Reject a callable JAX cannot use as a trace-cache key."""
    with pytest.raises(TypeError, match="identity-based equality and hashing"):
        resolve_core_program(
            program=_program(function=_UnhashableCore()),
            tile_widths={"action": 1},
        )


def test_core_program_requires_a_weakrefable_raw_callable() -> None:
    """Reject a route JAX cannot use as a stable raw trace-cache key."""
    with pytest.raises(TypeError, match="weak-referenceable"):
        resolve_core_program(
            program=_program(function=_NonWeakrefableCore()),
            tile_widths={"action": 1},
        )


@pytest.mark.parametrize(
    ("arguments", "coordinate_extent", "message"),
    [
        ({"other": jnp.asarray([1.0, 2.0])}, 2, "coordinate.*choice.*missing"),
        ({"choice": jnp.ones((2, 1))}, 2, "coordinate.*choice.*one-dimensional"),
        ({"choice": jnp.asarray([])}, 1, "coordinate.*choice.*non-empty"),
        ({"choice": jnp.asarray([1.0, 2.0, 3.0])}, 2, "coordinate.*choice.*extent"),
    ],
    ids=["missing", "not-one-dimensional", "empty", "extent-mismatch"],
)
def test_coordinate_arguments_match_the_declared_product(
    *,
    arguments: Mapping[str, object],
    coordinate_extent: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_core_program(
            program=_program(
                arguments=arguments,
                coordinate_extent=coordinate_extent,
            ),
            tile_widths={"action": 1},
        )


def test_streamable_axis_requires_an_explicit_planner_width() -> None:
    with pytest.raises(ValueError, match=r"[Tt]ile width.*required"):
        resolve_core_program(program=_program())


def test_width_keyword_must_be_accepted_by_the_core_function() -> None:
    def core_without_width(*, choice: jax.Array) -> jax.Array:
        return choice[0]

    with pytest.raises(TypeError, match=r"width keyword.*function"):
        resolve_core_program(
            program=_program(function=core_without_width),
            tile_widths={"action": 1},
        )
