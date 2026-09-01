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
