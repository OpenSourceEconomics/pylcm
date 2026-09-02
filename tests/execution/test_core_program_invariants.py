"""Fail-closed contracts for solver-declared core programs."""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionRequirements,
    CoreProgram,
    ReductionSemantics,
    StreamableProductAxis,
    _TargetValueAccess,
    _TargetValueAccessAware,
    resolve_core_program,
)
from _lcm.execution.output_layout import (
    UNPLANNED,
    VALUE,
    PlannedCore,
    ResolvedOutputLayout,
    planned_input_transfer_plan,
)
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    resolve_value_transfer,
)
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


def _core_with_values(
    *,
    choice: jax.Array,
    _test_action_tile_width: int,
    **_value_inputs: object,
) -> jax.Array:
    """Keep transfer inputs dynamic while reusing the scalar test core."""
    return _core(
        choice=choice,
        _test_action_tile_width=_test_action_tile_width,
    )


def _value_program(
    *,
    arguments: Mapping[str, object],
    accesses: tuple[_TargetValueAccess, ...],
) -> CoreProgram:
    """Build a program with one canonical axis and exact target-value reads."""
    return CoreProgram(
        function=_core_with_values,
        arguments=arguments,
        requirements=CoreExecutionRequirements(
            streamable_axes=(
                StreamableProductAxis(
                    name="action",
                    coordinate_names=("choice",),
                    coordinate_extents=(2,),
                    canonical_order="c",
                    reduction=HARD_MAX_REDUCTION,
                    width_keyword=_WIDTH_KEYWORD,
                ),
            ),
            target_value_accesses=accesses,
        ),
        output_roles=VALUE,
    )


def _access_and_transfer(
    *,
    value: jax.Array,
    source_period: int = 0,
    source_regime: str = "source",
    target_regime: str = "target",
    channel: ValueInputChannel = ValueInputChannel.NEXT_REGIME_VALUE,
    path: tuple[str | int, ...] = ("target",),
    source_sharding: jax.sharding.Sharding | None = None,
    kind: ValueTransferKind = ValueTransferKind.ALIGNED_LOCAL,
) -> tuple[_TargetValueAccess, ResolvedValueTransfer]:
    """Build one matched logical declaration and concrete transfer."""
    target_period = (
        source_period
        if channel is ValueInputChannel.SAME_PERIOD_VALUE
        else source_period + 1
    )
    target = ValueArtifactAddress(
        kind=ValueArtifactKind.REGIME_VALUE,
        period=target_period,
        regime=target_regime,
    )
    source = ValueConsumerAddress(
        source_period=source_period,
        source_regime=source_regime,
        core_key="main",
        channel=channel,
        path=path,
    )
    access = _TargetValueAccess(target=target, source=source)
    transfer = resolve_value_transfer(
        target=target,
        source=source,
        kind=kind,
        stored_template=value,
        source_sharding=value.sharding if source_sharding is None else source_sharding,
    )
    return access, transfer


@dataclass(frozen=True, kw_only=True)
class _AccessProvider:
    accesses: tuple[_TargetValueAccess, ...]

    def target_value_accesses(self, *, core_key: str) -> tuple[_TargetValueAccess, ...]:
        assert core_key == "main"
        return self.accesses


def test_exact_target_value_accesses_are_structural_and_allow_artifact_fan_out() -> (
    None
):
    value = jnp.asarray([3.0, 4.0])
    next_access, next_transfer = _access_and_transfer(value=value)
    edge_access, edge_transfer = _access_and_transfer(
        value=value,
        channel=ValueInputChannel.EDGE_REFERENCE_VALUE,
    )
    accesses = (next_access, edge_access)
    provider = _AccessProvider(accesses=accesses)
    program = _value_program(
        arguments={
            "choice": jnp.asarray([1.0, 2.0]),
            ValueInputChannel.NEXT_REGIME_VALUE.value: {"target": value},
            ValueInputChannel.EDGE_REFERENCE_VALUE.value: {"target": value},
        },
        accesses=accesses,
    )

    assert isinstance(provider, _TargetValueAccessAware)
    assert provider.target_value_accesses(core_key="main") == accesses
    resolved = resolve_core_program(
        program=program,
        tile_widths={"action": 1},
        input_transfer_plan=(edge_transfer, next_transfer),
    )

    assert program.requirements.target_value_accesses == accesses
    assert resolved.input_transfer_plan == (next_transfer, edge_transfer)
    resolved_values = cast(
        "Mapping[str, object]",
        resolved.arguments[ValueInputChannel.NEXT_REGIME_VALUE.value],
    )
    assert resolved_values["target"] is value
    assert next_access.target == edge_access.target
    assert next_access.source != edge_access.source


def test_duplicate_target_value_argument_path_is_rejected() -> None:
    value = jnp.asarray([3.0, 4.0])
    access, _transfer = _access_and_transfer(value=value)
    program = _value_program(
        arguments={
            "choice": jnp.asarray([1.0, 2.0]),
            ValueInputChannel.NEXT_REGIME_VALUE.value: {"target": value},
        },
        accesses=(access, access),
    )

    with pytest.raises(ValueError, match="duplicate target-value argument path"):
        resolve_core_program(program=program, tile_widths={"action": 1})


@pytest.mark.parametrize(
    ("value_arguments", "message"),
    [
        ({}, "input channel.*missing"),
        (
            {ValueInputChannel.NEXT_REGIME_VALUE.value: {"other": jnp.ones(2)}},
            "argument path.*missing",
        ),
    ],
    ids=["missing-channel", "missing-path"],
)
def test_target_value_access_path_must_exist_in_dynamic_arguments(
    *, value_arguments: Mapping[str, object], message: str
) -> None:
    value = jnp.asarray([3.0, 4.0])
    access, _transfer = _access_and_transfer(value=value)
    program = _value_program(
        arguments={"choice": jnp.asarray([1.0, 2.0]), **value_arguments},
        accesses=(access,),
    )

    with pytest.raises(ValueError, match=message):
        resolve_core_program(program=program, tile_widths={"action": 1})


def test_each_target_value_access_requires_its_exact_resolved_transfer() -> None:
    value = jnp.asarray([3.0, 4.0])
    access, _transfer = _access_and_transfer(value=value)
    program = _value_program(
        arguments={
            "choice": jnp.asarray([1.0, 2.0]),
            ValueInputChannel.NEXT_REGIME_VALUE.value: {"target": value},
        },
        accesses=(access,),
    )

    with pytest.raises(ValueError, match=r"one-to-one.*missing"):
        resolve_core_program(program=program, tile_widths={"action": 1})

    _other_access, other_transfer = _access_and_transfer(
        value=value,
        target_regime="other",
        path=("other",),
    )
    with pytest.raises(ValueError, match=r"one-to-one.*unexpected"):
        resolve_core_program(
            program=program,
            tile_widths={"action": 1},
            input_transfer_plan=(other_transfer,),
        )


def test_transfer_specialization_reuses_periods_but_distinguishes_representation() -> (
    None
):
    value = jnp.asarray([3.0, 4.0])
    first_access, first_transfer = _access_and_transfer(
        value=value,
        source_period=0,
    )
    later_access, later_transfer = _access_and_transfer(
        value=value,
        source_period=1,
    )
    arguments = {
        "choice": jnp.asarray([1.0, 2.0]),
        ValueInputChannel.NEXT_REGIME_VALUE.value: {"target": value},
    }
    first = resolve_core_program(
        program=_value_program(arguments=arguments, accesses=(first_access,)),
        tile_widths={"action": 1},
        input_transfer_plan=(first_transfer,),
    )
    later = resolve_core_program(
        program=_value_program(arguments=arguments, accesses=(later_access,)),
        tile_widths={"action": 1},
        input_transfer_plan=(later_transfer,),
    )

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("device",))
    source_sharding = jax.NamedSharding(mesh, jax.P())
    _copy_access, copy_transfer = _access_and_transfer(
        value=value,
        source_sharding=source_sharding,
        kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
    )
    copied = resolve_core_program(
        program=_value_program(arguments=arguments, accesses=(first_access,)),
        tile_widths={"action": 1},
        input_transfer_plan=(copy_transfer,),
    )

    assert first_transfer != later_transfer
    assert first.specialization_key == later.specialization_key
    assert first.specialization_key != copied.specialization_key
    copied_values = cast(
        "Mapping[str, object]",
        copied.arguments[ValueInputChannel.NEXT_REGIME_VALUE.value],
    )
    assert cast("jax.Array", copied_values["target"]).sharding == source_sharding


def test_resolver_rejects_stale_transfer_metadata() -> None:
    value = jnp.asarray([3.0, 4.0])
    access, _transfer = _access_and_transfer(value=value)
    arguments = {
        "choice": jnp.asarray([1.0, 2.0]),
        ValueInputChannel.NEXT_REGIME_VALUE.value: {"target": value},
    }
    program = _value_program(arguments=arguments, accesses=(access,))

    _shape_access, shape_transfer = _access_and_transfer(value=jnp.ones(3))
    with pytest.raises(ValueError, match="shape mismatch"):
        resolve_core_program(
            program=program,
            tile_widths={"action": 1},
            input_transfer_plan=(shape_transfer,),
        )

    _dtype_access, dtype_transfer = _access_and_transfer(
        value=jnp.asarray([3, 4], dtype=jnp.int32)
    )
    with pytest.raises(TypeError, match="dtype mismatch"):
        resolve_core_program(
            program=program,
            tile_widths={"action": 1},
            input_transfer_plan=(dtype_transfer,),
        )

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("device",))
    named = jax.NamedSharding(mesh, jax.P())
    named_value = jax.device_put(value, named)
    _sharding_access, sharding_transfer = _access_and_transfer(value=named_value)
    with pytest.raises(ValueError, match="stored-sharding mismatch"):
        resolve_core_program(
            program=program,
            tile_widths={"action": 1},
            input_transfer_plan=(sharding_transfer,),
        )


def test_planned_core_applies_and_retains_its_absolute_input_transfer_plan() -> None:
    value = jnp.asarray([3.0, 4.0])
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("device",))
    source_sharding = jax.NamedSharding(mesh, jax.P())
    _access, transfer = _access_and_transfer(
        value=value,
        source_sharding=source_sharding,
        kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
    )
    layout = ResolvedOutputLayout(
        out_shardings=source_sharding,
        compilation_key=("test-layout", source_sharding),
        expected_value_shape=value.shape,
        expected_value_dtype=value.dtype,
        expected_dissolution_shape=None,
        expected_dissolution_dtype=None,
    )

    def compiled(**kwargs: object) -> object:
        values = cast(
            "Mapping[str, object]", kwargs[ValueInputChannel.NEXT_REGIME_VALUE.value]
        )
        return values["target"]

    planned = PlannedCore(
        compiled=compiled,
        layout=layout,
        input_transfer_plan=(transfer,),
    )
    output = cast(
        "jax.Array",
        planned(**{ValueInputChannel.NEXT_REGIME_VALUE.value: {"target": value}}),
    )

    assert planned.input_transfer_plan == (transfer,)
    assert planned_input_transfer_plan(planned) == (transfer,)
    assert planned_input_transfer_plan(object()) is UNPLANNED
    assert output.sharding == source_sharding
    assert np.array_equal(output, value)


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
