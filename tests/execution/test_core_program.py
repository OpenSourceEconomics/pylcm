"""Tests for planner-owned CoreProgram declarations and resolution."""

import functools
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from _lcm.execution.core_program import (
    CoreExecutionRequirements,
    CoreProgram,
    CoreProgramAware,
    ReductionSemantics,
    ResolvedCoreProgram,
    StreamableProductAxis,
    resolve_core_program,
)
from _lcm.execution.output_layout import (
    DISSOLUTION_FLAG,
    VALUE,
    resolve_output_layout,
)
from _lcm.solution.action_reduction import (
    COLLECTIVE_HARD_MAX_REDUCTION,
    HARD_MAX_REDUCTION,
)
from _lcm.solution.action_streaming import (
    GridSearchEV1ActionReduction,
)
from _lcm.solution.backward_induction import (
    _assert_lowered_output_roles,
    _build_continuation_templates,
    _initial_tile_widths,
)
from lcm.solvers import GridSearch
from lcm.typing import ContinuousState, FloatND
from tests.regime_building.test_collective_feasibility_is_shared import (
    _make_model as _build_collective_model,
)
from tests.test_models import taste_shocks_toy
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid,
    utility,
)

_WIDTH_KEYWORD = "_lcm_action_block_width"


def _hard_max_core(
    *,
    first: jnp.ndarray,
    second: jnp.ndarray,
    _lcm_action_block_width: int,
) -> FloatND:
    """Reduce a 2x3 C-order action product using the planner-bound width."""
    first_cells = jnp.repeat(first, second.shape[0])
    second_cells = jnp.tile(second, first.shape[0])
    values = jnp.where(
        ((first_cells == 0) & (second_cells == 30))
        | ((first_cells == 1) & (second_cells == 10)),
        9.0,
        0.0,
    )
    feasible = jnp.ones(values.shape, dtype=bool)
    action_ids = jnp.arange(values.shape[0], dtype=jnp.int32)
    accumulator = HARD_MAX_REDUCTION.initialize(value_template=jnp.asarray(0.0))
    for start in range(0, values.shape[0], _lcm_action_block_width):
        stop = start + _lcm_action_block_width
        accumulator = HARD_MAX_REDUCTION.add(
            accumulator=accumulator,
            values=values[start:stop],
            feasible=feasible[start:stop],
            action_ids=action_ids[start:stop],
        )
    return HARD_MAX_REDUCTION.finalize(accumulator=accumulator).best_value


def _axis(
    *,
    coordinate_names: tuple[str, ...] = ("first", "second"),
    coordinate_extents: tuple[int, ...] = (2, 3),
    canonical_order: Literal["c"] = "c",
    reduction: ReductionSemantics = HARD_MAX_REDUCTION,
) -> StreamableProductAxis:
    return StreamableProductAxis(
        name="action",
        coordinate_names=coordinate_names,
        coordinate_extents=coordinate_extents,
        canonical_order=canonical_order,
        reduction=reduction,
        width_keyword=_WIDTH_KEYWORD,
    )


def _program(
    *,
    axis: StreamableProductAxis | None = None,
    arguments: Mapping[str, object] | None = None,
) -> CoreProgram:
    if arguments is None:
        arguments = {
            "first": jnp.asarray([0, 1]),
            "second": jnp.asarray([10, 20, 30]),
        }
    return CoreProgram(
        function=_hard_max_core,
        arguments=arguments,
        requirements=CoreExecutionRequirements(
            streamable_axes=(_axis() if axis is None else axis,)
        ),
        output_roles=VALUE,
    )


def _eval_resolved_shape(resolved: ResolvedCoreProgram) -> object:
    """Evaluate abstract output with the resolver's static choices bound."""
    bound = functools.partial(resolved.function, **resolved.static_kwargs)
    return jax.eval_shape(bound, **resolved.arguments)


@dataclass(frozen=True, kw_only=True)
class _Provider:
    program: CoreProgram

    def build_core_program(
        self,
        *,
        core_key: str,
        arguments: Mapping[str, object],
    ) -> CoreProgram | None:
        assert core_key == "main"
        assert arguments is self.program.arguments
        return self.program


def test_core_program_aware_is_an_optional_structural_seam() -> None:
    program = _program()
    provider = _Provider(program=program)

    assert isinstance(provider, CoreProgramAware)
    assert (
        provider.build_core_program(core_key="main", arguments=program.arguments)
        is program
    )


def test_explicit_core_program_cannot_omit_output_roles_on_unsharded_template() -> None:
    template = jnp.zeros((2,), dtype=jnp.float32)
    program = CoreProgram(
        function=lambda *, value: value,
        arguments={"value": template},
        requirements=CoreExecutionRequirements(),
        output_roles=None,
    )

    with pytest.raises(ValueError, match=r"output.?roles"):
        resolve_output_layout(
            kernel=object(),
            core_key="main",
            value_template=template,
            state_order=("state",),
            output_roles=program.output_roles,
        )


def _wrong_shape_value_core(*, value: jax.Array) -> jax.Array:
    return jnp.zeros((value.shape[0] + 1,), dtype=value.dtype)


def _wrong_dtype_value_core(*, value: jax.Array) -> jax.Array:
    return jnp.zeros(value.shape, dtype=jnp.int32)


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (_wrong_shape_value_core, r"value.*shape"),
        (_wrong_dtype_value_core, r"value.*dtype"),
    ],
    ids=["wrong-shape", "wrong-dtype"],
)
def test_explicit_value_program_rejects_wrong_lowered_metadata_before_compile(
    *, function: Callable[..., object], message: str
) -> None:
    template = jnp.zeros((2,), dtype=jnp.float32)
    program = CoreProgram(
        function=function,
        arguments={"value": template},
        requirements=CoreExecutionRequirements(),
        output_roles=VALUE,
    )
    resolved = resolve_core_program(program=program, tile_widths={})
    lowered = jax.jit(resolved.function).lower(**resolved.arguments)

    with pytest.raises(TypeError, match=message):
        _assert_lowered_output_roles(
            lowered=lowered,
            output_roles=program.output_roles,
            value_template=template,
            label="test core",
        )


def test_ordinary_singleton_grid_search_declares_action_core_program() -> None:
    """An ordinary solve core declares its canonical action product to planning."""

    def resources(*, liquid: ContinuousState) -> FloatND:
        return liquid

    model = make_alive_dead_model(
        n_periods=2,
        n_liquid=3,
        liquid_max=4.0,
        n_consumption=4,
        alive_functions={"utility": utility, "resources": resources},
        liquid_law=next_liquid,
        alive_solver=GridSearch(),
        constraints={"feasible": feasible},
    )
    flat_params = model._process_params(
        {
            "alive": {
                "utility": {"crra": 2.0},
                "koopmans_aggregator": {"discount_factor": 0.95},
                "alive": {
                    "next_liquid": {"return_liquid": 0.0, "income": 0.0},
                    "next_regime": {"final_age_alive": 1.0},
                },
                "dead": {
                    "next_liquid": {"return_liquid": 0.0, "income": 0.0},
                    "next_regime": {"final_age_alive": 1.0},
                },
            },
            "dead": {"utility": {"crra": 2.0}},
        }
    )
    next_V, next_continuation, _next_edges = _build_continuation_templates(
        regimes=model._regimes,
        flat_params=flat_params,
    )
    regime = model._regimes["alive"]
    kernel = regime.solution.period_kernels[0]
    arguments = kernel.build_lower_args(
        core_key="main",
        state_action_space=regime.solution.state_action_space(
            regime_params=flat_params["alive"]
        ),
        next_regime_to_V_arr=next_V,
        next_regime_to_continuation=next_continuation,
        flat_params=flat_params,
        period=0,
        ages=model.ages,
    )

    assert isinstance(kernel, CoreProgramAware)
    program = kernel.build_core_program(core_key="main", arguments=arguments)

    assert program is not None
    assert isinstance(program.arguments, MappingProxyType)
    assert tuple(program.arguments) == tuple(arguments)
    assert all(program.arguments[name] is value for name, value in arguments.items())
    assert program.output_roles is VALUE
    assert program.requirements.streamable_axes == (
        StreamableProductAxis(
            name="action",
            coordinate_names=("consumption",),
            coordinate_extents=(4,),
            canonical_order="c",
            reduction=HARD_MAX_REDUCTION,
            width_keyword=_WIDTH_KEYWORD,
        ),
    )
    with pytest.raises(TypeError):
        cast("dict[str, object]", program.arguments)["injected"] = jnp.asarray(0)

    resolved = resolve_core_program(program=program, tile_widths={"action": 2})
    output = _eval_resolved_shape(resolved)

    assert jax.tree.structure(output) == jax.tree.structure(program.output_roles)
    assert isinstance(output, jax.ShapeDtypeStruct)
    assert output.shape == next_V["alive"].shape


def test_collective_grid_search_declares_household_action_core_program() -> None:
    """An eligible collective core declares its shared reduction and output tree."""
    model = _build_collective_model()
    flat_params = model._process_params({"discount_factor": 0.95})
    next_V, next_continuation, _next_edges = _build_continuation_templates(
        regimes=model._regimes,
        flat_params=flat_params,
    )
    regime = model._regimes["couple"]
    kernel = regime.solution.period_kernels[0]
    arguments = kernel.build_lower_args(
        core_key="main",
        state_action_space=regime.solution.state_action_space(
            regime_params=flat_params["couple"]
        ),
        next_regime_to_V_arr=next_V,
        next_regime_to_continuation=next_continuation,
        flat_params=flat_params,
        period=0,
        ages=model.ages,
    )

    assert isinstance(kernel, CoreProgramAware)
    program = kernel.build_core_program(core_key="main", arguments=arguments)

    assert program is not None
    assert program.output_roles == (VALUE, DISSOLUTION_FLAG)
    assert program.requirements.streamable_axes == (
        StreamableProductAxis(
            name="action",
            coordinate_names=("work",),
            coordinate_extents=(2,),
            canonical_order="c",
            reduction=COLLECTIVE_HARD_MAX_REDUCTION,
            width_keyword=_WIDTH_KEYWORD,
        ),
    )

    resolved = resolve_core_program(program=program, tile_widths={"action": 1})
    output = _eval_resolved_shape(resolved)

    assert jax.tree.structure(output) == jax.tree.structure(program.output_roles)
    assert isinstance(output, tuple)
    assert isinstance(output[0], jax.ShapeDtypeStruct)
    assert isinstance(output[1], jax.ShapeDtypeStruct)
    assert output[0].shape == next_V["couple"].shape
    assert output[1].shape == next_V["couple"].shape[:-1]
    assert output[1].dtype == jnp.bool_


def test_ev1_grid_search_declares_composite_action_core_program() -> None:
    """An EV1 core declares one flat axis with branch-max/logsum semantics."""
    model = taste_shocks_toy.get_model()
    flat_params = model._process_params(
        taste_shocks_toy.get_params(scale=0.2, discount_factor=0.95)
    )
    next_V, next_continuation, _next_edges = _build_continuation_templates(
        regimes=model._regimes,
        flat_params=flat_params,
    )
    regime = model._regimes["alive"]
    kernel = regime.solution.period_kernels[0]
    arguments = kernel.build_lower_args(
        core_key="main",
        state_action_space=regime.solution.state_action_space(
            regime_params=flat_params["alive"]
        ),
        next_regime_to_V_arr=next_V,
        next_regime_to_continuation=next_continuation,
        flat_params=flat_params,
        period=0,
        ages=model.ages,
    )

    assert isinstance(kernel, CoreProgramAware)
    program = kernel.build_core_program(core_key="main", arguments=arguments)

    assert program is not None
    assert program.output_roles is VALUE
    assert len(program.requirements.streamable_axes) == 1
    axis = program.requirements.streamable_axes[0]
    assert axis.coordinate_names == ("work", "consumption")
    assert axis.coordinate_extents == (2, 8)
    assert axis.canonical_order == "c"
    assert isinstance(axis.reduction, GridSearchEV1ActionReduction)
    assert axis.reduction.semantic_key == (
        "grid-search-ev1-action-reduction",
        1,
        1,
        ("hard-max", 1),
        ("logsumexp", 1),
    )

    resolved = resolve_core_program(program=program, tile_widths={"action": 5})
    output = _eval_resolved_shape(resolved)

    assert jax.tree.structure(output) == jax.tree.structure(program.output_roles)
    assert isinstance(output, jax.ShapeDtypeStruct)
    assert output.shape == next_V["alive"].shape


def test_resolver_binds_width_without_adding_a_dynamic_argument() -> None:
    program = _program()

    with pytest.raises(TypeError, match=_WIDTH_KEYWORD):
        program.function(**program.arguments)

    resolved_three = resolve_core_program(program=program, tile_widths={"action": 3})
    resolved_four = resolve_core_program(program=program, tile_widths={"action": 4})

    assert isinstance(resolved_three, ResolvedCoreProgram)
    assert _WIDTH_KEYWORD not in program.arguments
    assert _WIDTH_KEYWORD not in resolved_three.arguments
    assert resolved_three.specialization_key != resolved_four.specialization_key
    assert resolved_three.function is resolved_four.function
    assert resolved_three.static_kwargs == {_WIDTH_KEYWORD: 3}
    assert resolved_four.static_kwargs == {_WIDTH_KEYWORD: 4}
    assert resolved_three.tile_widths == {"action": 3}
    assert resolved_four.tile_widths == {"action": 4}

    for resolved in (resolved_three, resolved_four):
        result = resolved.function(**resolved.arguments, **resolved.static_kwargs)
        assert_array_equal(result, 9.0)

    jitted = jax.jit(
        resolved_three.function,
        static_argnames=tuple(resolved_three.static_kwargs),
    )
    compiled = jitted.lower(
        **resolved_three.arguments, **resolved_three.static_kwargs
    ).compile()
    assert_array_equal(compiled(**resolved_three.arguments), 9.0)


def test_resolver_requires_a_planner_width_for_each_streamable_axis() -> None:
    with pytest.raises(ValueError, match=r"[Tt]ile width.*required"):
        resolve_core_program(program=_program())


@pytest.mark.parametrize(
    ("coordinate_extent", "error", "message"),
    [
        (2.5, TypeError, r"coordinate extents must be integers"),
        (True, TypeError, r"coordinate extents must be integers"),
        (0, ValueError, r"coordinate extents must be positive"),
        (-1, ValueError, r"coordinate extents must be positive"),
    ],
    ids=["float", "bool", "zero", "negative"],
)
def test_initial_tile_widths_rejects_invalid_coordinate_extents_fail_closed(
    *,
    coordinate_extent: object,
    error: type[Exception],
    message: str,
) -> None:
    """AOT bootstrap validates product declarations before doing width arithmetic."""
    axis = _axis(coordinate_names=("first",), coordinate_extents=(2,))
    object.__setattr__(axis, "coordinate_extents", (coordinate_extent,))
    program = _program(
        axis=axis,
        arguments={"first": jnp.asarray([0, 1])},
    )

    with pytest.raises(error, match=message):
        _initial_tile_widths(program=program)


def test_initial_tile_widths_preserves_bounded_power_of_two_policy() -> None:
    assert _initial_tile_widths(program=_program()) == {"action": 4}


def test_program_and_resolution_snapshot_their_input_mappings() -> None:
    raw_arguments: dict[str, object] = {
        "first": jnp.asarray([0, 1]),
        "second": jnp.asarray([10, 20, 30]),
    }
    program = _program(arguments=raw_arguments)
    raw_arguments["injected"] = jnp.asarray(-1)

    requested_widths = {"action": 3}
    resolved = resolve_core_program(
        program=program,
        tile_widths=requested_widths,
    )
    requested_widths["action"] = 4

    assert "injected" not in program.arguments
    assert "injected" not in resolved.arguments
    assert resolved.static_kwargs == {_WIDTH_KEYWORD: 3}
    assert resolved.tile_widths == {"action": 3}
    with pytest.raises(TypeError):
        cast("dict[str, object]", program.arguments)["new"] = jnp.asarray(0)
    with pytest.raises(TypeError):
        cast("dict[str, object]", resolved.arguments)["new"] = jnp.asarray(0)
    with pytest.raises(TypeError):
        cast("dict[str, int]", resolved.static_kwargs)[_WIDTH_KEYWORD] = 4
    with pytest.raises(TypeError):
        cast("dict[str, int]", resolved.tile_widths)["action"] = 4


@pytest.mark.parametrize(
    ("coordinate_names", "coordinate_extents", "message"),
    [
        (("first", "first"), (2, 3), "duplicate"),
        (("first", "second"), (2,), "coordinate.*extent"),
        (("first", "second"), (2, 0), "positive"),
        (("first", "second"), (2, -1), "positive"),
    ],
    ids=["duplicate-name", "extent-count", "zero-extent", "negative-extent"],
)
def test_resolver_rejects_invalid_canonical_product_declarations(
    *,
    coordinate_names: tuple[str, ...],
    coordinate_extents: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_core_program(
            program=_program(
                axis=_axis(
                    coordinate_names=coordinate_names,
                    coordinate_extents=coordinate_extents,
                )
            ),
            tile_widths={"action": 1},
        )


def test_resolver_rejects_a_non_canonical_product_order() -> None:
    with pytest.raises(ValueError, match=r"canonical.*c"):
        resolve_core_program(
            program=_program(
                axis=_axis(canonical_order=cast("Literal['c']", "fortran"))
            ),
            tile_widths={"action": 1},
        )


@pytest.mark.parametrize(
    ("width", "error", "message"),
    [
        (True, TypeError, "width.*integer"),
        (1.5, TypeError, "width.*integer"),
        (0, ValueError, "width.*positive"),
        (-1, ValueError, "width.*positive"),
        (7, ValueError, "width.*extent"),
    ],
    ids=["bool", "float", "zero", "negative", "beyond-extent"],
)
def test_resolver_rejects_invalid_widths(
    *,
    width: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        resolve_core_program(
            program=_program(),
            tile_widths={"action": cast("int", width)},
        )


def test_resolver_rejects_a_width_for_an_unknown_axis() -> None:
    with pytest.raises(ValueError, match=r"unknown.*axis"):
        resolve_core_program(
            program=_program(),
            tile_widths={"other": 2},
        )


def test_resolver_rejects_a_dynamic_argument_colliding_with_the_width() -> None:
    arguments = {
        "first": jnp.asarray([0, 1]),
        "second": jnp.asarray([10, 20, 30]),
        _WIDTH_KEYWORD: 3,
    }

    with pytest.raises(ValueError, match=r"width keyword.*arguments"):
        resolve_core_program(
            program=_program(arguments=arguments),
            tile_widths={"action": 3},
        )


@dataclass(frozen=True, kw_only=True)
class _UnhashableReductionSemantics:
    @property
    def semantic_key(self) -> Hashable:
        return cast("Hashable", [])


@pytest.mark.parametrize(
    ("reduction", "message"),
    [
        (cast("ReductionSemantics", object()), "stable semantic_key"),
        (_UnhashableReductionSemantics(), "semantic_key.*hashable"),
    ],
    ids=["missing-semantic-key", "unhashable-semantic-key"],
)
def test_resolver_rejects_reduction_without_a_stable_semantic_key(
    *, reduction: ReductionSemantics, message: str
) -> None:
    axis = _axis()
    object.__setattr__(axis, "reduction", reduction)
    with pytest.raises(TypeError, match=message):
        resolve_core_program(
            program=_program(axis=axis),
            tile_widths={"action": 1},
        )
