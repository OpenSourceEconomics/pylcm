"""Streamed GridSearch preserves every value-dependent input channel."""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    MaterializedCoreProgram,
    StreamableProductAxis,
    core_program_graph,
    initial_core_tile_widths,
    materialize_core_program,
    resolve_core_program,
)
from _lcm.execution.output_layout import VALUE
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueTransferKind,
    resolve_value_transfer,
)
from _lcm.regime_building.max_Q_over_a import (
    get_max_Q_over_a,
    get_streaming_max_Q_over_a,
)
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from _lcm.solution.backward_induction import (
    _build_continuation_templates,
    _edge_kwargs,
)
from _lcm.solution.grid_search import (
    _GridSearchArgumentBuilder,
    _GridSearchPeriodKernel,
    _target_value_accesses,
)
from lcm import Model
from tests.simulation.test_aot_collective_and_gated import _make_consent_model
from tests.simulation.test_aot_same_period_refs import _make_participation_model


def _materialize_program(
    *,
    model: Model,
    regime_name: str,
    params: Mapping[str, Any],
) -> tuple[_GridSearchArgumentBuilder, MaterializedCoreProgram]:
    """Build exactly the program arguments used by backward induction."""
    flat_params = model._process_params(params)
    next_V, next_continuation, next_edges = _build_continuation_templates(
        regimes=model._regimes,
        flat_params=flat_params,
    )
    regime = model._regimes[regime_name]
    edge_kwargs = cast(
        "dict[str, Any]",
        _edge_kwargs(
            regime=regime,
            regime_name=regime_name,
            next_edge_to_V_arr=next_edges,
        ),
    )
    kernel = regime.solution.period_kernels[0]
    assert isinstance(kernel, _GridSearchPeriodKernel)
    program = core_program_graph(kernel=kernel)["main"]
    context = CoreBuildContext(
        state_action_space=regime.solution.state_action_space(
            regime_params=flat_params[regime_name]
        ),
        next_regime_to_V_arr=next_V,
        next_regime_to_continuation=next_continuation,
        flat_params=flat_params,
        period=0,
        ages=model.ages,
        edge_regime_to_V_arr=edge_kwargs.get("edge_regime_to_V_arr"),
    )
    builder = cast("_GridSearchArgumentBuilder", program.argument_builder)
    return builder, materialize_core_program(program=program, context=context)


def _assert_tree_equal(*, actual: object, expected: object) -> None:
    """Compare every numerical output leaf exactly."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def _aligned_transfer_plan(
    *, program: MaterializedCoreProgram
) -> tuple[ResolvedValueTransfer, ...]:
    """Resolve identity adapters for this test's already local JAX arrays."""
    result: list[ResolvedValueTransfer] = []
    for access in program.requirements.target_value_accesses:
        leaf: object = program.arguments[access.source.channel.value]
        for segment in access.source.path:
            assert isinstance(leaf, Mapping)
            leaf = leaf[segment]
        assert isinstance(leaf, jax.Array)
        result.append(
            resolve_value_transfer(
                target=access.target,
                source=access.source,
                kind=ValueTransferKind.ALIGNED_LOCAL,
                stored_template=leaf,
                source_sharding=leaf.sharding,
            )
        )
    return tuple(result)


def _participation_case() -> Model:
    """Build the same-period participation fixture."""
    return _make_participation_model(n_subjects=None)


def _consent_case() -> Model:
    """Build the gated-transition consent fixture."""
    return _make_consent_model(n_subjects=None)


@pytest.mark.parametrize(
    (
        "model_factory",
        "regime_name",
        "params",
        "expected_channels",
        "expected_disposition",
        "expected_reason",
    ),
    [
        pytest.param(
            _participation_case,
            "couple",
            {
                "couple": {
                    "koopmans_aggregator": {"discount_factor": 0.95},
                },
                "couple_terminal": {},
                "single_f": {
                    "koopmans_aggregator": {"discount_factor": 0.95},
                },
                "single_f_terminal": {},
            },
            (("single_f",), (), ()),
            CoreExecutionDisposition.DENSE,
            "deliberately_dense:collective_resource_regression",
            id="same-period-reference",
        ),
        pytest.param(
            _consent_case,
            "single",
            {"discount_factor": 0.95},
            ((), ("single_terminal",), ("married_terminal",)),
            CoreExecutionDisposition.PLANNED,
            None,
            id="edge-reference-and-gated-target",
        ),
    ],
)
def test_value_dependent_model_declares_its_required_program_disposition(
    *,
    model_factory: Callable[[], Model],
    regime_name: str,
    params: Mapping[str, Any],
    expected_channels: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    expected_disposition: CoreExecutionDisposition,
    expected_reason: str | None,
) -> None:
    """Processed reference routes retain channels on their one native program."""
    builder, program = _materialize_program(
        model=model_factory(),
        regime_name=regime_name,
        params=params,
    )

    assert (
        builder.same_period_ref_regimes,
        builder.edge_reference_regimes,
        builder.edge_target_regimes,
    ) == expected_channels
    assert program.disposition is expected_disposition
    assert program.disposition_reason == expected_reason
    assert bool(program.requirements.streamable_axes) is (
        expected_disposition is CoreExecutionDisposition.PLANNED
    )
    transfer_plan = (
        _aligned_transfer_plan(program=program)
        if expected_disposition is CoreExecutionDisposition.PLANNED
        else ()
    )
    resolved = resolve_core_program(
        program=program,
        tile_widths=initial_core_tile_widths(program=program),
        input_transfer_plan=transfer_plan,
    )
    actual = resolved.function(**resolved.arguments, **resolved.static_kwargs)
    if expected_disposition is CoreExecutionDisposition.DENSE:
        expected = program.function(**program.arguments)
        _assert_tree_equal(actual=actual, expected=expected)
    else:
        assert jax.tree.leaves(actual)


def _observable_Q_and_F(
    *,
    choice: jax.Array,
    next_regime_to_V_arr: Mapping[str, jax.Array],
    same_period_regime_to_V_arr: Mapping[str, jax.Array],
    same_period_regime_to_params: Mapping[str, Mapping[str, jax.Array]],
    edge_reference_regime_to_V_arr: Mapping[str, jax.Array],
    edge_reference_regime_to_params: Mapping[str, Mapping[str, jax.Array]],
) -> tuple[jax.Array, jax.Array]:
    """Make every value-dependent input observable in candidate values."""
    value = (
        choice
        + next_regime_to_V_arr["target"][0]
        + same_period_regime_to_V_arr["reference"][0]
        + same_period_regime_to_params["reference"]["offset"]
        + edge_reference_regime_to_V_arr["outside"][0]
        + edge_reference_regime_to_params["outside"]["offset"]
    )
    return value, choice != 1.0


def _observable_route() -> tuple[Callable[..., object], MaterializedCoreProgram]:
    """Build a dense oracle and a non-production streamed reference program."""
    dense = get_max_Q_over_a(
        Q_and_F=_observable_Q_and_F,
        batch_sizes={},
        action_names=("choice",),
        state_names=(),
    )
    streamed = get_streaming_max_Q_over_a(
        Q_and_F=_observable_Q_and_F,
        batch_sizes={},
        action_names=("choice",),
        state_names=(),
    )
    arguments: Mapping[str, object] = MappingProxyType(
        {
            "choice": jnp.asarray([0.0, 1.0, 2.0]),
            "next_regime_to_V_arr": {"target": jnp.asarray([10.0])},
            "same_period_regime_to_V_arr": {"reference": jnp.asarray([20.0])},
            "same_period_regime_to_params": {
                "reference": {"offset": jnp.asarray(30.0)}
            },
            "edge_reference_regime_to_V_arr": {"outside": jnp.asarray([40.0])},
            "edge_reference_regime_to_params": {
                "outside": {"offset": jnp.asarray(50.0)}
            },
        }
    )
    declaration = CoreProgram(
        name="main",
        function=streamed,
        argument_builder=lambda _context: arguments,
        requirements=CoreExecutionRequirements(
            streamable_axes=(
                StreamableProductAxis(
                    name="action",
                    coordinate_names=("choice",),
                    coordinate_extents=(3,),
                    canonical_order="c",
                    reduction=HARD_MAX_REDUCTION,
                    width_keyword="_lcm_action_block_width",
                ),
            ),
            target_value_accesses=_target_value_accesses(
                regime_name="source",
                period=0,
                target_regimes=("target",),
                same_period_ref_regimes=("reference",),
                edge_reference_regimes=("outside",),
                edge_target_regimes=("target",),
            ),
        ),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.PLANNED,
    )
    program = materialize_core_program(
        program=declaration,
        context=CoreBuildContext(
            state_action_space=object(),
            next_regime_to_V_arr={},
            next_regime_to_continuation={},
            flat_params={},
            period=0,
            ages=object(),
        ),
    )
    return dense, program


@pytest.mark.parametrize("width", [1, 2], ids=["unit", "partial-tail"])
def test_value_dependent_reference_matches_dense_eager_jit_and_aot(width: int) -> None:
    """The reference stream preserves support and every value-dependent input."""
    dense_function, program = _observable_route()
    dense = dense_function(**program.arguments)
    resolved = resolve_core_program(
        program=program,
        tile_widths={"action": width},
        input_transfer_plan=_aligned_transfer_plan(program=program),
    )
    eager = resolved.function(**resolved.arguments, **resolved.static_kwargs)
    jitted_function = jax.jit(
        resolved.function, static_argnames=tuple(resolved.static_kwargs)
    )
    jitted = jitted_function(**resolved.arguments, **resolved.static_kwargs)
    compiled = jitted_function.lower(
        **resolved.arguments, **resolved.static_kwargs
    ).compile()
    aot = compiled(**resolved.arguments)

    for actual in (eager, jitted, aot):
        _assert_tree_equal(actual=actual, expected=dense)
