"""NB-EGM widths belong to the execution planner, not solver configuration."""

from dataclasses import replace
from functools import partial
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
import pytest

from _lcm.egm.published_policy import NBEGMGridPolicy
from _lcm.egm.upper_envelope.query import ComparisonArithmetic
from _lcm.execution.core_program import (
    CoreBuildContext,
    MaterializedCoreProgram,
    _value_read_argument_leaf,
    core_program_graph,
    materialize_core_program,
    resolve_core_program,
)
from _lcm.execution.value_transfer import ResolvedValueTransfer, ValueTransferKind
from _lcm.execution.workspace_planning import bootstrap_width
from lcm import ExecutionConfig, LinSpacedGrid, Model, Regime
from lcm.exceptions import ExecutionPlanningError, RegimeInitializationError
from lcm.solver_api import SolutionResult
from lcm.solvers import (
    BRANCH_AXIS,
    CELL_AXIS,
    INTERVAL_AXIS,
    NBEGM,
    STOCHASTIC_NODE_AXIS,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON, assert_agrees_to_ulp
from tests.solution._nbegm_direct_oracle import ride_along_kernel
from tests.test_models import (
    nbegm_multi_discrete_toy,
    nbegm_next_asset_cliff_toy,
    nbegm_ride_along_toy,
    nbegm_stochastic_node_toy,
    nbegm_tax_toy,
)


@pytest.mark.parametrize(
    "field",
    [
        "stochastic_node_batch_size",
        "envelope_segment_block_size",
        "interval_batch_size",
        "cell_block_size",
        "branch_batch_size",
    ],
)
def test_nbegm_has_no_width_field(field: str) -> None:
    """A solver constructor refuses every removed execution-width field."""
    invalid: dict[str, Any] = {field: 2}
    with pytest.raises(TypeError, match=field):
        NBEGM(savings_grid=LinSpacedGrid(start=0, stop=5, n_points=8), **invalid)


def test_ride_along_program_declares_the_cell_mesh() -> None:
    """Only the two independent kind cells are partitionable on this route."""
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_periods=3,
        n_liquid=8,
        n_savings=8,
        nbegm_overrides={"envelope_arithmetic": "ordinary"},
    )
    kernels = model._regimes["alive"].solution.period_kernels
    for kernel in kernels.values():
        for program in core_program_graph(kernel=kernel).values():
            assert program.requirements.axis_names == ("cell",)
            (axis,) = program.requirements.tiled_axes
            assert axis.state_names == ("kind",)
            assert axis.extent == 2


def test_liquid_reading_continuation_declares_its_interval_reduction() -> None:
    """The two liquid intervals use the stable-identity envelope fold."""
    model = nbegm_next_asset_cliff_toy.build_model(
        variant="nbegm",
        n_periods=3,
        n_liquid=8,
        n_savings=8,
        envelope_arithmetic="ordinary",
    )
    kernel = model._regimes["alive"].solution.period_kernels[0]
    for program in core_program_graph(kernel=kernel).values():
        assert program.requirements.axis_names == ("interval", "cell")
        (axis,) = program.requirements.reduced_axes
        assert axis.extent == 2
        assert axis.reduction.semantic_key == ("interval-envelope", 1)


def test_discrete_branch_axis_declares_the_full_action_product() -> None:
    """Two binary actions form four branches before the household maximum."""
    model = nbegm_multi_discrete_toy.build_model(
        variant="nbegm",
        n_actions=2,
        n_periods=3,
        n_liquid=8,
        n_savings=8,
        envelope_arithmetic="ordinary",
    )
    kernel = model._regimes["alive"].solution.period_kernels[0]
    for program in core_program_graph(kernel=kernel).values():
        assert program.requirements.axis_names == ("stochastic_node", "branch", "cell")
        axis = next(
            axis for axis in program.requirements.reduced_axes if axis.name == "branch"
        )
        assert axis.coordinate_names == ("buy_private", "claim_benefit")
        assert axis.coordinate_extents == (2, 2)


def test_stochastic_node_axis_spans_the_child_income_grid() -> None:
    """The five-node expectation declares exactly the child's shared income mesh."""
    model = nbegm_stochastic_node_toy.build_model(
        variant="nbegm",
        n_periods=3,
        n_liquid=8,
        n_savings=8,
        envelope_arithmetic="ordinary",
    )
    kernel = model._regimes["alive"].solution.period_kernels[0]
    for program in core_program_graph(kernel=kernel).values():
        assert program.requirements.axis_names == ("stochastic_node", "cell")
        (axis,) = program.requirements.reduced_axes
        assert axis.coordinate_names == ("income",)
        assert axis.coordinate_extents == (5,)
        assert axis.reduction.semantic_key == ("weighted-expectation", 1)


def _small_model(
    *,
    route: str,
    arithmetic: ComparisonArithmetic,
    widths: dict[str, int],
) -> tuple[Model, dict]:
    """Build small real routes with the same planner interface as a user model."""
    config = ExecutionConfig(axis_widths=widths)
    if route == CELL_AXIS:
        return (
            nbegm_ride_along_toy.build_model(
                variant="nbegm",
                n_periods=3,
                n_liquid=12,
                n_savings=16,
                execution_config=config,
                nbegm_overrides={"envelope_arithmetic": arithmetic},
            ),
            nbegm_ride_along_toy.build_params(),
        )
    if route == INTERVAL_AXIS:
        return (
            nbegm_next_asset_cliff_toy.build_model(
                variant="nbegm",
                n_periods=3,
                n_liquid=12,
                n_savings=16,
                execution_config=config,
                envelope_arithmetic=arithmetic,
            ),
            nbegm_next_asset_cliff_toy.build_params(),
        )
    if route == BRANCH_AXIS:
        return (
            nbegm_multi_discrete_toy.build_model(
                variant="nbegm",
                n_periods=3,
                n_liquid=12,
                n_savings=16,
                execution_config=config,
                n_actions=2,
                envelope_arithmetic=arithmetic,
            ),
            nbegm_multi_discrete_toy.build_params(),
        )
    return (
        nbegm_stochastic_node_toy.build_model(
            variant="nbegm",
            n_periods=3,
            n_liquid=12,
            n_savings=16,
            execution_config=config,
            envelope_arithmetic=arithmetic,
        ),
        nbegm_stochastic_node_toy.build_params(),
    )


def _assert_arrays_agree(*, actual: object, expected: object) -> None:
    """Keep topology, dtypes, masks, and discrete decisions exact; bound levels."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for (path, got), want in zip(
        jax.tree_util.tree_flatten_with_path(actual)[0],
        jax.tree.leaves(expected),
        strict=True,
    ):
        got_array, want_array = np.asarray(got), np.asarray(want)
        assert got_array.shape == want_array.shape
        assert got_array.dtype == want_array.dtype
        if np.issubdtype(want_array.dtype, np.inexact):
            np.testing.assert_array_equal(
                np.isfinite(got_array), np.isfinite(want_array)
            )
            finite = np.isfinite(want_array)
            assert_agrees_to_ulp(
                got=got_array,
                expected=want_array,
                n_ulp=64,
                operand_magnitude=(
                    float(np.abs(want_array[finite]).max()) if finite.any() else None
                ),
                err_msg=f"leaf={jax.tree_util.keystr(path)}",
            )
        else:
            np.testing.assert_array_equal(got_array, want_array)


def _assert_solutions_agree(
    *, actual: SolutionResult, expected: SolutionResult
) -> None:
    """Values and every retained replay payload keep their addressed layout."""
    assert actual.values.keys() == expected.values.keys()
    for period, regimes in expected.values.items():
        assert actual.values[period].keys() == regimes.keys()
        for regime, values in regimes.items():
            _assert_arrays_agree(actual=actual.values[period][regime], expected=values)
    assert actual.replay_artifacts.keys() == expected.replay_artifacts.keys()
    for ref, payload in expected.replay_artifacts.items():
        actual_payload = actual.replay_artifacts[ref]
        _assert_arrays_agree(actual=actual_payload, expected=payload)
        if isinstance(payload, NBEGMGridPolicy) and payload.branch_value is not None:
            assert isinstance(actual_payload, NBEGMGridPolicy)
            np.testing.assert_array_equal(
                np.argmax(np.asarray(actual_payload.branch_value), axis=0),
                np.argmax(np.asarray(payload.branch_value), axis=0),
            )


@pytest.mark.parametrize(
    "axis", [CELL_AXIS, INTERVAL_AXIS, BRANCH_AXIS, STOCHASTIC_NODE_AXIS]
)
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize(
    "arithmetic",
    [
        "ordinary",
        pytest.param(
            "certified",
            marks=pytest.mark.requires_exact_affine_kernel(
                reason=EXACT_KERNEL_SKIP_REASON
            ),
        ),
    ],
)
def test_planner_width_preserves_values_and_replay(
    *,
    axis: str,
    width: int,
    arithmetic: ComparisonArithmetic,
) -> None:
    """Actual planned solves preserve decisions and agree at working precision."""
    model, params = _small_model(route=axis, arithmetic=arithmetic, widths={axis: 1000})
    expected = model.solve(params=params, log_level="off")
    model, params = _small_model(
        route=axis, arithmetic=arithmetic, widths={axis: width}
    )
    actual = model.solve(params=params, log_level="off")
    _assert_solutions_agree(actual=actual, expected=expected)


def test_no_ride_route_declares_no_nbegm_partition_axes() -> None:
    """A one-dimensional schedule does not inherit unused ride-along controls."""
    model = nbegm_tax_toy.build_model(
        variant="nbegm",
        n_periods=2,
        n_liquid=8,
        n_savings=8,
        envelope_arithmetic="ordinary",
    )
    for kernel in model._regimes["alive"].solution.period_kernels.values():
        for program in core_program_graph(kernel=kernel).values():
            assert not program.requirements.axes


def test_co_mapped_carry_states_are_not_inner_cell_axes() -> None:
    """A co-mapped sole ride leaves no independent inner cell mesh."""
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_periods=3,
        n_liquid=8,
        n_savings=8,
        distributed_kind=True,
        nbegm_overrides={"envelope_arithmetic": "ordinary"},
    )
    for kernel in model._regimes["alive"].solution.period_kernels.values():
        for program in core_program_graph(kernel=kernel).values():
            assert CELL_AXIS not in program.requirements.axis_names


@pytest.mark.parametrize(
    "axis", [CELL_AXIS, INTERVAL_AXIS, BRANCH_AXIS, STOCHASTIC_NODE_AXIS]
)
def test_width_changes_the_lowered_computation(axis: str) -> None:
    """A declared width changes executable work, beyond the planner's metadata."""
    model, params = _small_model(route=axis, arithmetic="ordinary", widths={})
    kernel, context = ride_along_kernel(model=model, params=params, period=0)
    materialized = materialize_core_program(
        program=core_program_graph(kernel=kernel)["replay"],
        context=CoreBuildContext(**context),
    )
    full_widths = {item.name: item.extent for item in materialized.requirements.axes}
    bodies = []
    for widths in (full_widths, {**full_widths, axis: 1}):
        resolved = resolve_core_program(
            program=materialized,
            tile_widths=widths,
            input_transfer_plan=_aligned_transfer_plan(program=materialized),
        )
        lowered = jax.jit(partial(resolved.function, **resolved.static_kwargs)).lower(
            **resolved.arguments
        )
        bodies.append(str(lowered.compiler_ir(dialect="stablehlo")))
    assert bodies[0] != bodies[1]


def _aligned_transfer_plan(
    *, program: MaterializedCoreProgram
) -> tuple[ResolvedValueTransfer, ...]:
    """Preserve every captured input leaf at its actual stored layout."""
    transfers = []
    for read in program.requirements.value_reads:
        leaf = _value_read_argument_leaf(program=program, read=read)
        assert isinstance(leaf, jax.Array)
        transfers.append(
            ResolvedValueTransfer(
                target=read.target,
                source=read.source,
                kind=ValueTransferKind.ALIGNED_LOCAL,
                stored_sharding=leaf.sharding,
                source_sharding=leaf.sharding,
                expected_shape=leaf.shape,
                expected_dtype=leaf.dtype,
            )
        )
    return tuple(transfers)


def _constant_one() -> float:
    return 1.0


@pytest.mark.parametrize("route", [INTERVAL_AXIS, BRANCH_AXIS])
@pytest.mark.parametrize("slot", ["states", "functions"])
def test_planner_width_names_cannot_be_user_names(*, route: str, slot: str) -> None:
    """Internal width names use the separator public declarations reserve."""
    model, _ = _small_model(route=route, arithmetic="ordinary", widths={})
    kernel = model._regimes["alive"].solution.period_kernels[0]
    program = core_program_graph(kernel=kernel)["replay"]
    for axis in program.requirements.axes:
        value = (
            LinSpacedGrid(start=1.0, stop=2.0, n_points=2)
            if slot == "states"
            else _constant_one
        )
        declarations: dict[str, Any] = {slot: {axis.width_keyword: value}}
        with pytest.raises(RegimeInitializationError, match="reserved separator"):
            Regime(transition=None, **declarations)


def test_interval_coordinates_do_not_replace_a_legal_user_state() -> None:
    """Materialization retains user data beside the distinct interval coordinates."""
    name = "_lcm_interval_indices"
    grid = LinSpacedGrid(start=11.0, stop=13.0, n_points=3)
    Regime(transition=None, states={name: grid})
    model, params = _small_model(route=INTERVAL_AXIS, arithmetic="ordinary", widths={})
    kernel, context = ride_along_kernel(model=model, params=params, period=0)
    points = grid.to_jax()
    space = context["state_action_space"]
    materialized = materialize_core_program(
        program=core_program_graph(kernel=kernel)["replay"],
        context=CoreBuildContext(
            **{
                **context,
                "state_action_space": replace(
                    space, states=MappingProxyType({**space.states, name: points})
                ),
            }
        ),
    )
    assert materialized.arguments[name] is points
    interval = next(
        axis
        for axis in materialized.requirements.reduced_axes
        if axis.name == INTERVAL_AXIS
    )
    (coordinate_name,) = interval.coordinate_names
    assert coordinate_name != name
    np.testing.assert_array_equal(
        materialized.arguments[coordinate_name], np.arange(interval.extent)
    )


@pytest.mark.parametrize(
    "axis", ["envelope_segment", BRANCH_AXIS, INTERVAL_AXIS, STOCHASTIC_NODE_AXIS]
)
def test_a_ride_only_model_refuses_axes_it_does_not_use(axis: str) -> None:
    """The legal-name union contains declarations, including no retired segment axis."""
    with pytest.raises(ExecutionPlanningError, match=axis):
        _small_model(route=CELL_AXIS, arithmetic="ordinary", widths={axis: 1})


@pytest.mark.parametrize(
    "arithmetic",
    [
        "ordinary",
        pytest.param(
            "certified",
            marks=pytest.mark.requires_exact_affine_kernel(
                reason=EXACT_KERNEL_SKIP_REASON
            ),
        ),
    ],
)
def test_default_interval_stream_preserves_the_dense_period(
    arithmetic: ComparisonArithmetic,
) -> None:
    """Default streaming agrees with the preserved one-shot step on one carry."""
    model, params = _small_model(route=INTERVAL_AXIS, arithmetic=arithmetic, widths={})
    kernel, context = ride_along_kernel(model=model, params=params, period=0)
    assert kernel.cliff_candidates
    child = context["next_regime_to_continuation"]["alive"]
    assert child.breakpoints is not None
    assert np.isfinite(np.asarray(child.breakpoints)).any()
    materialized = materialize_core_program(
        program=core_program_graph(kernel=kernel)["replay"],
        context=CoreBuildContext(**context),
    )
    resolved = resolve_core_program(
        program=materialized,
        input_transfer_plan=_aligned_transfer_plan(program=materialized),
        tile_widths={
            axis.name: bootstrap_width(extent=axis.extent)
            for axis in materialized.requirements.axes
        },
    )
    streamed = jax.jit(partial(resolved.function, **resolved.static_kwargs))(
        **resolved.arguments
    )
    dense_keywords = {**resolved.static_kwargs, "__lcm_interval_width__": 0}
    dense = jax.jit(partial(resolved.function, **dense_keywords))(**resolved.arguments)
    _assert_arrays_agree(actual=streamed, expected=dense)
