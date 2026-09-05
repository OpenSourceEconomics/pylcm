"""The NEGM kernel publishes a native two-program graph.

`keeper` is the inner passive DC-EGM's own program under a new name: it solves
the regime once with the durable stock held at its no-adjustment level.
`outer_sweep` is one deliberately dense program that sweeps the exogenous outer
grid inside the compiled program, binding the outer post-decision per node,
takes the exact maximum of the keeper value and every node value, and stacks the
keeper carry with every node carry on the candidate axis after lifting each
into common cash on hand. Its builder delegates to the inner adjuster's builder
with the first outer node bound and adds the outer nodes and the credited-cost
shifts; the keeper's value and carry reach it through the internal edge the
graph declares, so calling the kernel runs the keeper and hands its outputs to
the sweep under the declared argument names. The compiled sweep agrees with a
keeper-then-per-node loop to the ULP for every outer batch size, the batch size
being a vmap width and nothing else.
"""

import functools
import logging
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.outer_envelope import build_stacked_outer_carry
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    InternalInputRef,
    InternalOutputSpec,
    MaterializedCoreProgram,
    ProgramScope,
    core_program_graph,
    materialize_core_program,
)
from _lcm.execution.internal_outputs import (
    internal_input_templates,
    topological_program_order,
)
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.solution import backward_induction, period_replay
from _lcm.solution.negm import (
    _COH_SHIFTS,
    _KEEPER_CARRY,
    _KEEPER_VALUE,
    _OUTER_NODES,
    _NodeSolver,
    _OuterCostAtCell,
    _with_outer_post_decision,
)
from _lcm.solution.period_replay import replay_period
from _lcm.typing import FlatParams
from lcm.solver_api import EGM_CONTINUATION, KernelOutput
from tests.conftest import X64_ENABLED, assert_agrees_to_ulp
from tests.solution._nbegm_direct_oracle import ride_along_kernel
from tests.test_models import negm_kinked_toy

_REGIME = "alive"
_PERIOD = 1
_PARAMS: dict[str, Any] = {"discount_factor": 0.95, "alive": {}}
_LOGGER = logging.getLogger(__name__)
_SWEEP_REASON = "deliberately_dense:negm_outer_candidates_retained_not_reduced"
_KEEPER_REASON = "deliberately_dense:dcegm_solver_owned_node_and_grid_batching"
_N_OUTER = negm_kinked_toy.N_AZ
# The sweep's block width only reschedules the `lax.map` over the outer nodes,
# leaving every operation and its operand order untouched; the compiled sweep
# and the per-node loop differ only by the vectorized kernel XLA emits per block
# width — a gap of a few ULP, not of an economic magnitude.
_INVARIANCE_ULP = 16


@pytest.fixture(scope="module")
def captured() -> tuple[Any, dict[str, Any]]:
    """The kinked toy's NEGM kernel at one period and the solve's inputs to it."""
    return ride_along_kernel(
        model=negm_kinked_toy.build_model(),
        params=_PARAMS,
        regime_name=_REGIME,
        period=_PERIOD,
    )


def _build_context(context: Mapping[str, Any]) -> CoreBuildContext:
    return CoreBuildContext(
        state_action_space=context["state_action_space"],
        next_regime_to_V_arr=context["next_regime_to_V_arr"],
        next_regime_to_continuation=context["next_regime_to_continuation"],
        flat_params=context["flat_params"],
        period=context["period"],
        ages=context["ages"],
    )


def _compiled_cores(*, kernel: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    """Compile every program of the graph the way the solve loop does.

    Programs are visited so every producer precedes its consumers, and each
    consumer is lowered against the abstract templates of the internal inputs it
    declares, exactly as the solve loop lowers them.
    """
    build_context = _build_context(context)
    graph = core_program_graph(kernel=kernel)
    compiled: dict[str, Any] = {}
    producers: dict[str, MaterializedCoreProgram] = {}
    for name in topological_program_order(graph=graph):
        materialized = materialize_core_program(
            program=graph[name], context=build_context
        )
        producers[name] = materialized
        templates = internal_input_templates(program=materialized, producers=producers)
        compiled[name] = (
            jax.jit(materialized.function)
            .lower(**materialized.arguments, **templates)
            .compile()
        )
    return compiled


def _call(*, kernel: Any, context: Mapping[str, Any]) -> KernelOutput:
    return cast(
        "KernelOutput",
        kernel(
            compiled_cores=_compiled_cores(kernel=kernel, context=context),
            logger=_LOGGER,
            **context,
        ),
    )


def _keeper_then_per_node_loop(
    *, kernel: Any, context: Mapping[str, Any]
) -> tuple[Any, EGMCarry]:
    """The keeper followed by one adjuster solve per outer node.

    The reference the compiled sweep replaces: the keeper program, then the
    adjuster program once per exogenous node with the outer post-decision bound
    into the regime's params and the kernel's fixed params supplied, the
    running exact maximum of the values, and the candidate stack of the carries.
    """
    build_context = _build_context(context)
    keeper = materialize_core_program(
        program=core_program_graph(kernel=kernel.keeper_kernel)["main"],
        context=build_context,
    )
    V_arr, keeper_carry = jax.jit(keeper.function)(**keeper.arguments)
    adjuster_program = core_program_graph(kernel=kernel.adjuster_kernel)["main"]
    adjuster_core = jax.jit(adjuster_program.function)
    carries: list[EGMCarry] = []
    for index in range(kernel.outer_grid_values.shape[0]):
        bound = replace(
            build_context,
            flat_params=_with_outer_post_decision(
                flat_params=context["flat_params"],
                regime_name=_REGIME,
                outer_post_decision=kernel.outer_post_decision,
                value=kernel.outer_grid_values[index],
            ),
        )
        adjuster = materialize_core_program(program=adjuster_program, context=bound)
        node_value, node_carry = adjuster_core(
            **adjuster.arguments, **kernel.fixed_sweep_kwargs
        )
        V_arr = jax.numpy.maximum(V_arr, node_value)
        carries.append(node_carry)
    coh_shifts = kernel.coh_shift_func(
        durable_values=kernel.durable_grid_values,
        outer_values=kernel.outer_grid_values,
        **context["flat_params"][_REGIME],
    )
    carry = build_stacked_outer_carry(
        keeper_carry=keeper_carry,
        adjuster_carries=tuple(carries),
        coh_shifts=coh_shifts,
        durable_axis=kernel.durable_axis_in_carry,
    )
    return V_arr, carry


def test_the_graph_publishes_the_keeper_and_the_outer_sweep(*, captured):
    kernel, _ = captured
    graph = core_program_graph(kernel=kernel)

    assert tuple(graph) == ("keeper", "outer_sweep")
    keeper, sweep = graph["keeper"], graph["outer_sweep"]
    assert keeper.disposition is CoreExecutionDisposition.DENSE
    assert keeper.disposition_reason == _KEEPER_REASON
    assert sweep.disposition is CoreExecutionDisposition.DENSE
    assert sweep.disposition_reason == _SWEEP_REASON
    assert keeper.scope is ProgramScope.VALUES_ONLY
    assert sweep.scope is ProgramScope.ANY
    assert sweep.requirements.streamable_axes == ()
    assert sweep.requirements.target_value_accesses == ()


def test_the_keeper_program_is_the_inner_keepers_program_under_a_new_name(*, captured):
    kernel, _ = captured
    keeper = core_program_graph(kernel=kernel)["keeper"]
    inner = core_program_graph(kernel=kernel.keeper_kernel)["main"]

    assert keeper.name == "keeper"
    assert keeper.function is inner.function
    assert keeper.argument_builder is inner.argument_builder
    assert keeper.output_roles == inner.output_roles


def test_the_sweep_publishes_the_value_and_the_stacked_carry_on_the_durable_axis(
    *, captured
):
    """Every carry row leads with the durable axis; the candidate axis replicates."""
    kernel, _ = captured
    value_role, carry_roles = cast(
        "tuple[Any, Any]", core_program_graph(kernel=kernel)["outer_sweep"].output_roles
    )

    row = StateAxesLeading(state_names=("illiquid",))
    assert value_role is VALUE
    assert isinstance(carry_roles, EGMCarry)
    assert (
        carry_roles.endog_grid,
        carry_roles.value,
        carry_roles.marginal_utility,
        carry_roles.taste_shock_scale,
        carry_roles.breakpoints,
        carry_roles.policy,
    ) == (row, row, row, StateAxesLeading(state_names=(), shape=()), None, None)


def test_the_sweep_declares_the_keepers_value_and_carry_as_internal_inputs(*, captured):
    """The sweep names the keeper's outputs; the keeper publishes them by label."""
    kernel, _ = captured
    graph = core_program_graph(kernel=kernel)

    assert graph["keeper"].internal_outputs == (
        InternalOutputSpec(label="value", path=(0,)),
        InternalOutputSpec(label="carry", path=(1,)),
    )
    assert dict(graph["outer_sweep"].requirements.internal_inputs) == {
        _KEEPER_VALUE: InternalInputRef(producer="keeper", label="value"),
        _KEEPER_CARRY: InternalInputRef(producer="keeper", label="carry"),
    }


def test_the_sweep_builder_binds_the_first_node_and_the_sweeps_own_inputs(*, captured):
    """Keeper outputs reach the sweep only through the declared internal edge."""
    kernel, context = captured
    arguments = cast(
        "Mapping[str, Any]",
        materialize_core_program(
            program=core_program_graph(kernel=kernel)["outer_sweep"],
            context=_build_context(context),
        ).arguments,
    )

    assert _KEEPER_VALUE not in arguments
    assert _KEEPER_CARRY not in arguments
    assert "next_regime_to_V_arr" not in arguments
    np.testing.assert_array_equal(arguments[_OUTER_NODES], kernel.outer_grid_values)
    assert arguments[kernel.outer_post_decision] == kernel.outer_grid_values[0]
    assert arguments[_COH_SHIFTS].shape == (
        kernel.durable_grid_values.shape[0],
        _N_OUTER,
    )


def test_the_sweep_transport_keys_live_in_an_engine_only_namespace():
    """No name a model can declare is reserved for the sweep's own inputs.

    Public regime, function, state, and action names cannot contain the `__`
    qualified-name separator, so a key that starts with it is unreachable
    from any supported model.
    """
    keys = {_KEEPER_VALUE, _KEEPER_CARRY, _OUTER_NODES, _COH_SHIFTS}

    assert all(key.startswith("__lcm_negm_") for key in keys)
    assert len(keys) == 4


def test_the_kernel_returns_a_public_output_with_the_stacked_continuation(*, captured):
    kernel, context = captured
    output = _call(kernel=kernel, context=context)

    assert isinstance(output, KernelOutput)
    assert set(output.continuations) == {EGM_CONTINUATION}
    assert not output.replay
    carry = cast("EGMCarry", output.continuations[EGM_CONTINUATION])
    assert carry.endog_grid.shape[-2] == _N_OUTER + 1


def _carry_leaves_with_paths(carry: EGMCarry) -> list[tuple[str, Any]]:
    paths = jax.tree.leaves(
        jax.tree.map_with_path(lambda path, _leaf: str(path), carry)
    )
    return list(zip(paths, jax.tree.leaves(carry), strict=True))


_WIDTHS = [0, 1, 2, 3, 5, _N_OUTER]


@pytest.mark.parametrize("outer_batch_size", _WIDTHS)
def test_the_compiled_sweep_value_agrees_with_the_per_node_loop(
    *, captured, outer_batch_size: int
):
    """The value is the exact maximum over candidates, so it agrees to a few ULP.

    The sweep and the per-node loop evaluate the same operations in the same
    operand order; they differ only by the vectorized kernel XLA emits for the
    sweep's block width, a gap of a few ULP rather than of an economic
    magnitude. A partition-dependent reduction would move the value by orders
    of magnitude more.
    """
    kernel, context = captured
    expected_value, _ = _keeper_then_per_node_loop(kernel=kernel, context=context)
    output = _call(
        kernel=replace(kernel, outer_batch_size=outer_batch_size), context=context
    )

    assert_agrees_to_ulp(
        got=output.value, expected=expected_value, n_ulp=_INVARIANCE_ULP
    )


@pytest.mark.parametrize("outer_batch_size", _WIDTHS)
def test_the_compiled_sweep_carry_rows_agree_with_the_per_node_loop(
    *, captured, outer_batch_size: int
):
    """Under x64, block width preserves every carry row's support exactly.

    NaN and infinity placement is representation-level support metadata: a
    parent continuation read turns the non-NaN prefix into its valid length.
    Under float64 it agrees exactly with the keeper-then-per-node reference at
    every width, and the finite rows agree to a few ULP at the row bank's
    operand scale. At float32 the row support is a rounded structural decision
    of the compiled inner adjuster: the per-node jit and a sequential scan of
    the same program already differ on a handful of near-tie cells, so no
    compiled sweep can reproduce that reference exactly, and what float32 owes
    is the value-level statement checked end to end below.
    """
    if not X64_ENABLED:
        pytest.skip("x64 run only; float32 support is checked end to end")
    kernel, context = captured
    _, expected_carry = _keeper_then_per_node_loop(kernel=kernel, context=context)
    output = _call(
        kernel=replace(kernel, outer_batch_size=outer_batch_size), context=context
    )

    got_carry = cast("EGMCarry", output.continuations[EGM_CONTINUATION])
    assert jax.tree.structure(got_carry) == jax.tree.structure(expected_carry)
    for (path, expected), (_path, got) in zip(
        _carry_leaves_with_paths(expected_carry),
        _carry_leaves_with_paths(got_carry),
        strict=True,
    ):
        got_arr = np.asarray(got)
        expected_arr = np.asarray(expected)
        for label, predicate in (
            ("NaN", np.isnan),
            ("positive-infinity", np.isposinf),
            ("negative-infinity", np.isneginf),
        ):
            np.testing.assert_array_equal(
                predicate(got_arr),
                predicate(expected_arr),
                err_msg=f"{path}: {label} support differs by outer_batch_size",
            )
        finite = expected_arr[np.isfinite(expected_arr)]
        assert_agrees_to_ulp(
            got=got_arr,
            expected=expected_arr,
            n_ulp=_INVARIANCE_ULP,
            err_msg=path,
            operand_magnitude=float(np.max(np.abs(finite))) if finite.size else None,
        )


@pytest.mark.parametrize("outer_batch_size", [1, 2, 3, 5, 7, _N_OUTER])
def test_block_width_leaves_every_periods_solved_values_within_ulp(
    *, outer_batch_size: int
):
    """The parent-read differential: block width is not observable in any period.

    Every earlier period reads the sweep's carry through its valid prefix, so a
    width-dependent support decision would surface as a value change upstream.
    Solving the kinked toy at each width and comparing every period's value
    array with the one-block solve bounds that effect at both precisions; the
    finiteness pattern of the values must agree exactly.
    """
    reference = negm_kinked_toy.build_model().solve(params=_PARAMS, log_level="off")
    solution = negm_kinked_toy.build_model(outer_batch_size=outer_batch_size).solve(
        params=_PARAMS, log_level="off"
    )

    assert solution.values.keys() == reference.values.keys()
    for period, regime_to_value in reference.values.items():
        for regime, expected in regime_to_value.items():
            got = solution.values[period][regime]
            np.testing.assert_array_equal(
                np.isfinite(np.asarray(got)),
                np.isfinite(np.asarray(expected)),
                err_msg=f"period {period}, regime {regime}: finiteness differs",
            )
            assert_agrees_to_ulp(
                got=got,
                expected=expected,
                n_ulp=_INVARIANCE_ULP,
                err_msg=f"period {period}, regime {regime}",
            )


def _fixed_flat_params() -> FlatParams:
    return cast(
        "FlatParams",
        MappingProxyType({_REGIME: MappingProxyType({"final_age_alive": 30.0})}),
    )


def test_fixed_params_bind_into_the_sweep_the_keeper_and_the_shift(*, captured):
    kernel, _ = captured
    bound = kernel.with_fixed_params(fixed_flat_params=_fixed_flat_params())
    graph = core_program_graph(kernel=bound)

    def keywords(name: str) -> Mapping[str, Any]:
        return cast("functools.partial[Any]", graph[name].function).keywords

    assert keywords("outer_sweep")["final_age_alive"] == 30.0
    assert keywords("keeper")["final_age_alive"] == 30.0
    assert isinstance(bound.coh_shift_func, functools.partial)
    assert bound.coh_shift_func.keywords["final_age_alive"] == 30.0


def test_periods_sharing_one_inner_core_share_one_sweep_lowering_key():
    """An age-invariant regime compiles its sweep once, with or without fixed params."""
    kernels = negm_kinked_toy.build_model()._regimes[_REGIME].solution.period_kernels
    first, second = (kernels[period] for period in sorted(kernels)[:2])
    fixed = _fixed_flat_params()

    def key(kernel: Any) -> Any:
        return backward_induction._func_dedup_key(
            func=core_program_graph(kernel=kernel)["outer_sweep"].function
        )

    assert key(first) == key(second)
    assert key(first.with_fixed_params(fixed_flat_params=fixed)) == key(
        second.with_fixed_params(fixed_flat_params=fixed)
    )


def test_a_replay_lowers_the_dense_programs_the_solve_ran(*, monkeypatch, tmp_path):
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", f"{_REGIME}@{_PERIOD}")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    solution = negm_kinked_toy.build_model().solve(params=_PARAMS, log_level="off")
    dispositions: list[CoreExecutionDisposition] = []
    original = period_replay.core_program_graph

    def record_graph(**kwargs: Any) -> Any:
        graph = original(**kwargs)
        dispositions.extend(program.disposition for program in graph.values())
        return graph

    monkeypatch.setattr(period_replay, "core_program_graph", record_graph)
    replay = replay_period(directory=tmp_path / f"{_REGIME}@{_PERIOD}")

    assert dispositions == [CoreExecutionDisposition.DENSE] * 2
    assert_agrees_to_ulp(
        got=np.asarray(replay.output.value),
        expected=np.asarray(solution.values[_PERIOD][_REGIME]),
        n_ulp=1,
    )


def test_the_node_solver_compares_by_identity_so_an_array_field_cannot_break_it() -> (
    None
):
    """Two node solvers built from equal arguments are distinct objects.

    The solver carries the adjuster's argument tree, whose leaves are arrays, so
    comparing two solvers field by field would take the truth value of an array and
    raise. Comparison is by identity instead.
    """
    first = _NodeSolver(
        inner_core=_unused_inner_core,
        outer_post_decision="new_illiquid",
        adjuster_arguments=MappingProxyType({"wealth": jnp.arange(3.0)}),
    )
    second = _NodeSolver(
        inner_core=_unused_inner_core,
        outer_post_decision="new_illiquid",
        adjuster_arguments=MappingProxyType({"wealth": jnp.arange(3.0)}),
    )

    assert first != second


def test_the_outer_cost_cell_compares_by_identity_when_a_param_is_an_array() -> None:
    """Two cost cells built from equal params are distinct objects.

    The cell carries the regime's flat params, which may hold arrays, so comparing
    two cells field by field would take the truth value of an array and raise.
    Comparison is by identity instead.
    """
    first = _OuterCostAtCell(
        cost_func=_unused_cost_func,
        cost_arg_names=frozenset({"illiquid"}),
        durable_state_name="illiquid",
        outer_post_decision="new_illiquid",
        params=MappingProxyType({"interest_rate": jnp.arange(3.0)}),
    )
    second = _OuterCostAtCell(
        cost_func=_unused_cost_func,
        cost_arg_names=frozenset({"illiquid"}),
        durable_state_name="illiquid",
        outer_post_decision="new_illiquid",
        params=MappingProxyType({"interest_rate": jnp.arange(3.0)}),
    )

    assert first != second


def _unused_inner_core(**kwargs: object) -> tuple[Any, Any]:
    """Stand in for the adjuster program; the comparison tests never call it."""
    raise AssertionError(kwargs)


def _unused_cost_func(**kwargs: object) -> Any:
    """Stand in for the cost DAG; the comparison tests never call it."""
    raise AssertionError(kwargs)
