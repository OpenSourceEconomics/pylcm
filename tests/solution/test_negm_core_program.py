"""The NEGM kernel publishes a native two-program graph.

`keeper` is the inner passive DC-EGM's own program under a new name: it solves
the regime once with the durable stock held at its no-adjustment level.
`outer_sweep` is one deliberately dense program that sweeps the exogenous outer
grid inside the compiled program, binding the outer post-decision per node,
takes the exact maximum of the keeper value and every node value, and stacks the
keeper carry with every node carry on the candidate axis after lifting each
into common cash on hand. Its builder delegates to the inner adjuster's builder
with the first outer node bound and adds the outer nodes, the credited-cost
shifts, and placeholders in the keeper outputs' shape; calling the kernel runs
the keeper, replaces exactly those placeholders, and dispatches the sweep. The
compiled sweep agrees with a keeper-then-per-node loop to the ULP for every
outer batch size, the batch size being a vmap width and nothing else.
"""

import ast
import functools
import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import jax
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.outer_envelope import build_stacked_outer_carry
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    ProgramScope,
    core_program_graph,
    materialize_core_program,
)
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.solution import backward_induction, negm, period_replay
from _lcm.solution.negm import _with_outer_post_decision
from _lcm.solution.period_replay import replay_period
from _lcm.typing import FlatParams
from lcm.solver_api import EGM_CONTINUATION, KernelOutput
from tests.conftest import X64_ENABLED, assert_agrees_to_ulp
from tests.solution._nbegm_direct_oracle import ride_along_kernel
from tests.test_models import negm_bequest_toy, negm_kinked_toy

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
    """Compile every program of the graph the way the solve loop does."""
    build_context = _build_context(context)
    return {
        name: jax.jit(
            materialize_core_program(program=program, context=build_context).function
        )
        for name, program in core_program_graph(kernel=kernel).items()
    }


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
    V_arr, keeper_carry, _ = jax.jit(keeper.function)(**keeper.arguments)
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
        node_value, node_carry, _ = adjuster_core(
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
    assert {keeper.scope, sweep.scope} == {ProgramScope.ANY}
    assert sweep.requirements.streamable_axes == ()
    assert sweep.requirements.target_value_accesses == ()


@pytest.mark.parametrize(
    "legacy_name", ["cores", "core", "build_lower_args", "_outer_nodes"]
)
def test_no_legacy_core_authority_survives_on_the_kernel(*, captured, legacy_name):
    kernel, _ = captured
    assert not hasattr(kernel, legacy_name)


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


def test_the_sweep_builder_binds_the_first_node_and_keeper_shaped_placeholders(
    *, captured
):
    kernel, context = captured
    build_context = _build_context(context)
    arguments = cast(
        "Mapping[str, Any]",
        materialize_core_program(
            program=core_program_graph(kernel=kernel)["outer_sweep"],
            context=build_context,
        ).arguments,
    )
    keeper = materialize_core_program(
        program=core_program_graph(kernel=kernel)["keeper"], context=build_context
    )
    keeper_value, keeper_carry, _ = jax.jit(keeper.function)(**keeper.arguments)

    assert "next_regime_to_V_arr" not in arguments
    np.testing.assert_array_equal(arguments["outer_nodes"], kernel.outer_grid_values)
    assert arguments[kernel.outer_post_decision] == kernel.outer_grid_values[0]
    assert arguments["coh_shifts"].shape == (
        kernel.durable_grid_values.shape[0],
        _N_OUTER,
    )
    assert (arguments["keeper_value"].shape, arguments["keeper_value"].dtype) == (
        keeper_value.shape,
        keeper_value.dtype,
    )
    assert jax.tree.structure(arguments["keeper_carry"]) == jax.tree.structure(
        keeper_carry
    )
    assert jax.tree.map(
        lambda placeholder, leaf: (
            placeholder.shape == leaf.shape and placeholder.dtype == leaf.dtype
        ),
        arguments["keeper_carry"],
        keeper_carry,
    ) == jax.tree.map(lambda _leaf: True, keeper_carry)


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


@pytest.mark.parametrize("outer_batch_size", [0, 1, 3, _N_OUTER])
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


@pytest.mark.parametrize("outer_batch_size", [0, 1, 3, _N_OUTER])
def test_the_compiled_sweep_carry_rows_agree_with_the_per_node_loop(
    *, captured, outer_batch_size: int
):
    """Every carry row agrees with the loop's to a few ULP at the row bank's scale.

    A row is formed from operands at the bank's own scale: the savings nodes,
    the consumption the Euler inversion returns, and the credited-cost lift
    that adds shifts of the grid's magnitude, so the spacing at that scale is
    the unit the rows are compared in. Dead cells must coincide exactly. At
    float32 a row lands on adjacent neighbours across block widths and a dead-
    cell decision at a near tie follows it, so the row bank is compared under
    float64 only; the value test above covers both precisions.
    """
    if not X64_ENABLED:
        pytest.skip("x64 run only")
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
        finite = np.asarray(expected)[np.isfinite(np.asarray(expected))]
        assert_agrees_to_ulp(
            got=got,
            expected=expected,
            n_ulp=_INVARIANCE_ULP,
            err_msg=path,
            operand_magnitude=float(np.max(np.abs(finite))) if finite.size else None,
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


def test_the_loop_never_repairs_the_negm_value(*, monkeypatch):
    def refuse(**_kwargs: object) -> None:
        msg = "the NEGM value must be born in its planned layout"
        raise AssertionError(msg)

    monkeypatch.setattr(backward_induction, "_repair_unplanned_kernel_value", refuse)

    negm_bequest_toy.build_negm_model().solve(params=_PARAMS, log_level="off")


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
        got=np.asarray(replay.result.V_arr),
        expected=np.asarray(solution.values[_PERIOD][_REGIME]),
        n_ulp=1,
    )


def test_negm_holds_no_legacy_result_type_and_no_per_node_python_loop():
    module = ast.parse(Path(negm.__file__).read_text())
    names = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
    defined = {
        node.name
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    assert "KernelResult" not in names
    assert defined.isdisjoint({"cores", "build_lower_args", "_outer_nodes"})
