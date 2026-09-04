"""A DC-EGM period publishes one native dense core program and a public output.

The DC-EGM kernel owns its stochastic-node and grid batching, so its single
program `main` is deliberately dense; it reads its continuation from the child
carries alone and declares no target value access. The program publishes the
value array, the carry a parent interpolates, and the off-grid simulation policy,
each described by the state axes that lead it, and the kernel returns them as a
public `KernelOutput` whose continuation and replay channels the solve loop
consumes. The NEGM composite reads its keeper and adjuster children through that
public output rather than through a legacy result.
"""

import functools
import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import jax
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    ProgramScope,
    core_program_graph,
    materialize_core_program,
)
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.solution import period_replay
from _lcm.solution.period_replay import replay_period
from lcm.solver_api import (
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    ArtifactRef,
    KernelOutput,
    OmissionReason,
    ResultRetention,
)
from tests.conftest import assert_agrees_to_ulp
from tests.solution._nbegm_direct_oracle import ride_along_kernel
from tests.solution.test_egm_passive import _get_model as _passive_model
from tests.solution.test_egm_passive import _get_params as _passive_params
from tests.test_models.deterministic.dcegm_variants import (
    get_full_model,
    get_full_params,
)

_N_PERIODS = 4
_REGIME = "working_life"
_PERIOD = 1
_LOGGER = logging.getLogger(__name__)
_DENSE_REASON = "deliberately_dense:dcegm_solver_owned_node_and_grid_batching"


def _full_kernel() -> tuple[Any, dict[str, Any]]:
    """The worker regime's DC-EGM kernel of the full model and its solve inputs."""
    return ride_along_kernel(
        model=get_full_model(solver="dcegm", n_periods=_N_PERIODS),
        params=get_full_params(n_periods=_N_PERIODS),
        regime_name=_REGIME,
        period=_PERIOD,
    )


def _passive_kernel() -> tuple[Any, dict[str, Any]]:
    """The kernel of a regime with a passive state and a discrete action."""
    return ride_along_kernel(
        model=_passive_model("dcegm"),
        params=_passive_params(),
        regime_name=_REGIME,
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


def _run(*, kernel: Any, context: Mapping[str, Any]) -> tuple:
    program = core_program_graph(kernel=kernel)["main"]
    materialized = materialize_core_program(
        program=program, context=_build_context(context)
    )
    return tuple(jax.jit(materialized.function)(**materialized.arguments))


def test_the_graph_publishes_one_dense_main_program():
    kernel, _ = _full_kernel()
    graph = core_program_graph(kernel=kernel)

    assert tuple(graph) == ("main",)
    program = graph["main"]
    assert program.disposition is CoreExecutionDisposition.DENSE
    assert program.disposition_reason == _DENSE_REASON
    assert program.scope is ProgramScope.ANY
    assert program.requirements.streamable_axes == ()
    assert program.requirements.target_value_accesses == ()


def test_main_publishes_the_value_the_carry_and_the_policy_by_their_row_axes():
    """Carry and policy rows lead with the discrete and passive state axes."""
    kernel, _ = _passive_kernel()
    value_role, carry_roles, policy_roles = cast(
        "tuple[Any, Any, Any]", core_program_graph(kernel=kernel)["main"].output_roles
    )

    row = StateAxesLeading(state_names=("skill",))
    assert value_role is VALUE
    assert isinstance(carry_roles, EGMCarry)
    assert (
        carry_roles.endog_grid,
        carry_roles.value,
        carry_roles.marginal_utility,
    ) == (
        row,
        row,
        row,
    )
    assert carry_roles.taste_shock_scale == StateAxesLeading(state_names=(), shape=())
    assert carry_roles.breakpoints is None
    assert carry_roles.policy is None
    assert isinstance(policy_roles, EGMSimPolicy)
    assert (
        policy_roles.endog_grid,
        policy_roles.policy,
        policy_roles.value,
        policy_roles.marginal_utility,
    ) == (row, row, row, row)
    assert policy_roles.row_discrete_state_names == ()
    assert policy_roles.row_passive_state_names == ("skill",)
    assert policy_roles.row_discrete_action_names == ("labor_supply",)


@pytest.mark.parametrize("build", [_full_kernel, _passive_kernel])
def test_the_declared_roles_share_the_runtime_outputs_pytree_structure(*, build):
    """The role tree is the output tree with roles for leaves, aux data included."""
    kernel, context = build()
    roles = core_program_graph(kernel=kernel)["main"].output_roles

    outputs = _run(kernel=kernel, context=context)

    assert jax.tree.structure(roles) == jax.tree.structure(outputs)


def test_the_builder_omits_target_values_and_filters_the_carry():
    kernel, context = _full_kernel()
    materialized = materialize_core_program(
        program=core_program_graph(kernel=kernel)["main"],
        context=_build_context(context),
    )

    assert "next_regime_to_V_arr" not in materialized.arguments
    carry = cast(
        "Mapping[str, Any]", materialized.arguments["next_regime_to_continuation"]
    )
    assert set(carry) == set(kernel.stateful_targets)


def test_the_kernel_publishes_its_continuation_and_its_policy_on_public_channels():
    kernel, context = _full_kernel()
    materialized = materialize_core_program(
        program=core_program_graph(kernel=kernel)["main"],
        context=_build_context(context),
    )

    output = kernel(
        compiled_cores={"main": materialized.function}, **context, logger=_LOGGER
    )

    assert isinstance(output, KernelOutput)
    assert tuple(output.continuations) == (EGM_CONTINUATION,)
    assert isinstance(output.continuations[EGM_CONTINUATION], EGMCarry)
    assert tuple(output.replay) == (SIMULATION_POLICY,)
    assert isinstance(output.replay[SIMULATION_POLICY], EGMSimPolicy)
    assert not output.solve_time_artifacts
    assert not output.auxiliary
    value, carry, policy = _run(kernel=kernel, context=context)
    np.testing.assert_array_equal(np.asarray(output.value), np.asarray(value))
    for got, expected in zip(
        jax.tree.leaves(
            (output.continuations[EGM_CONTINUATION], output.replay[SIMULATION_POLICY])
        ),
        jax.tree.leaves((carry, policy)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_with_fixed_params_rebinds_the_program():
    kernel, _ = _full_kernel()
    program = core_program_graph(kernel=kernel)["main"]
    fixed = MappingProxyType({_REGIME: MappingProxyType({"discount_factor": 0.9})})

    bound = kernel.with_fixed_params(fixed_flat_params=fixed)
    bound_program = core_program_graph(kernel=bound)["main"]

    function = cast("functools.partial", bound_program.function)
    assert isinstance(function, functools.partial)
    assert function.func is program.function
    assert function.keywords["discount_factor"] == 0.9
    assert kernel.with_fixed_params(fixed_flat_params=MappingProxyType({})) is kernel


def test_a_replay_lowers_the_dense_program_the_solve_ran(*, monkeypatch, tmp_path):
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", f"{_REGIME}@{_PERIOD}")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    solution = get_full_model(solver="dcegm", n_periods=_N_PERIODS).solve(
        params=get_full_params(n_periods=_N_PERIODS), log_level="off"
    )
    dispositions: list[CoreExecutionDisposition] = []
    real_graph = period_replay.core_program_graph

    def record_graph(**kwargs: Any) -> Any:
        graph = real_graph(**kwargs)
        dispositions.extend(program.disposition for program in graph.values())
        return graph

    monkeypatch.setattr(period_replay, "core_program_graph", record_graph)
    replay = replay_period(directory=tmp_path / f"{_REGIME}@{_PERIOD}")

    assert dispositions == [CoreExecutionDisposition.DENSE]
    assert_agrees_to_ulp(
        got=np.asarray(replay.output.value),
        expected=np.asarray(solution.values[_PERIOD][_REGIME]),
        n_ulp=1,
    )


@pytest.mark.parametrize(
    "retention", [ResultRetention.VALUES, ResultRetention.VALUES_AND_REPLAY]
)
def test_a_regime_without_a_policy_read_route_omits_the_policy_as_not_applicable(
    *, retention: ResultRetention
):
    """A passive state keeps simulation on the grid argmax, whatever the retention."""
    model = _passive_model("dcegm")
    solution = model.solve(
        params=_passive_params(), log_level="off", retention=retention
    )
    period = model._regimes[_REGIME].active_periods[0]
    policy_ref = ArtifactRef(period=period, regime=_REGIME, key=SIMULATION_POLICY)

    assert model._regimes[_REGIME].simulation.egm_policy_read is None
    assert not solution.replay_artifacts.project(SIMULATION_POLICY)
    assert solution.omissions[policy_ref] is OmissionReason.NOT_APPLICABLE
