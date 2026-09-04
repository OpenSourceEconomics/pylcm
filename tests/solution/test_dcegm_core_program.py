"""A DC-EGM period publishes output-specialized dense core programs.

The DC-EGM kernel owns its stochastic-node and grid batching, so both variants
are deliberately dense and read continuation from child carries alone. `main`
publishes only value and carry; `replay` additionally publishes the off-grid
policy when its exact retention key is selected. The NEGM composite consumes the
values variant through the public output rather than through a legacy result.
"""

import functools
import logging
from collections.abc import Mapping
from dataclasses import replace
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


def _run(*, kernel: Any, context: Mapping[str, Any], name: str = "main") -> tuple:
    program = core_program_graph(kernel=kernel)[name]
    materialized = materialize_core_program(
        program=program, context=_build_context(context)
    )
    return tuple(jax.jit(materialized.function)(**materialized.arguments))


def test_the_graph_publishes_dense_values_and_replay_variants():
    kernel, _ = _full_kernel()
    graph = core_program_graph(kernel=kernel)

    assert tuple(graph) == ("main", "replay")
    assert graph["main"].scope is ProgramScope.VALUES_ONLY
    assert graph["main"].retained_artifact_keys == ()
    assert graph["replay"].scope is ProgramScope.REPLAY
    assert graph["replay"].retained_artifact_keys == (SIMULATION_POLICY,)
    assert graph["replay"].retained_artifact_payload_types == {
        SIMULATION_POLICY: EGMSimPolicy
    }
    assert graph["replay"].replaces_program == "main"
    for program in graph.values():
        assert program.disposition is CoreExecutionDisposition.DENSE
        assert program.disposition_reason == _DENSE_REASON
        assert program.requirements.streamable_axes == ()
        assert program.requirements.target_value_accesses == ()


def test_model_authority_rejects_a_policy_type_conflicting_with_the_route() -> None:
    model = get_full_model(solver="dcegm", n_periods=_N_PERIODS)
    kernel = model._regimes[_REGIME].solution.period_kernels[_PERIOD]
    graph = dict(core_program_graph(kernel=kernel))
    graph["replay"] = replace(
        graph["replay"],
        retained_artifact_payload_types={SIMULATION_POLICY: tuple},
    )
    object.__setattr__(kernel, "_core_programs", MappingProxyType(graph))

    with pytest.raises(TypeError, match="producer and built-in replay route disagree"):
        model.solve(
            params=get_full_params(n_periods=_N_PERIODS),
            log_level="off",
            retention=ResultRetention.VALUES,
        )


def test_variants_publish_only_their_retained_outputs_by_their_row_axes():
    """Carry and policy rows lead with the discrete and passive state axes."""
    kernel, _ = _passive_kernel()
    value_role, carry_roles, policy_roles = cast(
        "tuple[Any, Any, Any]",
        core_program_graph(kernel=kernel)["replay"].output_roles,
    )
    assert core_program_graph(kernel=kernel)["main"].output_roles == (
        value_role,
        carry_roles,
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
@pytest.mark.parametrize("name", ["main", "replay"])
def test_the_declared_roles_share_the_runtime_outputs_pytree_structure(
    *, build, name: str
):
    """The role tree is the output tree with roles for leaves, aux data included."""
    kernel, context = build()
    roles = core_program_graph(kernel=kernel)[name].output_roles

    outputs = _run(kernel=kernel, context=context, name=name)

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


def test_the_kernel_publishes_only_the_selected_public_channels():
    kernel, context = _full_kernel()
    graph = core_program_graph(kernel=kernel)
    main = materialize_core_program(
        program=graph["main"], context=_build_context(context)
    )
    replay = materialize_core_program(
        program=graph["replay"], context=_build_context(context)
    )

    values_output = kernel(
        compiled_cores={"main": main.function}, **context, logger=_LOGGER
    )
    replay_output = kernel(
        compiled_cores={"replay": replay.function}, **context, logger=_LOGGER
    )

    assert isinstance(values_output, KernelOutput)
    assert tuple(values_output.continuations) == (EGM_CONTINUATION,)
    assert not values_output.replay
    assert isinstance(replay_output, KernelOutput)
    assert tuple(replay_output.continuations) == (EGM_CONTINUATION,)
    assert isinstance(replay_output.continuations[EGM_CONTINUATION], EGMCarry)
    assert tuple(replay_output.replay) == (SIMULATION_POLICY,)
    assert isinstance(replay_output.replay[SIMULATION_POLICY], EGMSimPolicy)
    assert not replay_output.solve_time_artifacts
    assert not replay_output.auxiliary
    value, carry, policy = _run(kernel=kernel, context=context, name="replay")
    np.testing.assert_array_equal(np.asarray(replay_output.value), np.asarray(value))
    for got, expected in zip(
        jax.tree.leaves(
            (
                replay_output.continuations[EGM_CONTINUATION],
                replay_output.replay[SIMULATION_POLICY],
            )
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
    assert set(core_program_graph(kernel=bound)) == {"main", "replay"}
    assert kernel.with_fixed_params(fixed_flat_params=MappingProxyType({})) is kernel


def test_a_replay_lowers_the_dense_program_the_solve_ran(*, monkeypatch, tmp_path):
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", f"{_REGIME}@{_PERIOD}")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    solution = get_full_model(solver="dcegm", n_periods=_N_PERIODS).solve(
        params=get_full_params(n_periods=_N_PERIODS), log_level="off"
    )
    selected_names: list[str] = []
    real_select = period_replay.select_programs

    def record_select(**kwargs: Any) -> Any:
        selected = real_select(**kwargs)
        selected_names.extend(selected)
        return selected

    monkeypatch.setattr(period_replay, "select_programs", record_select)
    replay = replay_period(directory=tmp_path / f"{_REGIME}@{_PERIOD}")

    assert selected_names == ["main"]
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
