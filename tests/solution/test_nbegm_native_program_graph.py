"""A ride-along NB-EGM period publishes one native core-program graph.

The graph is the kernel's sole core authority. `main` solves the period for a
values-only retention and publishes the value array and the carry; `replay` is the
same tile-local body and additionally publishes the consumption policy and, for a
discrete ride action, the conditional branch banks. The engine dispatches exactly one
of them per solve, chosen by the result retention.
"""

import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import jax
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    ProgramScope,
    core_program_graph,
    materialize_core_program,
)
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from tests.solution._nbegm_direct_oracle import ride_along_kernel
from tests.test_models import (
    nbegm_jump_ride_along_toy,
    nbegm_ride_along_toy,
    nbegm_ride_discrete_toy,
)

_SMALL: dict[str, Any] = {"n_liquid": 12, "n_savings": 16}


def _smooth_kernel(**overrides: Any) -> tuple[Any, dict[str, Any]]:
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm", n_periods=3, **_SMALL, **overrides
    )
    return ride_along_kernel(model=model, params=nbegm_ride_along_toy.build_params())


def _discrete_kernel() -> tuple[Any, dict[str, Any]]:
    model = nbegm_ride_discrete_toy.build_model(
        variant="nbegm", n_periods=3, n_liquid=12, n_savings=16, n_consumption=24
    )
    return ride_along_kernel(model=model, params=nbegm_ride_discrete_toy.build_params())


def _jump_kernel() -> tuple[Any, dict[str, Any]]:
    model = nbegm_jump_ride_along_toy.build_model(
        variant="nbegm", n_periods=4, **_SMALL
    )
    return ride_along_kernel(
        model=model, params=nbegm_jump_ride_along_toy.build_params()
    )


def _value_state_order(kernel: Any) -> tuple[str, ...]:
    """The published value array's state order: ride axes with liquid at its slot."""
    spec = kernel.schedule_spec
    order = list(spec.ride_along_state_names)
    order.insert(spec.liquid_axis_pos, spec.liquid_state_name)
    return tuple(order)


def _roles(*, kernel: Any, name: str) -> tuple[Any, ...]:
    """The named program's output-role tuple."""
    return cast("tuple[Any, ...]", core_program_graph(kernel=kernel)[name].output_roles)


def _build_context(context: Mapping[str, Any]) -> CoreBuildContext:
    return CoreBuildContext(
        state_action_space=context["state_action_space"],
        next_regime_to_V_arr=context["next_regime_to_V_arr"],
        next_regime_to_continuation=context["next_regime_to_continuation"],
        flat_params=context["flat_params"],
        period=context["period"],
        ages=context["ages"],
    )


def _run(*, kernel: Any, context: Mapping[str, Any], name: str) -> tuple:
    program = core_program_graph(kernel=kernel)[name]
    materialized = materialize_core_program(
        program=program, context=_build_context(context)
    )
    return tuple(jax.jit(materialized.function)(**materialized.arguments))


def test_the_graph_publishes_exactly_a_values_only_main_and_a_replay_program():
    kernel, _ = _smooth_kernel()
    graph = core_program_graph(kernel=kernel)

    assert tuple(graph) == ("main", "replay")
    assert graph["main"].scope is ProgramScope.VALUES_ONLY
    assert graph["replay"].scope is ProgramScope.REPLAY
    for program in graph.values():
        assert program.disposition is CoreExecutionDisposition.PLANNED
        assert program.disposition_reason is None
        assert program.requirements.streamable_axes == ()
        assert program.requirements.target_value_accesses == ()


def _carry_role_leaves(roles: EGMCarry) -> dict[str, object]:
    return {
        "endog_grid": roles.endog_grid,
        "value": roles.value,
        "marginal_utility": roles.marginal_utility,
        "taste_shock_scale": roles.taste_shock_scale,
        "breakpoints": roles.breakpoints,
        "policy": roles.policy,
    }


def test_main_publishes_the_value_and_a_ride_axes_leading_carry():
    kernel, _ = _smooth_kernel()
    value_role, carry_roles = _roles(kernel=kernel, name="main")

    ride = StateAxesLeading(state_names=("kind",))
    assert value_role is VALUE
    assert isinstance(carry_roles, EGMCarry)
    assert _carry_role_leaves(carry_roles) == {
        "endog_grid": ride,
        "value": ride,
        "marginal_utility": ride,
        "taste_shock_scale": StateAxesLeading(state_names=(), shape=()),
        "breakpoints": None,
        "policy": ride,
    }


def test_a_jump_schedule_carries_breakpoints_and_no_policy_rows():
    kernel, _ = _jump_kernel()
    _, carry_roles = _roles(kernel=kernel, name="main")

    ride = StateAxesLeading(state_names=("kind",))
    assert carry_roles.breakpoints == ride
    assert carry_roles.policy is None


def test_replay_adds_the_policy_on_the_value_layout():
    kernel, _ = _smooth_kernel()

    main_roles = _roles(kernel=kernel, name="main")
    replay_roles = _roles(kernel=kernel, name="replay")
    assert replay_roles[:2] == main_roles
    assert replay_roles[2:] == (
        StateAxesLeading(state_names=_value_state_order(kernel)),
    )


def test_replay_adds_the_branch_banks_for_a_discrete_ride_action():
    kernel, _ = _discrete_kernel()
    replay_roles = _roles(kernel=kernel, name="replay")

    state_order = _value_state_order(kernel)
    bank = StateAxesLeading(state_names=state_order, n_free_leading_axes=1)
    assert replay_roles[2:] == (
        StateAxesLeading(state_names=state_order),
        bank,
        bank,
    )


def test_the_builder_omits_target_values_and_filters_the_carry():
    kernel, context = _smooth_kernel()
    materialized = materialize_core_program(
        program=core_program_graph(kernel=kernel)["main"],
        context=_build_context(context),
    )

    assert "next_regime_to_V_arr" not in materialized.arguments
    carry = cast(
        "Mapping[str, Any]", materialized.arguments["next_regime_to_continuation"]
    )
    assert set(carry) == set(kernel.stateful_targets)


def test_with_fixed_params_rebinds_both_programs():
    kernel, _ = _smooth_kernel()
    graph = core_program_graph(kernel=kernel)
    fixed = MappingProxyType({kernel.regime_name: MappingProxyType({"beta": 0.9})})

    bound = kernel.with_fixed_params(fixed_flat_params=fixed)
    bound_graph = core_program_graph(kernel=bound)

    for name, program in graph.items():
        function = bound_graph[name].function
        assert isinstance(function, functools.partial)
        assert function.func is program.function
        assert function.keywords["beta"] == 0.9
    main_bound = cast("functools.partial", bound_graph["main"].function)
    replay_bound = cast("functools.partial", bound_graph["replay"].function)
    assert main_bound.keywords == replay_bound.keywords
    assert kernel.with_fixed_params(fixed_flat_params=MappingProxyType({})) is kernel


def test_main_and_replay_publish_the_same_value_and_carry():
    kernel, context = _smooth_kernel()
    main_value, main_carry = _run(kernel=kernel, context=context, name="main")
    replay_value, replay_carry, _policy = _run(
        kernel=kernel, context=context, name="replay"
    )

    np.testing.assert_array_equal(np.asarray(main_value), np.asarray(replay_value))
    for main_leaf, replay_leaf in zip(
        jax.tree.leaves(main_carry), jax.tree.leaves(replay_carry), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(main_leaf), np.asarray(replay_leaf))
