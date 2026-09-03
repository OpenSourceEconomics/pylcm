"""A plain EGM period publishes one native dense core program.

The one-row kernel has no product axis to stream over and reads its continuation
from the child's carry alone, so its single program `main` is deliberately dense
and declares no target value access. The graph is the kernel's sole core
authority: the runtime call builds its arguments through the declared builder,
the builder refuses a law of motion that cannot be inverted, the value and the
carry are born in their planned placement, and a replay lowers the same program.
"""

import functools
import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

import jax
import numpy as np
import pytest

import lcm
from _lcm.egm.carry import EGMCarry
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
from lcm.exceptions import RegimeInitializationError
from lcm.solvers import EGM
from tests.conftest import assert_agrees_to_ulp
from tests.solution._nbegm_direct_oracle import ride_along_kernel
from tests.solution.test_egm_solver import _SAVINGS_GRID, _model, _params

_REGIME = "saving"
_PERIOD = 1
_LOGGER = logging.getLogger(__name__)


def _kernel() -> tuple[Any, dict[str, Any]]:
    """The EGM kernel of the middle active period and the inputs the solve gave it."""
    return ride_along_kernel(
        model=_model(solver=EGM(savings_grid=_SAVINGS_GRID)),
        params=_params(),
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


def test_the_graph_publishes_one_dense_main_program():
    kernel, _ = _kernel()
    graph = core_program_graph(kernel=kernel)

    assert tuple(graph) == ("main",)
    program = graph["main"]
    assert program.disposition is CoreExecutionDisposition.DENSE
    assert program.disposition_reason == (
        "deliberately_dense:egm_one_row_no_product_axis"
    )
    assert program.scope is ProgramScope.ANY
    assert program.requirements.streamable_axes == ()
    assert program.requirements.target_value_accesses == ()


@pytest.mark.parametrize(
    "legacy_name", ["cores", "core", "build_lower_args", "unwrapped_core"]
)
def test_no_legacy_core_authority_survives_on_the_kernel(*, legacy_name: str):
    kernel, _ = _kernel()
    assert not hasattr(kernel, legacy_name)


def test_main_publishes_the_value_and_a_one_row_carry():
    kernel, _ = _kernel()
    value_role, carry_roles = cast(
        "tuple[Any, Any]", core_program_graph(kernel=kernel)["main"].output_roles
    )

    row = StateAxesLeading(state_names=())
    assert value_role is VALUE
    assert isinstance(carry_roles, EGMCarry)
    assert carry_roles.endog_grid == row
    assert carry_roles.value == row
    assert carry_roles.marginal_utility == row
    assert carry_roles.taste_shock_scale == StateAxesLeading(state_names=(), shape=())
    assert carry_roles.breakpoints is None
    assert carry_roles.policy is None


def test_the_runtime_call_hands_the_core_exactly_the_builders_arguments():
    kernel, context = _kernel()
    program = core_program_graph(kernel=kernel)["main"]
    materialized = materialize_core_program(
        program=program, context=_build_context(context)
    )
    received: list[Mapping[str, Any]] = []

    def recording_core(**kwargs: Any) -> Any:
        received.append(kwargs)
        return program.function(**kwargs)

    kernel(compiled_cores={"main": recording_core}, **context, logger=_LOGGER)

    assert len(received) == 1
    assert set(received[0]) == set(materialized.arguments)
    for name, expected in materialized.arguments.items():
        np.testing.assert_array_equal(
            np.asarray(received[0][name]), np.asarray(expected), err_msg=name
        )


def test_the_builder_refuses_a_law_falling_in_savings():
    kernel, context = _kernel()
    flat_params = {
        regime: {
            name: (-1.5 if name.endswith("return_liquid") else value)
            for name, value in regime_params.items()
        }
        for regime, regime_params in context["flat_params"].items()
    }
    assert any(name.endswith("return_liquid") for name in flat_params[_REGIME]), (
        "the fixture carries no liquid return to flip; test is inert"
    )

    with pytest.raises(RegimeInitializationError, match="falls as savings rise"):
        materialize_core_program(
            program=core_program_graph(kernel=kernel)["main"],
            context=_build_context({**context, "flat_params": flat_params}),
        )


def test_with_fixed_params_rebinds_the_program_and_its_builder():
    kernel, context = _kernel()
    program = core_program_graph(kernel=kernel)["main"]
    fixed = MappingProxyType({_REGIME: MappingProxyType({"crra": 2.0})})

    bound = kernel.with_fixed_params(fixed_flat_params=fixed)
    bound_program = core_program_graph(kernel=bound)["main"]

    function = cast("functools.partial", bound_program.function)
    assert isinstance(function, functools.partial)
    assert function.func is program.function
    assert function.keywords["crra"] == 2.0
    assert bound_program.argument_builder is not program.argument_builder
    free_arguments = materialize_core_program(
        program=bound_program,
        context=_build_context(
            {
                **context,
                "flat_params": {
                    regime: {
                        name: value
                        for name, value in regime_params.items()
                        if name != "crra"
                    }
                    for regime, regime_params in context["flat_params"].items()
                },
            }
        ),
    ).arguments
    assert "crra" not in free_arguments
    assert kernel.with_fixed_params(fixed_flat_params=MappingProxyType({})) is kernel


def test_a_replay_lowers_the_dense_program_the_solve_ran(*, monkeypatch, tmp_path):
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", f"{_REGIME}@{_PERIOD}")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    solution = _model(solver=EGM(savings_grid=_SAVINGS_GRID)).solve(
        params=_params(), log_level="off"
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


def test_the_kernel_runs_under_jit_from_its_declared_program():
    """The declared program, jitted on the builder's arguments, is the period solve."""
    kernel, context = _kernel()
    program = core_program_graph(kernel=kernel)["main"]
    materialized = materialize_core_program(
        program=program, context=_build_context(context)
    )

    value, carry = jax.jit(materialized.function)(**materialized.arguments)
    output = kernel(
        compiled_cores={"main": materialized.function}, **context, logger=_LOGGER
    )

    np.testing.assert_array_equal(np.asarray(value), np.asarray(output.value))
    for got, expected in zip(
        jax.tree.leaves(carry),
        jax.tree.leaves(next(iter(output.continuations.values()))),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_a_single_liquid_nbegm_kernel_declares_its_feasibility_breakpoints():
    """The one-row kernel NB-EGM builds for a single liquid axis publishes the same
    graph, with a breakpoints row for the feasibility boundaries its carry carries."""
    from tests.test_nbegm_constraint_validation import (  # noqa: PLC0415
        _build_smooth_model,
        _smooth_params,
    )

    declaration = cast("Callable[..., object]", lcm.ref("liquid") >= 4.0)
    model = _build_smooth_model(constraints={"asset_test": declaration})
    kernel, _ = ride_along_kernel(
        model=model, params=_smooth_params(asset_limit=None), regime_name="alive"
    )
    graph = core_program_graph(kernel=kernel)

    assert tuple(graph) == ("main",)
    assert graph["main"].disposition_reason == (
        "deliberately_dense:egm_one_row_no_product_axis"
    )
    _, carry_roles = cast("tuple[Any, Any]", graph["main"].output_roles)
    assert carry_roles.breakpoints == StateAxesLeading(state_names=())
    assert carry_roles.policy is None
