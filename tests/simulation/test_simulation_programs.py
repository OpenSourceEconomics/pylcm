"""Simulation declares its per-regime work as planner programs."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import pytest

from _lcm.execution.core_program import CoreExecutionDisposition
from _lcm.regime_building.max_Q_over_a import get_argmax_and_max_Q_over_a
from _lcm.simulation.programs import SUBJECT_AXIS, _StreamedArgmaxQOverA
from _lcm.typing import ArgmaxQOverAFunction, QAndFFunction
from lcm.solvers import ACTION_PRODUCT_AXIS
from lcm.typing import BoolND, FloatND
from tests.conftest import assert_agrees_to_ulp
from tests.simulation.test_dispatch_counts import WITNESSES

# A regime whose solve kernel streams its action product, one whose collective
# kernel keeps the canonical dense reducer, and one whose routing a host loop
# drives through the gated-edge folds.
_STREAMED = ("multi_regime", "work")
_COLLECTIVE = ("dissolution", "married")
_GATED_ROUTE = ("dissolution", "married")


def _program(*, witness: str, regime: str, family: str):
    """Return the first program of one regime's declared family."""
    model, _, _ = WITNESSES[witness]()
    programs = getattr(model._regimes[regime].simulation.programs, family)
    return programs[next(iter(programs))]


def test_decision_program_declares_action_product_and_subject_axes() -> None:
    """The decision program streams the action product and tiles subjects."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="decision")
    assert program.requirements.axis_names == (ACTION_PRODUCT_AXIS, SUBJECT_AXIS)


def test_decision_program_is_planned() -> None:
    """The engine, not the regime builder, owns the decision program's widths."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="decision")
    assert program.disposition is CoreExecutionDisposition.PLANNED


def test_collective_decision_program_declares_the_subject_axis_only() -> None:
    """A collective decision tiles subjects and keeps its action product dense."""
    program = _program(witness=_COLLECTIVE[0], regime=_COLLECTIVE[1], family="decision")
    assert program.requirements.axis_names == (SUBJECT_AXIS,)


def test_transition_program_declares_the_subject_axis_only() -> None:
    """A law of motion tiles subjects and reduces no axis."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="transition")
    assert program.requirements.axis_names == (SUBJECT_AXIS,)


def test_route_program_declares_the_subject_axis_only() -> None:
    """A regime-transition draw tiles subjects and reduces no axis."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="route")
    assert program.requirements.axis_names == (SUBJECT_AXIS,)


def test_gated_route_program_is_host_driven() -> None:
    """A gated regime cedes its routing width to the host loop that drives it."""
    program = _program(witness=_GATED_ROUTE[0], regime=_GATED_ROUTE[1], family="route")
    assert program.disposition is CoreExecutionDisposition.HOST_DRIVEN


def test_gated_route_program_declares_no_execution_axis() -> None:
    """A host-driven route names no axis, because it cedes every width."""
    program = _program(witness=_GATED_ROUTE[0], regime=_GATED_ROUTE[1], family="route")
    assert program.requirements.axis_names == ()


def test_terminal_regime_declares_no_route_program() -> None:
    """A regime that draws no successor publishes no routing program."""
    model, _, _ = WITNESSES[_STREAMED[0]]()
    assert model._regimes["dead"].simulation.programs.route == {}


def test_decision_program_declares_its_continuation_reads() -> None:
    """A decision names every stored value leaf it reads across the boundary."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="decision")
    assert len(program.requirements.value_reads) == 2


def _toy_Q_and_F(
    *,
    next_regime_to_V_arr: Mapping[str, FloatND],
    consumption: FloatND,
    work: FloatND,
    wealth: FloatND,
) -> tuple[FloatND, BoolND]:
    """Return a toy action value and its feasibility at one state cell."""
    value = (
        jnp.asarray(wealth + 2.0 * consumption - consumption**2 + 0.5 * work)
        + next_regime_to_V_arr["target"]
    )
    return value, jnp.asarray(consumption <= wealth)


_TOY_ACTIONS = {
    "work": jnp.array([0.0, 1.0]),
    "consumption": jnp.array([0.1, 0.4, 0.9, 1.3, 1.8, 2.4, 3.1]),
}


def _reducers() -> tuple[ArgmaxQOverAFunction, _StreamedArgmaxQOverA]:
    """Return the dense canonical reducer and its streamed counterpart."""
    toy = cast("QAndFFunction", _toy_Q_and_F)
    dense = get_argmax_and_max_Q_over_a(
        Q_and_F=toy,
        action_names=("work", "consumption"),
        state_names=("wealth",),
        n_discrete_action_axes=1,
    )
    streamed = _StreamedArgmaxQOverA(
        Q_and_F=toy,
        action_names=("work", "consumption"),
        q_and_f_arg_names=frozenset(
            {"next_regime_to_V_arr", "consumption", "work", "wealth"}
        ),
        action_width_keyword="_lcm_action_block_width",
    )
    return dense, streamed


@pytest.mark.parametrize("width", [1, 2, 3, 5, 14])
@pytest.mark.parametrize("wealth", [0.05, 1.0, 2.5])
def test_streamed_decision_chooses_the_dense_reducers_action(
    *, width: int, wealth: float
) -> None:
    """Streaming the action product at any width picks the dense winner."""
    dense, streamed = _reducers()
    continuation = MappingProxyType({"target": jnp.asarray(0.25)})
    dense_index, _ = dense(
        next_regime_to_V_arr=continuation, wealth=jnp.asarray(wealth), **_TOY_ACTIONS
    )
    streamed_index, _ = streamed(
        next_regime_to_V_arr=continuation,
        wealth=jnp.asarray(wealth),
        _lcm_action_block_width=width,
        **_TOY_ACTIONS,
    )
    assert int(streamed_index) == int(dense_index)


@pytest.mark.parametrize("width", [1, 2, 3, 5, 14])
@pytest.mark.parametrize("wealth", [0.05, 1.0, 2.5])
def test_streamed_decision_reports_the_dense_reducers_value(
    *, width: int, wealth: float
) -> None:
    """Streaming the action product at any width reports the dense maximum."""
    dense, streamed = _reducers()
    continuation = MappingProxyType({"target": jnp.asarray(0.25)})
    _, dense_value = dense(
        next_regime_to_V_arr=continuation, wealth=jnp.asarray(wealth), **_TOY_ACTIONS
    )
    _, streamed_value = streamed(
        next_regime_to_V_arr=continuation,
        wealth=jnp.asarray(wealth),
        _lcm_action_block_width=width,
        **_TOY_ACTIONS,
    )
    assert_agrees_to_ulp(got=streamed_value, expected=dense_value, n_ulp=1)


@pytest.mark.parametrize("width", [1, 3, 14])
def test_streamed_decision_publishes_identity_zero_when_nothing_is_feasible(
    *, width: int
) -> None:
    """An all-infeasible cell publishes the identity the dense reducer publishes."""
    dense, streamed = _reducers()
    continuation = MappingProxyType({"target": jnp.asarray(0.25)})
    infeasible = jnp.asarray(-1.0)
    dense_index, _ = dense(
        next_regime_to_V_arr=continuation, wealth=infeasible, **_TOY_ACTIONS
    )
    streamed_index, _ = streamed(
        next_regime_to_V_arr=continuation,
        wealth=infeasible,
        _lcm_action_block_width=width,
        **_TOY_ACTIONS,
    )
    assert int(streamed_index) == int(dense_index)


def test_the_toy_decision_does_not_trivially_pick_the_first_action() -> None:
    """The oracle differential is exercised at a non-trivial winning action."""
    dense, _ = _reducers()
    dense_index, _ = dense(
        next_regime_to_V_arr=MappingProxyType({"target": jnp.asarray(0.25)}),
        wealth=jnp.asarray(2.5),
        **_TOY_ACTIONS,
    )
    assert int(dense_index) != 0
