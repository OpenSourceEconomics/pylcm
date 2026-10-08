"""The last clock position, terminal roots, and terminal problems evaluated on demand.

A nonterminal problem required at the last age has no next age and fails, even
when its law gives the self-loop zero probability there. A named terminal exit
ends the same schedule cleanly. A terminal regime is a valid start at any age,
including in a model with a single age, and its utility is evaluated only at
the ages some start requires.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidValueFunctionError, ModelInitializationError
from lcm.typing import ContinuousState, FloatND, ScalarInt
from tests.test_demand_worklists import _gated_model

_WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
_PARAMS = {"discount_factor": 0.9}
_AGES = AgeGrid(start=25, inclusive_stop=75, step="10Y")


@categorical(ordered=False)
class _LifeId:
    working: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _DeadId:
    dead: ScalarInt


def _utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _nonterminal() -> Regime:
    return Regime(
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _utility},
    )


def _terminal(utility: Any = _utility) -> Regime:
    return Regime(
        states={"wealth": _WEALTH},
        functions={"utility": utility},
    )


def _never(age: float) -> FloatND:
    """Keep the zero self-loop conditional on the runtime source coordinate."""
    return jnp.zeros_like(age, dtype=float)


def _always() -> FloatND:
    return jnp.asarray(1.0)


_ZERO_SELF_LOOP = {
    "working": StochasticTransition(func=_never),
    "dead": StochasticTransition(func=_always),
}
_EVERY_SOURCE_AGE = AgeRange(exclusive_stop=75)
_EXIT_EDGES = {"dead": _EVERY_SOURCE_AGE}
_SELF_LOOP_EDGES = {"working": _EVERY_SOURCE_AGE, "dead": _EVERY_SOURCE_AGE}
_UNTIL_EXIT_EDGES = {"working": AgeRange(exclusive_stop=65), "dead": _EVERY_SOURCE_AGE}


def _life_model(
    *,
    working_edges: Any,
    initial_nodes: Any,
    working_law: Any = None,
    dead: Regime | None = None,
):
    return Model(
        regimes={
            "working": _nonterminal(),
            "retirement": _nonterminal(),
            "dead": dead or _terminal(),
        },
        ages=_AGES,
        regime_id_class=_LifeId,
        initial_nodes=initial_nodes,
        edges={
            "working": working_edges
            if working_law is None
            else Transition(targets=working_edges, law=working_law),
            "retirement": {"dead": 65},
        },
    )


def _solved_pairs(*, model: Model, params: dict) -> frozenset[tuple[Any, str]]:
    values = model.solve(params=params, log_level="off").values
    assert model.ages is not None
    return frozenset(
        (model.ages.exact_values[period], name)
        for period, by_regime in values.items()
        for name in by_regime
    )


def test_one_period_terminal_root_model_solves() -> None:
    """A model with one age and one terminal regime solves its single start."""
    model = Model(
        regimes={"dead": _terminal()},
        ages=AgeGrid(exact_values=(75,)),
        regime_id_class=_DeadId,
        initial_nodes={75: "dead"},
        edges={},
    )
    np.testing.assert_array_equal(
        np.asarray(model.solve(params={}, log_level="off").values[0]["dead"]),
        np.asarray([0.0, 1.0]),
    )


def test_terminal_only_model_over_several_ages_solves_its_root() -> None:
    """A terminal-only model is solved exactly at its declared start."""
    model = Model(
        regimes={"dead": _terminal()},
        ages=AgeGrid(start=25, inclusive_stop=45, step="10Y"),
        regime_id_class=_DeadId,
        initial_nodes={35: "dead"},
        edges={},
    )
    assert _solved_pairs(model=model, params={}) == frozenset({(35, "dead")})


def test_terminal_root_at_the_final_age_solves() -> None:
    """A terminal start at the last age is valid and solved only there."""
    model = _life_model(
        working_edges=_EXIT_EDGES,
        initial_nodes={75: "dead"},
    )
    assert _solved_pairs(model=model, params={}) == frozenset({(75, "dead")})


def test_unused_final_age_nonterminal_regime_builds() -> None:
    """A nonterminal regime without a last-age edge is harmless while unrequired."""
    model = _life_model(
        working_edges=_EXIT_EDGES,
        initial_nodes={65: "working"},
    )
    assert model.reachability.nodes == frozenset({(65, "working"), (75, "dead")})


def test_requested_final_age_nonterminal_regime_fails() -> None:
    """Requiring the same regime at the last age names the start and age."""
    with pytest.raises(
        ModelInitializationError,
        match=r"requires 'working' at age 75, which is nonterminal at the last age",
    ):
        _life_model(
            working_edges=_EXIT_EDGES,
            initial_nodes={75: "working"},
        )


def test_zero_probability_self_loop_at_the_horizon_fails_structurally() -> None:
    """A self-loop with runtime probability zero still requires a last-age problem."""
    with pytest.raises(
        ModelInitializationError,
        match=r"\(65, 'working'\) requires 'working' at age 75, which is nonterminal",
    ):
        _life_model(
            working_law=_ZERO_SELF_LOOP,
            working_edges=_SELF_LOOP_EDGES,
            initial_nodes={25: "working"},
        )


def test_named_terminal_exit_at_the_horizon_solves() -> None:
    """Replacing the last-source law by a named terminal exit ends the schedule."""
    model = _life_model(
        working_law=ByAge.until(
            stop_age_exclusive=75, law=_ZERO_SELF_LOOP, then="dead"
        ),
        working_edges=_UNTIL_EXIT_EDGES,
        initial_nodes={25: "working"},
    )
    assert _solved_pairs(model=model, params=_PARAMS) == frozenset(
        {(age, "working") for age in (25, 35, 45, 55, 65)}
        | {(age, "dead") for age in (35, 45, 55, 65, 75)}
    )


def test_named_terminal_exit_keeps_terminality() -> None:
    """The exit does not make its target nonterminal or the source terminal."""
    model = _life_model(
        working_law=ByAge.until(
            stop_age_exclusive=75, law=_ZERO_SELF_LOOP, then="dead"
        ),
        working_edges=_UNTIL_EXIT_EDGES,
        initial_nodes={25: "working"},
    )
    assert (
        model.graph.laws["working"].terminal,
        model.graph.laws["dead"].terminal,
    ) == (False, True)


def _finite_only_at_75(*, wealth: ContinuousState, age: float) -> FloatND:
    return jnp.where(age == 75, wealth, jnp.nan)


@pytest.mark.parametrize(
    "initial_nodes",
    [{65: "retirement"}, {75: "dead"}],
    ids=["reached-at-75", "final-terminal-root"],
)
def test_terminal_utility_is_not_evaluated_at_unrequired_ages(
    initial_nodes: Any,
) -> None:
    """A terminal utility that is NaN off age 75 passes the debug NaN check."""
    model = _life_model(
        working_edges=_EXIT_EDGES,
        initial_nodes=initial_nodes,
        dead=_terminal(_finite_only_at_75),
    )
    params = _PARAMS if "retirement" in initial_nodes.values() else {}
    values = model.solve(params=params, log_level="debug").values
    np.testing.assert_array_equal(np.asarray(values[5]["dead"]), np.asarray([0.0, 1.0]))


def test_terminal_utility_required_at_an_unfinite_age_trips_the_nan_check() -> None:
    """The control: requiring the same terminal regime at 55 reports its NaN."""
    model = _life_model(
        working_edges=_EXIT_EDGES,
        initial_nodes={55: "dead"},
        dead=_terminal(_finite_only_at_75),
    )
    with pytest.raises(InvalidValueFunctionError, match="dead"):
        model.solve(params={}, log_level="debug")


def test_gate_reference_through_a_terminal_regime_reads_its_value() -> None:
    """A terminal regime read only as a gate reference is solved to its utility."""
    values = _gated_model(phased_fallback=False).solve(params=_PARAMS, log_level="off")
    np.testing.assert_array_equal(
        np.asarray(values.values[1]["reference"]), np.asarray([0.0, 1.0])
    )
