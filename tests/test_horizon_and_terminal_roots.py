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
    ByAge,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidValueFunctionError, ModelInitializationError
from lcm.typing import ContinuousState, FloatND, ScalarInt
from tests.test_demand_worklists import _gated_model

_WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
_PARAMS = {"discount_factor": 0.9}
_AGES = AgeGrid(start=25, stop=75, step="10Y")


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


def _nonterminal(transition: Any) -> Regime:
    return Regime(
        regime_transitions=transition,
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _utility},
    )


def _terminal(utility: Any = _utility) -> Regime:
    return Regime(
        regime_transitions=None,
        states={"wealth": _WEALTH},
        functions={"utility": utility},
    )


def _never() -> FloatND:
    return jnp.asarray(0.0)


def _always() -> FloatND:
    return jnp.asarray(1.0)


_ZERO_SELF_LOOP = {
    "working": MarkovTransition(func=_never),
    "dead": MarkovTransition(func=_always),
}


def _life_model(*, working: Any, initial_regimes: Any, dead: Regime | None = None):
    return Model(
        regimes={
            "working": _nonterminal(working),
            "retirement": _nonterminal(ByAge(cases={65: "dead"})),
            "dead": dead or _terminal(),
        },
        ages=_AGES,
        regime_id_class=_LifeId,
        initial_regimes=initial_regimes,
    )


def _solved_pairs(*, model: Model, params: dict) -> frozenset[tuple[Any, str]]:
    values = model.solve(params=params, log_level="off").values
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
        initial_regimes={75: "dead"},
    )
    np.testing.assert_array_equal(
        np.asarray(model.solve(params={}, log_level="off").values[0]["dead"]),
        np.asarray([0.0, 1.0]),
    )


def test_terminal_only_model_over_several_ages_solves_its_root() -> None:
    """A terminal-only model is solved exactly at its declared start."""
    model = Model(
        regimes={"dead": _terminal()},
        ages=AgeGrid(start=25, stop=45, step="10Y"),
        regime_id_class=_DeadId,
        initial_regimes={35: "dead"},
    )
    assert _solved_pairs(model=model, params={}) == frozenset({(35, "dead")})


def test_terminal_root_at_the_final_age_solves() -> None:
    """A terminal start at the last age is valid and solved only there."""
    model = _life_model(
        working=ByAge(cases={}, default="dead"), initial_regimes={75: "dead"}
    )
    assert _solved_pairs(model=model, params={}) == frozenset({(75, "dead")})


def test_unused_final_age_default_law_builds() -> None:
    """A fallback law selected at the last age is harmless while unrequired."""
    model = _life_model(
        working=ByAge(cases={}, default="dead"), initial_regimes={65: "working"}
    )
    assert model.reachability.nodes == frozenset({(65, "working"), (75, "dead")})


def test_requested_final_age_default_law_fails() -> None:
    """Requiring the same fallback law at the last age names the start and age."""
    with pytest.raises(
        ModelInitializationError,
        match=r"requires 'working' at age 75, which is nonterminal at the last age",
    ):
        _life_model(
            working=ByAge(cases={}, default="dead"), initial_regimes={75: "working"}
        )


def test_zero_probability_self_loop_at_the_horizon_fails_structurally() -> None:
    """A self-loop with runtime probability zero still requires a last-age problem."""
    with pytest.raises(
        ModelInitializationError,
        match=r"\(65, 'working'\) requires 'working' at age 75, which is nonterminal",
    ):
        _life_model(working=_ZERO_SELF_LOOP, initial_regimes={25: "working"})


def test_named_terminal_exit_at_the_horizon_solves() -> None:
    """Replacing the last-source law by a named terminal exit ends the schedule."""
    model = _life_model(
        working=ByAge.until(stop_age_exclusive=75, law=_ZERO_SELF_LOOP, then="dead"),
        initial_regimes={25: "working"},
    )
    assert _solved_pairs(model=model, params=_PARAMS) == frozenset(
        {(age, "working") for age in (25, 35, 45, 55, 65)}
        | {(age, "dead") for age in (35, 45, 55, 65, 75)}
    )


def test_named_terminal_exit_keeps_terminality() -> None:
    """The exit does not make its target nonterminal or the source terminal."""
    model = _life_model(
        working=ByAge.until(stop_age_exclusive=75, law=_ZERO_SELF_LOOP, then="dead"),
        initial_regimes={25: "working"},
    )
    assert (
        model._engine_user_regimes["working"].terminal,
        model._engine_user_regimes["dead"].terminal,
    ) == (False, True)


def _finite_only_at_75(*, wealth: ContinuousState, age: float) -> FloatND:
    return jnp.where(age == 75, wealth, jnp.nan)


@pytest.mark.parametrize(
    "initial_regimes",
    [{65: "retirement"}, {75: "dead"}],
    ids=["reached-at-75", "final-terminal-root"],
)
def test_terminal_utility_is_not_evaluated_at_unrequired_ages(
    initial_regimes: Any,
) -> None:
    """A terminal utility that is NaN off age 75 passes the debug NaN check."""
    model = _life_model(
        working=ByAge(cases={}, default="dead"),
        initial_regimes=initial_regimes,
        dead=_terminal(_finite_only_at_75),
    )
    params = _PARAMS if "retirement" in initial_regimes.values() else {}
    values = model.solve(params=params, log_level="debug").values
    np.testing.assert_array_equal(np.asarray(values[5]["dead"]), np.asarray([0.0, 1.0]))


def test_terminal_utility_required_at_an_unfinite_age_trips_the_nan_check() -> None:
    """The control: requiring the same terminal regime at 55 reports its NaN."""
    model = _life_model(
        working=ByAge(cases={}, default="dead"),
        initial_regimes={55: "dead"},
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
