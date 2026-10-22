"""The missing-handoff error names the declaration that put the state on the target.

A state carried only by a target regime reaches it either through the target's
own `Regime(states=...)` or through `Model(states=...)`, broadcast and kept on
the target because the target uses it. A source reaching that target without
carrying the state or giving it an entry law is refused, and the message says
which of the two declarations the target carries the state from.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import StateEntry
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    work: ScalarInt
    unemployed: ScalarInt
    retired: ScalarInt


@categorical(ordered=True)
class PensionGroup:
    low: ScalarInt
    high: ScalarInt


def _utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _retired_utility(*, wealth: ContinuousState, pension_group: ScalarInt) -> FloatND:
    return wealth + pension_group


def _next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def _next_pension_group(wealth: ContinuousState) -> ScalarInt:
    return jnp.where(wealth > 5.0, PensionGroup.high, PensionGroup.low)


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _third() -> FloatND:
    return jnp.asarray(1 / 3)


def _one() -> FloatND:
    return jnp.asarray(1.0)


def _build_model(
    *,
    retired_states: Mapping[str, StateEntry],
    model_states: Mapping[str, StateEntry],
) -> Model:
    """Build a model where 'unemployed' reaches 'retired' without an entry law."""
    wealth = LinSpacedGrid(start=1.0, stop=10.0, n_points=5)
    consumption = LinSpacedGrid(start=1.0, stop=5.0, n_points=3)
    work_law = ByAge(
        cases={
            0: {
                "work": StochasticTransition(func=_third),
                "unemployed": StochasticTransition(func=_third),
                "retired": StochasticTransition(func=_third),
            },
            1: {"retired": StochasticTransition(func=_one)},
        }
    )
    unemployed_law = ByAge(
        cases={
            0: {
                "work": StochasticTransition(func=_half),
                "unemployed": StochasticTransition(func=_half),
            },
            1: {"retired": StochasticTransition(func=_one)},
        }
    )
    return Model(
        regimes={
            "work": Regime(
                states={"wealth": wealth},
                actions={"consumption": consumption},
                state_transitions={
                    "wealth": _next_wealth,
                    "pension_group": {"retired": _next_pension_group},
                },
                functions={"utility": _utility},
            ),
            "unemployed": Regime(
                states={"wealth": wealth},
                actions={"consumption": consumption},
                state_transitions={"wealth": _next_wealth},
                functions={"utility": _utility},
            ),
            "retired": Regime(
                states={"wealth": wealth, **retired_states},
                functions={"utility": _retired_utility},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "work"},
        edges={
            "work": Transition(
                targets={"work": 0, "unemployed": 0, "retired": (0, 1)},
                law=work_law,
            ),
            "unemployed": Transition(
                targets={"work": 0, "unemployed": 0, "retired": 1},
                law=unemployed_law,
            ),
        },
        states=model_states,
    )


_PENSION_GROUP = DiscreteGrid(PensionGroup)


@pytest.mark.parametrize(
    ("retired_states", "model_states", "phrase"),
    [
        pytest.param(
            {"pension_group": _PENSION_GROUP},
            {},
            r"'retired' carries state 'pension_group' from its own "
            r"`Regime\(states=\.\.\.\)` declaration",
            id="regime_level",
        ),
        pytest.param(
            {},
            {"pension_group": _PENSION_GROUP},
            r"'retired' carries state 'pension_group' from the model-level "
            r"`Model\(states=\.\.\.\)` declaration",
            id="model_level",
        ),
    ],
)
def test_handoff_error_names_the_declaration_the_target_carries_the_state_from(
    *,
    retired_states: Mapping[str, StateEntry],
    model_states: Mapping[str, StateEntry],
    phrase: str,
) -> None:
    """The refusal of 'unemployed' -> 'retired' names the declaring slot."""
    with pytest.raises(
        ModelInitializationError,
        match=rf"source 'unemployed' -> .*target 'retired'.*{phrase}",
    ):
        _build_model(retired_states=retired_states, model_states=model_states)
