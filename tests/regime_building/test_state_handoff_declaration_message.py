"""The missing-handoff error names the declaration that put the state on the target.

A state carried only by a target regime reaches it either through the target's
own `Regime(states=...)` or through `Model(states=...)`, broadcast and kept on
the target because the target uses it. A source reaching that target without
carrying the state or giving it an entry law is refused, and the message says
which of the two declarations the target carries the state from.

Only the model's state slot establishes the model-level origin: a model action
of the same name, masked on the target, leaves a regime-local state local.
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
    Phased,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import StateEntry
from lcm.typing import (
    ActionName,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)


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


@categorical(ordered=False)
class ProvenanceRegimeId:
    source: ScalarInt
    target: ScalarInt
    terminal: ScalarInt


def _provenance_action_utility(x: ContinuousAction) -> FloatND:
    return x


def _provenance_other_action_utility(choice: ContinuousAction) -> FloatND:
    return choice


def _provenance_state_utility(x: ContinuousState) -> FloatND:
    return x


def _provenance_imputation() -> FloatND:
    return jnp.asarray(1.0)


def _provenance_terminal_utility() -> FloatND:
    return jnp.asarray(0.0)


def _build_provenance_model(
    *, carried: bool, action_name: ActionName, duplicate_state: bool = False
) -> Model:
    """Reach a target-local state 'x' from a source that has no state 'x'.

    The model declares an action `action_name`, which the target masks with
    `None`, and no states unless `duplicate_state` repeats the target's
    declaration at the model level. With `carried`, 'x' exists only in the
    simulation phase, so the target is nonterminal and has its own outgoing
    law; that law does not supply the missing incoming handoff.
    """
    grid = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)
    declaration = (
        Phased(solve=_provenance_imputation, simulate=grid) if carried else grid
    )
    source_utility = (
        _provenance_action_utility
        if action_name == "x"
        else _provenance_other_action_utility
    )
    return Model(
        regimes={
            "source": Regime(functions={"utility": source_utility}),
            "target": Regime(
                states={"x": declaration},
                actions={action_name: None},
                state_transitions={"x": fixed_transition("x")},
                functions={"utility": _provenance_state_utility},
            ),
            "terminal": Regime(
                actions={action_name: None},
                functions={"utility": _provenance_terminal_utility},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=ProvenanceRegimeId,
        initial_nodes={0: "source"},
        edges={"source": {"target": 0}, "target": {"terminal": 1}},
        actions={action_name: grid},
        states={"x": declaration} if duplicate_state else {},
    )


def _provenance_handoff_message(*, carried: bool, action_name: ActionName) -> str:
    """Return the missing-handoff refusal of 'source' -> 'target'."""
    with pytest.raises(ModelInitializationError) as captured:
        _build_provenance_model(carried=carried, action_name=action_name)
    message = str(captured.value)
    assert "source 'source' ->" in message
    assert "target 'target' retains" in message
    assert "source does not carry 'x' and defines no entry law" in message
    assert "simulation phase" in message
    assert ("solution phase" in message) is not carried
    return message


@pytest.mark.parametrize("carried", [False, True], ids=["grid", "phased"])
def test_handoff_error_never_names_a_model_state_declaration_the_model_lacks(
    *, carried: bool
) -> None:
    """A masked model action named like the state does not make it a model state."""
    message = _provenance_handoff_message(carried=carried, action_name="x")
    assert "`Model(states=...)`" not in message


@pytest.mark.parametrize("carried", [False, True], ids=["grid", "phased"])
@pytest.mark.parametrize("action_name", ["x", "choice"], ids=["alias", "unrelated"])
def test_handoff_error_names_the_regime_declaration_beside_masked_model_actions(
    *, carried: bool, action_name: ActionName
) -> None:
    """The target's own `Regime(states=...)` stays the named declaration."""
    message = _provenance_handoff_message(carried=carried, action_name=action_name)
    assert (
        "'target' carries state 'x' from its own `Regime(states=...)` declaration"
    ) in message


@pytest.mark.parametrize("carried", [False, True], ids=["grid", "phased"])
def test_state_declared_on_both_model_and_regime_is_refused_before_handoff(
    *, carried: bool
) -> None:
    """Identical declarations at both levels are ambiguous, not a precedence rule."""
    with pytest.raises(
        ModelInitializationError,
        match=r"Ambiguous specification for states\['x'\] in regime 'target'",
    ):
        _build_provenance_model(
            carried=carried, action_name="choice", duplicate_state=True
        )
