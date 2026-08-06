"""`next_<name>` names a value, never a parameter.

Next-period values exist inside a transition law and inside whatever feeds one. Every
other function — utility, a constraint, the Koopmans aggregator, a helper only those
reach — is evaluated on this period's state-action grid, ahead of the transitions, so
an argument spelled that way has nothing behind it there. Binding it to a parameter
would answer a next-period question with a constant supplied at solve time, which is
silent rather than wrong-looking, so the model rejects it at construction.

A constraint on next-period assets is therefore written on this period's variables —
`assets - consumption >= 0` — and a law reading a sibling law's output is unaffected.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidNameError
from lcm.phased import Phased
from lcm.typing import ScalarFloat, ScalarInt

_WEALTH = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)
_SHOCK = NormalIIDProcess(n_points=3, gauss_hermite=False, mu=1.0, sigma=0.5, n_std=2.0)


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _to_target() -> ScalarFloat:
    return jnp.float32(1)


def _keep_wealth(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


def _wealth_and_shock(wealth: ScalarFloat, shock: ScalarFloat) -> ScalarFloat:
    return wealth + shock


def _build(
    *,
    functions,
    koopmans_aggregator=None,
    states=None,
    state_transitions=None,
) -> Model:
    """Two-regime model whose source is configurable, target always terminal."""
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                states={"wealth": _WEALTH} if states is None else states,
                state_transitions=(
                    {"wealth": {"target": _keep_wealth}}
                    if state_transitions is None
                    else state_transitions
                ),
                functions=functions,
                koopmans_aggregator=koopmans_aggregator,
            ),
            "target": Regime(
                transition=None,
                states={"wealth": _WEALTH, "shock": _SHOCK},
                functions={"utility": _wealth_and_shock},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )


def _utility_of_a_moved_state(
    wealth: ScalarFloat, next_wealth: ScalarFloat
) -> ScalarFloat:
    return wealth + next_wealth


def _utility_of_a_target_only_draw(next_shock: ScalarFloat) -> ScalarFloat:
    return next_shock


def _utility_of_a_helper(helper: ScalarFloat) -> ScalarFloat:
    return helper


def _helper_reading_a_next_name(next_shock: ScalarFloat) -> ScalarFloat:
    return next_shock


def _constraint_of_a_next_name(next_shock: ScalarFloat) -> ScalarFloat:
    return next_shock > 0


def _plain_utility(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


@pytest.mark.parametrize(
    ("functions", "offender"),
    [
        ({"utility": _utility_of_a_target_only_draw}, "utility"),
        (
            {"utility": _utility_of_a_helper, "helper": _helper_reading_a_next_name},
            "helper",
        ),
    ],
    ids=["directly", "through_a_helper"],
)
def test_a_value_belonging_to_a_target_may_not_be_read_outside_a_transition(
    functions, offender
) -> None:
    """The offending consumer is named, and the message says where the value lives."""
    with pytest.raises(InvalidNameError, match=offender):
        _build(functions=functions)


def test_a_state_this_regime_moves_is_also_rejected_outside_a_transition() -> None:
    """Declaring the law does not put its output in scope before it runs.

    Utility and constraints are evaluated on this period's state-action grid,
    ahead of the transitions, so `next_wealth` has no value there even though the
    regime computes one later in the period.
    """
    with pytest.raises(InvalidNameError, match="next_wealth"):
        _build(
            functions={"utility": _utility_of_a_moved_state},
            state_transitions={"wealth": _keep_wealth},
        )


def test_a_constraint_may_not_read_a_targets_draw() -> None:
    """A constraint is evaluated before any target is entered."""
    with pytest.raises(InvalidNameError, match="next_shock"):
        Model(
            regimes={
                "source": Regime(
                    transition={"target": MarkovTransition(_to_target)},
                    active=lambda age: age < 22,
                    states={"wealth": _WEALTH},
                    state_transitions={"wealth": {"target": _keep_wealth}},
                    functions={"utility": _plain_utility},
                    constraints={"affordable": _constraint_of_a_next_name},
                ),
                "target": Regime(
                    transition=None,
                    states={"wealth": _WEALTH, "shock": _SHOCK},
                    functions={"utility": _wealth_and_shock},
                ),
            },
            ages=AgeGrid(start=20, stop=22, step="Y"),
            regime_id_class=RegimeId,
        )


def _aggregator_reading_a_next_name(
    utility: ScalarFloat, CE: ScalarFloat, next_wealth: ScalarFloat
) -> ScalarFloat:
    return utility + CE + next_wealth


@pytest.mark.parametrize(
    "aggregator",
    [
        _aggregator_reading_a_next_name,
        Phased(
            solve=_aggregator_reading_a_next_name,
            simulate=_aggregator_reading_a_next_name,
        ),
    ],
    ids=["plain", "phased"],
)
def test_a_koopmans_aggregator_may_not_read_a_next_name(aggregator) -> None:
    """The aggregator combines this period's utility with an already-formed CE."""
    with pytest.raises(InvalidNameError, match="koopmans_aggregator"):
        _build(functions={"utility": _plain_utility}, koopmans_aggregator=aggregator)


def _next_aime(wealth: ScalarFloat) -> ScalarFloat:
    return 0.1 * wealth


def _next_wealth_reading_a_sibling(
    wealth: ScalarFloat, next_aime: ScalarFloat
) -> ScalarFloat:
    return 0.5 * wealth + next_aime


def _wealth_and_aime(wealth: ScalarFloat, aime: ScalarFloat) -> ScalarFloat:
    return wealth + aime


def test_a_transition_law_may_still_read_a_next_name() -> None:
    """A law reading a sibling law's output is the case the prefix exists for.

    At the top corner `wealth = 4, aime = 1` the source pays `4 + 1 = 5`, hands over
    `next_aime = 0.4` and `next_wealth = 0.5 * 4 + 0.4 = 2.4`, and the target pays
    `2.4 + 0.4 = 2.8` on those values — both inside their grids, so no
    extrapolation enters.
    """
    aime = LinSpacedGrid(start=0.0, stop=1.0, n_points=3)
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                states={"wealth": _WEALTH, "aime": aime},
                state_transitions={
                    "wealth": {"target": _next_wealth_reading_a_sibling},
                    "aime": {"target": _next_aime},
                },
                functions={"utility": _wealth_and_aime},
            ),
            "target": Regime(
                transition=None,
                states={"wealth": _WEALTH, "aime": aime},
                functions={"utility": _wealth_and_aime},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )

    V = model.solve(
        params={"source": {"koopmans_aggregator": {"discount_factor": 1.0}}},
        log_level="off",
    )

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]).max(), np.array(5.0 + 2.8), atol=1e-5
    )
