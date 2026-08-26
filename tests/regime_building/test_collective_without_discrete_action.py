"""Collective regimes need no discrete action.

The household argmax runs over whatever action product a regime declares. That
product may be continuous, and it may be empty — a terminal regime whose payoff
is fully determined by its states has no decision to take. Neither case needs a
placeholder `DiscreteGrid` action, and both produce the ordinary per-stakeholder
readout: each stakeholder's own value at the shared household choice.

Both models are stateless and equally weighted, so every value below is an exact
small-integer expression rather than a tolerance-bounded approximation.
"""

import jax.numpy as jnp
import numpy as np

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import ContinuousAction, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

DISCOUNT_FACTOR = 0.95

AGES = AgeGrid(start=0, stop=2, step="Y")

CONSUMPTION_GRID = LinSpacedGrid(start=0.0, stop=1.0, n_points=3)


@categorical(ordered=False)
class CoupleRegimeId:
    couple: ScalarInt  # code 0
    couple_terminal: ScalarInt  # code 1


def _next_couple_regime() -> ScalarInt:
    return CoupleRegimeId.couple_terminal


def _flow_utility_f() -> FloatND:
    return jnp.asarray(1.0)


def _flow_utility_m() -> FloatND:
    return jnp.asarray(2.0)


def _terminal_utility_f() -> FloatND:
    return jnp.asarray(10.0)


def _terminal_utility_m() -> FloatND:
    return jnp.asarray(4.0)


def _consumption_utility_f(consumption: ContinuousAction) -> FloatND:
    return consumption


def _consumption_utility_m(consumption: ContinuousAction) -> FloatND:
    return 2.0 * consumption


def _params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "couple": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
        "couple_terminal": {},
    }


def test_collective_regime_with_no_action_solves_to_its_own_utilities() -> None:
    """A zero-action collective regime stores each stakeholder's own utility.

    Neither regime offers a choice, so the household argmax is over a single
    empty cell. The terminal pair is the terminal utilities `(10, 4)`, and the
    period-0 pair adds the discounted continuation to each stakeholder's own
    flow payoff: `(1 + 0.95 * 10, 2 + 0.95 * 4) = (10.5, 5.8)`.
    """
    couple = Regime(
        transition=_next_couple_regime,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={},
        actions={},
        functions={"utility_f": _flow_utility_f, "utility_m": _flow_utility_m},
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={},
        actions={},
        functions={
            "utility_f": _terminal_utility_f,
            "utility_m": _terminal_utility_m,
        },
    )
    model = Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=AGES,
        regime_id_class=CoupleRegimeId,
    )

    solution = model.solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[1]["couple_terminal"]),
        np.array([10.0, 4.0]),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["couple"]),
        np.array([10.5, 5.8]),
        decimal=DECIMAL_PRECISION,
    )


def test_collective_regime_with_only_a_continuous_action_solves() -> None:
    """A continuous-only collective regime maximizes over the continuous grid.

    Both stakeholders' payoffs rise in `consumption`, so the equally weighted
    household objective is maximized at the top grid point `1.0` in both
    periods. The terminal pair is `(1, 2)`, and the period-0 pair is
    `(1 + 0.95 * 1, 2 + 0.95 * 2) = (1.95, 3.9)`.
    """
    couple = Regime(
        transition=_next_couple_regime,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={},
        actions={"consumption": CONSUMPTION_GRID},
        functions={
            "utility_f": _consumption_utility_f,
            "utility_m": _consumption_utility_m,
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={},
        actions={"consumption": CONSUMPTION_GRID},
        functions={
            "utility_f": _consumption_utility_f,
            "utility_m": _consumption_utility_m,
        },
    )
    model = Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=AGES,
        regime_id_class=CoupleRegimeId,
    )

    solution = model.solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[1]["couple_terminal"]),
        np.array([1.0, 2.0]),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["couple"]),
        np.array([1.95, 3.9]),
        decimal=DECIMAL_PRECISION,
    )
