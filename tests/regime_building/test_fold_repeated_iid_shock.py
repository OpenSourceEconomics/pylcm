"""A folded IID shock may be redrawn every period of the same regime.

An IID shock that enters only its own period's payoff carries no information
forward, so integrating its axis out of the stored value costs nothing even when
the regime is active for many periods and redraws it each time. This is the case
the memory saving exists for: the stored value loses the shock's axis in every
period rather than in one.

The unfolded twin is the oracle throughout. Its continuation sums the target's
node-indexed value against the process's own weights; the folded model takes that
same sum one step earlier, so the two agree on every decision and every value and
differ only in the shape of what is stored.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.transition import MarkovTransition
from lcm.typing import ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

N_POINTS = 5
OUTSIDE_OPTION = 0.2
DISCOUNT_FACTOR = 0.9
SEED = 7
LAST_ALIVE_AGE = 2
WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
AGES = AgeGrid(start=0, stop=3, step="Y")


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


def _probability_alive(age: float) -> FloatND:
    return jnp.where(age < LAST_ALIVE_AGE, 1.0, 0.0)


def _probability_dead(age: float) -> FloatND:
    return jnp.where(age < LAST_ALIVE_AGE, 0.0, 1.0)


def _utility(
    *, wealth: ContinuousState, wage_shock: FloatND, work: DiscreteAction
) -> FloatND:
    """Working pays the period's own shock; leisure pays a fixed outside option."""
    return jnp.where(work == 1, wage_shock, OUTSIDE_OPTION) + 0.0 * wealth


def _build_model(*, fold: bool) -> Model:
    alive = Regime(
        transition={
            "alive": MarkovTransition(_probability_alive),
            "dead": MarkovTransition(_probability_dead),
        },
        active=lambda age: age < 3,
        states={
            "wealth": WEALTH,
            "wage_shock": NormalIIDProcess(
                n_points=N_POINTS,
                gauss_hermite=False,
                mu=0.0,
                n_std=3.0,
                sigma=1.0,
                fold=fold,
            ),
        },
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _utility},
    )
    dead = Regime(
        transition=None,
        active=lambda age: age >= LAST_ALIVE_AGE,
        functions={"utility": lambda: jnp.asarray(0.0)},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AGES,
        regime_id_class=RegimeId,
    )


def _params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "alive": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
        "dead": {},
    }


def _initial_conditions() -> dict[str, jnp.ndarray]:
    return {
        "age": jnp.array([0.0, 0.0, 0.0]),
        "wealth": jnp.array([0.0, 1.0, 0.0]),
        "wage_shock": jnp.array([1.5, -1.5, 0.0]),
        "regime_id": jnp.array([RegimeId.alive] * 3),
    }


def _simulate(*, fold: bool) -> pd.DataFrame:
    return (
        _build_model(fold=fold)
        .simulate(
            params=_params(),
            initial_conditions=_initial_conditions(),
            period_to_regime_to_V_arr=None,
            log_level="debug",
            seed=SEED,
        )
        .to_dataframe()
    )


def test_a_shock_redrawn_every_period_loses_its_axis_in_every_period() -> None:
    """`alive` stores `(n_wealth,)` folded where it stores `(n_nodes, n_wealth)` not.

    The saving is per period, so the shape is asserted in each period the regime
    is active rather than only in the one that declares the shock.
    """
    unfolded = _build_model(fold=False).solve(params=_params(), log_level="debug")
    folded = _build_model(fold=True).solve(params=_params(), log_level="debug")

    alive_periods = [period for period, arrays in folded.items() if "alive" in arrays]

    assert len(alive_periods) == 3
    for period in alive_periods:
        assert unfolded[period]["alive"].shape == (N_POINTS, len(WEALTH.to_jax()))
        assert folded[period]["alive"].shape == (len(WEALTH.to_jax()),)


def test_the_folded_value_is_the_unfolded_value_averaged_over_the_nodes() -> None:
    """Folding is the same quadrature the continuation would have applied.

    An axis integrated out one step earlier than it used to be must reproduce
    the number the later reduction produced, in every period the regime is
    active — not only in the last one, where there is no continuation to get
    wrong.
    """
    unfolded = _build_model(fold=False).solve(params=_params(), log_level="debug")
    folded = _build_model(fold=True).solve(params=_params(), log_level="debug")
    weights = np.asarray(
        NormalIIDProcess(
            n_points=N_POINTS, gauss_hermite=False, mu=0.0, n_std=3.0, sigma=1.0
        ).get_transition_probs()[0]
    )

    for period, arrays in folded.items():
        if "alive" not in arrays:
            continue
        np.testing.assert_array_almost_equal(
            np.asarray(arrays["alive"]),
            weights @ np.asarray(unfolded[period]["alive"]),
            decimal=DECIMAL_PRECISION,
        )


def test_the_folded_panel_matches_its_unfolded_twin() -> None:
    """Every simulated cell agrees, so the fold changed storage and nothing else.

    The shock is redrawn in each of the three active periods from the same seed,
    so this covers the redraw, the action it induces, and the value read at that
    action — in a period whose continuation is itself folded.
    """
    pd.testing.assert_frame_equal(_simulate(fold=True), _simulate(fold=False))


def test_every_active_period_redraws_the_shock() -> None:
    """A subject meets a different shock each period, not one carried forward.

    A panel that reused one draw would still match an equally broken twin, so
    the redraw is pinned directly.
    """
    alive_rows = _simulate(fold=True).query("regime_name == 'alive'")

    assert alive_rows["wage_shock"].notna().all()
    per_subject_draws = alive_rows.groupby("subject_id")["wage_shock"].nunique()
    assert (per_subject_draws == 3).all()
