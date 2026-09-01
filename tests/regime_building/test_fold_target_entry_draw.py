"""Simulation draws a folded shock a subject meets on entering its regime.

`fold=True` removes a shock's axis from the STORED value, not the shock from the
world. A subject entering a regime whose only state is such a shock still meets a
realization, and that regime's utility and policy read it. The fold is therefore
a storage optimization and nothing else: the simulated panel of a folded model
matches its unfolded twin cell for cell, while the folded regime's value function
carries one fewer axis.
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
from lcm.typing import ContinuousState, DiscreteAction, FloatND, ScalarInt

N_POINTS = 5
OUTSIDE_OPTION = 0.2
DISCOUNT_FACTOR = 0.9
SEED = 7
WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
AGES = AgeGrid(start=0, stop=3, step="Y")


@categorical(ordered=False)
class RegimeId:
    start: ScalarInt
    bonus: ScalarInt
    terminal: ScalarInt


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


def _next_bonus() -> ScalarInt:
    return RegimeId.bonus


def _next_terminal() -> ScalarInt:
    return RegimeId.terminal


def _utility_start(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    """No payoff of its own — `start` exists only to route into `bonus`."""
    return 0.0 * wealth + 0.0 * work


def _utility_bonus(*, bonus_shock: FloatND, work: DiscreteAction) -> FloatND:
    """Working pays the realized bonus; leisure pays a fixed outside option."""
    return jnp.where(work == 1, bonus_shock, OUTSIDE_OPTION)


def _build_model(*, fold: bool) -> Model:
    start = Regime(
        transition=_next_bonus,
        active=lambda age: age < 1,
        states={"wealth": WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _utility_start},
    )
    bonus = Regime(
        transition=_next_terminal,
        active=lambda age: (age >= 1) & (age < 2),
        states={
            "bonus_shock": NormalIIDProcess(
                n_points=N_POINTS,
                gauss_hermite=False,
                mu=0.0,
                n_std=3.0,
                sigma=1.0,
                fold=fold,
            )
        },
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _utility_bonus},
    )
    terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        functions={"utility": lambda: jnp.asarray(0.0)},
    )
    return Model(
        regimes={"start": start, "bonus": bonus, "terminal": terminal},
        ages=AGES,
        regime_id_class=RegimeId,
    )


def _params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "start": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
        "bonus": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
        "terminal": {},
    }


def _initial_conditions() -> dict[str, jnp.ndarray]:
    return {
        "age": jnp.array([0.0, 0.0, 0.0, 0.0]),
        "wealth": jnp.array([0.0, 1.0, 0.0, 1.0]),
        "regime_id": jnp.array([RegimeId.start] * 4),
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


def test_folding_removes_the_axis_from_the_stored_value() -> None:
    """The folded regime stores a scalar where the unfolded one stores a node axis.

    This is the control for the panel comparison below: without it, two identical
    panels would also be consistent with `fold=True` doing nothing at all.
    """
    unfolded = _build_model(fold=False).solve(params=_params(), log_level="debug")
    folded = _build_model(fold=True).solve(params=_params(), log_level="debug")

    assert unfolded[1]["bonus"].shape == (N_POINTS,)
    assert folded[1]["bonus"].shape == ()


def test_the_folded_target_period_realizes_a_shock() -> None:
    """A subject in the folded regime has a shock and a value, not an unset cell.

    Nothing draws the entry shock unless the simulate phase builds the target's
    entry law, and a subject that reaches the regime without one carries an unset
    shock into its own utility.
    """
    entered = _simulate(fold=True).query("regime_name == 'bonus'")

    assert entered["bonus_shock"].notna().all()
    assert entered["value"].notna().all()


def test_the_folded_panel_matches_its_unfolded_twin() -> None:
    """Every simulated cell agrees, so the fold changed storage and nothing else.

    Both models draw from the same seed against the same entry law, so agreement
    here covers the realized shock, the discrete action it induces, and the value
    read at that action — not just that the run produced numbers.
    """
    pd.testing.assert_frame_equal(_simulate(fold=True), _simulate(fold=False))


def test_the_realized_shock_decides_the_action() -> None:
    """Working is chosen exactly where the drawn bonus beats the outside option.

    A panel whose shocks were all unset, or all equal, would satisfy the frame
    comparison against an equally broken twin; this pins the policy to the draw.
    """
    entered = _simulate(fold=True).query("regime_name == 'bonus'")

    np.testing.assert_array_equal(
        (entered["work"] == "work").to_numpy(),
        (entered["bonus_shock"] > OUTSIDE_OPTION).to_numpy(),
    )
    assert entered["work"].nunique() == 2
