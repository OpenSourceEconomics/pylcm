"""Model construction preserves the exact utility each stakeholder declares."""

from typing import Literal

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    Model,
    Phased,
    Regime,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.typing import DiscreteState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    """The household, its terminal regime, and an independent singleton."""

    couple: ScalarInt
    couple_terminal: ScalarInt
    single: ScalarInt


@categorical(ordered=True)
class Bonus:
    """Two category codes for the singleton's state."""

    low: ScalarInt
    high: ScalarInt


def _solve_utility() -> FloatND:
    return jnp.asarray(1.0)


def _simulate_utility() -> FloatND:
    return jnp.asarray(2.0)


def _other_utility() -> FloatND:
    return jnp.asarray(3.0)


def _single_utility(*, bonus: DiscreteState) -> FloatND:
    return jnp.asarray(bonus, dtype=float)


@pytest.mark.parametrize("value_grid", [False, True], ids=["codes", "values"])
def test_model_preserves_an_unchanged_phased_stakeholder(*, value_grid: bool) -> None:
    """A singleton's value grid cannot replace a household's authored utility."""
    body = Phased(solve=_solve_utility, simulate=_simulate_utility)
    declaration = CollectiveUtility(utilities={"f": _other_utility, "m": body})
    couple = Regime(functions={"utility": declaration})
    terminal = Regime(
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _other_utility, "m": _solve_utility}
            )
        }
    )
    bonus_grid = (
        DiscreteGrid(values=[10, 20])
        if value_grid
        else DiscreteGrid(category_class=Bonus)
    )
    single = Regime(
        states={"bonus": bonus_grid}, functions={"utility": _single_utility}
    )
    model = Model(
        regimes={"couple": couple, "couple_terminal": terminal, "single": single},
        edges={"couple": {"couple_terminal": 0}},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: ("couple", "single")},
        enable_jit=False,
    )

    finalized = model.user_regimes["couple"]
    assert finalized.functions["utility"] is declaration
    assert finalized.decomposed_functions["utility_m"] is body
    assert body.solve is _solve_utility
    assert body.simulate is _simulate_utility
    assert model.user_regimes["single"].states["bonus"] is bonus_grid
    assert bonus_grid.values == ((10, 20) if value_grid else None)


@pytest.mark.parametrize("changed_phase", ["solve", "simulate"])
def test_replacing_one_phase_of_an_owned_stakeholder_is_refused(
    *, changed_phase: Literal["solve", "simulate"]
) -> None:
    """Changing either child of a stakeholder's declared body violates ownership."""
    body = Phased(solve=_solve_utility, simulate=_simulate_utility)
    regime = Regime(
        functions={
            "utility": CollectiveUtility(utilities={"f": _other_utility, "m": body})
        }
    )
    replacement = Phased(
        solve=_other_utility if changed_phase == "solve" else body.solve,
        simulate=_other_utility if changed_phase == "simulate" else body.simulate,
    )
    assert replacement is not body
    assert (replacement.solve is body.solve) is (changed_phase != "solve")
    assert (replacement.simulate is body.simulate) is (changed_phase != "simulate")
    engine_functions = {**regime.decomposed_functions, "utility_m": replacement}

    with pytest.raises(
        RegimeInitializationError, match="different body for 'utility_m'"
    ):
        regime.with_engine_functions(engine_functions=engine_functions)
