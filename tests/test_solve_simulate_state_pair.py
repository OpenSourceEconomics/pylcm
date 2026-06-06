"""Phase-variant state: imputed during solve, seeded and evolved during simulate.

A `SolveSimulateStatePair` placed in `Regime.states` gives a quantity two roles:
- solve: a derived function (here, pension wealth imputed from AIME) that never
  becomes a grid dimension, so the solve grid is unchanged;
- simulate: a genuine state seeded from the initial conditions and carried
  forward each period via its transition, read by simulate functions instead of
  the solve-phase imputation.
"""

from collections.abc import Mapping
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, SolveSimulateStatePair, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age < 62, RegimeId.working, RegimeId.dead)


def _impute_pension_wealth(aime: float) -> float:
    """Solve-phase pension wealth: imputed from AIME."""
    return aime * 0.1


def _evolve_pension_wealth(pension_wealth: float) -> float:
    """Simulate-phase pension wealth: grows by a fixed factor each period."""
    return pension_wealth * 1.03


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _next_wealth(wealth: float, consumption: float, pension_wealth: float) -> float:
    return wealth - consumption + pension_wealth


def _next_aime(aime: float) -> float:
    return aime


def _consumption_leq_wealth(consumption: float, wealth: float) -> bool:
    return consumption <= wealth


def _working_active(age: float) -> bool:
    return age < 64


def _build_pension_regime() -> UserRegime:
    """A non-terminal regime whose pension wealth is a `SolveSimulateStatePair`."""
    return UserRegime(
        transition=_next_regime,
        active=_working_active,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
            "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
            "pension_wealth": SolveSimulateStatePair(
                solve=_impute_pension_wealth,
                grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
                transition=_evolve_pension_wealth,
            ),
        },
        state_transitions={
            "wealth": _next_wealth,
            "aime": _next_aime,
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        constraints={"feasible_consumption": _consumption_leq_wealth},
        functions={"utility": _utility},
    )


_DEAD = UserRegime(transition=None, functions={"utility": lambda: 0.0})


def _build_pension_model(*, pension_as_pair: bool) -> Model:
    """A two-regime model with pension wealth as a state pair or a plain function.

    With `pension_as_pair=False` pension wealth is an ordinary derived function
    of AIME, present in both phases; this is the behavior the state-pair's solve
    phase must reproduce exactly.
    """
    if pension_as_pair:
        working = _build_pension_regime()
    else:
        working = UserRegime(
            transition=_next_regime,
            active=_working_active,
            states={
                "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
                "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
            },
            state_transitions={"wealth": _next_wealth, "aime": _next_aime},
            actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
            constraints={"feasible_consumption": _consumption_leq_wealth},
            functions={"utility": _utility, "pension_wealth": _impute_pension_wealth},
        )
    return Model(
        regimes={"working": working, "dead": _DEAD},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=RegimeId,
    )


def test_regime_accepts_state_pair_in_states() -> None:
    """A `SolveSimulateStatePair` is a valid value in `Regime.states`."""
    regime = _build_pension_regime()
    assert isinstance(regime.states["pension_wealth"], SolveSimulateStatePair)


def test_state_pair_in_state_transitions_raises() -> None:
    """A pair carries its own transition; listing it in `state_transitions` errors."""
    with pytest.raises(RegimeInitializationError, match="carry their own transition"):
        UserRegime(
            transition=_next_regime,
            states={
                "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
                "pension_wealth": SolveSimulateStatePair(
                    solve=_impute_pension_wealth,
                    grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
                    transition=_evolve_pension_wealth,
                ),
            },
            state_transitions={
                "aime": lambda aime: aime,
                "pension_wealth": _evolve_pension_wealth,
            },
            functions={"utility": lambda pension_wealth: pension_wealth},
        )


def test_solve_grid_excludes_state_pair() -> None:
    """A state pair is a function in solve, so it is not a solve grid dimension."""
    model = _build_pension_model(pension_as_pair=True)
    solve_state_names = model._regimes["working"].variables.state_names
    assert set(solve_state_names) == {"wealth", "aime"}


def _solve_pension_model(model: Model) -> Mapping[int, Mapping[str, FloatND]]:
    params = cast("dict[str, Any]", model.get_params_template())
    params["working"]["H"]["discount_factor"] = 0.95
    return model.solve(params=params, log_level="debug")


def test_state_pair_solves_like_plain_function() -> None:
    """Imputing pension wealth via a state pair gives the same value function as
    imputing it via an ordinary derived function."""
    pair_solution = _solve_pension_model(_build_pension_model(pension_as_pair=True))
    plain_solution = _solve_pension_model(_build_pension_model(pension_as_pair=False))

    for period, regime_to_V in plain_solution.items():
        for regime_name, expected_V in regime_to_V.items():
            assert bool(jnp.allclose(pair_solution[period][regime_name], expected_V))


def _simulate_pension(model: Model, *, pension_seed: list[float]) -> pd.DataFrame:
    params = cast("dict[str, Any]", model.get_params_template())
    params["working"]["H"]["discount_factor"] = 0.95
    n = len(pension_seed)
    result = model.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions={
            "wealth": jnp.full(n, 50.0),
            "aime": jnp.full(n, 20.0),
            "pension_wealth": jnp.asarray(pension_seed),
            "age": jnp.full(n, 60.0),
            "regime_id": jnp.array([RegimeId.working] * n),
        },
    )
    working = result.to_dataframe().query('regime_name == "working"')
    return working.set_index(["subject_id", "period"]).sort_index()


@pytest.mark.xfail(
    reason="Milestone B: true pension wealth as a simulate state not yet wired.",
    strict=True,
)
def test_simulate_seeds_and_evolves_true_pension_wealth() -> None:
    """Pension wealth is seeded from the initial conditions and evolved by its own
    transition each period (the true value, not the AIME imputation)."""
    sim = _simulate_pension(
        _build_pension_model(pension_as_pair=True), pension_seed=[5.0, 15.0]
    )
    pension = sim["pension_wealth"]
    np.testing.assert_allclose(pension.loc[(0, 0)], 5.0)
    np.testing.assert_allclose(pension.loc[(1, 0)], 15.0)
    np.testing.assert_allclose(pension.loc[(0, 1)], 5.0 * 1.03)
    np.testing.assert_allclose(pension.loc[(1, 1)], 15.0 * 1.03)


@pytest.mark.xfail(
    reason="Milestone B: true pension wealth as a simulate state not yet wired.",
    strict=True,
)
def test_simulate_decides_on_imputed_but_accounts_on_true_pension() -> None:
    """Two subjects with equal AIME impute equal pension, so they choose the same
    consumption; their realized next wealth differs by exactly the true-pension gap."""
    sim = _simulate_pension(
        _build_pension_model(pension_as_pair=True), pension_seed=[5.0, 15.0]
    )
    consumption = sim["consumption"]
    np.testing.assert_allclose(consumption.loc[(0, 0)], consumption.loc[(1, 0)])

    wealth = sim["wealth"]
    np.testing.assert_allclose(wealth.loc[(1, 1)] - wealth.loc[(0, 1)], 15.0 - 5.0)
