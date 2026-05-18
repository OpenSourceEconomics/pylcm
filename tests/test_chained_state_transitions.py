"""A state transition can consume another transition's output.

The model defines `next_aime` and `next_wealth`, with `next_wealth`
referencing `next_aime` in its signature. dags resolves the chain at
evaluation time. These tests assert the chain produces the
mathematically expected next-period values in solve and simulate.
"""

import jax.numpy as jnp
import numpy as np

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, categorical
from lcm.api.regime import Regime as UserRegime
from lcm.typing import DiscreteAction, FloatND, ScalarInt


@categorical(ordered=False)
class _LaborSupply:
    work: ScalarInt
    rest: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    active: ScalarInt
    dead: ScalarInt


def _next_aime(aime: float, labor_supply: DiscreteAction) -> FloatND:
    """AIME accumulates by 1 unit when working, 0 otherwise."""
    return aime + jnp.where(labor_supply == _LaborSupply.work, 1.0, 0.0)


def _next_wealth(wealth: float, consumption: float, next_aime: FloatND) -> FloatND:
    """Next-period wealth depends on next-period AIME (the chained transition)."""
    return wealth - consumption + 0.1 * next_aime


def _utility(consumption: float, labor_supply: DiscreteAction) -> FloatND:
    disutility = jnp.where(labor_supply == _LaborSupply.work, 0.5, 0.0)
    return jnp.log(jnp.maximum(consumption, 1e-6)) - disutility


def _next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, _RegimeId.dead, _RegimeId.active)


_active = UserRegime(
    transition=_next_regime,
    actions={
        "labor_supply": DiscreteGrid(_LaborSupply),
        "consumption": LinSpacedGrid(start=0.5, stop=2.0, n_points=3),
    },
    states={
        "aime": LinSpacedGrid(start=0.0, stop=4.0, n_points=3),
        "wealth": LinSpacedGrid(start=0.5, stop=5.0, n_points=3),
    },
    state_transitions={
        "aime": _next_aime,
        "wealth": _next_wealth,
    },
    functions={"utility": _utility},
    active=lambda age: age < 2,
)


_dead = UserRegime(transition=None, functions={"utility": lambda: jnp.array(0.0)})


def _build_model() -> Model:
    return Model(
        regimes={"active": _active, "dead": _dead},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
    )


def test_solve_with_chained_transitions_returns_finite_value_function() -> None:
    """`solve()` returns a finite value function for the active regime.

    With `discount_factor=0.9` and a 2-period horizon, each state in the
    active regime must produce a finite expected value: the chain
    `next_aime → next_wealth` resolves and feeds back into the agent's
    next-period continuation value.
    """
    model = _build_model()
    params = {"discount_factor": 0.9, "final_age_alive": 1.0}

    period_to_regime_to_V_arr = model.solve(params=params)

    V_active = period_to_regime_to_V_arr[0]["active"]
    assert jnp.all(jnp.isfinite(V_active))


def test_simulate_with_chained_transitions_yields_expected_next_wealth() -> None:
    """`next_wealth_t = wealth_t - c_t + 0.1 * next_aime_t` holds in simulation.

    For each subject, `next_aime` is the value of `_next_aime(aime, ls)` at
    the chosen labor supply, and `next_wealth` must equal
    `wealth - c + 0.1 * next_aime` exactly. Solving for the optimum, the
    test then checks that successive `wealth` values in the simulated
    DataFrame satisfy this identity, which can only hold if the chained
    dependency was wired correctly.
    """
    model = _build_model()
    params = {"discount_factor": 0.9, "final_age_alive": 1.0}
    initial_conditions = {
        "age": jnp.array([0.0, 0.0]),
        "aime": jnp.array([0.0, 1.0]),
        "wealth": jnp.array([2.0, 3.0]),
        "regime_id": jnp.array([_RegimeId.active, _RegimeId.active]),
    }

    df = (
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=None,
        )
        .to_dataframe()
        .query('regime_name == "active"')
        .sort_values(["subject_id", "period"])
        .reset_index(drop=True)
    )

    for subject_id in df["subject_id"].unique():
        rows = df.loc[df["subject_id"] == subject_id].sort_values("period")
        for i in range(len(rows) - 1):
            prev = rows.iloc[i]
            curr = rows.iloc[i + 1]
            work = prev["labor_supply"] == "work"
            expected_next_aime = float(prev["aime"]) + (1.0 if work else 0.0)
            expected_next_wealth = (
                float(prev["wealth"])
                - float(prev["consumption"])
                + 0.1 * expected_next_aime
            )
            np.testing.assert_allclose(
                float(curr["aime"]), expected_next_aime, atol=1e-6
            )
            np.testing.assert_allclose(
                float(curr["wealth"]), expected_next_wealth, atol=1e-6
            )
