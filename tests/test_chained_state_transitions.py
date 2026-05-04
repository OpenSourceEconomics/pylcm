"""End-to-end check that one state transition can consume another's output.

dags resolves dependencies between state-transition functions when they
appear in the merged transitions+functions dict that
`get_next_state_function_for_solution` builds. The blocker fixed here is in
the upstream `create_regime_params_template`: it must not classify
`next_<state>` names as regime-level fixed_params, otherwise param resolution
fails before dags ever runs.
"""

import jax.numpy as jnp

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import DiscreteAction, FloatND, ScalarInt


@categorical(ordered=False)
class _LaborSupply:
    work: int
    rest: int


@categorical(ordered=False)
class _RegimeId:
    active: int
    dead: int


def _next_aime(aime: float, labor_supply: DiscreteAction) -> FloatND:
    """AIME accumulates only when working."""
    return aime + jnp.where(labor_supply == _LaborSupply.work, 1.0, 0.0)


def _next_wealth(wealth: float, consumption: float, next_aime: FloatND) -> FloatND:
    """Next-period wealth depends on next-period AIME (the chained transition).

    The economically interesting use is `pia = f(next_aime)` feeding
    next_wealth via a pension correction. Here we keep the dependency simple
    so the test focuses on the wiring, not the economics.
    """
    return wealth - consumption + 0.1 * next_aime


def _utility(consumption: float, labor_supply: DiscreteAction) -> FloatND:
    disutility = jnp.where(labor_supply == _LaborSupply.work, 0.5, 0.0)
    return jnp.log(jnp.maximum(consumption, 1e-6)) - disutility


def _next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, _RegimeId.dead, _RegimeId.active)


_active = Regime(
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


_dead = Regime(transition=None, functions={"utility": lambda: jnp.array(0.0)})


def _build_model() -> Model:
    return Model(
        regimes={"active": _active, "dead": _dead},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
    )


def test_solve_resolves_chain_via_dags() -> None:
    """`solve()` runs and dags wires `next_aime → next_wealth` correctly.

    Before the fix, `_resolve_fixed_params` raised
    `InvalidParamsError: Missing required parameter:
    'active__next_wealth__next_aime'` because `create_regime_params_template`
    classified `next_aime` (a `next_<state>` reference inside another
    transition's signature) as a regime-level fixed_param.
    """
    model = _build_model()
    params = {
        "discount_factor": 0.9,
        "final_age_alive": 1.0,
    }
    period_to_regime_to_V_arr = model.solve(params=params)
    for regime_to_V_arr in period_to_regime_to_V_arr.values():
        for V_arr in regime_to_V_arr.values():
            assert not jnp.any(jnp.isnan(V_arr))
