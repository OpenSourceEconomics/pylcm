"""Simulation stores the same physical entry values that solve prices.

Solve indexes a target's value function along a private node axis, but that is a
representation detail of the interpolation. The physical next-period values —
the entry itself and anything computed from it — are one law, so the two phases
must publish the same numbers.
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.typing import FloatND, ScalarFloat, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_ENTRY = 1.5
_DEPENDENT = 2.0 * _ENTRY


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _one_probability() -> FloatND:
    return jnp.asarray(1.0)


def _source_is_early(age: float) -> bool:
    return age < 22


def _enter_shock() -> ScalarFloat:
    return jnp.asarray(_ENTRY)


def _double_the_entry(next_shock: ScalarFloat) -> ScalarFloat:
    return 2.0 * next_shock


def _wealth_utility(wealth: ScalarFloat, shock: ScalarFloat) -> FloatND:
    return wealth + 0.0 * shock


PARAMS = {
    "source": {
        "utility": {},
        "koopmans_aggregator": {"discount_factor": 1.0},
        "target": {"next_regime": {}, "next_shock": {}, "next_wealth": {}},
    },
    "target": {"utility": {}},
}


def _build_model() -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions={
                    "shock": {"target": _enter_shock},
                    "wealth": {"target": _double_the_entry},
                },
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                states={
                    "shock": NormalIIDProcess(
                        n_points=3, gauss_hermite=False, mu=1.0, sigma=0.5, n_std=2.0
                    ),
                    "wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=11),
                },
                functions={"utility": _wealth_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )


def test_simulation_stores_the_physical_entry_and_its_dependent_state() -> None:
    """The simulated target row holds `shock = 1.5` and `wealth = 3.0`.

    A node index would be one of `(0, 1, 2)` for the entry and one of `(0, 2, 4)`
    for the law reading it, so either substitution is visible in the stored
    states rather than only in a value.
    """
    model = _build_model()
    result = model.simulate(
        params=PARAMS,
        initial_conditions={
            "regime_id": jnp.array([RegimeId.source]),
            "age": jnp.array([20.0]),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    df = result.to_dataframe()

    target_rows = df.query("regime_name == 'target'")
    np.testing.assert_almost_equal(
        target_rows["shock"].to_numpy(), _ENTRY, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_almost_equal(
        target_rows["wealth"].to_numpy(), _DEPENDENT, decimal=DECIMAL_PRECISION
    )


def test_solve_prices_the_state_simulation_stores() -> None:
    """The source's value equals the target utility at the state simulation stores.

    Terminal utility is `wealth`, so solve's continuation must be the same `3.0`
    that the simulated target row carries — one physical law, one number.
    """
    model = _build_model()
    solution = model.solve(params=PARAMS, log_level="debug")
    period = max(p for p in solution if "source" in solution[p])
    got = float(np.asarray(solution[period]["source"]).ravel()[0])

    np.testing.assert_almost_equal(got, _DEPENDENT, decimal=DECIMAL_PRECISION)
