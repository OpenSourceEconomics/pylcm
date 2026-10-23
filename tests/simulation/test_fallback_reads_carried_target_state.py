"""A simulate-phase fallback reads a state the gated target carries in simulation.

A carried state `Phased(solve=imputation, simulate=Grid)` is a derived function
while the model solves and a genuine state while it simulates. A gate's
simulate-phase fallback projects the realized landing onto the fallback
regime's states, so it may read the target's carried state as the state it is
there, and the routed row keeps its exact value.
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    ByAge,
    DeterministicTransition,
    Gate,
    LinSpacedGrid,
    Model,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.transition import StochasticTransition
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_WEALTH = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)
_YEARS = LinSpacedGrid(start=0.0, stop=20.0, n_points=5)


@categorical(ordered=False)
class RegimeId:
    worker: ScalarInt
    retired: ScalarInt
    hardship: ScalarInt
    shelter: ScalarInt
    dead: ScalarInt


def _certain(wealth: ContinuousState) -> FloatND:
    return jnp.ones_like(wealth)


def _zero(wealth: ContinuousState) -> FloatND:
    return jnp.zeros_like(wealth)


def _wealth_payoff(wealth: ContinuousState) -> FloatND:
    return wealth


def _impute_years(wealth: ContinuousState) -> ContinuousState:
    """Solve-phase years: imputed from wealth, never a solve axis."""
    return 5.0 * wealth


def _keep_years(years: ContinuousState) -> ContinuousState:
    return years


def _well_off(wealth: ContinuousState) -> BoolND:
    return wealth > 1.5


def _whole_wealth(wealth: ContinuousState) -> ContinuousState:
    return wealth


def _half_wealth(wealth: ContinuousState) -> ContinuousState:
    return 0.5 * wealth


def _years_held(years: ContinuousState) -> ContinuousState:
    return years


def _to_dead(age: FloatND) -> ScalarInt:
    return jnp.full_like(age, RegimeId.dead, dtype=jnp.int32)


def _carried_years() -> Phased:
    return Phased(solve=_impute_years, simulate=_YEARS)


def _no_payoff() -> FloatND:
    return jnp.asarray(0.0)


_KEEP = {"wealth": fixed_transition("wealth"), "years": _keep_years}

_DYING = {
    name: Transition(targets={"dead": 1}, law=DeterministicTransition(func=_to_dead))
    for name in ("retired", "shelter")
}


def _make_model() -> Model:
    law = ByAge(cases={0: {"retired": StochasticTransition(func=_certain)}})
    gates = {
        "retired": Gate(
            predicate=_well_off,
            routes={
                "only": StakeholderRoute(
                    fallback=Phased(
                        solve=ProjectedRegimeValue(
                            regime="hardship", projection={"wealth": _whole_wealth}
                        ),
                        simulate=ProjectedRegimeValue(
                            regime="shelter",
                            projection={"wealth": _half_wealth, "years": _years_held},
                        ),
                    )
                )
            },
        )
    }
    return Model(
        regimes={
            "worker": Regime(
                states={"wealth": _WEALTH, "years": _carried_years()},
                state_transitions=_KEEP,
                functions={"utility": _zero},
            ),
            "retired": Regime(
                states={"wealth": _WEALTH, "years": _carried_years()},
                state_transitions=_KEEP,
                functions={"utility": _wealth_payoff},
            ),
            "hardship": Regime(
                states={"wealth": _WEALTH}, functions={"utility": _wealth_payoff}
            ),
            "shelter": Regime(
                states={"wealth": _WEALTH, "years": _carried_years()},
                state_transitions=_KEEP,
                functions={"utility": _wealth_payoff},
            ),
            "dead": Regime(functions={"utility": _no_payoff}),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "worker"},
        edges=Phased(
            solve={
                "worker": Transition(
                    targets={"retired": 0, "hardship": 0}, law=law, gates=gates
                ),
                **_DYING,
            },
            simulate={
                "worker": Transition(
                    targets={"retired": 0, "shelter": 0}, law=law, gates=gates
                ),
                **_DYING,
            },
        ),
    )


_PARAMS = {
    "worker": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "retired": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "hardship": {},
    "shelter": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "dead": {},
}


def test_a_routed_row_keeps_the_targets_carried_state() -> None:
    """The closed row lands in `shelter` with the 7 years it held, not an imputation."""
    model = _make_model()
    solution = model.solve(params=_PARAMS, log_level="off")
    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.asarray([1.0, 2.0]),
            "years": jnp.asarray([7.0, 3.0]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, model.regime_names_to_ids["worker"]),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(result.raw_results["shelter"][1].states["years"])[0],
        7.0,
        decimal=DECIMAL_PRECISION,
    )
